"""`tx` command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from transcribe import __version__
from transcribe.core import models, registry
from transcribe.core.config import Config, data_dir
from transcribe.speakers.voiceprint_db import VoiceprintDB
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Transcript

app = typer.Typer(help="Local live transcription with speaker identification.",
                  no_args_is_help=True)
speakers_app = typer.Typer(help="Manage known speakers (voiceprints).", no_args_is_help=True)
models_app = typer.Typer(help="List and download models.", no_args_is_help=True)
app.add_typer(speakers_app, name="speakers")
app.add_typer(models_app, name="models")
console = Console()
err = Console(stderr=True)  # status/progress; keeps stdout clean for piping

DbOption = Annotated[Path | None, typer.Option("--db", help="Voiceprint database path.")]


def _db(path: Path | None) -> VoiceprintDB:
    return VoiceprintDB(path or data_dir() / "voiceprints.sqlite")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command()
def plugins() -> None:
    """List installed plugins per group."""
    for group in registry.GROUPS:
        names = registry.available(group)
        console.print(f"[bold]{group}[/]: {', '.join(names) or '-'}")


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Audio/video file.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Whisper model.")] = None,
    lang: Annotated[str | None, typer.Option(
        "--lang", "-l", help="Allowed languages, comma-separated (e.g. de,en or de).")] = None,
    num_speakers: Annotated[int | None, typer.Option(
        "--speakers", "-n", min=1,
        help="Number of speakers, if known (most reliable diarization).")] = None,
    diarize: Annotated[bool, typer.Option(help="Identify speakers.")] = True,
    threshold: Annotated[float | None, typer.Option(
        "--diarize-threshold", help="Clustering threshold; lower finds more speakers.")] = None,
    device: Annotated[str | None, typer.Option(help="auto | cpu | cuda")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format.")] = "txt",
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write output here.")] = None,
    save_session: Annotated[bool, typer.Option(
        "--session/--no-session", help="Store a session folder.")] = True,
) -> None:
    """Transcribe an audio file with speaker diarization."""
    from transcribe.audio.file import load_audio
    from transcribe.core.offline import load_engines, transcribe_audio

    config = Config.load()
    if model:
        config.asr.model = model
    if device:
        config.asr.device = device
    if threshold is not None:
        config.speakers.diarize_threshold = threshold
    if lang:
        config.asr.languages = [x.strip() for x in lang.split(",") if x.strip()]
    exporter = registry.load("transcribe.exporters", fmt)()

    with err.status("Loading models (first run downloads them)..."):
        asr, diarizer = load_engines(config, diarize)
        audio = load_audio(file)

    columns = (TextColumn("{task.description:<11}"), BarColumn(), TextColumn(
        "{task.percentage:>3.0f}%"), TimeElapsedColumn())
    with Progress(*columns, console=err, transient=False) as progress:
        stages = ["diarize"] if diarize else []
        if len(config.asr.languages) > 1:
            stages.append("language")
        stages.append("transcribe")
        tasks = {name: progress.add_task(name, total=1.0) for name in stages}

        def report(stage: str, frac: float) -> None:
            if stage in tasks:
                progress.update(tasks[stage], completed=frac)

        transcript = transcribe_audio(audio, asr, diarizer, config.asr.languages,
                                      num_speakers, report)

    transcript.meta.update(title=file.stem, source=str(file.resolve()),
                           asr_model=config.asr.model)
    if save_session:
        path = session_store.new_session_dir(config.resolved_sessions_dir(), file.stem)
        session_store.save_session(path, transcript, {
            "source": str(file.resolve()), "asr_engine": config.asr.engine,
            "asr_model": config.asr.model, "languages": config.asr.languages,
            "diarizer": config.speakers.diarizer if diarize else None,
            "embedding_model": config.speakers.embedding_model if diarize else None,
        })
        err.print(f"Session saved to [bold]{path}[/]")
    _emit(exporter.render(transcript), out)


@app.command()
def export(
    session: Annotated[Path, typer.Argument(help="Session folder or transcript.json.")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="Exporter name.")] = "txt",
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Export a transcript to another format."""
    transcript = Transcript.load(session_store.resolve_transcript(session))
    exporter = registry.load("transcribe.exporters", fmt)()
    _emit(exporter.render(transcript), out)


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        console.out(text, end="", highlight=False)
    else:
        out.write_text(text, encoding="utf-8")
        err.print(f"Wrote {out}")


@models_app.command("list")
def models_list() -> None:
    """List downloadable models and whether they are cached."""
    table = Table("Name", "Kind", "Cached", "Description")
    for a in models.ASSETS.values():
        table.add_row(a.name, a.kind, "yes" if models.is_present(a.name) else "-", a.description)
    console.print(table)
    console.print(f"Cache: {models.models_dir()}  (Whisper models are fetched on first use)")


@models_app.command("download")
def models_download(
    names: Annotated[list[str] | None, typer.Argument(help="Asset names; default: defaults.")]
    = None,
) -> None:
    """Download model assets (default: the configured diarization models)."""
    config = Config.load()
    for name in names or [config.speakers.segmentation_model, config.speakers.embedding_model]:
        if name not in models.ASSETS:
            err.print(f"[red]Unknown model {name}[/]")
            raise typer.Exit(1)
        with Progress(console=err, transient=True) as progress:
            task = progress.add_task(name, total=None)

            def on_bytes(done: int, total: int, task: TaskID = task) -> None:
                progress.update(task, completed=done, total=total or None)

            models.ensure(name, on_bytes)
        err.print(f"{name}: {models.ensure(name)}")


@speakers_app.command("list")
def speakers_list(db: DbOption = None) -> None:
    """List known persons."""
    store = _db(db)
    table = Table("ID", "Name", "Consent", "Created")
    for p in store.persons():
        table.add_row(str(p.id), p.display_name, p.consent_at or "-", p.created_at)
    console.print(table)


@speakers_app.command("add")
def speakers_add(
    name: str,
    consent: Annotated[bool, typer.Option(help="Person consented to voiceprint storage.")] = False,
    db: DbOption = None,
) -> None:
    """Create a person entry (voiceprints are added via enrollment)."""
    p = _db(db).add_person(name, consent=consent)
    console.print(f"Added {p.display_name} (id {p.id})")


@speakers_app.command("rename")
def speakers_rename(old: str, new: str, db: DbOption = None) -> None:
    store = _db(db)
    p = _require(store, old)
    store.rename_person(p.id, new)
    console.print(f"Renamed {old} -> {new}")


@speakers_app.command("merge")
def speakers_merge(keep: str, drop: str, db: DbOption = None) -> None:
    """Move all voiceprints of DROP to KEEP and delete DROP."""
    store = _db(db)
    store.merge_persons(_require(store, keep).id, _require(store, drop).id)
    console.print(f"Merged {drop} into {keep}")


@speakers_app.command("delete")
def speakers_delete(
    name: str,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    db: DbOption = None,
) -> None:
    """Delete a person and all of their voiceprints."""
    store = _db(db)
    p = _require(store, name)
    if not yes:
        typer.confirm(f"Delete {name} and all voiceprints?", abort=True)
    store.delete_person(p.id)
    console.print(f"Deleted {name}")


def _require(store: VoiceprintDB, name: str):
    p = store.find_person(name)
    if p is None:
        console.print(f"[red]Unknown person: {name}[/]")
        raise typer.Exit(1)
    return p


if __name__ == "__main__":
    app()
