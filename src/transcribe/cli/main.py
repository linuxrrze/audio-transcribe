"""`tx` command line interface."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from transcribe import __version__
from transcribe.cli.speakers import speakers_app
from transcribe.core import models, registry
from transcribe.core.config import Config
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Transcript

app = typer.Typer(help="Local live transcription with speaker identification.",
                  no_args_is_help=True)
models_app = typer.Typer(help="List and download models.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(speakers_app, name="speakers")
console = Console()
err = Console(stderr=True)  # status/progress; keeps stdout clean for piping

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


ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Whisper model.")]
LangOpt = Annotated[str | None, typer.Option(
    "--lang", "-l", help="Allowed languages, comma-separated (e.g. de,en or de).")]
SpeakersOpt = Annotated[int | None, typer.Option(
    "--speakers", "-n", min=1, help="Number of speakers, if known (most reliable).")]
DiarizeOpt = Annotated[bool, typer.Option(help="Identify speakers.")]
ThresholdOpt = Annotated[float | None, typer.Option(
    "--diarize-threshold", help="Clustering threshold; lower finds more speakers.")]
DeviceOpt = Annotated[str | None, typer.Option(help="auto | cpu | cuda")]
FormatOpt = Annotated[str, typer.Option("--format", "-f", help="Output format.")]
OutOpt = Annotated[Path | None, typer.Option("--out", "-o", help="Write output here.")]


def _config(model: str | None = None, device: str | None = None, lang: str | None = None,
            threshold: float | None = None) -> Config:
    config = Config.load()
    if model:
        config.asr.model = model
    if device:
        config.asr.device = device
    if threshold is not None:
        config.speakers.diarize_threshold = threshold
    if lang:
        config.asr.languages = [x.strip() for x in lang.split(",") if x.strip()]
    return config


@contextmanager
def _stage_progress(stages: list[str]) -> Iterator[Callable[[str, float], None]]:
    columns = (TextColumn("{task.description:<11}"), BarColumn(),
               TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn())
    with Progress(*columns, console=err) as progress:
        tasks = {name: progress.add_task(name, total=1.0) for name in stages}

        def report(stage: str, frac: float) -> None:
            if stage in tasks:
                progress.update(tasks[stage], completed=frac)

        yield report


def _offline_stages(config: Config, diarize: bool) -> list[str]:
    stages = ["diarize"] if diarize else []
    if len(config.asr.languages) > 1:
        stages.append("language")
    return [*stages, "transcribe"]


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Audio/video file.")],
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    diarize: DiarizeOpt = True,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
    fmt: FormatOpt = "txt",
    out: OutOpt = None,
    save_session: Annotated[bool, typer.Option(
        "--session/--no-session", help="Store a session folder.")] = True,
) -> None:
    """Transcribe an audio file with speaker diarization."""
    from transcribe.audio.file import load_audio
    from transcribe.core.offline import load_engines, transcribe_audio

    config = _config(model, device, lang, threshold)
    exporter = registry.load("transcribe.exporters", fmt)()
    with err.status("Loading models (first run downloads them)..."):
        asr, diarizer = load_engines(config, diarize)
        audio = load_audio(file)
    with _stage_progress(_offline_stages(config, diarize)) as report:
        transcript = transcribe_audio(audio, asr, diarizer, config.asr.languages,
                                      num_speakers, report)

    transcript.meta.update(title=file.stem, source=str(file.resolve()),
                           asr_model=config.asr.model)
    path = (session_store.new_session_dir(config.resolved_sessions_dir(), file.stem)
            if save_session else None)
    if diarize and transcript.speakers:
        _identify(config, path, audio, transcript)
    if path is not None:
        session_store.save_session(path, transcript, {
            "source": str(file.resolve()), "asr_engine": config.asr.engine,
            "asr_model": config.asr.model, "languages": config.asr.languages,
            "diarizer": config.speakers.diarizer if diarize else None,
            "embedding_model": config.speakers.embedding_model if diarize else None,
        })
        err.print(f"Session saved to [bold]{path}[/]")
    _emit(exporter.render(transcript), out)


@app.command()
def live(
    file: Annotated[Path | None, typer.Option(
        "--file", exists=True, dir_okay=False,
        help="Replay a file instead of the microphone (for testing).")] = None,
    fast: Annotated[bool, typer.Option(help="Replay --file as fast as possible.")] = False,
    source_kind: Annotated[str, typer.Option(
        "--source", "-s", help="mic | system (what the computer plays) | both")] = "mic",
    input_device: Annotated[str | None, typer.Option(
        "--input", "-i", help="Input device index or name (see `tx devices`).")] = None,
    output_device: Annotated[str | None, typer.Option(
        "--output", help="System audio device (see `tx devices`); default: default output.")]
    = None,
    mix: Annotated[bool, typer.Option(
        help="With --source both: sum mic+system into one stream (cheaper, no echo "
             "suppression or channel separation).")] = False,
    name: Annotated[str, typer.Option(help="Session name.")] = "live",
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    diarize: DiarizeOpt = True,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
    refine: Annotated[bool, typer.Option(
        help="Re-run the full pipeline on the recording when stopped.")] = True,
    fmt: FormatOpt = "txt",
    out: OutOpt = None,
) -> None:
    """Transcribe live from the microphone; Ctrl+C stops (and refines)."""
    from rich.live import Live

    from transcribe.audio.factory import build_source
    from transcribe.cli.live_view import LiveView
    from transcribe.live.factory import build_live_session

    config = _config(model, device, lang, threshold)
    exporter = registry.load("transcribe.exporters", fmt)()
    try:
        source = build_source(source_kind, input_device, output_device, mix, file,
                              realtime=not fast)
    except (RuntimeError, ValueError) as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    if file is not None and name == "live":
        name = file.stem
    session_dir = session_store.new_session_dir(config.resolved_sessions_dir(), name)
    with err.status("Loading models (first run downloads them)..."):
        session = build_live_session(source, config, session_dir, num_speakers, diarize,
                                     meta={"title": name, "source": source.describe().name})
    view = LiveView()
    session.bus.subscribe(view)
    try:
        session.start()
    except RuntimeError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    with Live(view, console=err, refresh_per_second=8, transient=False):
        try:
            while not session.wait(0.2):
                pass
        except KeyboardInterrupt:
            pass
        try:
            transcript = session.stop(refine=refine)
        except KeyboardInterrupt:
            err.print("[yellow]Refinement aborted; live transcript kept.[/]")
            transcript = Transcript.load(session_dir / "transcript.live.json")
    err.print(f"Session saved to [bold]{session_dir}[/]")
    _emit(exporter.render(transcript), out)


@app.command()
def refine(
    session: Annotated[Path, typer.Argument(exists=True, file_okay=False,
                                            help="Session folder.")],
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
) -> None:
    """Re-run full diarization + transcription on a session's recording."""
    from transcribe.core.offline import load_engines, transcribe_audio
    from transcribe.live.session import refine_channels
    from transcribe.storage.recorder import load_recording, recorded_channels

    config = _config(model, device, lang, threshold)
    audio = {ch: load_recording(session, ch) for ch in recorded_channels(session)}
    audio = {ch: a for ch, a in audio.items() if len(a)}
    if not audio:
        err.print(f"[red]No recording in {session}[/]")
        raise typer.Exit(1)
    path = session_store.resolve_transcript(session)
    previous = Transcript.load(path) if path.exists() else Transcript()
    with err.status("Loading models..."):
        asr, diarizer = load_engines(config, diarize=True)

    def run(signal, report):
        return transcribe_audio(signal, asr, diarizer, config.asr.languages, num_speakers,
                                report)

    stages = [f"{ch} {st}".strip() for ch in audio for st in _offline_stages(config, True)]
    with _stage_progress(stages) as report:
        final = refine_channels(audio, run, previous, report)
    final.meta["asr_model"] = config.asr.model
    if final.speakers:
        _identify(config, session, audio if len(audio) > 1 else next(iter(audio.values())),
                  final)
    final.save(path)
    err.print(f"Refined transcript saved to [bold]{path}[/]")


def _identify(config: Config, session_dir: Path | None, audio, transcript: Transcript) -> None:
    """Store exemplars and label speakers that match known voiceprints."""
    from transcribe.speakers.embedder import SherpaEmbedder
    from transcribe.speakers.identity import process_session

    if not (config.speakers.identify_known or config.speakers.store_voiceprints):
        return
    with err.status("Identifying speakers..."):
        matches = process_session(session_dir, audio, transcript,
                                  SherpaEmbedder.from_config(config), config)
    for sid, m in matches.items():
        if m.decision.value != "unknown":
            verb = "is" if m.decision.value == "known" else "might be"
            err.print(f"{sid} {verb} [bold]{m.name}[/] (score {m.score:.2f})")


@app.command()
def devices() -> None:
    """List microphones and system-audio (output) devices."""
    from transcribe.audio.mic import list_input_devices
    from transcribe.audio.system import output_devices

    try:
        devs = list_input_devices()
    except RuntimeError as e:
        err.print(f"[red]{e}[/]")
        devs = []
    table = Table("Index", "Microphone / input", "Channels", "Rate", "Host API", "Default",
                  title="Inputs (--input)")
    for d in devs:
        table.add_row(str(d.index), d.name, str(d.channels), f"{d.default_samplerate:.0f}",
                      d.hostapi, "*" if d.is_default else "")
    console.print(table)
    outs = Table("ID", "System audio source", "Default", title="System audio (--output)")
    for o in output_devices():
        outs.add_row(o.id, o.name, "*" if o.is_default else "")
    console.print(outs)


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


if __name__ == "__main__":
    app()
