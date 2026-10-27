"""`tx` command line interface."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from typer import rich_utils

from transcribe import __version__
from transcribe.cli.live_view import stage_label
from transcribe.cli.sessions import sessions_app
from transcribe.cli.speakers import speakers_app
from transcribe.core import models, registry
from transcribe.core.config import Config, vocabulary
from transcribe.i18n import _
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Transcript

# typer's help screen: panel titles and markers in the UI language ("Usage:" and the
# --help text come from click and stay English)
rich_utils.ARGUMENTS_PANEL_TITLE = _("Arguments")
rich_utils.OPTIONS_PANEL_TITLE = _("Options")
rich_utils.COMMANDS_PANEL_TITLE = _("Commands")
rich_utils.REQUIRED_LONG_STRING = _("[required]")
rich_utils.DEFAULT_STRING = _("[default: {}]")
rich_utils.ENVVAR_STRING = _("[env var: {}]")

app = typer.Typer(help=_("Local live transcription with speaker identification."),
                  no_args_is_help=True)
models_app = typer.Typer(help=_("List and download models."), no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(speakers_app, name="speakers")
app.add_typer(sessions_app, name="sessions")
console = Console()
err = Console(stderr=True)  # status/progress; keeps stdout clean for piping

@app.command(help=_("Print the version."))
def version() -> None:
    console.print(__version__)


@app.command(help=_("Open the desktop app."))
def gui() -> None:
    from transcribe.gui.app import main

    raise typer.Exit(main())


@app.command(help=_("List installed plugins per group."))
def plugins() -> None:
    for group in registry.GROUPS:
        names = registry.available(group)
        console.print(f"[bold]{group}[/]: {', '.join(names) or '-'}")


ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help=_("Whisper model."))]
LangOpt = Annotated[str | None, typer.Option(
    "--lang", "-l", help=_("Allowed languages, comma-separated (e.g. de,en or de)."))]
SpeakersOpt = Annotated[int | None, typer.Option(
    "--speakers", "-n", min=1, help=_("Number of speakers, if known (most reliable)."))]
DiarizeOpt = Annotated[bool, typer.Option(help=_("Identify speakers."))]
ThresholdOpt = Annotated[float | None, typer.Option(
    "--diarize-threshold", help=_("Clustering threshold; lower finds more speakers."))]
DeviceOpt = Annotated[str | None, typer.Option(
    help="auto | cpu | cuda (faster-whisper) | gpu (whisper.cpp)")]
EngineOpt = Annotated[str | None, typer.Option(
    "--engine",
    help=_("ASR engine: faster-whisper | whisper.cpp (AMD/Intel/Apple GPUs)."))]
FormatOpt = Annotated[str, typer.Option("--format", "-f", help=_("Output format."))]
OutOpt = Annotated[Path | None, typer.Option("--out", "-o", help=_("Write output here."))]
VocabOpt = Annotated[str | None, typer.Option(
    "--vocab",
    help=_("Names/acronyms to recognise, comma-separated (adds to the config)."))]
VocabFileOpt = Annotated[Path | None, typer.Option(
    "--vocab-file", exists=True, dir_okay=False,
    help=_("Vocabulary file: one term per line, 'wrong -> right' for corrections."))]


def _config(model: str | None = None, device: str | None = None, lang: str | None = None,
            threshold: float | None = None, vocab: str | None = None,
            vocab_file: Path | None = None, engine: str | None = None) -> Config:
    from transcribe.core.vocabulary import Vocabulary

    config = Config.load()
    extra = Vocabulary.parse(vocab.replace(",", "\n")) if vocab else Vocabulary()
    if vocab_file:
        extra = extra.merged(Vocabulary.load(vocab_file))
    if extra:
        merged = Vocabulary(config.asr.vocabulary, config.asr.corrections).merged(extra)
        config.asr.vocabulary, config.asr.corrections = merged.terms, merged.corrections
    if engine:
        config.asr.engine = engine
    if model:
        config.asr.model = model
    if device:
        config.asr.device = device
    if threshold is not None:
        config.speakers.diarize_threshold = threshold
    if lang:
        config.asr.languages = [x.strip() for x in lang.split(",") if x.strip()]
    return config


class _PercentColumn(TaskProgressColumn):
    """Percentage, or nothing while the total is unknown (pulsing bar)."""

    def render(self, task):
        return Text("") if task.total is None else super().render(task)


@contextmanager
def _stage_progress(stages: list[str]) -> Iterator[Callable[[str, float], None]]:
    """One bar per stage; a stage's timer starts when it reports for the first time.

    Diarization reports nothing while it searches for speech (minutes for long files),
    so it starts as a pulsing bar ("finding speech…") until real progress arrives.
    """
    columns = (TextColumn("{task.description:<26}"), BarColumn(), _PercentColumn(),
               TimeElapsedColumn())
    with Progress(*columns, console=err) as progress:
        tasks = {name: progress.add_task(stage_label(name), total=1.0, start=False)
                 for name in stages}
        started: set[str] = set()

        def start(stage: str) -> None:
            if stage in started:
                return
            started.add(stage)
            if stage.endswith("diarize"):
                progress.update(tasks[stage], total=None,
                                description=_("{stage} (finding speech…)").format(
                                    stage=stage_label(stage)))
            progress.start_task(tasks[stage])

        if stages:
            start(stages[0])

        def report(stage: str, frac: float) -> None:
            if stage not in tasks:
                return
            for earlier in stages[: stages.index(stage)]:  # previous stages are done
                if earlier in started:
                    progress.update(tasks[earlier], total=1.0, completed=1.0,
                                    description=stage_label(earlier))
            start(stage)
            if frac > 0:
                progress.update(tasks[stage], total=1.0, completed=frac,
                                description=stage_label(stage))

        yield report


def _offline_stages(config: Config, diarize: bool) -> list[str]:
    stages = ["diarize"] if diarize else []
    if len(config.asr.languages) > 1:
        stages.append("language")
    stages.append("transcribe")
    if diarize and config.speakers.resegment:
        stages.append("speakers")
    return stages


@app.command(help=_("Transcribe an audio file with speaker diarization."))
def transcribe(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False,
                                         help=_("Audio/video file."))],
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    diarize: DiarizeOpt = True,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
    engine: EngineOpt = None,
    vocab: VocabOpt = None,
    vocab_file: VocabFileOpt = None,
    fmt: FormatOpt = "txt",
    out: OutOpt = None,
    save_session: Annotated[bool, typer.Option(
        "--session/--no-session", help=_("Store a session folder."))] = True,
    keep_audio: Annotated[bool | None, typer.Option(
        "--keep-audio/--no-keep-audio",
        help=_("Store the audio in the session folder, so `tx refine` can re-run it later "
               "(default: keep_file_audio in the config)."))] = None,
) -> None:
    from transcribe.audio.file import load_audio
    from transcribe.core.offline import load_engines, transcribe_audio

    config = _config(model, device, lang, threshold, vocab, vocab_file, engine)
    exporter = registry.load("transcribe.exporters", fmt)()
    with err.status(_("Loading models (first run downloads them)...")):
        asr, diarizer = load_engines(config, diarize)
        audio = load_audio(file)
    with _stage_progress(_offline_stages(config, diarize)) as report:
        transcript = transcribe_audio(audio, asr, diarizer, config.asr.languages,
                                      num_speakers, report, vocabulary=vocabulary(config))

    transcript.meta.update(title=file.stem, source=str(file.resolve()),
                           asr_model=config.asr.model)
    path = (session_store.new_session_dir(config.resolved_sessions_dir(), file.stem)
            if save_session else None)
    if diarize and transcript.speakers:
        _identify(config, path, audio, transcript)
    if path is not None:
        if config.keep_file_audio if keep_audio is None else keep_audio:
            from transcribe.storage.recorder import save_recording

            save_recording(path, audio)
        session_store.save_session(path, transcript, {
            "source": str(file.resolve()), "asr_engine": config.asr.engine,
            "asr_model": config.asr.model, "languages": config.asr.languages,
            "diarizer": config.speakers.diarizer if diarize else None,
            "embedding_model": config.speakers.embedding_model if diarize else None,
        })
        err.print(_("Session saved to {path}").format(path=f"[bold]{path}[/]"))
    _emit(exporter.render(transcript), out)


@app.command(help=_("Transcribe live from the microphone; Ctrl+C stops (and refines)."))
def live(
    file: Annotated[Path | None, typer.Option(
        "--file", exists=True, dir_okay=False,
        help=_("Replay a file instead of the microphone (for testing)."))] = None,
    fast: Annotated[bool, typer.Option(help=_("Replay --file as fast as possible."))] = False,
    source_kind: Annotated[str, typer.Option(
        "--source", "-s", help=_("mic | system (what the computer plays) | both"))] = "mic",
    input_device: Annotated[str | None, typer.Option(
        "--input", "-i", help=_("Input device index or name (see `tx devices`)."))] = None,
    output_device: Annotated[str | None, typer.Option(
        "--output",
        help=_("System audio device, or one application as app:<name or pid> (see `tx "
               "devices`); default: default output."))]
    = None,
    mix: Annotated[bool, typer.Option(
        help=_("With --source both: sum mic+system into one stream (cheaper, no echo "
               "suppression or channel separation)."))] = False,
    name: Annotated[str, typer.Option(help=_("Session name."))] = "live",
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    diarize: DiarizeOpt = True,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
    engine: EngineOpt = None,
    vocab: VocabOpt = None,
    vocab_file: VocabFileOpt = None,
    refine: Annotated[bool, typer.Option(
        help=_("Re-run the full pipeline on the recording when stopped."))] = True,
    fmt: FormatOpt = "txt",
    out: OutOpt = None,
) -> None:
    from rich.live import Live

    from transcribe.audio.apps import APP_PREFIX, app_available
    from transcribe.audio.factory import build_source
    from transcribe.cli.live_view import LiveView
    from transcribe.live.factory import build_live_session

    config = _config(model, device, lang, threshold, vocab, vocab_file, engine)
    exporter = registry.load("transcribe.exporters", fmt)()
    try:
        source = build_source(source_kind, input_device, output_device, mix, file,
                              realtime=not fast)
    except (RuntimeError, ValueError) as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    if (file is None and source_kind != "mic" and output_device
            and output_device.startswith(APP_PREFIX)
            and not app_available(output_device[len(APP_PREFIX):])):
        err.print(f"[yellow]{_('{app} has no audio output yet; its channel stays silent '
                               'until it plays.').format(app=output_device)}[/]")
    if file is not None and name == "live":
        name = file.stem
    session_dir = session_store.new_session_dir(config.resolved_sessions_dir(), name)
    with err.status(_("Loading models (first run downloads them)...")):
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
            err.print(f"[yellow]{_('Refinement aborted; live transcript kept.')}[/]")
            transcript = Transcript.load(session_dir / "transcript.live.json")
    err.print(_("Session saved to {path}").format(path=f"[bold]{session_dir}[/]"))
    _emit(exporter.render(transcript), out)


@app.command(help=_("Re-run full diarization + transcription on a session's recording."))
def refine(
    session: Annotated[Path, typer.Argument(exists=True, file_okay=False,
                                            help=_("Session folder."))],
    model: ModelOpt = None,
    lang: LangOpt = None,
    num_speakers: SpeakersOpt = None,
    threshold: ThresholdOpt = None,
    device: DeviceOpt = None,
    engine: EngineOpt = None,
    vocab: VocabOpt = None,
    vocab_file: VocabFileOpt = None,
) -> None:
    from transcribe.sessions import refine_session, session_audio

    config = _config(model, device, lang, threshold, vocab, vocab_file, engine)
    try:  # the audio first: a missing recording is reported before any progress bars
        audio = session_audio(session, keep=config.keep_file_audio)
        stages = [f"{ch} {st}".strip() if len(audio) > 1 else st
                  for ch in audio for st in _offline_stages(config, True)]
        with err.status(_("Loading models...")), _stage_progress(stages) as report:
            refine_session(session, config, num_speakers, report, audio=audio)
    except FileNotFoundError as e:
        err.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    err.print(_("Refined transcript saved to {path}").format(
        path=f"[bold]{session_store.resolve_transcript(session)}[/]"))


def _identify(config: Config, session_dir: Path | None, audio, transcript: Transcript) -> None:
    """Store exemplars and label speakers that match known voiceprints."""
    from transcribe.speakers.embedder import SherpaEmbedder
    from transcribe.speakers.identity import process_session

    if not (config.speakers.identify_known or config.speakers.store_voiceprints):
        return
    with err.status(_("Identifying speakers...")):
        matches = process_session(session_dir, audio, transcript,
                                  SherpaEmbedder.from_config(config), config)
    for sid, m in matches.items():
        if m.decision.value != "unknown":
            text = (_("{speaker} is {name} (score {score})") if m.decision.value == "known"
                    else _("{speaker} might be {name} (score {score})"))
            err.print(text.format(speaker=sid, name=f"[bold]{m.name}[/]",
                                  score=f"{m.score:.2f}"))


@app.command(help=_("List microphones, system-audio (output) devices and applications "
                    "playing audio."))
def devices() -> None:
    import platform

    from transcribe.audio.apps import list_audio_apps
    from transcribe.audio.mic import list_input_devices
    from transcribe.audio.system import output_devices

    try:
        devs = list_input_devices()
    except RuntimeError as e:
        err.print(f"[red]{e}[/]")
        devs = []
    table = Table(_("Index"), _("Microphone / input"), _("Channels"), _("Rate"),
                  _("Host API"), _("Default"), title=_("Inputs (--input)"))
    for d in devs:
        table.add_row(str(d.index), d.name, str(d.channels), f"{d.default_samplerate:.0f}",
                      d.hostapi, "*" if d.is_default else "")
    console.print(table)
    outs = Table(_("ID"), _("System audio source"), _("Default"),
                 title=_("System audio (--output)"))
    for o in output_devices():
        outs.add_row(o.id, o.name, "*" if o.is_default else "")
    console.print(outs)
    if platform.system() == "Windows":
        err.print(_("Capturing a single application is not supported on Windows yet."))
        return
    apps = Table(_("ID"), _("Application"), _("PID"), _("Playing"),
                 title=_("Applications (--output app:<name or pid>)"))
    for a in list_audio_apps():
        apps.add_row(a.device, a.name, str(a.pid or ""), "*" if a.playing else "")
    console.print(apps)


@app.command(help=_(
    "Show live signal levels of microphones and system-audio devices.\n\n"
    "Speak, and play something in the meeting app: the devices that move are the ones to "
    "pass to `tx live --input ... --output ...`."))
def levels(
    input_device: Annotated[list[str] | None, typer.Option(
        "--input", "-i", help=_("Only these inputs (index or name); repeatable."))] = None,
    output_device: Annotated[list[str] | None, typer.Option(
        "--output",
        help=_("Only these system-audio devices (see `tx devices`); repeatable."))]
    = None,
    seconds: Annotated[float | None, typer.Option(
        help=_("Stop after this many seconds (default: until Ctrl+C)."))] = None,
) -> None:
    import time

    from rich.live import Live

    from transcribe.audio.levels import DeviceEntry, LevelMonitor, device_source, list_devices
    from transcribe.cli.levels import levels_table

    entries, errors = list_devices()
    for e in errors:
        err.print(f"[yellow]{e}[/]")
    if input_device or output_device:
        from transcribe.audio.mic import resolve_device

        wanted_in = set()
        for d in input_device or []:
            try:
                wanted_in.add(str(resolve_device(d)))
            except ValueError as e:
                err.print(f"[red]{e}[/]")
                raise typer.Exit(1) from e
        known_out = {e.device for e in entries if e.kind == "output"}
        entries = [e for e in entries
                   if (e.kind == "input" and e.device in wanted_in)
                   or (e.kind == "output" and e.device in (output_device or []))]
        for o in output_device or []:  # allow ids not listed (e.g. a pid on macOS)
            if o not in known_out:
                entries.append(DeviceEntry(f"out:{o}", "output", o, o, False))
    if not entries:
        err.print(f"[red]{_('No audio devices found.')}[/]")
        raise typer.Exit(1)
    rows = [(e.key, f"{'🎤' if e.kind == 'input' else '🔊'} {e.name}", e.is_default)
            for e in entries]
    monitor = LevelMonitor({e.key: device_source(e) for e in entries}).start()
    end = time.monotonic() + seconds if seconds else None
    title = _("Audio levels - 🎤 inputs (--input), 🔊 system audio (--output); * = default")
    try:
        with Live(levels_table(rows, monitor.snapshot(), title), console=console,
                  refresh_per_second=10) as live:
            while end is None or time.monotonic() < end:
                live.update(levels_table(rows, monitor.snapshot(), title))
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()


@app.command(help=_("Export a transcript to another format."))
def export(
    session: Annotated[Path, typer.Argument(help=_("Session folder or transcript.json."))],
    fmt: Annotated[str, typer.Option("--format", "-f", help=_("Exporter name."))] = "txt",
    out: Annotated[Path | None, typer.Option("--out", "-o", help=_("Write output here."))]
    = None,
) -> None:
    transcript = Transcript.load(session_store.resolve_transcript(session))
    exporter = registry.load("transcribe.exporters", fmt)()
    _emit(exporter.render(transcript), out)


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        console.out(text, end="", highlight=False)
    else:
        out.write_text(text, encoding="utf-8")
        err.print(_("Wrote {path}").format(path=out))


@models_app.command("list", help=_("List downloadable models and whether they are cached."))
def models_list() -> None:
    table = Table(_("Name"), _("Kind"), _("Cached"), _("Description"))
    for a in models.ASSETS.values():
        table.add_row(a.name, a.kind, _("yes") if models.is_present(a.name) else "-",
                      a.description)
    console.print(table)
    from transcribe.core.config import required_models

    whisper, _assets = required_models(Config.load())
    for w in dict.fromkeys(whisper):
        state = _("yes") if models.model_present(w) else "-"
        console.print(_("{model}  cached: {state}").format(
            model=w if ":" in w else "whisper:" + w, state=state))
    console.print(_("Cache: {path}").format(path=models.models_dir()))


@models_app.command(
    "download",
    help=_("Download models (default: the configured Whisper and diarization models)."))
def models_download(
    names: Annotated[list[str] | None, typer.Argument(
        help=_("Asset names, whisper:<model> or ggml:<model>; default: everything the "
               "config needs."))] = None,
) -> None:
    from transcribe.core.config import required_models

    config = Config.load()
    if not names:
        whisper, assets = required_models(config)
        names = models.missing_models(whisper, assets)
        if not names:
            err.print(_("All configured models are present."))
            return
    for name in names:
        if not name.startswith(("whisper:", "ggml:")) and name not in models.ASSETS:
            err.print(f"[red]{_('Unknown model {name}').format(name=name)}[/]")
            raise typer.Exit(1)
    _download(names)


def _download(names: list[str]) -> None:
    """Download with one byte-progress bar per file; Ctrl+C cancels."""
    from rich.progress import DownloadColumn

    columns = (TextColumn("{task.description:<28}"), BarColumn(), _PercentColumn(),
               DownloadColumn())
    with Progress(*columns, console=err) as progress:
        tasks = {n: progress.add_task(n, total=None, start=False) for n in names}

        def on_bytes(name: str, done: int, total: int) -> None:
            task: TaskID = tasks[name]
            progress.start_task(task)
            progress.update(task, completed=done, total=total or None)

        try:
            models.download_all(names, on_bytes)
        except KeyboardInterrupt:
            err.print(f"[yellow]{_('Download cancelled.')}[/]")
            raise typer.Exit(1) from None
        except Exception as e:  # noqa: BLE001 - network errors etc.
            err.print(f"[red]{_('Download failed: {error}').format(error=e)}[/]")
            raise typer.Exit(1) from e
    err.print(_("Models stored in {path}").format(path=models.models_dir()))


def _interactive() -> bool:
    """Prompts only on a terminal (a script or pipe gets the recommendations)."""
    import sys

    return sys.stdin.isatty()


@app.command(help=_(
    "First-run setup: pick a model for this computer, languages and voiceprint privacy, "
    "then download the models. Without --yes it asks; given options are not asked."))
def setup(
    profile: Annotated[str | None, typer.Option(
        help=_("Model choice: fast | balanced | accurate | cuda | gpu (default: "
               "recommended for this computer)."))] = None,
    lang: Annotated[str | None, typer.Option(
        "--lang", "-l", help=_("Spoken languages: de,en | de | en."))] = None,
    ui_language: Annotated[str | None, typer.Option(
        help=_("Language of the app: auto | en | de."))] = None,
    voiceprints: Annotated[bool | None, typer.Option(
        "--voiceprints/--no-voiceprints",
        help=_("Store voiceprints and recognise known people."))] = None,
    download: Annotated[bool | None, typer.Option(
        "--download/--no-download", help=_("Download the models now."))] = None,
    yes: Annotated[bool, typer.Option(
        "--yes", "-y", help=_("Don't ask; use the recommendation for anything not given."))]
    = False,
) -> None:
    from rich.prompt import Confirm, Prompt

    from transcribe.core import setup as core_setup
    from transcribe.core.config import required_models
    from transcribe.i18n import LANGUAGES

    ask = not yes and _interactive()
    config = Config.load()
    hw = core_setup.detect_hardware()
    choices = core_setup.profiles_for(hw)
    recommended = core_setup.recommend(hw)
    if profile is not None and profile not in core_setup.PROFILES:
        err.print(f"[red]{_('Unknown profile {name}').format(name=profile)}[/]")
        raise typer.Exit(1)
    if lang is not None and lang not in core_setup.LANGUAGE_CHOICES:
        err.print(f"[red]{_('Unknown languages {lang}').format(lang=lang)}[/]")
        raise typer.Exit(1)
    if ui_language is not None and ui_language not in ("auto", *LANGUAGES):
        err.print(f"[red]{_('Unknown language {lang}').format(lang=ui_language)}[/]")
        raise typer.Exit(1)

    err.print(core_setup.hardware_text(hw))
    if not hw.has_gpu:
        err.print(f"[dim]{_(core_setup.NO_GPU_HINT)}[/]")
    if profile is None and ask:
        table = Table("#", _("Model"), _("Download"), _("Speed"))
        numbers = [str(i) for i in range(1, len(choices) + 1)]
        default = str(choices.index(recommended) + 1)
        if core_setup.config_path().exists():  # offer to keep hand-made settings
            matching = core_setup.matching_profile(config)
            numbers.insert(0, "0")
            default = str(choices.index(matching) + 1) if matching in choices else "0"
            table.add_row("0", _("Keep the current settings ({model})").format(
                model=config.asr.model), "", "")
        for i, p in enumerate(choices, 1):
            label = _(p.label) + (f" [bold]{_('(recommended)')}[/]" if p is recommended else "")
            table.add_row(str(i), label,
                          core_setup.format_size(core_setup.profile_size_mb(p, config)),
                          _(p.speed))
        err.print(table)
        number = Prompt.ask(_("Model"), console=err, choices=numbers, default=default)
        profile = choices[int(number) - 1].key if number != "0" else None
    if lang is None and ask:
        for value, label in core_setup.LANGUAGE_CHOICES.items():
            err.print(f"  [bold]{value}[/]: {_(label)}")
        current = ",".join(config.asr.languages)
        lang = Prompt.ask(_("Spoken languages"), console=err,
                          choices=list(core_setup.LANGUAGE_CHOICES),
                          default=current if current in core_setup.LANGUAGE_CHOICES
                          else "de,en")
    if ui_language is None and ask:
        ui_language = Prompt.ask(_("Language of the app"), console=err,
                                 choices=["auto", *LANGUAGES], default=config.ui.language)
    if voiceprints is None and ask:
        err.print(f"[dim]{_(core_setup.VOICEPRINT_NOTE)}[/]")
        voiceprints = Confirm.ask(_("Store voiceprints and recognise known people?"),
                                  console=err, default=config.speakers.store_voiceprints)
    chosen = core_setup.PROFILES[profile] if profile else (
        None if core_setup.config_path().exists() else recommended)
    core_setup.apply_choices(config, chosen, ui_language,
                             lang.split(",") if lang else None, voiceprints, voiceprints)
    config.save()
    err.print(_("Settings saved to {path}").format(path=core_setup.config_path()))
    missing = models.missing_models(*required_models(config))
    if not missing:
        err.print(_("All configured models are present."))
        return
    size = core_setup.format_size(sum(core_setup.size_mb(n) for n in missing))
    if download is None:
        download = Confirm.ask(_("Download {count} models now ({size})?").format(
            count=len(missing), size=size), console=err, default=True) if ask else True
    if download:
        _download(missing)
    else:
        err.print(_("Models download on first use, or with `tx models download`."))


if __name__ == "__main__":
    app()
