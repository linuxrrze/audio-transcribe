"""Session operations shared by the CLI and the GUI."""

from __future__ import annotations

import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config, vocabulary
from transcribe.live.session import LIVE_TRANSCRIPT, refine_channels
from transcribe.storage import editing, session_store
from transcribe.storage.recorder import MERGED, load_recording, recorded_channels, save_recording
from transcribe.storage.transcript_model import Speaker, Transcript

PLAYBACK_MIX = "playback.flac"
VERSIONS = "versions"  # earlier transcript.json files, kept when a session is refined
StageProgress = Callable[[str, float], None]


@dataclass(frozen=True)
class SessionInfo:
    path: Path
    title: str
    created: datetime
    duration_s: float
    speakers: int
    refined: bool
    live: bool


def transcript_path(session_dir: Path) -> Path:
    """The final transcript, or the live one of an unfinished/crashed session."""
    final = session_dir / session_store.TRANSCRIPT
    return final if final.exists() else session_dir / LIVE_TRANSCRIPT


def load(session_dir: Path) -> Transcript:
    return Transcript.load(transcript_path(session_dir))


def _created(path: Path) -> datetime:
    try:
        return datetime.strptime(path.name[:17], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return datetime.fromtimestamp(path.stat().st_mtime)


def list_sessions(root: Path) -> list[SessionInfo]:
    """Sessions under ``root``, newest first. Unreadable folders are skipped."""
    out = []
    for path in root.iterdir() if root.exists() else []:
        tp = transcript_path(path) if path.is_dir() else None
        if tp is None or not tp.exists():
            continue
        try:
            t = Transcript.load(tp)
        except (ValueError, OSError, KeyError):
            continue
        duration = t.meta.get("duration_s")
        out.append(SessionInfo(
            path, str(t.meta.get("title") or path.name), _created(path),
            float(duration) if isinstance(duration, int | float)
            else (t.segments[-1].t1 if t.segments else 0.0),
            len(t.speakers), bool(t.meta.get("refined")),
            live=bool(t.meta.get("channels")) or (path / LIVE_TRANSCRIPT).exists()))
    return sorted(out, key=lambda s: s.created, reverse=True)


def playback_audio(session_dir: Path) -> Path | None:
    """A single file to play: the recording, a cached mix of channels, or the source."""
    if (session_dir / MERGED).exists():
        return session_dir / MERGED
    channels = [c for c in recorded_channels(session_dir) if c]
    if channels:
        mix_path = session_dir / PLAYBACK_MIX
        if not mix_path.exists():
            import soundfile as sf

            signals = [load_recording(session_dir, c) for c in channels]
            mix = np.zeros(max(len(s) for s in signals), np.float32)
            for s in signals:
                mix[: len(s)] += s
            sf.write(mix_path, np.clip(mix, -1, 1), SAMPLE_RATE, subtype="PCM_16")
        return mix_path
    source = session_source(session_dir)
    return source if source is not None and source.exists() else None


def session_source(session_dir: Path) -> Path | None:
    """The original file of a `tx transcribe` session (it may have moved since)."""
    meta_file = session_dir / session_store.SESSION_META
    if not meta_file.exists():
        return None
    source = tomllib.loads(meta_file.read_text(encoding="utf-8")).get("source")
    return Path(source) if isinstance(source, str) and source else None


def session_audio(session_dir: Path, keep: bool = False) -> dict[str, np.ndarray]:
    """Channel id -> audio to refine: the recording, else the original file.

    File sessions from before ``keep_file_audio`` only know the source path; it is used
    while it still exists and, with ``keep``, stored as the session's recording."""
    audio = {ch: load_recording(session_dir, ch) for ch in recorded_channels(session_dir)}
    audio = {ch: a for ch, a in audio.items() if len(a)}
    if audio:
        return audio
    source = session_source(session_dir)
    if source is None:
        raise FileNotFoundError(f"No recording in {session_dir}")
    if not source.is_file():
        raise FileNotFoundError(
            f"No recording in {session_dir}, and its original file {source} no longer "
            "exists. Transcribe the file again (tx transcribe stores its audio).")
    from transcribe.audio.file import load_audio

    signal = load_audio(source)
    if keep and len(signal):
        save_recording(session_dir, signal)
    return {"": signal}


def rename_speaker(session_dir: Path, speaker_id: str, name: str) -> None:
    """Relabel a speaker (no voiceprint; see identity.name_speaker for enrollment)."""
    path = transcript_path(session_dir)
    t = Transcript.load(path)
    old = t.speakers[speaker_id]
    t.speakers[speaker_id] = Speaker(speaker_id, name, old.person_id, confirmed=True)
    t.save(path)


def decline_suggestion(session_dir: Path, speaker_id: str) -> None:
    path = transcript_path(session_dir)
    t = Transcript.load(path)
    t.speakers[speaker_id].suggestion = None
    t.save(path)


def merge_speakers(session_dir: Path, keep: str, drop: str,
                   config: Config | None = None) -> None:
    """Assign all of ``drop``'s speech to ``keep`` (fixes a diarization split)."""
    _edit(session_dir, config, lambda t: editing.merge(t, keep, drop))


def assign_segments(session_dir: Path, seg_ids: list[str], speaker: str,
                    new_label: str | None = None, config: Config | None = None) -> str:
    """Attribute whole lines to ``speaker`` (an id), or to a new speaker ``new_label``.

    Returns the speaker id."""
    target = speaker

    def apply(t: Transcript) -> editing.EditResult:
        nonlocal target
        if new_label:
            target = editing.add_speaker(t, new_label)
        return editing.assign(t, seg_ids, target)

    _edit(session_dir, config, apply)
    return target


def edit_text(session_dir: Path, seg_id: str, text: str,
              config: Config | None = None) -> list[str]:
    """Replace a line's text (word times kept where words are unchanged). Each line of
    ``text`` becomes its own segment; empty text deletes it. Returns the segment ids."""
    out: list[str] = []

    def apply(t: Transcript) -> editing.EditResult:
        result = editing.set_text(t, seg_id, text)
        if seg_id not in result.changed_audio or result.new_segments:
            out.extend([seg_id, *result.new_segments])
        return result

    _edit(session_dir, config, apply)
    return out


def load_numbered(session_dir: Path) -> Transcript:
    """The session's transcript with segment ids (assigned and saved if missing)."""
    path = transcript_path(session_dir)
    t = Transcript.load(path)
    ids = [s.id for s in t.segments]
    t.number_segments()
    if [s.id for s in t.segments] != ids:
        t.save(path)
    return t


def _edit(session_dir: Path, config: Config | None,
          apply: Callable[[Transcript], editing.EditResult]) -> None:
    from transcribe.speakers.crypto import make_cipher
    from transcribe.speakers.identity import update_exemplars

    path = transcript_path(session_dir)
    t = load_numbered(session_dir)
    before = {s.id: s.speaker_id for s in t.segments}
    result = apply(t)
    # speakers that lost speech: a concatenated exemplar of theirs may contain it
    affected = {before[i] for i in (*result.moved, *result.changed_audio)
                if before.get(i) and before[i] != result.moved.get(i)}
    config = config or Config.load()
    update_exemplars(session_dir, result, {a for a in affected if a},
                     make_cipher(config.speakers.encrypt_voiceprints))
    t.save(path)


def refine_session(session_dir: Path, config: Config, num_speakers: int | None = None,
                   progress: StageProgress | None = None, identify: bool = True,
                   audio: dict[str, np.ndarray] | None = None) -> Transcript:
    """Re-run diarization + transcription on a session's recording (all channels).

    ``audio``: from ``session_audio`` if the caller loaded it already. Replaces
    ``transcript.json`` and keeps the old one in ``versions/`` (``keep_version``); a live
    session also keeps ``transcript.live.json``."""
    from transcribe.core.offline import load_engines, transcribe_audio

    if audio is None:
        audio = session_audio(session_dir, keep=config.keep_file_audio)
    path = session_store.resolve_transcript(session_dir)
    previous = Transcript.load(transcript_path(session_dir)) if transcript_path(
        session_dir).exists() else Transcript()
    asr, diarizer = load_engines(config, diarize=True)

    def run(signal: np.ndarray, report: StageProgress) -> Transcript:
        return transcribe_audio(signal, asr, diarizer, config.asr.languages, num_speakers,
                                report, vocabulary=vocabulary(config))

    final = refine_channels(audio, run, previous, progress)
    final.meta["asr_model"] = config.asr.model
    if identify and final.speakers and (config.speakers.identify_known
                                        or config.speakers.store_voiceprints):
        from transcribe.speakers.embedder import SherpaEmbedder
        from transcribe.speakers.identity import process_session

        process_session(session_dir, audio if len(audio) > 1 else next(iter(audio.values())),
                        final, SherpaEmbedder.from_config(config), config)
    keep_version(path)
    final.save(path)
    return final


def keep_version(path: Path) -> Path | None:
    """Copy a transcript into ``versions/``, named by when it was last written."""
    if not path.exists():
        return None
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d_%H%M%S")
    target = path.parent / VERSIONS / f"{path.stem}.{stamp}{path.suffix}"
    suffix = 1
    while target.exists():
        target = target.with_name(f"{path.stem}.{stamp}_{suffix}{path.suffix}")
        suffix += 1
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(path, target)  # keeps the time; transcript.json stays until replaced
    return target
