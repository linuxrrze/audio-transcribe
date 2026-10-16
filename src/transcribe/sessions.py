"""Session operations shared by the CLI and the GUI."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config
from transcribe.live.session import LIVE_TRANSCRIPT, refine_channels
from transcribe.storage import session_store
from transcribe.storage.recorder import MERGED, load_recording, recorded_channels
from transcribe.storage.transcript_model import Speaker, Transcript

PLAYBACK_MIX = "playback.flac"
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
    meta_file = session_dir / session_store.SESSION_META
    if meta_file.exists():
        source = tomllib.loads(meta_file.read_text(encoding="utf-8")).get("source")
        if source and Path(source).exists():
            return Path(source)
    return None


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


def merge_speakers(session_dir: Path, keep: str, drop: str) -> None:
    """Assign all of ``drop``'s speech to ``keep`` (fixes a diarization split)."""
    path = transcript_path(session_dir)
    t = Transcript.load(path)
    if keep not in t.speakers or drop not in t.speakers:
        raise KeyError(f"Unknown speaker {keep if keep not in t.speakers else drop}")
    for seg in t.segments:
        if seg.speaker_id == drop:
            seg.speaker_id = keep
        for w in seg.words:
            if w.speaker_id == drop:
                w.speaker_id = keep
    del t.speakers[drop]
    t.save(path)


def refine_session(session_dir: Path, config: Config, num_speakers: int | None = None,
                   progress: StageProgress | None = None, identify: bool = True) -> Transcript:
    """Re-run diarization + transcription on a session's recording (all channels)."""
    from transcribe.core.offline import load_engines, transcribe_audio

    audio = {ch: load_recording(session_dir, ch) for ch in recorded_channels(session_dir)}
    audio = {ch: a for ch, a in audio.items() if len(a)}
    if not audio:
        raise FileNotFoundError(f"No recording in {session_dir}")
    path = session_store.resolve_transcript(session_dir)
    previous = Transcript.load(transcript_path(session_dir)) if transcript_path(
        session_dir).exists() else Transcript()
    asr, diarizer = load_engines(config, diarize=True)

    def run(signal: np.ndarray, report: StageProgress) -> Transcript:
        return transcribe_audio(signal, asr, diarizer, config.asr.languages, num_speakers,
                                report)

    final = refine_channels(audio, run, previous, progress)
    final.meta["asr_model"] = config.asr.model
    if identify and final.speakers and (config.speakers.identify_known
                                        or config.speakers.store_voiceprints):
        from transcribe.speakers.embedder import SherpaEmbedder
        from transcribe.speakers.identity import process_session

        process_session(session_dir, audio if len(audio) > 1 else next(iter(audio.values())),
                        final, SherpaEmbedder.from_config(config), config)
    final.save(path)
    return final
