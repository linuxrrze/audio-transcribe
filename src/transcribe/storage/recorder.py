"""Crash-safe session audio recording.

Audio is written as consecutive FLAC parts (``audio/part-00000.flac`` ...); every closed
part is a valid file, so a crash loses at most the part being written. ``finalize()``
merges the parts into ``audio.flac``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from transcribe.audio.base import SAMPLE_RATE

PARTS_DIR = "audio"
MERGED = "audio.flac"


class AudioRecorder:
    def __init__(self, session_dir: Path, part_seconds: float = 30.0) -> None:
        self.session_dir = session_dir
        self.parts_dir = session_dir / PARTS_DIR
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.part_samples = int(part_seconds * SAMPLE_RATE)
        self._file: Any = None  # soundfile.SoundFile (imported lazily)
        self._written = 0
        self._index = 0
        self.total_samples = 0

    def write(self, samples: np.ndarray) -> None:
        import soundfile as sf

        while len(samples):
            if self._file is None:
                path = self.parts_dir / f"part-{self._index:05d}.flac"
                self._file = sf.SoundFile(path, "w", SAMPLE_RATE, 1, "PCM_16", format="FLAC")
                self._index += 1
                self._written = 0
            take = min(len(samples), self.part_samples - self._written)
            self._file.write(samples[:take])
            self._written += take
            self.total_samples += take
            samples = samples[take:]
            if self._written >= self.part_samples:
                self._close_part()

    def _close_part(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def finalize(self) -> Path:
        """Close the current part and merge all parts into ``audio.flac``."""
        import soundfile as sf

        self._close_part()
        merged = self.session_dir / MERGED
        sf.write(merged, load_recording(self.session_dir), SAMPLE_RATE, subtype="PCM_16")
        for part in sorted(self.parts_dir.glob("part-*.flac")):
            part.unlink()
        self.parts_dir.rmdir()
        return merged


def load_recording(session_dir: Path) -> np.ndarray:
    """Load a session's audio: the merged file, or the parts left by an interrupted run."""
    import soundfile as sf

    merged = session_dir / MERGED
    if merged.exists():
        return sf.read(merged, dtype="float32")[0]
    parts = []
    for part in sorted((session_dir / PARTS_DIR).glob("part-*.flac")):
        try:
            parts.append(sf.read(part, dtype="float32")[0])
        except RuntimeError:  # last part of a crashed session may be truncated
            break
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
