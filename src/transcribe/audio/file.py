"""Audio files as a source. Decoding uses PyAV (bundled ffmpeg), so any common format works."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo


def load_audio(path: Path | str) -> np.ndarray:
    """Decode a file to 16 kHz mono float32."""
    from faster_whisper import decode_audio

    return np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE), dtype=np.float32)


class FileSource:
    """Replays a file as frames; ``realtime=True`` paces output like a live device."""

    def __init__(self, path: Path | str, realtime: bool = False, frame_ms: int = 100,
                 channel_id: str = "default") -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.frame_len = SAMPLE_RATE * frame_ms // 1000
        self.channel_id = channel_id
        self._audio: np.ndarray | None = None

    def describe(self) -> SourceInfo:
        return SourceInfo(name=f"file:{self.path.name}", channels=[self.channel_id],
                          realtime=self.realtime)

    def open(self) -> None:
        self._audio = load_audio(self.path)

    def frames(self) -> Iterator[AudioFrame]:
        if self._audio is None:
            raise RuntimeError("open() must be called first")
        start = time.monotonic()
        for i in range(0, len(self._audio), self.frame_len):
            t = i / SAMPLE_RATE
            if self.realtime:
                delay = start + t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield AudioFrame(self._audio[i : i + self.frame_len], t, self.channel_id)

    def close(self) -> None:
        self._audio = None
