"""Microphone input via PortAudio (sounddevice), resampled to 16 kHz mono."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo

# PortAudio is not thread-safe for opening/closing streams or querying devices; with
# several devices open at once (level meters) concurrent calls crash inside the host
# API (seen with PulseAudio: an assertion in libpulse). All such calls hold this lock.
PORTAUDIO_LOCK = threading.RLock()


def _sd():
    try:
        import sounddevice
    except OSError as e:  # PortAudio shared library missing (common on Linux)
        raise RuntimeError(
            "PortAudio not found. Install it (Debian/Ubuntu: apt install libportaudio2, "
            "Fedora: dnf install portaudio) - Windows/macOS wheels bundle it."
        ) from e
    except ImportError as e:
        raise RuntimeError("Microphone input needs the 'audio' extra: "
                           "pip install 'transcribe[audio]'") from e
    return sounddevice


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float
    hostapi: str
    is_default: bool


def list_input_devices() -> list[InputDevice]:
    with PORTAUDIO_LOCK:
        return _list_input_devices()


def _list_input_devices() -> list[InputDevice]:
    sd = _sd()
    hostapis = sd.query_hostapis()
    default_in = sd.default.device[0]
    return [
        InputDevice(i, d["name"], d["max_input_channels"], d["default_samplerate"],
                    hostapis[d["hostapi"]]["name"], i == default_in)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def resolve_device(device: str | int | None) -> int | None:
    """Accept an index, an exact name or a case-insensitive name substring."""
    if device is None or isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    devices = list_input_devices()
    for d in devices:
        if d.name == device:
            return d.index
    matches = [d for d in devices if device.lower() in d.name.lower()]
    if len(matches) == 1:
        return matches[0].index
    names = ", ".join(d.name for d in matches or devices)
    raise ValueError(f"Input device {device!r} is ambiguous or unknown; candidates: {names}")


class MicSource:
    def __init__(self, device: str | int | None = None, block_ms: int = 100,
                 channel_id: str = "mic") -> None:
        self.device = device
        self.block_ms = block_ms
        self.channel_id = channel_id
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: Any = None  # sounddevice.InputStream
        self._resampler: Any = None  # soxr.ResampleStream
        self.overflows = 0

    def describe(self) -> SourceInfo:
        return SourceInfo(name=f"mic:{self.device or 'default'}", channels=[self.channel_id])

    def open(self) -> None:
        import soxr

        with PORTAUDIO_LOCK:
            self._open(soxr)

    def _open(self, soxr) -> None:
        sd = _sd()
        index = resolve_device(self.device)
        info = sd.query_devices(index, "input")
        rate = int(info["default_samplerate"])
        if rate != SAMPLE_RATE:
            self._resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")

        def callback(indata, frames, time_info, status) -> None:  # PortAudio thread
            if status.input_overflow:
                self.overflows += 1
            self._queue.put(indata.mean(axis=1).astype(np.float32))  # never blocks

        self._stream = sd.InputStream(device=index, samplerate=rate, channels=min(
            2, info["max_input_channels"]), dtype="float32",
            blocksize=rate * self.block_ms // 1000, callback=callback)
        self._stream.start()

    def frames(self) -> Iterator[AudioFrame]:
        t = 0.0
        while True:
            block = self._queue.get()
            if block is None:
                break
            if self._resampler is not None:
                block = self._resampler.resample_chunk(block)
            if len(block):
                yield AudioFrame(block, t, self.channel_id)
                t += len(block) / SAMPLE_RATE

    def close(self) -> None:
        with PORTAUDIO_LOCK:
            stream, self._stream = self._stream, None
            if stream is not None:
                stream.stop()
                stream.close()
        self._queue.put(None)
