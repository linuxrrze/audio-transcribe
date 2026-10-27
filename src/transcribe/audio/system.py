"""System audio ("what the computer plays") per platform.

- Linux: monitor source of a PulseAudio/PipeWire sink, captured with ``parec`` (works with
  pipewire-pulse too).
- Windows: WASAPI loopback of the default (or chosen) speaker via ``soundcard``.
- macOS 14.2+: Core Audio process tap through the bundled ``transcribe-audiotap`` helper
  (needs the "System Audio Recording" permission). Older macOS: route output through a
  virtual device such as BlackHole and select it as microphone input.

A single application instead of a whole device: ``audio/apps.py`` (``app:<name>``).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, AudioSource, SourceInfo
from transcribe.audio.command import CommandSource


@dataclass(frozen=True)
class OutputDevice:
    id: str
    name: str
    is_default: bool


# --- Linux -------------------------------------------------------------------------------
def pulse_monitors() -> list[OutputDevice]:
    """Monitor sources of all sinks (PulseAudio / PipeWire-Pulse)."""
    if shutil.which("pactl") is None:
        return []
    try:
        default = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        rows = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True,
                              text=True, timeout=5).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for row in rows:
        parts = row.split("\t")
        if len(parts) > 1 and parts[1].endswith(".monitor"):
            out.append(OutputDevice(parts[1], parts[1][: -len(".monitor")],
                                    parts[1] == f"{default}.monitor"))
    return out


class PulseMonitorSource(CommandSource):
    def __init__(self, monitor: str | None = None, channel_id: str = "system") -> None:
        device = monitor or "@DEFAULT_MONITOR@"
        super().__init__(["parec", f"--device={device}", "--format=float32le",
                          f"--rate={SAMPLE_RATE}", "--channels=1", "--raw",
                          "--latency-msec=100"], channel_id, name=f"system:{device}")


# --- Windows -----------------------------------------------------------------------------
class WasapiLoopbackSource:
    """Loopback capture of a speaker (WASAPI) using the ``soundcard`` package."""

    def __init__(self, speaker: str | None = None, channel_id: str = "system",
                 block_ms: int = 100) -> None:
        self.speaker, self.channel_id, self.block_ms = speaker, channel_id, block_ms
        self._recorder: Any = None
        self._closed = False
        self._rate = 48000

    def describe(self) -> SourceInfo:
        return SourceInfo(name=f"system:{self.speaker or 'default'}", channels=[self.channel_id])

    def open(self) -> None:
        try:
            import soundcard as sc
        except ImportError as e:
            raise RuntimeError("System audio on Windows needs the 'system' extra "
                               "(soundcard)") from e
        spk = sc.get_speaker(self.speaker) if self.speaker else sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        self._recorder = mic.recorder(samplerate=self._rate, channels=1,
                                      blocksize=self._rate * self.block_ms // 1000)
        self._recorder.__enter__()

    def frames(self) -> Iterator[AudioFrame]:
        import soxr

        resampler = soxr.ResampleStream(self._rate, SAMPLE_RATE, 1, dtype="float32")
        n, t = self._rate * self.block_ms // 1000, 0.0
        while not self._closed and self._recorder is not None:
            block = self._recorder.record(numframes=n)
            mono = np.asarray(block, dtype=np.float32).reshape(len(block), -1).mean(axis=1)
            samples = resampler.resample_chunk(mono)
            if len(samples):
                yield AudioFrame(samples, t, self.channel_id)
                t += len(samples) / SAMPLE_RATE

    def close(self) -> None:
        self._closed = True
        if self._recorder is not None:
            self._recorder.__exit__(None, None, None)
            self._recorder = None


def wasapi_speakers() -> list[OutputDevice]:
    try:
        import soundcard as sc
    except ImportError:
        return []
    default = sc.default_speaker().name
    return [OutputDevice(s.name, s.name, s.name == default) for s in sc.all_speakers()]


# --- macOS -------------------------------------------------------------------------------
AUDIOTAP = "transcribe-audiotap"


def audiotap_path() -> Path | None:
    """The Swift helper: next to the frozen app, in PATH, or a local development build."""
    candidates = []
    if getattr(sys, "frozen", False):  # PyInstaller: next to the exe or in the data dir
        candidates.append(Path(sys.executable).parent / AUDIOTAP)
        candidates.append(Path(getattr(sys, "_MEIPASS", sys.executable)) / AUDIOTAP)
    if env := os.environ.get("TRANSCRIBE_AUDIOTAP"):
        candidates.append(Path(env))
    if found := shutil.which(AUDIOTAP):
        candidates.append(Path(found))
    repo = Path(__file__).resolve().parents[3] / "helpers" / "macos-audiotap" / ".build"
    candidates += [repo / "apple" / "Products" / "Release" / AUDIOTAP,  # universal build
                   repo / "release" / AUDIOTAP]
    return next((c for c in candidates if c.exists()), None)


class MacAudioTapSource(CommandSource):
    def __init__(self, pid: int | None = None, channel_id: str = "system") -> None:
        helper = audiotap_path()
        if helper is None:
            raise RuntimeError(
                "macOS system audio needs the transcribe-audiotap helper (macOS 14.2+; build "
                "it with helpers/macos-audiotap/build.sh). Alternative: install BlackHole, "
                "route output through it and use it as microphone input.")
        cmd = [str(helper), "--rate", str(SAMPLE_RATE)]
        if pid is not None:  # capture one application only (e.g. a meeting client)
            cmd += ["--pid", str(pid)]
        super().__init__(cmd, channel_id, name=f"system:{'pid ' + str(pid) if pid else 'all'}")


# --- factory -----------------------------------------------------------------------------
def output_devices() -> list[OutputDevice]:
    system = platform.system()
    if system == "Linux":
        return pulse_monitors()
    if system == "Windows":
        return wasapi_speakers()
    return [OutputDevice("all", "All system audio", True)] if audiotap_path() else []


def system_source(device: str | None = None, channel_id: str = "system") -> AudioSource:
    """What an output device plays, or one application's audio (``app:<name or pid>``,
    see ``audio/apps.py``)."""
    from transcribe.audio.apps import APP_PREFIX, app_source

    if device and device.startswith(APP_PREFIX):
        return app_source(device[len(APP_PREFIX):], channel_id)
    system = platform.system()
    if system == "Linux":
        if shutil.which("parec") is None:
            raise RuntimeError("System audio on Linux needs `parec` (pulseaudio-utils; works "
                               "with PipeWire's pulse server as well)")
        return PulseMonitorSource(device, channel_id)
    if system == "Windows":
        return WasapiLoopbackSource(device, channel_id)
    if system == "Darwin":
        return MacAudioTapSource(int(device) if device and device.isdigit() else None,
                                 channel_id)
    raise RuntimeError(f"System audio capture is not supported on {system}")
