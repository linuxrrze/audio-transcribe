"""Signal levels of audio devices - to check that the right devices are selected.

``LevelMonitor`` opens any number of sources in parallel (microphones and system-audio
outputs) and keeps the latest level of each. A device that fails to open only marks
its own entry with an error; the others keep running.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from transcribe.audio.base import AudioSource

FLOOR_DB = -60.0  # meters show -60..0 dBFS
SILENT_DB = -50.0  # below this the device counts as silent
PEAK_HOLD_S = 1.5


def to_db(x: float) -> float:
    return max(FLOOR_DB, 20 * math.log10(max(x, 1e-9)))


def level_db(samples: np.ndarray) -> tuple[float, float]:
    """(RMS dBFS, peak dBFS) of a block."""
    if not len(samples):
        return FLOOR_DB, FLOOR_DB
    return (to_db(float(np.sqrt(np.mean(np.square(samples))))),
            to_db(float(np.max(np.abs(samples)))))


def meter_fraction(db: float) -> float:
    """0..1 position of ``db`` on a -60..0 dBFS meter."""
    return min(1.0, max(0.0, (db - FLOOR_DB) / -FLOOR_DB))


@dataclass(frozen=True)
class Level:
    rms_db: float = FLOOR_DB
    peak_db: float = FLOOR_DB  # held for PEAK_HOLD_S
    max_db: float = FLOOR_DB  # loudest RMS since the monitor started
    receiving: bool = False  # frames arrived within the last second
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if not self.receiving:
            return "no data"  # e.g. WASAPI loopback while nothing plays
        return "silent" if self.rms_db < SILENT_DB else "signal"


@dataclass
class _State:
    rms_db: float = FLOOR_DB
    peak_db: float = FLOOR_DB
    peak_at: float = 0.0
    max_db: float = FLOOR_DB
    last_frame: float = 0.0
    error: str | None = None


class LevelMonitor:
    """Measure levels of named sources until ``stop()``.

    ``on_update(name, level)`` (optional) is called from the capture threads.
    """

    def __init__(self, sources: dict[str, AudioSource],
                 on_update: Callable[[str, Level], None] | None = None) -> None:
        self.sources = sources
        self.on_update = on_update
        self._state = {name: _State() for name in sources}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> LevelMonitor:
        for name, source in self.sources.items():
            t = threading.Thread(target=self._run, args=(name, source), daemon=True,
                                 name=f"level:{name}")
            t.start()
            self._threads.append(t)
        return self

    def _run(self, name: str, source: AudioSource) -> None:
        try:
            source.open()
            for frame in source.frames():
                if self._stop.is_set():
                    break
                self._update(name, frame.samples)
        except Exception as e:  # noqa: BLE001 - shown per device
            with self._lock:
                self._state[name].error = str(e) or type(e).__name__
            if self.on_update:
                self.on_update(name, self.level(name))
        finally:
            try:
                source.close()
            except Exception:  # noqa: BLE001, S110 - closing a failed device
                pass

    def _update(self, name: str, samples: np.ndarray) -> None:
        rms, peak = level_db(samples)
        now = time.monotonic()
        with self._lock:
            st = self._state[name]
            st.rms_db, st.last_frame = rms, now
            st.max_db = max(st.max_db, rms)
            if peak >= st.peak_db or now - st.peak_at > PEAK_HOLD_S:
                st.peak_db, st.peak_at = peak, now
        if self.on_update:
            self.on_update(name, self.level(name))

    def level(self, name: str) -> Level:
        now = time.monotonic()
        with self._lock:
            st = self._state[name]
            receiving = now - st.last_frame < 1.0
            return Level(st.rms_db if receiving else FLOOR_DB, st.peak_db, st.max_db,
                         receiving, st.error)

    def snapshot(self) -> dict[str, Level]:
        return {name: self.level(name) for name in self.sources}

    def stop(self) -> None:
        self._stop.set()
        for source in self.sources.values():
            try:
                source.close()  # unblocks sources waiting for data
            except Exception:  # noqa: BLE001, S110
                pass
        for t in self._threads:
            t.join(timeout=2)


@dataclass(frozen=True)
class DeviceEntry:
    key: str  # "in:<index>" or "out:<id>"
    kind: str  # "input" | "output"
    device: str  # value for --input / --output
    name: str
    is_default: bool


def list_devices() -> tuple[list[DeviceEntry], list[str]]:
    """All input and system-audio devices, plus errors from enumerating them."""
    from transcribe.audio.mic import list_input_devices
    from transcribe.audio.system import output_devices

    entries, errors = [], []
    try:
        monitors = {o.id for o in output_devices()}
        for d in list_input_devices():
            if d.name in monitors:  # PulseAudio monitors also appear as inputs
                continue
            entries.append(DeviceEntry(f"in:{d.index}", "input", str(d.index), d.name,
                                       d.is_default))
    except RuntimeError as e:
        errors.append(f"inputs: {e}")
    try:
        for o in output_devices():
            entries.append(DeviceEntry(f"out:{o.id}", "output", o.id, o.name, o.is_default))
    except RuntimeError as e:
        errors.append(f"outputs: {e}")
    return entries, errors


class _Unavailable:
    """Placeholder for a device whose source could not even be created."""

    def __init__(self, message: str) -> None:
        self.message = message

    def describe(self):
        from transcribe.audio.base import SourceInfo

        return SourceInfo("unavailable")

    def open(self) -> None:
        raise RuntimeError(self.message)

    def frames(self):
        return iter(())

    def close(self) -> None:
        pass


def device_source(entry: DeviceEntry) -> AudioSource:
    from transcribe.audio.mic import MicSource
    from transcribe.audio.system import system_source

    try:
        if entry.kind == "input":
            return MicSource(entry.device, channel_id=entry.key)
        return system_source(entry.device, channel_id=entry.key)
    except (RuntimeError, ValueError) as e:
        return _Unavailable(str(e))
