import threading
import time

import numpy as np
from rich.console import Console

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo
from transcribe.audio.levels import (
    FLOOR_DB,
    DeviceEntry,
    Level,
    LevelMonitor,
    device_source,
    level_db,
    meter_fraction,
)
from transcribe.cli.levels import levels_table, meter


class Tone:
    """Endless real-time-ish source with a fixed amplitude (0 = digital silence)."""

    def __init__(self, amplitude: float, channel: str = "x") -> None:
        self.amplitude, self.channel = amplitude, channel
        self._closed = threading.Event()

    def describe(self):
        return SourceInfo(self.channel, [self.channel])

    def open(self):
        pass

    def frames(self):
        t = np.arange(SAMPLE_RATE // 20) / SAMPLE_RATE
        block = (self.amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        while not self._closed.is_set():
            yield AudioFrame(block, 0.0, self.channel)
            time.sleep(0.02)

    def close(self):
        self._closed.set()


class Broken(Tone):
    def open(self):
        raise RuntimeError("device busy")


class Mute(Tone):
    """Delivers no frames at all (like WASAPI loopback while nothing plays)."""

    def frames(self):
        self._closed.wait()
        return iter(())


def test_level_math():
    rms, peak = level_db(np.full(100, 0.5, np.float32))
    assert abs(rms - (-6.02)) < 0.1 and abs(peak - (-6.02)) < 0.1
    assert level_db(np.zeros(10, np.float32)) == (FLOOR_DB, FLOOR_DB)
    assert meter_fraction(FLOOR_DB) == 0.0 and meter_fraction(0.0) == 1.0
    assert meter_fraction(-30.0) == 0.5


def test_monitor_reports_each_device_independently():
    mon = LevelMonitor({"loud": Tone(0.5), "quiet": Tone(0.0), "bad": Broken(0.5),
                        "mute": Mute(0.5)}).start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not mon.level("loud").receiving:
            time.sleep(0.02)
        time.sleep(0.2)
        snap = mon.snapshot()
    finally:
        start = time.monotonic()
        mon.stop()
        assert time.monotonic() - start < 3  # stop() unblocks waiting sources
    assert snap["loud"].status == "signal" and -12 < snap["loud"].rms_db < -6
    assert snap["quiet"].status == "silent"
    assert snap["bad"].status == "error" and "busy" in snap["bad"].error
    assert snap["mute"].status == "no data"


def test_device_source_wraps_construction_errors(monkeypatch):
    import transcribe.audio.system as system

    def boom(*a, **k):
        raise RuntimeError("helper missing")

    monkeypatch.setattr(system, "system_source", boom)
    src = device_source(DeviceEntry("out:x", "output", "x", "X", False))
    mon = LevelMonitor({"x": src}).start()
    time.sleep(0.2)
    mon.stop()
    assert mon.level("x").error == "helper missing"


def test_cli_table_renders_status_and_meter():
    console = Console(width=120, record=True)
    console.print(levels_table([("a", "🎤 Headset", True), ("b", "🔊 Speakers", False)],
                               {"a": Level(-20, -10, -15, True),
                                "b": Level(error="device busy")}))
    out = console.export_text()
    assert "Headset *" in out and "signal" in out and "device busy" in out
    assert meter(Level(-30, -30, -30, True), 10).plain.count("▮") == 5
