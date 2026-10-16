"""System-audio capture against a real PulseAudio/PipeWire server (Linux).

Skipped unless `pactl info` succeeds; no models needed.
"""

import shutil
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf

from transcribe.audio.base import SAMPLE_RATE
from transcribe.audio.system import PulseMonitorSource, output_devices


def _pulse_available() -> bool:
    if not (shutil.which("pactl") and shutil.which("parec") and shutil.which("paplay")):
        return False
    return subprocess.run(["pactl", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _pulse_available(), reason="no PulseAudio server")


def test_monitor_capture_records_what_is_played(tmp_path):
    t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(tmp_path / "tone.wav", tone, SAMPLE_RATE)
    default = next(d for d in output_devices() if d.is_default)

    src = PulseMonitorSource(default.id)
    src.open()
    player = subprocess.Popen(["paplay", str(tmp_path / "tone.wav")])
    got, start = [], time.monotonic()
    try:
        for frame in src.frames():
            got.append(frame.samples)
            if time.monotonic() - start > 4:
                break
    finally:
        src.close()
        player.wait(timeout=10)
    audio = np.concatenate(got)
    loud = np.abs(audio) > 0.1
    assert loud.sum() > SAMPLE_RATE  # > 1 s of the tone arrived
    # dominant frequency of a contiguous second of the tone is the 440 Hz we played
    start = int(np.argmax(loud))
    window = audio[start : start + SAMPLE_RATE]
    freq = np.argmax(np.abs(np.fft.rfft(window))) * SAMPLE_RATE / len(window)
    assert abs(freq - 440) < 5
