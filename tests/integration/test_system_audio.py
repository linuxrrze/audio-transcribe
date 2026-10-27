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


def test_app_capture_records_only_that_app(tmp_path):
    """Per-app capture (parec --monitor-stream): started before the app plays, it records
    the app's tone but not another app's, and keeps running after the app quits."""
    from transcribe.audio.apps import list_audio_apps
    from transcribe.audio.system import system_source

    t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    for name, freq in (("app", 440), ("other", 1000)):
        tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(tmp_path / f"{name}.wav", tone, SAMPLE_RATE)

    def play(app: str, file: str) -> subprocess.Popen:
        return subprocess.Popen(["paplay", f"--client-name={app}",
                                 f"--property=application.name={app}",
                                 str(tmp_path / file)])

    src = system_source("app:TranscribeToneApp")
    src.open()
    frames = src.frames()
    got = [next(frames).samples for _ in range(5)]  # nothing plays yet: silence
    players = [play("TranscribeToneApp", "app.wav"), play("OtherApp", "other.wav")]
    try:
        time.sleep(0.3)
        assert {"TranscribeToneApp", "OtherApp"} <= {a.name for a in list_audio_apps()}
        for frame in frames:
            got.append(frame.samples)
            if frame.t_start > 5.5:  # well beyond the 3 s tone: the app has quit
                break
    finally:
        src.close()
        for p in players:
            p.wait(timeout=10)
    audio = np.concatenate(got)
    assert not (np.abs(audio[: 5 * 1600]) > 0.01).any()  # before the app played
    loud = np.abs(audio) > 0.1
    assert loud.sum() > SAMPLE_RATE  # > 1 s of the tone arrived
    start = int(np.argmax(loud))
    window = audio[start : start + SAMPLE_RATE]
    freq = np.argmax(np.abs(np.fft.rfft(window))) * SAMPLE_RATE / len(window)
    assert abs(freq - 440) < 5  # this app's tone, not the other one's 1000 Hz
    assert not loud[-SAMPLE_RATE // 2 :].any()  # silence after the app quit, no error
