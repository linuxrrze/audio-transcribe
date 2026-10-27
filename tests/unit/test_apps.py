"""Per-application capture: pactl parsing, app matching, source selection, the paced
multi-stream source (with a fake capture process instead of parec)."""

import subprocess
import sys
import time

import numpy as np
import pytest

from transcribe.audio import apps
from transcribe.audio.apps import (
    AppAudioSource,
    SinkInput,
    _JitterBuffer,
    group_apps,
    match_app,
    parse_pactl_json,
    parse_pactl_text,
)
from transcribe.audio.base import SAMPLE_RATE

PACTL_JSON = """[{"index":42,"driver":"protocol-native.c","owner_module":"10","client":"25",
"sink":0,"sample_specification":"float32le 2ch 48000Hz","channel_map":"front-left,front-right",
"format":"pcm, format.sample_format = \\"\\\\\\"float32le\\\\\\"\\"","corked":false,
"mute":false,"volume":{"front-left":{"value":65536,"value_percent":"100%","db":"0.00 dB"}},
"balance":0.00,"buffer_latency_usec":0,"sink_latency_usec":0,"resample_method":"",
"properties":{"application.name":"Firefox","application.process.id":"4242",
"application.process.binary":"firefox","media.name":"AudioStream"}},
{"index":57,"driver":"PipeWire","owner_module":"4294967295","client":"80","sink":61,
"corked":true,"properties":{"application.name":"ZOOM VoiceEngine",
"application.process.id":"777","application.process.binary":"zoom"}},
{"index":58,"corked":false,"properties":{"media.name":"Playback"}}]"""

PACTL_TEXT = """Sink Input #42
\tDriver: protocol-native.c
\tOwner Module: 10
\tClient: 25
\tSink: 0
\tSample Specification: float32le 2ch 48000Hz
\tChannel Map: front-left,front-right
\tFormat: pcm, format.sample_format = "\\"float32le\\""
\tCorked: no
\tMute: no
\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
\tProperties:
\t\tmedia.name = "Meeting \\"Weekly\\""
\t\tapplication.name = "Firefox"
\t\tapplication.process.id = "4242"
\t\tapplication.process.binary = "firefox"

Sink Input #57
\tDriver: PipeWire
\tCorked: yes
\tProperties:
\t\tapplication.name = "ZOOM VoiceEngine"
\t\tapplication.process.id = "777"
\t\tapplication.process.binary = "zoom"
"""


def test_parse_pactl_json():
    inputs = parse_pactl_json(PACTL_JSON)
    assert inputs[0] == SinkInput(42, "Firefox", 4242, "firefox", False)
    assert inputs[1] == SinkInput(57, "ZOOM VoiceEngine", 777, "zoom", True)
    assert inputs[2] == SinkInput(58, "Playback", None, "", False)  # media name fallback
    assert parse_pactl_json("[]") == [] and parse_pactl_json("") == []


def test_parse_pactl_text():
    inputs = parse_pactl_text(PACTL_TEXT)
    assert inputs == [SinkInput(42, "Firefox", 4242, "firefox", False),
                      SinkInput(57, "ZOOM VoiceEngine", 777, "zoom", True)]
    assert parse_pactl_text("") == []


def _run_results(*results):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["env"]["LC_ALL"] == "C"  # untranslated text output
        code, out = results[len(calls) - 1]
        return subprocess.CompletedProcess(cmd, code, out, "")
    return run, calls


@pytest.mark.parametrize("first", [(1, ""), (0, "[{invalid")])  # old pactl, broken JSON
def test_pulse_sink_inputs_falls_back_to_text(monkeypatch, first):
    run, calls = _run_results(first, (0, PACTL_TEXT))
    monkeypatch.setattr(apps.shutil, "which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(apps.subprocess, "run", run)
    assert [s.index for s in apps.pulse_sink_inputs()] == [42, 57]
    assert calls == [["pactl", "-f", "json", "list", "sink-inputs"],
                     ["pactl", "list", "sink-inputs"]]


def test_pulse_sink_inputs_json(monkeypatch):
    run, calls = _run_results((0, PACTL_JSON))
    monkeypatch.setattr(apps.shutil, "which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(apps.subprocess, "run", run)
    assert len(apps.pulse_sink_inputs()) == 3 and len(calls) == 1
    monkeypatch.setattr(apps.shutil, "which", lambda tool: None)
    assert apps.pulse_sink_inputs() == []  # no pactl: no apps, no error


def test_match_and_group():
    inputs = [SinkInput(1, "Chromium", 10, "chromium"), SinkInput(2, "Chrome", 20, "chrome"),
              SinkInput(3, "Chrome", 20, "chrome", corked=True),
              SinkInput(4, "ZOOM VoiceEngine", 30, "zoom", corked=True)]
    assert [s.index for s in match_app("chrome", inputs)] == [2, 3]  # exact beats substring
    assert [s.index for s in match_app("chrom", inputs)] == [1, 2, 3]
    assert [s.index for s in match_app("zoom", inputs)] == [4]  # binary name
    assert [s.index for s in match_app("voiceengine", inputs)] == [4]
    assert [s.index for s in match_app("20", inputs)] == [2, 3]  # a pid
    assert match_app("teams", inputs) == []
    grouped = group_apps(inputs)
    assert [(a.name, a.playing) for a in grouped] == [
        ("Chrome", True), ("Chromium", True), ("ZOOM VoiceEngine", False)]
    assert grouped[0].device == "app:Chrome"


def test_app_available(monkeypatch):
    monkeypatch.setattr(apps, "list_audio_apps",
                        lambda: [apps.AudioApp("ZOOM VoiceEngine", 30, False)])
    assert apps.app_available("zoom") and apps.app_available("30")
    assert not apps.app_available("teams") and not apps.app_available("31")


def test_pulse_targets_use_monitor_stream():
    targets = apps.pulse_targets("firefox", lambda: parse_pactl_json(PACTL_JSON))()
    assert list(targets) == ["42"]
    assert "--monitor-stream=42" in targets["42"] and targets["42"][0] == "parec"


def test_mac_targets_resolve_name_to_pid():
    found = [apps.AudioApp("zoom.us", 777, True), apps.AudioApp("Safari", 5, False)]
    targets = apps.mac_targets("zoom", "/x/transcribe-audiotap", lambda: found)()
    assert targets == {"777": ["/x/transcribe-audiotap", "--rate", "16000", "--pid", "777"]}
    assert list(apps.mac_targets("123", "h", lambda: [])()) == ["123"]


def test_source_selection(monkeypatch):
    from transcribe.audio import system
    from transcribe.audio.factory import build_source
    from transcribe.audio.mixer import MixerSource

    monkeypatch.setattr(apps.shutil, "which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(apps.platform, "system", lambda: "Linux")
    src = system.system_source("app:zoom")
    assert isinstance(src, AppAudioSource) and src.describe().name == "app:zoom"
    mixer = build_source("both", None, "app:zoom")  # you on the mic, the app as "system"
    assert isinstance(mixer, MixerSource)
    assert isinstance(mixer.sources[1], AppAudioSource)
    assert mixer.sources[1].channel_id == "system"
    with pytest.raises(ValueError):
        system.system_source("app:")
    monkeypatch.setattr(apps.shutil, "which", lambda tool: None)
    with pytest.raises(RuntimeError, match="pactl"):
        system.system_source("app:zoom")
    monkeypatch.setattr(apps.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="not supported on Windows"):
        system.system_source("app:zoom")


def test_app_entries_in_device_list(monkeypatch):
    from transcribe.audio import levels

    monkeypatch.setattr(apps, "list_audio_apps",
                        lambda: [apps.AudioApp("Firefox", 1, True)])
    monkeypatch.setattr("transcribe.audio.system.output_devices", lambda: [])
    monkeypatch.setattr("transcribe.audio.mic.list_input_devices", lambda: [])
    entries, _errors = levels.list_devices()
    assert [(e.kind, e.device, e.name) for e in entries] == [
        ("output", "app:Firefox", "App: Firefox (playing)")]
    assert levels.list_devices(apps=False)[0] == []


def test_jitter_buffer():
    buf = _JitterBuffer(prebuffer=4, max_backlog=10)
    buf.push(np.ones(3, np.float32))
    assert buf.take(2) is None  # still prebuffering
    buf.push(np.ones(3, np.float32))
    assert buf.take(4).tolist() == [1, 1, 1, 1]
    out = buf.take(4)  # underrun: the rest, then silence; prebuffer again
    assert out.tolist() == [1, 1, 0, 0] and not buf.primed
    buf.push(np.arange(12, dtype=np.float32))  # too much backlog: oldest dropped
    assert buf.size <= 10


TONE = ("import sys, numpy as np; t = np.arange(8000) / 16000; "
        "sys.stdout.buffer.write((0.5 * np.sin(2 * np.pi * 440 * t)).astype('<f4').tobytes())")


def test_app_source_waits_for_the_app_and_survives_its_exit():
    """Silence until the app appears, its audio while it plays, silence after it quits."""
    start = time.monotonic()

    def targets():  # the "app" starts playing after 0.3 s, for 0.5 s, then exits
        elapsed = time.monotonic() - start
        return {"7": [sys.executable, "-c", TONE]} if 0.3 < elapsed < 0.6 else {}

    src = AppAudioSource("test", targets, poll_s=0.05, prebuffer_ms=100, retry_s=60)
    src.open()
    got = []
    try:
        for frame in src.frames():
            got.append(frame.samples)
            if frame.t_start >= 1.9:
                break
    finally:
        src.close()
    audio = np.concatenate(got)
    assert len(audio) == 20 * 1600  # real-time pacing never stalls
    loud = np.abs(audio) > 0.1
    assert not loud[: int(0.3 * SAMPLE_RATE)].any()  # nothing before the app plays
    assert loud.sum() > 0.3 * SAMPLE_RATE  # most of the 0.5 s tone arrived
    assert not loud[-int(0.4 * SAMPLE_RATE):].any()  # silence after it exited
    assert src.captured == {"7"}
    assert not src._procs  # nothing left running


def test_app_source_retries_failed_capture_later():
    calls = []

    def targets():
        calls.append(1)
        return {"9": [sys.executable, "-c", "import sys; sys.exit(3)"]}

    src = AppAudioSource("broken", targets, poll_s=0.02, retry_s=60)
    src.open()
    frames = src.frames()
    for _ in range(5):
        assert not np.any(next(frames).samples)  # silence, no exception
    src.close()
    assert "9" in src._failed and len(calls) > 3  # not restarted every poll
