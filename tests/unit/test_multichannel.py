import sys

import numpy as np
import pytest

from tests.unit.fakes import FakeEmbedder, ScriptedWorld, W, fake_vad, utterance
from tests.unit.test_live_tracker import threshold_cluster
from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo
from transcribe.audio.command import CommandSource
from transcribe.audio.mixer import MixDownSource, MixerSource
from transcribe.core.events import SegmentRemoved
from transcribe.live.echo import is_echo, suppress_echo
from transcribe.live.session import LiveSession
from transcribe.live.streamer import StreamerConfig
from transcribe.speakers.live_tracker import LiveSpeakerTracker
from transcribe.storage.recorder import recorded_channels
from transcribe.storage.transcript_model import Segment, Word


class Chan:
    """Array-backed source with a fixed channel id."""

    def __init__(self, audio, channel, frame=0.1, stop_after=None):
        self.audio, self.channel, self.n = audio, channel, int(frame * SAMPLE_RATE)
        self.stop_after = stop_after

    def describe(self):
        return SourceInfo(self.channel, [self.channel], realtime=False)

    def open(self):
        pass

    def frames(self):
        end = len(self.audio) if self.stop_after is None else int(self.stop_after * SAMPLE_RATE)
        for i in range(0, end, self.n):
            yield AudioFrame(self.audio[i : i + self.n], i / SAMPLE_RATE, self.channel)

    def close(self):
        pass


def test_mixer_keeps_channels():
    mixer = MixerSource([Chan(np.ones(1600, np.float32), "mic"),
                         Chan(np.full(3200, 0.5, np.float32), "system")])
    assert mixer.describe().channels == ["mic", "system"]
    mixer.open()
    got = {}
    for f in mixer.frames():
        got[f.channel_id] = got.get(f.channel_id, 0) + len(f.samples)
    assert got == {"mic": 1600, "system": 3200}


def test_mixer_rejects_duplicate_channels():
    with pytest.raises(ValueError):
        MixerSource([Chan(np.zeros(10), "mic"), Chan(np.zeros(10), "mic")])


def test_mixdown_sums_and_pads_silent_channel():
    # "system" delivers nothing after 0.2 s (like WASAPI loopback during silence).
    a = np.full(SAMPLE_RATE, 0.25, np.float32)
    mix = MixDownSource(MixerSource([Chan(a, "mic"), Chan(a, "system", stop_after=0.2)]))
    mix.open()
    out = np.concatenate([f.samples for f in mix.frames()])
    assert len(out) == SAMPLE_RATE
    assert np.allclose(out[: int(0.2 * SAMPLE_RATE)], 0.5)
    assert np.allclose(out[int(0.6 * SAMPLE_RATE) :], 0.25)


def test_command_source_reads_float_pcm():
    script = ("import sys, struct; "
              "sys.stdout.buffer.write(struct.pack('<4000f', *([0.5] * 4000)))")
    src = CommandSource([sys.executable, "-c", script], channel_id="system")
    src.open()
    samples = np.concatenate([f.samples for f in src.frames()])
    src.close()
    assert len(samples) == 4000 and np.allclose(samples, 0.5)


def test_command_source_reports_failures():
    src = CommandSource([sys.executable, "-c", "import sys; sys.exit(3)"])
    src.open()
    with pytest.raises(RuntimeError, match="exited with 3"):
        list(src.frames())


def seg(ch, t0, t1, text):
    return Segment(t0, t1, [Word(" " + text, t0, t1)], channel_id=ch)


def test_echo_detection():
    system = seg("system", 1.0, 4.0, "we should meet again next week")
    assert is_echo(seg("mic", 1.2, 4.1, "we should meet again next week"), system)
    assert is_echo(seg("mic", 1.2, 4.1, "we should meet again next"), system)
    assert not is_echo(seg("mic", 1.2, 4.1, "yes that works for me"), system)  # crosstalk
    assert not is_echo(seg("mic", 9.0, 12.0, "we should meet again next week"), system)
    kept, dropped = suppress_echo([system, seg("mic", 1.1, 4.0, "we should meet again")])
    assert kept == [system] and len(dropped) == 1


def two_channel_world():
    remote = utterance("good morning everyone how are you", 0.3, "en")
    local = utterance("guten morgen mir geht es gut", 3.5, "de")
    echo = [W(w.text, w.t0 + 0.05, w.t1 + 0.05, w.lang) for w in remote]  # mic hears speakers
    system_world, mic_world = ScriptedWorld(remote), ScriptedWorld(echo + local)
    return system_world, mic_world


class RoutedAsr:
    """Fake ASR answering from the world of the channel being decoded."""

    name = "routed"

    def __init__(self, worlds):
        self.worlds, self.current = worlds, None

    def detect_language(self, audio, languages):
        return next(iter(self.worlds.values())).detect_language(audio, languages)

    def transcribe(self, audio, offset=0.0, language=None, prompt=None, progress=None):
        return self.worlds[self.current].transcribe(audio, offset, language)


def test_two_channel_session_suppresses_echo(tmp_path):
    system_world, mic_world = two_channel_world()
    source = MixerSource([Chan(mic_world.audio(6.5), "mic"),
                          Chan(system_world.audio(6.5), "system")])
    asr = RoutedAsr({"mic": mic_world, "system": system_world})
    session = LiveSession(source, asr, tmp_path, ["de", "en"], fake_vad,
                          embedder=FakeEmbedder(),
                          tracker=LiveSpeakerTracker(threshold_cluster),
                          streamer_config=StreamerConfig(step=0.5))
    for ch, st in session.streamers.items():  # route decodes to the channel's world
        original = st.asr

        class Bound:
            name = "bound"

            def __init__(self, channel, inner=original):
                self.channel, self.inner = channel, inner

            def detect_language(self, audio, languages):
                return self.inner.detect_language(audio, languages)

            def transcribe(self, audio, offset=0.0, language=None, prompt=None,
                           progress=None):
                self.inner.current = self.channel
                return self.inner.transcribe(audio, offset, language)

        st.asr = Bound(ch)
    events = []
    session.bus.subscribe(events.append)
    final = session.run_sync()
    texts = [(s.channel_id, s.text) for s in final.segments]
    assert texts == [("system", "good morning everyone how are you"),
                     ("mic", "guten morgen mir geht es gut")]
    assert final.meta["echo_segments_dropped"] == 1
    assert sorted(recorded_channels(tmp_path)) == ["mic", "system"]
    removed = [e for e in events if isinstance(e, SegmentRemoved)]
    assert len(removed) <= 1  # echo dropped on arrival, or withdrawn when system arrived
