import json

import pytest

from tests.unit.fakes import FakeEmbedder, ScriptedWorld, fake_vad, utterance
from tests.unit.test_live_tracker import threshold_cluster
from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo
from transcribe.core.events import FinalSegment, SpeakerDetected, StatusChanged
from transcribe.live.session import LIVE_TRANSCRIPT, LiveSession
from transcribe.live.streamer import StreamerConfig
from transcribe.speakers.live_tracker import LiveSpeakerTracker
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


class ArraySource:
    def __init__(self, audio, frame=0.1):
        self.audio, self.n = audio, int(frame * SAMPLE_RATE)
        self.closed = False

    def describe(self):
        return SourceInfo("array")

    def open(self):
        pass

    def frames(self):
        for i in range(0, len(self.audio), self.n):
            if self.closed:
                return
            yield AudioFrame(self.audio[i : i + self.n], i / SAMPLE_RATE)

    def close(self):
        self.closed = True


WORDS = (utterance("hallo zusammen wie geht es", 0.3, "de")
         + utterance("fine thanks and you", 3.0, "en")
         + utterance("sehr gut danke", 5.5, "de"))


def make_session(tmp_path, refine=None):
    world = ScriptedWorld(WORDS)
    session = LiveSession(ArraySource(world.audio(7.5)), world, tmp_path, ["de", "en"],
                          fake_vad, embedder=FakeEmbedder(),
                          tracker=LiveSpeakerTracker(threshold_cluster), refine=refine,
                          streamer_config=StreamerConfig(step=0.5))
    events = []
    session.bus.subscribe(events.append)
    return session, events


def check_live_result(transcript, tmp_path):
    assert [(s.speaker_id, s.lang, s.text) for s in transcript.segments] == [
        ("S1", "de", "hallo zusammen wie geht es"), ("S2", "en", "fine thanks and you"),
        ("S1", "de", "sehr gut danke")]
    saved = json.loads((tmp_path / LIVE_TRANSCRIPT).read_text())
    assert len(saved["segments"]) == 3
    assert (tmp_path / "audio.flac").exists() and (tmp_path / "transcript.json").exists()


def test_run_sync(tmp_path):
    session, events = make_session(tmp_path)
    final = session.run_sync()
    check_live_result(final, tmp_path)
    assert [e.speaker_id for e in events if isinstance(e, SpeakerDetected)] == ["S1", "S2"]
    assert isinstance(events[-1], StatusChanged) and events[-1].status == "stopped"


def test_threaded_start_wait_stop(tmp_path):
    session, events = make_session(tmp_path)
    session.start()
    assert session.wait(timeout=30)
    final = session.stop()
    check_live_result(final, tmp_path)
    assert sum(isinstance(e, FinalSegment) for e in events) == 3


def test_refinement_replaces_transcript_and_keeps_live_ids(tmp_path):
    def refine(audio, progress):
        assert abs(len(audio) / SAMPLE_RATE - 7.5) < 0.01
        progress("transcribe", 1.0)
        # Refined diarization uses its own ids (swapped) - must be mapped back.
        segs = [Segment(0.3, 2.3, [Word(" Hallo zusammen.", 0.3, 2.3, speaker_id="S2")],
                        speaker_id="S2"),
                Segment(3.0, 4.6, [Word(" Fine, thanks.", 3.0, 4.6, speaker_id="S1")],
                        speaker_id="S1")]
        return Transcript(segments=segs, speakers={"S1": Speaker("S1", "Speaker 1"),
                                                   "S2": Speaker("S2", "Speaker 2")})

    session, events = make_session(tmp_path, refine=refine)
    final = session.run_sync()
    assert [(s.speaker_id, s.text) for s in final.segments] == [
        ("S1", "Hallo zusammen."), ("S2", "Fine, thanks.")]
    assert final.meta["refined"] is True
    assert Transcript.load(tmp_path / "transcript.json").segments[0].text == "Hallo zusammen."
    assert len(Transcript.load(tmp_path / LIVE_TRANSCRIPT).segments) == 3  # live kept
    assert any(isinstance(e, StatusChanged) and e.status == "refining" for e in events)


def test_processing_error_surfaces_on_stop(tmp_path):
    session, _ = make_session(tmp_path)

    def boom(*a, **k):
        raise ValueError("asr exploded")

    session.streamer.asr.transcribe = boom
    session.start()
    session.wait(timeout=30)
    with pytest.raises(RuntimeError, match="live processing failed"):
        session.stop()
