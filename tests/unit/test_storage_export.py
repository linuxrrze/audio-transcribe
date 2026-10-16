from transcribe.core import registry
from transcribe.core.config import Config
from transcribe.core.events import EventBus, PartialText
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


def make_transcript():
    return Transcript(
        segments=[
            Segment(0.0, 1.5, [Word(" Hallo", 0.0, 0.7), Word(" zusammen.", 0.7, 1.5)],
                    speaker_id="S1", lang="de"),
            Segment(3725.25, 3727.0, [Word(" Hi!", 3725.25, 3727.0)], speaker_id="S2"),
        ],
        speakers={"S1": Speaker("S1", "Alice", 1), "S2": Speaker("S2", "Speaker 2")},
    )


def test_roundtrip(tmp_path):
    t = make_transcript()
    p = tmp_path / "transcript.json"
    t.save(p)
    assert Transcript.load(p) == t


def test_exporters_via_registry():
    t = make_transcript()
    txt = registry.load("transcribe.exporters", "txt")().render(t)
    assert txt.splitlines() == ["[00:00:00] Alice: Hallo zusammen.",
                                "[01:02:05] Speaker 2: Hi!"]
    srt = registry.load("transcribe.exporters", "srt")().render(t)
    assert "01:02:05,250 --> 01:02:07,000\n[Speaker 2] Hi!" in srt
    vtt = registry.load("transcribe.exporters", "vtt")().render(t)
    assert vtt.startswith("WEBVTT") and "<v Alice>Hallo zusammen." in vtt


def test_config_roundtrip(tmp_path):
    c = Config()
    c.asr.model = "large-v3-turbo"
    p = tmp_path / "config.toml"
    c.save(p)
    assert Config.load(p) == c


def test_event_bus_threaded():
    bus, got = EventBus(), []
    bus.subscribe(got.append)
    bus.start()
    bus.publish(PartialText(t=1.0, text="hi"))
    bus.stop()
    assert got == [PartialText(t=1.0, text="hi")]


def test_exporters_omit_label_without_speaker():
    t = Transcript(segments=[Segment(1.0, 2.0, [Word(" Hallo.", 1.0, 2.0)])])
    for name in ["txt", "md", "srt", "vtt"]:
        out = registry.load("transcribe.exporters", name)().render(t)
        assert "Unknown" not in out and "Hallo." in out
