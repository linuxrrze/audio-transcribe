"""GUI tests (offscreen). Skipped when PySide6 or its system libraries are unavailable."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication
except ImportError as e:  # missing extra or libGL & co.
    pytest.skip(f"Qt unavailable: {e}", allow_module_level=True)

from tests.unit.fakes import FakeEmbedder, ScriptedWorld, fake_vad, utterance  # noqa: E402
from tests.unit.test_live_session import ArraySource  # noqa: E402
from tests.unit.test_live_tracker import threshold_cluster  # noqa: E402
from transcribe.core.config import Config  # noqa: E402
from transcribe.live.session import LiveSession  # noqa: E402
from transcribe.live.streamer import StreamerConfig  # noqa: E402
from transcribe.speakers.live_tracker import LiveSpeakerTracker  # noqa: E402
from transcribe.storage.transcript_model import (  # noqa: E402
    Segment,
    Speaker,
    Transcript,
    Word,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(app, until, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if until():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    c = Config()
    c.sessions_dir = str(tmp_path / "sessions")
    c.speakers.identify_known = False
    c.speakers.store_voiceprints = False
    return c


def test_transcript_view_updates(app):
    from transcribe.gui.transcript_view import TranscriptView

    view = TranscriptView()
    view.add_segment("a", 0.0, 1.0, "S1", "hello there", "mic")
    view.add_segment("b", 2.0, 3.0, "S2", "general kenobi", "mic")
    view.set_partial("mic", "still", "talking")
    view.set_speaker(Speaker("S2", "Obi-Wan"))
    view._render()
    text = view.toPlainText()
    assert "hello there" in text and "Obi-Wan: general kenobi" in text
    assert "still talking" in text
    view.relabel_segment("b", "S1")
    view.remove_segment("a")
    view._render()
    assert "hello there" not in view.toPlainText()
    assert "Speaker 1: general kenobi" in view.toPlainText()
    seeks = []
    view.seek_requested.connect(seeks.append)
    view._on_anchor(QUrl("seek:2.000"))
    assert seeks == [2.0]


def test_speaker_panel_suggestion_and_merge(app, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from transcribe.gui.speaker_panel import SpeakerPanel

    panel = SpeakerPanel()
    panel.set_speakers({"S1": Speaker("S1", "Speaker 1", suggestion="Anna", score=0.7),
                        "S2": Speaker("S2", "Bob")})
    panel.select("S1")
    assert not panel.accept_btn.isHidden() and "Anna" in panel.hint.text()
    accepted, merged = [], []
    panel.accept_requested.connect(accepted.append)
    panel.merge_requested.connect(lambda k, d: merged.append((k, d)))
    panel.accept_btn.click()
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Bob (S2)", True))
    panel.merge_btn.click()
    assert accepted == ["S1"] and merged == [("S2", "S1")]


def test_live_tab_start_stop(app, config, tmp_path):
    from transcribe.gui.live_tab import LiveTab

    words = (utterance("hallo zusammen wie geht es", 0.3, "de")
             + utterance("fine thanks and you", 3.0, "en"))
    world = ScriptedWorld(words)

    def source_factory(kind, inp, out, mix):
        return ArraySource(world.audio(5.5))

    def session_factory(source, cfg, session_dir, num, diarize, meta):
        return LiveSession(source, world, session_dir, ["de", "en"], fake_vad,
                           embedder=FakeEmbedder(),
                           tracker=LiveSpeakerTracker(threshold_cluster),
                           streamer_config=StreamerConfig(step=0.5), meta=meta)

    tab = LiveTab(config, source_factory=source_factory, session_factory=session_factory)
    finished = []
    tab.session_finished.connect(finished.append)
    tab.start()
    assert pump(app, lambda: tab.running and tab.view.segment_count() == 2)
    assert tab.panel.list.count() == 2 and set(tab.meters) == {"default"}
    tab._name_live("S2", "Fiona", enroll=False)  # user names a speaker while recording
    tab.stop()
    assert pump(app, lambda: finished)
    assert not tab.running and tab.start_btn.text() == "Start"
    assert (finished[0] / "transcript.json").exists()
    from transcribe.sessions import load

    assert pump(app, lambda: load(finished[0]).speakers["S2"].label == "Fiona")
    assert load(finished[0]).speakers["S2"].confirmed
    tab.view._render()
    assert "Fiona: fine thanks and you" in tab.view.toPlainText()


def make_session(root, name="2026-01-02_100000_demo"):
    import soundfile as sf

    d = root / name
    d.mkdir(parents=True)
    t = Transcript(segments=[
        Segment(0.0, 1.0, [Word(" Hallo.", 0.0, 1.0, speaker_id="S1")], "S1", id="1"),
        Segment(1.5, 2.5, [Word(" Hi!", 1.5, 2.5, speaker_id="S2")], "S2", id="2")],
        speakers={"S1": Speaker("S1", "Speaker 1"), "S2": Speaker("S2", "Speaker 2")},
        meta={"title": "Demo", "duration_s": 3.0})
    t.save(d / "transcript.json")
    sf.write(d / "audio.flac", np.zeros(48000, np.float32), 16000)
    return d


def test_sessions_tab_lists_opens_and_edits(app, config, tmp_path):
    from transcribe.gui.sessions_tab import SessionsTab
    from transcribe.sessions import load

    d = make_session(tmp_path / "sessions")
    tab = SessionsTab(config)
    assert tab.list.count() == 1 and "Demo" in tab.list.item(0).text()
    tab.list.setCurrentRow(0)
    assert tab.current == d and tab.panel.list.count() == 2
    assert tab.play_btn.isEnabled()  # audio.flac found
    tab._name("S1", "Anna", enroll=False, consent=False)
    assert load(d).speakers["S1"].label == "Anna"
    tab._merge("S1", "S2")
    t = load(d)
    assert set(t.speakers) == {"S1"} and {s.speaker_id for s in t.segments} == {"S1"}


def test_main_window_builds(app, config):
    from transcribe.gui.main_window import MainWindow

    w = MainWindow(config)
    assert w.tabs.count() == 2
    w.close()
