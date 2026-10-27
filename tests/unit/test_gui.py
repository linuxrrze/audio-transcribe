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


def _delete_widgets(app):
    """Delete every top-level widget now, while Qt is intact.

    Widgets left alive until interpreter exit are destroyed in arbitrary order, which
    segfaulted at exit in ~80 % of runs (after all tests had passed).
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    for w in app.topLevelWidgets():
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    _delete_widgets(application)
    application.shutdown()


@pytest.fixture(autouse=True)
def _cleanup_widgets(request):
    yield
    if "app" in request.fixturenames:
        _delete_widgets(request.getfixturevalue("app"))


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
                           tracker=LiveSpeakerTracker(),
                           streamer_config=StreamerConfig(step=0.5), meta=meta)

    tab = LiveTab(config, source_factory=source_factory, session_factory=session_factory,
                  preview=False)
    finished = []
    tab.session_finished.connect(finished.append)
    tab.start()
    assert pump(app, lambda: tab.running and tab.view.segment_count() == 2)
    assert tab.panel.list.count() == 2
    # while recording: naming works, merging needs the finished transcript
    assert not tab.panel.rename_btn.isHidden() and tab.panel.merge_btn.isHidden()
    assert tab.mic_meter.status != "no data"  # session levels drive the meter
    tab._name_live("S2", "Fiona", enroll=False)  # user names a speaker while recording
    tab.stop()
    assert pump(app, lambda: finished)
    assert not tab.running and tab.start_btn.text() == "Start"
    assert (finished[0] / "transcript.json").exists()
    # stopped: no dead buttons, a link to the Sessions tab instead
    assert tab.panel.rename_btn.isHidden() and tab.panel.merge_btn.isHidden()
    assert not tab.panel.note.isHidden() and "Sessions tab" in tab.panel.note.text()
    opened = []
    tab.open_session_requested.connect(opened.append)
    tab.panel.link_activated.emit("sessions")
    assert opened == [finished[0]]
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


def test_live_tab_previews_selected_devices(app, config):
    from tests.unit.test_levels import Tone
    from transcribe.gui.live_tab import LiveTab

    opened = []

    def level_source(entry):
        opened.append((entry.kind, entry.device))
        return Tone(0.5 if entry.kind == "input" else 0.05)

    tab = LiveTab(config, level_source_factory=level_source)
    tab.show()
    assert pump(app, lambda: tab.mic_meter.status == "signal")
    assert tab.sys_meter.status == "no data"  # source "mic": system audio not monitored
    tab.source.setCurrentIndex(tab.source.findData("both"))
    assert pump(app, lambda: tab.sys_meter.status == "signal")
    assert tab.sys_meter.rms_db < tab.mic_meter.rms_db
    assert ("output", None) in opened
    tab.hide()
    assert tab._preview is None  # hidden tab releases the devices


def test_device_check_dialog_use_selects_device(app, config):
    from tests.unit.test_levels import Broken, Tone
    from transcribe.audio.levels import DeviceEntry
    from transcribe.gui.level_meter import DeviceCheckDialog
    from transcribe.gui.live_tab import LiveTab

    entries = [DeviceEntry("in:3", "input", "3", "USB Headset", False),
               DeviceEntry("out:zoom", "output", "zoom", "Zoom output", False),
               DeviceEntry("out:hdmi", "output", "hdmi", "HDMI", True)]
    tab = LiveTab(config, preview=False)
    tab.input.addItem("USB Headset", "3")
    tab.output.addItem("Zoom output", "zoom")
    dlg = DeviceCheckDialog(tab.select_device, devices=(entries, []),
                            source_factory=lambda e: Broken(0) if e.device == "hdmi"
                            else Tone(0.3))
    assert pump(app, lambda: dlg.meters["in:3"].status == "signal"
                and dlg.meters["out:hdmi"].status == "error")
    tab.select_device(entries[1])
    assert tab.output.currentData() == "zoom"
    assert tab.source.currentData() == "both"  # switched so system audio is captured
    dlg.reject()


def test_settings_roundtrip_and_vocabulary_dialog(app, config, monkeypatch):
    from transcribe.gui import settings_dialog
    from transcribe.gui.settings_dialog import SettingsDialog, VocabularyDialog

    config.asr.vocabulary = ["Justus"]
    config.asr.corrections = {"Fanfress": "Fentriss"}
    dlg = SettingsDialog(config)
    assert dlg.tabs.count() == 5
    assert dlg.vocab_label.text() == "1 terms, 1 corrections"
    assert not dlg.wcpp_dtw.isEnabled()  # whisper.cpp options only for that engine
    dlg.engine.setCurrentText("whisper.cpp")
    assert dlg.wcpp_dtw.isEnabled()

    # the vocabulary is edited in its own dialog
    vocab = VocabularyDialog(dlg.vocab)
    assert vocab.editor.toPlainText() == "Justus\nFanfress -> Fentriss"
    vocab.editor.setPlainText("Justus\nDSGVO, FAU\nHitschcock -> Hitchcock")
    assert "3 terms, 1 corrections" in vocab.status.text()
    vocab.editor.setPlainText("\n".join(f"Name{i:03d}" for i in range(120)))
    assert "Only the first" in vocab.status.text()  # more than fits Whisper's prompt

    class Accepting(VocabularyDialog):
        def exec(self):
            self.editor.setPlainText("Justus\nDSGVO, FAU\nHitschcock -> Hitchcock")
            return True

    monkeypatch.setattr(settings_dialog, "VocabularyDialog", Accepting)
    dlg._edit_vocabulary()
    dlg.beam_size.setValue(3)
    dlg.speaker_count.setCurrentIndex(1)
    dlg.step.setValue(0.5)
    dlg.high.setValue(0.9)
    dlg.low.setValue(0.95)  # never above the auto-label threshold
    c = dlg.apply()
    assert c.asr.engine == "whisper.cpp" and c.asr.beam_size == 3
    assert c.asr.vocabulary == ["Justus", "DSGVO", "FAU"]
    assert c.asr.corrections == {"Hitschcock": "Hitchcock"}
    assert c.speakers.speaker_count == "silhouette" and c.live.step == 0.5
    assert c.speakers.match_threshold_low == 0.9
    assert c.live.refine_model == ""  # "(same as live)" is stored as ""
    assert c.ui.language == "auto"


def test_settings_language_and_refine_model(app, config):
    from transcribe.gui.settings_dialog import SettingsDialog

    config.live.refine_model = "large-v3-turbo"
    dlg = SettingsDialog(config)
    assert dlg.refine_model.currentData() == "large-v3-turbo"
    dlg.ui_language.setCurrentIndex(dlg.ui_language.findData("de"))
    assert dlg.ui_language.currentText() == "Deutsch"
    c = dlg.apply()
    assert c.ui.language == "de" and c.live.refine_model == "large-v3-turbo"
    dlg.refine_model.setCurrentIndex(0)
    assert dlg.apply().live.refine_model == ""


def test_settings_decimals_accept_point_and_comma(app, config):
    from PySide6.QtCore import QLocale
    from PySide6.QtTest import QTest

    from transcribe.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(config)
    step = dlg.step  # seconds, with a " s" suffix
    for language in (QLocale.Language.German, QLocale.Language.English):
        step.setLocale(QLocale(language))
        for typed in ("0.5", "0,5"):  # the other separator used to be dropped: 5 s
            step.lineEdit().selectAll()
            step.lineEdit().del_()
            QTest.keyClicks(step.lineEdit(), typed)
            step.interpretText()
            assert step.value() == 0.5, (language, typed)


def test_voiceprints_dialog(app, config, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    from transcribe.gui import voiceprints_dialog as vd
    from transcribe.speakers.voiceprint_db import VoiceprintDB

    db = VoiceprintDB(tmp_path / "vp.sqlite")
    anna = db.add_person("Anna", consent=True)
    db.add_exemplars(anna.id, "m1", [np.ones(4, np.float32), np.arange(4, dtype=np.float32)],
                     [6.0, 4.0], source_session="2026-01-01_120000_demo")
    db.add_person("Ben")
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    dlg = vd.VoiceprintsDialog(config, db=db)
    assert dlg.table.rowCount() == 2
    info = dlg.findChildren(vd.QLabel)[0].text()
    assert "2 voiceprint(s) were made with another voice model" in info  # m1 != titanet
    assert [dlg.table.item(0, c).text() for c in range(4)] == ["Anna", "2", "0.2 min",
                                                               dlg.table.item(0, 3).text()]
    assert dlg.table.item(1, 3).text() == "no consent"
    assert dlg.table.item(0, 4).text() == "2026-01-01_120000_demo"

    dlg.table.selectRow(1)  # Ben
    assert dlg.buttons["rename"].isEnabled()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Benjamin", True))
    dlg._rename()
    dlg._consent()  # Yes: consent recorded
    assert db.find_person("Benjamin").consent_at
    monkeypatch.setattr(QInputDialog, "getMultiLineText",
                        lambda *a, **k: ("Podcast guest", True))
    dlg._notes()
    assert db.find_person("Benjamin").notes == "Podcast guest"
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Anna", True))
    dlg._merge()  # Benjamin is Anna
    assert [p.display_name for p in db.persons()] == ["Anna"]

    dlg.table.selectRow(0)
    details = vd.VoiceprintDetailsDialog(db, db.find_person("Anna"))
    assert details.table.rowCount() == 2
    details.table.selectRow(0)
    details._delete()
    assert len(db.voiceprint_info(anna.id)) == 1

    out = tmp_path / "export.json"
    monkeypatch.setattr(vd.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    dlg._export()
    assert out.exists() and "Exported 1" in dlg.status.text()
    dlg._consent()  # withdraw: voiceprints deleted
    assert db.voiceprint_info(anna.id) == [] and not db.find_person("Anna").consent_at
    monkeypatch.setattr(vd.QFileDialog, "getOpenFileName", lambda *a, **k: (str(out), ""))
    dlg._import()
    assert len(db.voiceprint_info(anna.id)) == 1
    dlg._delete()
    assert db.persons() == []


def test_sessions_tab_line_corrections(app, config, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    from transcribe.gui.sessions_tab import SessionsTab
    from transcribe.sessions import load

    d = make_session(tmp_path / "sessions")
    tab = SessionsTab(config)
    tab.list.setCurrentRow(0)
    view = tab.view
    assert view.editable and len(view._order) == len(load(d).segments)
    first, second = view._order[0], view._order[1]
    # the block under a point maps to its segment
    view.resize(600, 400)
    rect = view.cursorRect(view.textCursor())
    assert view.segments_at(rect.center()) == [first]

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Carlos", True))
    view.assign_requested.emit([first, second], "")
    t = load(d)
    carlos = next(sid for sid, s in t.speakers.items() if s.label == "Carlos")
    assert [s.speaker_id for s in t.segments[:2]] == [carlos, carlos]
    assert tab.panel.list.count() == len(t.speakers)

    monkeypatch.setattr(QInputDialog, "getMultiLineText",
                        lambda *a, **k: ("Erste Zeile\nzweite Zeile", True))
    view.edit_requested.emit(first)
    t = load(d)
    assert [s.text for s in t.segments[:2]] == ["Erste Zeile", "zweite Zeile"]
    assert len(view._order) == len(t.segments)


def test_transcript_view_follows_new_lines_until_scrolled_up(app):
    from transcribe.gui.transcript_view import TranscriptView

    view = TranscriptView()
    view.resize(400, 200)
    view.show()
    bar = view.verticalScrollBar()

    def add(i):
        view.add_segment(f"s{i}", float(i), i + 0.5, "S1", f"line {i} " + "word " * 12)
        view._render()
        app.processEvents()

    for i in range(40):
        add(i)
    assert bar.maximum() > 0 and bar.value() == bar.maximum()  # follows the newest line
    bar.setValue(bar.maximum() // 2)  # the user scrolls up to read
    position = bar.value()
    for i in range(40, 45):
        add(i)
    assert bar.value() == position  # not pulled away while reading
    bar.setValue(bar.maximum())  # back at the end: follows again
    for i in range(45, 50):
        add(i)
    assert bar.value() == bar.maximum()
    t = Transcript(segments=[Segment(float(i), i + 0.5, [Word(f" w{i}" * 20, float(i),
                                                              i + 0.5)], "S1", id=str(i))
                             for i in range(40)], speakers={"S1": Speaker("S1", "Anna")})
    view.set_transcript(t)  # a stored transcript opens at the top and stays there
    app.processEvents()
    assert bar.value() == 0


def test_german_ui_with_qt_standard_buttons(app, config):
    from PySide6.QtCore import QLocale, QTranslator
    from PySide6.QtWidgets import QDialogButtonBox

    from transcribe import i18n
    from transcribe.gui.app import install_qt_translations
    from transcribe.gui.settings_dialog import SettingsDialog
    from transcribe.gui.voiceprints_dialog import _decimal

    i18n.set_language("de")
    try:
        assert install_qt_translations(app)
        dlg = SettingsDialog(config)
        assert dlg.windowTitle() == "Einstellungen"
        save = dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
        assert save.text().replace("&", "") == "Speichern"  # Qt's own translation
        assert _decimal(0.2) == "0,2" and _decimal(12) == "12,0"
    finally:
        for t in app.findChildren(QTranslator):
            app.removeTranslator(t)
        QLocale.setDefault(QLocale(QLocale.Language.English))
        i18n.set_language("en")


def test_wayland_textinput_noise_is_silenced(monkeypatch):
    from transcribe.gui.app import quiet_qt_logging

    monkeypatch.delenv("QT_LOGGING_RULES", raising=False)
    quiet_qt_logging()
    assert os.environ["QT_LOGGING_RULES"] == "qt.qpa.wayland.textinput=false"
    monkeypatch.setenv("QT_LOGGING_RULES", "qt.multimedia.*=false")
    quiet_qt_logging()  # keeps the user's own rules
    assert os.environ["QT_LOGGING_RULES"] == ("qt.multimedia.*=false;"
                                              "qt.qpa.wayland.textinput=false")
    monkeypatch.setenv("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=true")
    quiet_qt_logging()  # the user wants to see them
    assert os.environ["QT_LOGGING_RULES"] == "qt.qpa.wayland.textinput=true"


def test_live_tab_offers_apps_as_system_audio(app, config, monkeypatch):
    from transcribe.audio.levels import DeviceEntry
    from transcribe.gui import live_tab
    from transcribe.gui.live_tab import LiveTab

    devices = [DeviceEntry("out:hdmi", "output", "hdmi", "HDMI", True),
               DeviceEntry("out:app:Zoom", "output", "app:Zoom", "App: Zoom (playing)", False)]
    monkeypatch.setattr(live_tab, "list_devices", lambda: (list(devices), []))
    tab = LiveTab(config, preview=False)
    items = [tab.output.itemData(i) for i in range(tab.output.count())]
    assert items == [None, "hdmi", None, "app:Zoom"]  # default, devices, separator, apps
    tab.output.setCurrentIndex(tab.output.findData("app:Zoom"))
    devices.pop()  # Zoom stopped playing: the selection stays
    tab._refresh_apps()
    assert tab.output.currentData() == "app:Zoom"
    tab.select_device(DeviceEntry("out:app:Teams", "output", "app:Teams", "App: Teams", False))
    assert tab.output.currentData() == "app:Teams"
    assert tab.source.currentData() == "both"


def test_setup_wizard_saves_downloads_and_retries(app, config, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from transcribe.core import models
    from transcribe.core.setup import Hardware
    from transcribe.gui.setup_wizard import SetupWizard

    path = tmp_path / "config.toml"
    monkeypatch.setattr(models, "missing_models",
                        lambda *a: ["whisper:base", "nemo-titanet-small"])
    fail = [True]

    def download(names, progress, cancel):  # in the download thread, like the real one
        for name in names:
            progress(name, 50, 100)
            if fail[0] and name == "nemo-titanet-small":
                raise OSError("network down")
            progress(name, 100, 100)

    w = SetupWizard(config, hardware=Hardware(4), download=download, path=path)
    assert w.profile_buttons["fast"].isChecked()  # recommended for 4 cores, no GPU
    assert "(recommended)" in w.profile_buttons["fast"].text()
    assert "cuda" not in w.profile_buttons
    w.profile_buttons["accurate"].setChecked(True)
    w._next()
    w.languages.setCurrentIndex(w.languages.findData("de"))
    w.store.setChecked(False)
    w._next()  # saves the settings, shows the download page
    saved = Config.load(path)
    assert (saved.asr.model, saved.live.refine_model) == ("small", "large-v3-turbo")
    assert saved.asr.languages == ["de"] and not saved.speakers.store_voiceprints
    assert config.live.refine_model == "large-v3-turbo"  # the app's config object too
    assert list(w.bars) == ["whisper:base", "nemo-titanet-small"]
    assert w.next_btn.text() == "Download"

    w._next()
    assert pump(app, lambda: not w.downloading)
    assert "network down" in w.status.text() and w.next_btn.text() == "Retry"
    assert w.bars["whisper:base"].value() == 1000
    assert 0 < w.total.value() < 1000
    fail[0] = False
    w._next()  # retry: only what is still missing
    assert pump(app, lambda: not w.downloading)
    assert w.total.value() == 1000 and w.next_btn.text() == "Finish"
    w._next()
    assert w.result() == QDialog.DialogCode.Accepted


def test_setup_wizard_cancel_and_keep_current(app, config, tmp_path, monkeypatch):
    from transcribe.core import models
    from transcribe.core.setup import Hardware
    from transcribe.gui.setup_wizard import SetupWizard

    path = tmp_path / "config.toml"
    config.asr.model = "medium"  # hand-made settings ...
    config.save(path)
    monkeypatch.setattr(models, "missing_models", lambda *a: ["whisper:medium"])

    def download(names, progress, cancel):
        progress(names[0], 10, 100)
        cancel.wait(10)
        raise models.Cancelled(names[0])

    w = SetupWizard(config, hardware=Hardware(16, cuda_devices=1), download=download,
                    path=path)
    assert w.profile_buttons[""].isChecked()  # ... are kept by default
    assert "cuda" in w.profile_buttons
    w._next()
    w._next()
    assert Config.load(path).asr.model == "medium"
    w._next()
    assert pump(app, lambda: w.bars["whisper:medium"].value() == 100)
    assert w.cancel_btn.isVisibleTo(w) and not w.next_btn.isEnabled()
    w.cancel_download()
    assert pump(app, lambda: not w.downloading)
    assert "cancelled" in w.status.text() and w.next_btn.text() == "Retry"
    w.reject()  # "Later"
