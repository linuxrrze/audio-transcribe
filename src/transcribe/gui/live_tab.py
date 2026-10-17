"""Live tab: choose sources, start/stop, live transcript, speakers, level meters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from transcribe.audio.base import AudioSource
from transcribe.audio.levels import DeviceEntry, LevelMonitor, device_source, list_devices
from transcribe.core.config import Config
from transcribe.core.events import (
    AudioLevel,
    Event,
    FinalSegment,
    PartialText,
    SegmentRelabeled,
    SegmentRemoved,
    SpeakerDetected,
    SpeakerIdentified,
    StatusChanged,
)
from transcribe.gui.bridge import EventBridge, run_job
from transcribe.gui.dialogs import NameSpeakerDialog
from transcribe.gui.level_meter import DeviceCheckDialog, LevelMeter
from transcribe.gui.speaker_panel import SpeakerPanel
from transcribe.gui.transcript_view import TranscriptView
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Speaker

SOURCES = [("Microphone", "mic"), ("System audio (what the computer plays)", "system"),
           ("Microphone + system audio (calls)", "both")]


class LiveTab(QWidget):
    session_finished = Signal(object)  # session dir (Path)
    open_session_requested = Signal(object)  # session dir: show it in the Sessions tab
    status_message = Signal(str)

    def __init__(self, config: Config, parent=None,
                 source_factory: Callable | None = None,
                 session_factory: Callable | None = None,
                 level_source_factory: Callable[[DeviceEntry], AudioSource] | None = None,
                 preview: bool = True) -> None:
        super().__init__(parent)
        from transcribe.audio.factory import build_source
        from transcribe.live.factory import build_live_session

        self.config = config
        self.source_factory = source_factory or build_source
        self.session_factory = session_factory or build_live_session
        self.level_source_factory = level_source_factory or device_source
        self.preview_enabled = preview
        self._preview: LevelMonitor | None = None
        self._preview_keys: dict[str, LevelMeter] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self.session = None
        self.session_dir: Path | None = None
        self.pending_names: dict[str, tuple[str, bool]] = {}  # sid -> (name, enroll)
        self.bridge = EventBridge(self)
        self.bridge.received.connect(self._on_event)
        self._unsubscribe: Callable[[], None] | None = None

        self.source = QComboBox()
        for label, key in SOURCES:
            self.source.addItem(label, key)
        self.input = QComboBox()
        self.output = QComboBox()
        self.mix = QCheckBox("Mix into one stream (cheaper)")
        self.speakers = QSpinBox()
        self.speakers.setRange(0, 20)
        self.speakers.setSpecialValueText("auto")
        self.start_btn = QPushButton("Start")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.toggle)
        self.source.currentIndexChanged.connect(self._update_controls)
        self.mic_meter, self.sys_meter = LevelMeter(), LevelMeter()
        self.check_btn = QPushButton("Check devices…")
        self.check_btn.setToolTip("Show signal levels of all microphones and outputs")
        self.check_btn.clicked.connect(self.check_devices)
        self.refresh_devices()
        for combo in (self.source, self.input, self.output):
            combo.currentIndexChanged.connect(self._restart_preview)

        self.status = QLabel("Ready")
        self.view = TranscriptView()
        self.panel = SpeakerPanel()
        self.panel.rename_requested.connect(self._rename)
        self.panel.accept_requested.connect(self._accept)
        self.panel.decline_requested.connect(self._decline)
        self.panel.link_activated.connect(self._open_in_sessions)
        self._live_panel()

        def with_meter(combo: QComboBox, meter: LevelMeter) -> QHBoxLayout:
            row = QHBoxLayout()
            row.addWidget(combo, 3)
            row.addWidget(meter, 1)
            return row

        form = QFormLayout()
        form.addRow("Source:", self.source)
        form.addRow("Microphone:", with_meter(self.input, self.mic_meter))
        form.addRow("System audio:", with_meter(self.output, self.sys_meter))
        form.addRow("", self.mix)
        form.addRow("Speakers:", self.speakers)
        controls = QHBoxLayout()
        controls.addLayout(form, 1)
        right = QVBoxLayout()
        right.addWidget(self.start_btn)
        right.addWidget(self.check_btn)
        right.addStretch(1)
        controls.addLayout(right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)
        self._update_controls()

    # --- devices -----------------------------------------------------------------------
    def refresh_devices(self) -> None:
        entries, errors = list_devices()
        for combo, kind, default in ((self.input, "input", "Default microphone"),
                                     (self.output, "output", "Default output")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(default, None)
            for e in entries:
                if e.kind == kind:
                    combo.addItem(e.name, e.device)
            combo.setToolTip("\n".join(x for x in errors if x.startswith(kind)))
            combo.blockSignals(False)

    def select_device(self, entry: DeviceEntry) -> None:
        """Select ``entry`` in its combo and switch the source to include it."""
        combo = self.input if entry.kind == "input" else self.output
        index = combo.findData(entry.device)
        if index >= 0:
            combo.setCurrentIndex(index)
        kind = self.source.currentData()
        wanted = "mic" if entry.kind == "input" else "system"
        if kind not in (wanted, "both"):
            self.source.setCurrentIndex(self.source.findData("both"))

    def check_devices(self) -> None:
        self._stop_preview()  # the dialog opens every device itself
        DeviceCheckDialog(self.select_device, self,
                          source_factory=self.level_source_factory).exec()
        self._restart_preview()

    # --- level preview while idle -------------------------------------------------------
    def _selected_entries(self) -> dict[str, tuple[DeviceEntry, LevelMeter]]:
        kind = self.source.currentData()
        out = {}
        if kind in ("mic", "both"):
            dev = self.input.currentData()
            out["mic"] = (DeviceEntry("mic", "input", dev, self.input.currentText(), False),
                          self.mic_meter)
        if kind in ("system", "both"):
            dev = self.output.currentData()
            out["system"] = (DeviceEntry("system", "output", dev, self.output.currentText(),
                                         False), self.sys_meter)
        return out

    def _restart_preview(self, *_) -> None:
        self._stop_preview()
        if not self.preview_enabled or self.running or not self.isVisible():
            return
        selected = self._selected_entries()
        self._preview_keys = {key: meter for key, (_, meter) in selected.items()}
        self._preview = LevelMonitor({key: self.level_source_factory(entry)
                                      for key, (entry, _) in selected.items()}).start()
        self._preview_timer.start()

    def _stop_preview(self, reset_meters: bool = True) -> None:
        self._preview_timer.stop()
        if self._preview is not None:
            self._preview.stop()
            self._preview = None
        if reset_meters:
            for meter in (self.mic_meter, self.sys_meter):
                meter.reset()

    def _refresh_preview(self) -> None:
        if self._preview is not None:
            for key, level in self._preview.snapshot().items():
                self._preview_keys[key].set_level(level)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._restart_preview()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().hideEvent(event)
        # only release the devices: at application exit the meters may already be gone
        self._stop_preview(reset_meters=False)

    def _update_controls(self) -> None:
        kind = self.source.currentData()
        running = self.session is not None
        for w in (self.source, self.speakers):
            w.setEnabled(not running)
        self.input.setEnabled(not running and kind in ("mic", "both"))
        self.output.setEnabled(not running and kind in ("system", "both"))
        self.mix.setEnabled(not running and kind == "both")
        self.check_btn.setEnabled(not running)
        self.mic_meter.setEnabled(kind in ("mic", "both"))
        self.sys_meter.setEnabled(kind in ("system", "both"))

    # --- start / stop ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.session is not None

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        self._stop_preview()  # release the devices for the session
        try:
            source = self.source_factory(self.source.currentData(), self.input.currentData(),
                                         self.output.currentData(), self.mix.isChecked())
        except (RuntimeError, ValueError) as e:
            QMessageBox.warning(self, "Audio source", str(e))
            self._restart_preview()
            return
        session_dir = session_store.new_session_dir(self.config.resolved_sessions_dir(), "live")
        self.session_dir = session_dir
        num = self.speakers.value() or None
        self.view.clear_all()
        self.panel.set_speakers({})
        self._live_panel()
        self.pending_names.clear()
        self.start_btn.setEnabled(False)
        self._set_status("Loading models (first run downloads them)…")
        run_job(lambda: self.session_factory(source, self.config, session_dir, num,
                                             True, {"title": "Live session",
                                                    "source": source.describe().name}),
                self._started, self._failed, self)

    def _started(self, session) -> None:
        self.session = session
        self._unsubscribe = self.bridge.attach(session.bus)
        try:
            session.start()
        except RuntimeError as e:
            self.session = None
            self._failed(str(e))
            return
        self.start_btn.setText("Stop")
        self.start_btn.setEnabled(True)
        self._update_controls()

    def stop(self) -> None:
        if self.session is None:
            return
        self.start_btn.setEnabled(False)
        self._set_status("Stopping…")
        session = self.session
        run_job(lambda: session.stop(refine=self.config.live.refine), self._stopped,
                self._failed, self)

    def _stopped(self, transcript) -> None:
        session_dir = self.session_dir
        if self._unsubscribe:
            self._unsubscribe()
        self.session = None
        self.view.set_transcript(transcript)
        self.panel.set_speakers(transcript.speakers)
        # the session is closed: speakers are edited in the Sessions tab
        self.panel.set_actions(rename=False, merge=False, suggestions=False)
        self.panel.set_note('Session saved. <a href="sessions">Name, merge or correct '
                            'speakers in the Sessions tab</a>.')
        self.start_btn.setText("Start")
        self.start_btn.setEnabled(True)
        self._update_controls()
        self._set_status(f"Saved to {session_dir}")
        self._restart_preview()
        if self.pending_names and session_dir is not None:
            self._apply_pending_names(session_dir)
        self.session_finished.emit(session_dir)

    def _apply_pending_names(self, session_dir: Path) -> None:
        from transcribe.speakers.identity import name_speaker

        names, self.pending_names = dict(self.pending_names), {}

        def apply() -> list[str]:
            problems = []
            for sid, (name, enroll) in names.items():
                try:
                    name_speaker(session_dir, session_store.resolve_transcript(session_dir),
                                 sid, name, self.config, consent=enroll, enroll=enroll)
                except (KeyError, ValueError, PermissionError, FileNotFoundError) as e:
                    problems.append(f"{name}: {e}")
                    name_speaker(session_dir, session_store.resolve_transcript(session_dir),
                                 sid, name, self.config, consent=False, enroll=False)
            return problems

        def done(problems: list[str]) -> None:
            from transcribe.sessions import load

            final = load(session_dir)
            self.view.set_transcript(final)
            self.panel.set_speakers(final.speakers)
            if problems:
                QMessageBox.warning(self, "Voiceprints", "\n".join(problems))

        run_job(apply, done, self._failed, self)

    def _failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Stop" if self.running else "Start")
        self._update_controls()
        self._set_status("Error")
        if not self.running:
            self._restart_preview()
        QMessageBox.critical(self, "Error", message)

    # --- events ------------------------------------------------------------------------
    def _on_event(self, event: Event) -> None:
        session = self.session
        match event:
            case PartialText():
                self.view.set_partial(event.channel_id, event.committed, event.pending)
            case FinalSegment(segment=seg):
                self.view.add_segment(seg.id, seg.t0, seg.t1, seg.speaker_id, seg.text,
                                      seg.channel_id)
                if session is not None and seg.speaker_id in session.transcript.speakers:
                    self._show_speaker(session.transcript.speakers[seg.speaker_id])
            case SegmentRelabeled():
                self.view.relabel_segment(event.segment_id, event.speaker_id)
            case SegmentRemoved():
                self.view.remove_segment(event.segment_id)
            case SpeakerDetected():
                self._show_speaker(Speaker(event.speaker_id, event.label))
            case SpeakerIdentified():
                if session is not None and event.speaker_id in session.transcript.speakers:
                    self._show_speaker(session.transcript.speakers[event.speaker_id])
            case AudioLevel():
                # "system" -> system meter; "mic", a mixed stream or a file -> mic meter
                meter = self.sys_meter if event.channel_id == "system" else self.mic_meter
                meter.set_db(event.rms_db)
            case StatusChanged():
                text = event.status.capitalize()
                if event.detail:
                    text += f": {event.detail}"
                if event.progress is not None:
                    text += f" {event.progress:.0%}"
                self._set_status(text)

    def _show_speaker(self, speaker: Speaker) -> None:
        pending = self.pending_names.get(speaker.id)
        if pending:  # user naming wins over automatic labels during the session
            speaker = Speaker(speaker.id, pending[0], speaker.person_id, confirmed=True)
        self.view.set_speaker(speaker)
        self.panel.update_speaker(speaker)

    # --- speaker actions (live: applied in memory, voiceprints learned after stop) -------
    def _live_panel(self) -> None:
        """While recording, speakers can be named (and suggestions answered); merging
        needs the finished transcript."""
        self.panel.set_actions(rename=True, merge=False, suggestions=True)
        self.panel.set_note("")

    def _open_in_sessions(self, _link: str = "") -> None:
        if self.session_dir is not None:
            self.open_session_requested.emit(self.session_dir)

    def _speaker(self, sid: str) -> Speaker | None:
        if self.session is not None:
            return self.session.transcript.speakers.get(sid)
        return None

    def _rename(self, sid: str) -> None:
        spk = self._speaker(sid)
        if spk is None:
            return
        dlg = NameSpeakerDialog(spk.label, spk.suggestion,
                                store_allowed=self.config.speakers.store_voiceprints,
                                parent=self)
        if dlg.exec():
            name, enroll = dlg.result_values()
            self._name_live(sid, name, enroll)

    def _accept(self, sid: str) -> None:
        spk = self._speaker(sid)
        if spk and spk.suggestion:  # known person: consent already on record
            self._name_live(sid, spk.suggestion, self.config.speakers.store_voiceprints)

    def _decline(self, sid: str) -> None:
        spk = self._speaker(sid)
        if spk:
            spk.suggestion = None
            self._show_speaker(spk)

    def _name_live(self, sid: str, name: str, enroll: bool) -> None:
        spk = self._speaker(sid)
        if spk is None:
            return
        spk.label, spk.confirmed, spk.suggestion = name, True, None
        self.pending_names[sid] = (name, enroll)
        self._show_speaker(spk)

    def _set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status_message.emit(text)

    def shutdown(self) -> None:
        """Stop a running session synchronously (window closing)."""
        self._stop_preview()
        if self.session is not None:
            self.session.stop(refine=False)
            self.session = None
