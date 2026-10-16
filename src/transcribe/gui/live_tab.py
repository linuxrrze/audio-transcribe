"""Live tab: choose sources, start/stop, live transcript, speakers, level meters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

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
from transcribe.gui.speaker_panel import SpeakerPanel
from transcribe.gui.transcript_view import TranscriptView
from transcribe.storage import session_store
from transcribe.storage.transcript_model import Speaker

SOURCES = [("Microphone", "mic"), ("System audio (what the computer plays)", "system"),
           ("Microphone + system audio (calls)", "both")]


class LiveTab(QWidget):
    session_finished = Signal(object)  # session dir (Path)
    status_message = Signal(str)

    def __init__(self, config: Config, parent=None,
                 source_factory: Callable | None = None,
                 session_factory: Callable | None = None) -> None:
        super().__init__(parent)
        from transcribe.audio.factory import build_source
        from transcribe.live.factory import build_live_session

        self.config = config
        self.source_factory = source_factory or build_source
        self.session_factory = session_factory or build_live_session
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
        self.refresh_devices()

        self.meters: dict[str, QProgressBar] = {}
        self.meter_box = QHBoxLayout()
        self.status = QLabel("Ready")
        self.view = TranscriptView()
        self.panel = SpeakerPanel()
        self.panel.rename_requested.connect(self._rename)
        self.panel.accept_requested.connect(self._accept)
        self.panel.decline_requested.connect(self._decline)
        self.panel.merge_requested.connect(
            lambda *_: QMessageBox.information(self, "Merge speakers",
                                               "Merging is available once the session has "
                                               "stopped (Sessions tab)."))

        form = QFormLayout()
        form.addRow("Source:", self.source)
        form.addRow("Microphone:", self.input)
        form.addRow("System audio:", self.output)
        form.addRow("", self.mix)
        form.addRow("Speakers:", self.speakers)
        controls = QHBoxLayout()
        controls.addLayout(form, 1)
        right = QVBoxLayout()
        right.addWidget(self.start_btn)
        right.addLayout(self.meter_box)
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
        self.input.clear()
        self.input.addItem("Default microphone", None)
        try:
            from transcribe.audio.mic import list_input_devices

            for d in list_input_devices():
                self.input.addItem(d.name, str(d.index))
        except RuntimeError as e:
            self.input.setToolTip(str(e))
        self.output.clear()
        self.output.addItem("Default output", None)
        try:
            from transcribe.audio.system import output_devices

            for o in output_devices():
                self.output.addItem(o.name, o.id)
        except RuntimeError as e:
            self.output.setToolTip(str(e))

    def _update_controls(self) -> None:
        kind = self.source.currentData()
        running = self.session is not None
        for w in (self.source, self.speakers):
            w.setEnabled(not running)
        self.input.setEnabled(not running and kind in ("mic", "both"))
        self.output.setEnabled(not running and kind in ("system", "both"))
        self.mix.setEnabled(not running and kind == "both")

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
        try:
            source = self.source_factory(self.source.currentData(), self.input.currentData(),
                                         self.output.currentData(), self.mix.isChecked())
        except (RuntimeError, ValueError) as e:
            QMessageBox.warning(self, "Audio source", str(e))
            return
        session_dir = session_store.new_session_dir(self.config.resolved_sessions_dir(), "live")
        self.session_dir = session_dir
        num = self.speakers.value() or None
        self.view.clear_all()
        self.panel.set_speakers({})
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
        self._build_meters(session.channels)
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
        self.start_btn.setText("Start")
        self.start_btn.setEnabled(True)
        self._update_controls()
        self._set_status(f"Saved to {session_dir}")
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
                if bar := self.meters.get(event.channel_id):
                    bar.setValue(max(0, min(60, int(event.rms_db + 60))))
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

    def _build_meters(self, channels: list[str]) -> None:
        for bar in self.meters.values():
            bar.deleteLater()
        self.meters = {}
        for ch in channels:
            bar = QProgressBar()
            bar.setRange(0, 60)
            bar.setTextVisible(True)
            bar.setFormat(ch)
            bar.setMaximumWidth(160)
            self.meter_box.addWidget(bar)
            self.meters[ch] = bar

    # --- speaker actions (live: applied in memory, voiceprints learned after stop) -------
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
        if self.session is not None:
            self.session.stop(refine=False)
            self.session = None
