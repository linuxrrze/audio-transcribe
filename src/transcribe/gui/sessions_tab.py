"""Sessions tab: browse past sessions, play back synced to the text, fix speakers, export."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from transcribe import sessions
from transcribe.core import registry
from transcribe.core.config import Config
from transcribe.export.base import fmt_clock
from transcribe.gui.bridge import run_job
from transcribe.gui.dialogs import NameSpeakerDialog
from transcribe.gui.speaker_panel import SpeakerPanel
from transcribe.gui.transcript_view import TranscriptView
from transcribe.storage.transcript_model import Transcript


class _Progress(QObject):
    changed = Signal(str)


class SessionsTab(QWidget):
    status_message = Signal(str)

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.current: Path | None = None
        self.transcript: Transcript | None = None
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._selected)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        self.view = TranscriptView()
        self.panel = SpeakerPanel()
        self.panel.rename_requested.connect(self._rename)
        self.panel.merge_requested.connect(self._merge)
        self.panel.accept_requested.connect(self._accept)
        self.panel.decline_requested.connect(self._decline)

        self.player, self.audio_out = self._make_player()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(lambda ms: self.player and self.player.setPosition(ms))
        self.time = QLabel("--:--:--")
        self.view.seek_requested.connect(self._seek)

        self.format = QComboBox()
        self.format.addItems(registry.available("transcribe.exporters"))
        self.format.setCurrentText("txt")
        export = QPushButton("Export…")
        export.clicked.connect(self._export)
        self.refine_speakers = QSpinBox()
        self.refine_speakers.setRange(0, 20)
        self.refine_speakers.setSpecialValueText("auto")
        self.refine_btn = QPushButton("Refine")
        self.refine_btn.setToolTip("Re-run diarization + transcription on the recording")
        self.refine_btn.clicked.connect(self._refine)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Sessions</b>"))
        left.addWidget(self.list, 1)
        left.addWidget(refresh)
        left_w = QWidget()
        left_w.setLayout(left)
        transport = QHBoxLayout()
        transport.addWidget(self.play_btn)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.time)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Speakers:"))
        actions.addWidget(self.refine_speakers)
        actions.addWidget(self.refine_btn)
        actions.addStretch(1)
        actions.addWidget(self.format)
        actions.addWidget(export)
        center = QVBoxLayout()
        center.addWidget(self.view, 1)
        center.addLayout(transport)
        center.addLayout(actions)
        center_w = QWidget()
        center_w.setLayout(center)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(center_w)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(1, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.reload()

    # --- list ---------------------------------------------------------------------------
    def reload(self, select: Path | None = None) -> None:
        keep = select or self.current
        self.list.blockSignals(True)
        self.list.clear()
        for info in sessions.list_sessions(self.config.resolved_sessions_dir()):
            flags = (" ✓" if info.refined else "") + (" ●" if info.live else "")
            item = QListWidgetItem(f"{info.title}{flags}\n{info.created:%Y-%m-%d %H:%M}  "
                                   f"{fmt_clock(info.duration_s)}  {info.speakers} speakers")
            item.setData(Qt.ItemDataRole.UserRole, str(info.path))
            self.list.addItem(item)
            if keep and info.path == keep:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        if keep:
            self.open(keep)

    def _selected(self, item: QListWidgetItem | None, _prev=None) -> None:
        if item is not None:
            self.open(Path(item.data(Qt.ItemDataRole.UserRole)))

    def open(self, path: Path) -> None:
        self.current = path
        self.transcript = sessions.load(path)
        self.view.set_transcript(self.transcript)
        self.panel.set_speakers(self.transcript.speakers)
        audio = sessions.playback_audio(path)
        self.play_btn.setEnabled(audio is not None and self.player is not None)
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(str(audio)) if audio else QUrl())

    # --- playback -----------------------------------------------------------------------
    def _make_player(self):
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except ImportError:
            return None, None
        player, out = QMediaPlayer(self), QAudioOutput(self)
        player.setAudioOutput(out)
        player.positionChanged.connect(self._position)
        player.durationChanged.connect(lambda ms: self.slider.setRange(0, ms))
        return player, out

    def _toggle_play(self) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if self.player is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setText("Play")
        else:
            self.player.play()
            self.play_btn.setText("Pause")

    def _seek(self, seconds: float) -> None:
        if self.player is not None:
            self.player.setPosition(int(seconds * 1000))
            if self.play_btn.text() == "Play":
                self._toggle_play()

    def _position(self, ms: int) -> None:
        self.slider.setValue(ms)
        self.time.setText(fmt_clock(ms / 1000))
        self.view.highlight_at(ms / 1000)

    # --- speakers -----------------------------------------------------------------------
    def _after_edit(self) -> None:
        if self.current:
            self.open(self.current)

    def _rename(self, sid: str) -> None:
        if self.current is None or self.transcript is None:
            return
        from transcribe.speakers.identity import EXEMPLARS, EXEMPLARS_ENC

        spk = self.transcript.speakers[sid]
        has_exemplars = any((self.current / n).exists() for n in (EXEMPLARS, EXEMPLARS_ENC))
        dlg = NameSpeakerDialog(spk.label, spk.suggestion, can_learn=has_exemplars,
                                store_allowed=self.config.speakers.store_voiceprints,
                                parent=self)
        if dlg.exec():
            name, enroll = dlg.result_values()
            self._name(sid, name, enroll, consent=enroll)

    def _accept(self, sid: str) -> None:
        if self.transcript and (name := self.transcript.speakers[sid].suggestion):
            self._name(sid, name, self.config.speakers.store_voiceprints, consent=False)

    def _name(self, sid: str, name: str, enroll: bool, consent: bool) -> None:
        from transcribe.speakers.identity import name_speaker

        session = self.current
        assert session is not None
        path = sessions.transcript_path(session)
        try:
            if enroll:
                added = name_speaker(session, path, sid, name, self.config, consent=consent)
                self.status_message.emit(f"{name}: {added} voiceprint(s) stored")
            else:
                sessions.rename_speaker(session, sid, name)
        except (ValueError, PermissionError, FileNotFoundError, KeyError) as e:
            QMessageBox.warning(self, "Voiceprint not stored",
                                f"{e}\n\nThe speaker is renamed without learning the voice.")
            sessions.rename_speaker(session, sid, name)
        self._after_edit()

    def _decline(self, sid: str) -> None:
        if self.current:
            sessions.decline_suggestion(self.current, sid)
            self._after_edit()

    def _merge(self, keep: str, drop: str) -> None:
        if self.current:
            sessions.merge_speakers(self.current, keep, drop)
            self._after_edit()

    # --- export / refine ----------------------------------------------------------------
    def _export(self) -> None:
        if self.transcript is None or self.current is None:
            return
        exporter = registry.load("transcribe.exporters", self.format.currentText())()
        default = str(Path.home() / f"{self.current.name}.{exporter.extension}")
        path, _ = QFileDialog.getSaveFileName(self, "Export transcript", default)
        if path:
            Path(path).write_text(exporter.render(self.transcript), encoding="utf-8")
            self.status_message.emit(f"Exported to {path}")

    def _refine(self) -> None:
        if self.current is None:
            return
        session, num = self.current, self.refine_speakers.value() or None
        progress = _Progress(self)
        progress.changed.connect(self.status_message.emit)
        self.refine_btn.setEnabled(False)

        def work():
            return sessions.refine_session(
                session, self.config, num,
                lambda stage, frac: progress.changed.emit(f"Refining: {stage} {frac:.0%}"))

        def done(_):
            self.refine_btn.setEnabled(True)
            self.status_message.emit("Refined")
            self.reload(session)

        def failed(msg: str):
            self.refine_btn.setEnabled(True)
            QMessageBox.critical(self, "Refine failed", msg)

        run_job(work, done, failed, self)
