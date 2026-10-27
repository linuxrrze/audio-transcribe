"""Main window: Live and Sessions tabs, settings, voiceprints."""

from __future__ import annotations

from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from transcribe import __version__
from transcribe.core.config import Config
from transcribe.gui.live_tab import LiveTab
from transcribe.gui.sessions_tab import SessionsTab
from transcribe.gui.settings_dialog import SettingsDialog
from transcribe.gui.voiceprints_dialog import VoiceprintsDialog
from transcribe.i18n import _


class MainWindow(QMainWindow):
    def __init__(self, config: Config, live_tab: LiveTab | None = None) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Transcribe")
        self.resize(1100, 700)
        self.live = live_tab or LiveTab(config)
        self.sessions = SessionsTab(config)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.live, _("Live"))
        self.tabs.addTab(self.sessions, _("Sessions"))
        self.setCentralWidget(self.tabs)
        self.live.session_finished.connect(self.sessions.reload)
        self.live.open_session_requested.connect(self._show_session)
        self.live.status_message.connect(self.statusBar().showMessage)
        self.sessions.status_message.connect(self.statusBar().showMessage)

        menu = self.menuBar().addMenu(_("&File"))
        for text, slot in [(_("&Settings…"), self._settings),
                           (_("&Voiceprints…"), self._people),
                           (_("&Download models…"), self.setup), (_("&Quit"), self.close)]:
            action = QAction(text, self)
            action.triggered.connect(slot)
            menu.addAction(action)
        about = QAction(_("&About"), self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "Transcribe", f"Transcribe {__version__}\n" + _(
                "Local transcription with speaker identification. Audio and voiceprints "
                "never leave this computer.")))
        self.menuBar().addMenu(_("&Help")).addAction(about)

    def _show_session(self, session_dir) -> None:
        self.sessions.reload(session_dir)
        self.tabs.setCurrentWidget(self.sessions)

    def _settings(self) -> None:
        if self.live.running:
            QMessageBox.information(self, _("Settings"), _("Stop the live session first."))
            return
        if SettingsDialog(self.config, self).exec():
            self.sessions.reload()  # the sessions folder may have changed

    def setup(self) -> None:
        """The setup wizard: model for this hardware, languages, privacy, download."""
        from transcribe.gui.setup_wizard import SetupWizard

        if self.live.running:
            QMessageBox.information(self, _("Download models"),
                                    _("Stop the live session first."))
            return
        SetupWizard(self.config, self).exec()

    def _people(self) -> None:
        try:
            VoiceprintsDialog(self.config, self).exec()
        except RuntimeError as e:  # e.g. encrypted database without its key
            QMessageBox.warning(self, _("Voiceprints"), str(e))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.live.running:
            answer = QMessageBox.question(self, _("Quit"),
                                          _("A live session is running. Stop it and quit?"))
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.live.shutdown()  # also stops the idle level preview
        event.accept()
