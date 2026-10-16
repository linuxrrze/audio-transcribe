"""Main window: Live and Sessions tabs, settings, known people."""

from __future__ import annotations

from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from transcribe import __version__
from transcribe.core.config import Config
from transcribe.gui.dialogs import KnownPeopleDialog, SettingsDialog
from transcribe.gui.live_tab import LiveTab
from transcribe.gui.sessions_tab import SessionsTab


class MainWindow(QMainWindow):
    def __init__(self, config: Config, live_tab: LiveTab | None = None) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Transcribe")
        self.resize(1100, 700)
        self.live = live_tab or LiveTab(config)
        self.sessions = SessionsTab(config)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.live, "Live")
        self.tabs.addTab(self.sessions, "Sessions")
        self.setCentralWidget(self.tabs)
        self.live.session_finished.connect(self.sessions.reload)
        self.live.status_message.connect(self.statusBar().showMessage)
        self.sessions.status_message.connect(self.statusBar().showMessage)

        menu = self.menuBar().addMenu("&File")
        for text, slot in [("&Settings…", self._settings), ("&Known people…", self._people),
                           ("&Quit", self.close)]:
            action = QAction(text, self)
            action.triggered.connect(slot)
            menu.addAction(action)
        about = QAction("&About", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "Transcribe", f"Transcribe {__version__}\nLocal transcription with speaker "
            "identification. Audio and voiceprints never leave this computer."))
        self.menuBar().addMenu("&Help").addAction(about)

    def _settings(self) -> None:
        if self.live.running:
            QMessageBox.information(self, "Settings", "Stop the live session first.")
            return
        SettingsDialog(self.config, self).exec()

    def _people(self) -> None:
        try:
            KnownPeopleDialog(self.config, self).exec()
        except RuntimeError as e:
            QMessageBox.warning(self, "Known people", str(e))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.live.running:
            answer = QMessageBox.question(self, "Quit",
                                          "A live session is running. Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.live.shutdown()
        event.accept()
