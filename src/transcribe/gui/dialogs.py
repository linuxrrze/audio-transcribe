"""Dialogs: naming a speaker (with consent), first-run model download.

Settings: ``settings_dialog``; stored voiceprints: ``voiceprints_dialog``.
"""

from __future__ import annotations

import html

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class NameSpeakerDialog(QDialog):
    """Name + optional voiceprint enrollment, which requires explicit consent."""

    def __init__(self, current: str, suggestion: str | None = None, can_learn: bool = True,
                 store_allowed: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Name speaker")
        self.name = QLineEdit(suggestion or ("" if current.startswith("Speaker ") else current))
        self.name.setPlaceholderText("Name")
        self.learn = QCheckBox("Remember this voice for future sessions")
        self.consent = QCheckBox("The person agreed to storing their voiceprint")
        self.learn.setEnabled(can_learn and store_allowed)
        self.learn.setChecked(can_learn and store_allowed)
        self.learn.toggled.connect(self._update)
        self.consent.toggled.connect(self._update)
        info = QLabel("Voiceprints are biometric data. They stay on this computer and can "
                      "be deleted at any time (File → Voiceprints).")
        info.setWordWrap(True)
        if not store_allowed:
            info.setText("Voiceprint storage is disabled in the settings.")
        info.setStyleSheet("color: palette(placeholder-text);")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        heading = QLabel(f"<b>Who is {html.escape(current)}?</b>")
        name_row = QFormLayout()
        name_row.addRow("Name:", self.name)
        # A vertical layout, not a form: word-wrapped labels in a QFormLayout get too little
        # height on some platforms and overlap the checkbox above them.
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)
        layout.addWidget(heading)
        layout.addLayout(name_row)
        layout.addWidget(self.learn)
        layout.addWidget(self.consent)
        layout.addSpacing(6)
        layout.addWidget(info)
        layout.addSpacing(6)
        layout.addWidget(self.buttons)
        info.setMinimumWidth(440)  # room for the title and the consent text
        self.name.textChanged.connect(self._update)
        self._update()

    def _update(self, *_) -> None:
        self.consent.setEnabled(self.learn.isChecked())
        ok = bool(self.name.text().strip()) and (not self.learn.isChecked()
                                                 or self.consent.isChecked())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def result_values(self) -> tuple[str, bool]:
        """(name, enroll) - enroll implies consent was given."""
        return self.name.text().strip(), self.learn.isChecked() and self.consent.isChecked()


class ModelSetupDialog(QDialog):
    """First run: download the models the configuration needs (shows progress)."""

    def __init__(self, missing: list[str], parent=None) -> None:
        from PySide6.QtWidgets import QProgressBar

        super().__init__(parent)
        self.setWindowTitle("Download models")
        self.missing = list(missing)
        intro = QLabel("Transcribe runs entirely on this computer. These models are needed "
                       "once (several hundred MB in total):")
        intro.setWordWrap(True)
        self.items = QLabel("\n".join(f"• {m}" for m in self.missing))
        self.bar = QProgressBar()
        self.bar.setRange(0, len(self.missing))
        self.current = QLabel("")
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.start)
        later = QPushButton("Later")
        later.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(later)
        row.addWidget(self.download_btn)
        layout = QVBoxLayout(self)
        for w in (intro, self.items, self.bar, self.current):
            layout.addWidget(w)
        layout.addLayout(row)

    def start(self) -> None:
        from transcribe.core import models
        from transcribe.gui.bridge import Job

        self.download_btn.setEnabled(False)
        names = list(self.missing)
        job = Job(lambda: [models.download(n) for n in names], self)

        # progress per finished item is approximated by polling the cache
        def poll() -> None:
            done = sum(models.model_present(n) if ":" in n else models.is_present(n)
                       for n in names)
            self.bar.setValue(done)
            if done < len(names):
                self.current.setText(f"Downloading {names[done]}…")

        from PySide6.QtCore import QTimer

        timer = QTimer(self)
        timer.timeout.connect(poll)
        timer.start(1000)
        def done(_result: object) -> None:
            timer.stop()
            self.bar.setValue(len(names))
            self.accept()

        def failed(message: str) -> None:
            timer.stop()
            self.current.setText(message)
            self.download_btn.setEnabled(True)

        job.done.connect(done)
        job.failed.connect(failed)
        job.start()
