"""Dialogs: naming a speaker (with consent).

Settings: ``settings_dialog``; stored voiceprints: ``voiceprints_dialog``; first-run setup
and model download: ``setup_wizard``.
"""

from __future__ import annotations

import html

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from transcribe.i18n import _, display_label


class NameSpeakerDialog(QDialog):
    """Name + optional voiceprint enrollment, which requires explicit consent."""

    def __init__(self, current: str, suggestion: str | None = None, can_learn: bool = True,
                 store_allowed: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Name speaker"))
        self.name = QLineEdit(suggestion or ("" if current.startswith("Speaker ") else current))
        self.name.setPlaceholderText(_("Name"))
        self.learn = QCheckBox(_("Remember this voice for future sessions"))
        self.consent = QCheckBox(_("The person agreed to storing their voiceprint"))
        self.learn.setEnabled(can_learn and store_allowed)
        self.learn.setChecked(can_learn and store_allowed)
        self.learn.toggled.connect(self._update)
        self.consent.toggled.connect(self._update)
        info = QLabel(_("Voiceprints are biometric data. They stay on this computer and "
                        "can be deleted at any time (File → Voiceprints)."))
        info.setWordWrap(True)
        if not store_allowed:
            info.setText(_("Voiceprint storage is disabled in the settings."))
        info.setStyleSheet("color: palette(placeholder-text);")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        who = _("Who is {speaker}?").format(speaker=display_label(current))
        heading = QLabel(f"<b>{html.escape(who)}</b>")
        name_row = QFormLayout()
        name_row.addRow(_("Name:"), self.name)
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

    def _update(self, *_args) -> None:
        self.consent.setEnabled(self.learn.isChecked())
        ok = bool(self.name.text().strip()) and (not self.learn.isChecked()
                                                 or self.consent.isChecked())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def result_values(self) -> tuple[str, bool]:
        """(name, enroll) - enroll implies consent was given."""
        return self.name.text().strip(), self.learn.isChecked() and self.consent.isChecked()

