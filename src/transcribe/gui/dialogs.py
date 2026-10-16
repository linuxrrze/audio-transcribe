"""Dialogs: naming a speaker (with consent), settings, known people."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from transcribe.core.config import Config
from transcribe.core.vocabulary import Vocabulary

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]


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
                      "be deleted at any time (Known people).")
        info.setWordWrap(True)
        if not store_allowed:
            info.setText("Voiceprint storage is disabled in the settings.")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Name:", self.name)
        form.addRow(self.learn)
        form.addRow(self.consent)
        form.addRow(info)
        form.addRow(self.buttons)
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


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config
        self.model = QComboBox()
        self.model.addItems(WHISPER_MODELS)
        self.model.setEditable(True)
        self.model.setCurrentText(config.asr.model)
        self.refine_model = QComboBox()
        self.refine_model.addItems(["(same as live)", *WHISPER_MODELS])
        self.refine_model.setCurrentText(config.live.refine_model or "(same as live)")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.device.setCurrentText(config.asr.device)
        self.languages = QLineEdit(",".join(config.asr.languages))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(config.speakers.diarize_threshold)
        self.vocabulary = QPlainTextEdit(Vocabulary.from_config(
            config.asr.vocabulary, config.asr.corrections).to_text())
        self.vocabulary.setPlaceholderText("Justus\nFentriss\nDSGVO\nFanfress -> Fentriss")
        self.vocabulary.setToolTip("Names, acronyms, jargon - one per line. "
                                   "'wrong -> right' corrects a word the model mishears.")
        self.vocabulary.setFixedHeight(90)
        self.refine = QCheckBox("Refine transcript when a live session stops")
        self.refine.setChecked(config.live.refine)
        self.identify = QCheckBox("Recognise known people")
        self.identify.setChecked(config.speakers.identify_known)
        self.store = QCheckBox("Store voiceprints (needed to learn voices)")
        self.store.setChecked(config.speakers.store_voiceprints)
        self.encrypt = QCheckBox("Encrypt voiceprints (key in OS keyring)")
        self.encrypt.setChecked(config.speakers.encrypt_voiceprints)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Whisper model (live):", self.model)
        form.addRow("Model for refinement:", self.refine_model)
        form.addRow("Compute device:", self.device)
        form.addRow("Languages:", self.languages)
        form.addRow("Speaker threshold:", self.threshold)
        form.addRow("Vocabulary:", self.vocabulary)
        form.addRow(self.refine)
        form.addRow(QLabel("<b>Voiceprints (biometric data)</b>"))
        form.addRow(self.identify)
        form.addRow(self.store)
        form.addRow(self.encrypt)
        form.addRow(buttons)

    def apply(self) -> Config:
        c = self.config
        c.asr.model = self.model.currentText().strip() or c.asr.model
        refine = self.refine_model.currentText()
        c.live.refine_model = "" if refine.startswith("(") else refine
        c.asr.device = self.device.currentText()
        c.asr.languages = [x.strip() for x in self.languages.text().split(",") if x.strip()]
        c.speakers.diarize_threshold = self.threshold.value()
        vocab = Vocabulary.parse(self.vocabulary.toPlainText())
        c.asr.vocabulary, c.asr.corrections = vocab.terms, vocab.corrections
        c.live.refine = self.refine.isChecked()
        c.speakers.identify_known = self.identify.isChecked()
        c.speakers.store_voiceprints = self.store.isChecked()
        c.speakers.encrypt_voiceprints = self.encrypt.isChecked()
        return c

    def _save(self) -> None:
        self.apply().save()
        self.accept()


class KnownPeopleDialog(QDialog):
    """Persons with stored voiceprints: delete or withdraw consent (GDPR)."""

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        from transcribe.speakers.identity import open_db

        self.setWindowTitle("Known people")
        self.resize(560, 320)
        self.db = open_db(config)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Voiceprints", "Consent since"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        delete = QPushButton("Delete person…")
        delete.clicked.connect(self._delete)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(delete)
        row.addStretch(1)
        row.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(row)
        self.reload()

    def reload(self) -> None:
        counts = self.db.voiceprint_counts()
        persons = self.db.persons()
        self.table.setRowCount(len(persons))
        for r, p in enumerate(persons):
            for c, value in enumerate([p.display_name, str(counts.get(p.id, 0)),
                                       p.consent_at or "-"]):
                self.table.setItem(r, c, QTableWidgetItem(value))

    def _delete(self) -> None:
        item = self.table.item(self.table.currentRow(), 0)
        if item is None:
            return
        name = item.text()
        answer = QMessageBox.question(self, "Delete person",
                                      f"Delete {name} and all stored voiceprints?")
        if answer == QMessageBox.StandardButton.Yes:
            person = self.db.find_person(name)
            if person:
                self.db.delete_person(person.id)
            self.reload()

    def done(self, result: int) -> None:
        self.db.close()
        super().done(result)


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
            done = len(names) - len(models.missing_models(
                [n.split(":", 1)[1] for n in names if n.startswith("whisper:")],
                [n for n in names if not n.startswith("whisper:")]))
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
