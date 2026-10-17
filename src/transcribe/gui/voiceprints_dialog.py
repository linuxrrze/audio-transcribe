"""Manage stored voiceprints: people, consent, enrollment, import/export (GDPR-friendly)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from transcribe.core.config import Config, data_dir
from transcribe.gui.bridge import Job, run_job
from transcribe.speakers.voiceprint_db import Person, VoiceprintDB


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.verticalHeader().setVisible(False)
    return t


def _date(iso: str | None) -> str:
    return iso[:10] if iso else "-"


class VoiceprintDetailsDialog(QDialog):
    """One person's voiceprints: when, from where, how long; delete single ones."""

    def __init__(self, db: VoiceprintDB, person: Person, parent=None) -> None:
        super().__init__(parent)
        self.db, self.person = db, person
        self.setWindowTitle(f"Voiceprints of {person.display_name}")
        self.resize(640, 360)
        self.table = _table(["Added", "Source", "Speech", "Channel", "Model"])
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        delete = QPushButton("Delete selected")
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
        infos = self.db.voiceprint_info(self.person.id)
        self.table.setRowCount(len(infos))
        for r, vp in enumerate(infos):
            values = [vp.created_at.replace("T", " ")[:16], vp.source_session or "-",
                      f"{vp.duration_s:.1f} s", vp.channel_type, vp.model_id]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, vp.id)
                self.table.setItem(r, c, item)

    def _delete(self) -> None:
        ids = sorted({i.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)
                      for i in self.table.selectedIndexes()})
        if ids and QMessageBox.question(
                self, "Delete voiceprints",
                f"Delete {len(ids)} voiceprint(s) of {self.person.display_name}?"
        ) == QMessageBox.StandardButton.Yes:
            for vp_id in ids:
                self.db.delete_voiceprint(vp_id)
            self.reload()


class VoiceprintsDialog(QDialog):
    """Everyone with stored voiceprints - add, enroll, rename, merge, consent, delete."""

    def __init__(self, config: Config, parent=None, db: VoiceprintDB | None = None) -> None:
        super().__init__(parent)
        from transcribe.speakers.identity import open_db

        self.config = config
        self.setWindowTitle("Voiceprints")
        self.resize(900, 460)
        self.db = db or open_db(config)
        self._job: Job | None = None
        s = config.speakers
        info = QLabel(
            f"Database: {data_dir() / 'voiceprints.sqlite'} - "
            + ("encrypted" if self.db.cipher.encrypted else "not encrypted")
            + f" - current model: {s.embedding_model}"
            + ("" if s.store_voiceprints else
               " - <b>storing new voiceprints is disabled in the settings</b>"))
        stale = (sum(self.db.voiceprint_counts().values())
                 - sum(self.db.voiceprint_counts(s.embedding_model).values()))
        if stale:  # e.g. enrolled with WeSpeaker ResNet34 before TitaNet became the default
            info.setText(info.text() + f"<br><b>{stale} voiceprint(s) were made with another "
                         "voice model and are not used</b> - enroll these people again "
                         "(Voiceprints… shows the model of each).")
        info.setWordWrap(True)
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.table = _table(["Name", "Voiceprints", "Speech", "Consent since", "Sources",
                             "Notes"])
        header = self.table.horizontalHeader()
        for col in (4, 5):  # long free text: share the remaining width
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda _item: self._details())
        self.status = QLabel()
        side = QVBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key, text, slot, tip in [
            ("add", "Add person…", self._add, "Create an entry (voiceprints come later)"),
            ("enroll", "Enroll from file…", self._enroll,
             "Learn a voice from a recording of this person speaking alone"),
            ("details", "Voiceprints…", self._details, "List and delete single voiceprints"),
            ("rename", "Rename…", self._rename, ""),
            ("merge", "Merge into…", self._merge, "Two entries are the same person"),
            ("consent", "Consent…", self._consent, "Record or withdraw consent"),
            ("notes", "Notes…", self._notes, ""),
            ("delete", "Delete…", self._delete, "Delete the person and all voiceprints"),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            side.addWidget(b)
            self.buttons[key] = b
        side.addStretch(1)
        body = QHBoxLayout()
        body.addWidget(self.table, 1)
        body.addLayout(side)
        import_btn = QPushButton("Import…")
        import_btn.clicked.connect(self._import)
        export_btn = QPushButton("Export…")
        export_btn.setToolTip("Selected people, or everyone if none is selected")
        export_btn.clicked.connect(self._export)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(import_btn)
        bottom.addWidget(export_btn)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addLayout(body, 1)
        layout.addLayout(bottom)
        self.reload()

    # --- table --------------------------------------------------------------------------
    def reload(self, select: str | None = None) -> None:
        stats = self.db.person_stats()
        persons = self.db.persons()
        self.table.setRowCount(len(persons))
        for r, p in enumerate(persons):
            st = stats.get(p.id)
            sources = st.sources if st else []
            values = [p.display_name, str(st.voiceprints if st else 0),
                      f"{st.seconds / 60:.1f} min" if st else "-",
                      _date(p.consent_at) if p.consent_at else "no consent",
                      ", ".join(sources[:3]) + (f" (+{len(sources) - 3})"
                                                if len(sources) > 3 else ""),
                      p.notes]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, p.id)
                if c == 4 and sources:
                    item.setToolTip("\n".join(sources))
                self.table.setItem(r, c, item)
            if p.display_name == select:
                self.table.selectRow(r)
        self._update_buttons()

    def selected(self) -> list[Person]:
        ids = {i.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)
               for i in self.table.selectedIndexes()}
        return [self.db.get_person(pid) for pid in sorted(ids)]

    def _one(self) -> Person | None:
        sel = self.selected()
        return sel[0] if len(sel) == 1 else None

    def _update_buttons(self) -> None:
        n = len(self.selected())
        for key in ("details", "rename", "merge", "consent", "notes"):
            self.buttons[key].setEnabled(n == 1)
        self.buttons["delete"].setEnabled(n >= 1)
        self.buttons["merge"].setEnabled(n == 1 and self.table.rowCount() > 1)
        self.buttons["enroll"].setEnabled(self._job is None)

    def _ask_consent(self, name: str) -> bool:
        return QMessageBox.question(
            self, "Consent",
            f"Has {name} agreed to storing their voiceprint on this computer?\n\n"
            "Voiceprints are biometric data (GDPR Art. 9); store them only with consent."
        ) == QMessageBox.StandardButton.Yes

    # --- actions ------------------------------------------------------------------------
    def _add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add person", "Name:")
        name = name.strip()
        if not ok or not name:
            return
        if self.db.find_person(name):
            QMessageBox.information(self, "Add person", f"{name} already exists.")
            return
        self.db.add_person(name, consent=self._ask_consent(name))
        self.reload(name)

    def _enroll(self) -> None:
        person = self._one()
        name = person.display_name if person else ""
        if not name:
            name, ok = QInputDialog.getText(self, "Enroll", "Name of the person speaking:")
            name = name.strip()
            if not ok or not name:
                return
        path, _ = QFileDialog.getOpenFileName(
            self, f"Recording of {name} speaking alone", str(Path.home()),
            "Audio (*.wav *.mp3 *.m4a *.m4b *.flac *.ogg *.opus);;All files (*)")
        if not path:
            return
        existing = self.db.find_person(name)
        consent = bool(existing and existing.consent_at) or self._ask_consent(name)
        if not consent:
            return
        config = self.config

        def work() -> tuple[int, int]:
            # own connection: sqlite objects must stay in the thread that created them
            from transcribe.audio.file import load_audio
            from transcribe.speakers.identity import enroll_audio, open_db

            db = open_db(config)
            try:
                return enroll_audio(db, name, load_audio(path), True, config,
                                    source=Path(path).name)
            finally:
                db.close()

        def done(result: tuple[int, int]) -> None:
            self._job = None
            self.status.setText(f"{name}: {result[1]} voiceprint(s) added from "
                                f"{result[0]} piece(s) of speech.")
            self.reload(name)

        def failed(message: str) -> None:
            self._job = None
            self.status.setText("")
            self._update_buttons()
            QMessageBox.warning(self, "Enroll", message)

        self.status.setText(f"Learning {name}'s voice from {Path(path).name}…")
        self._job = run_job(work, done, failed, self)
        self._update_buttons()

    def _details(self) -> None:
        if person := self._one():
            VoiceprintDetailsDialog(self.db, person, self).exec()
            self.reload(person.display_name)

    def _rename(self) -> None:
        person = self._one()
        if person is None:
            return
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=person.display_name)
        name = name.strip()
        if not ok or not name or name == person.display_name:
            return
        if self.db.find_person(name):
            QMessageBox.information(self, "Rename",
                                    f"{name} already exists - use Merge into… instead.")
            return
        self.db.rename_person(person.id, name)
        self.reload(name)

    def _merge(self) -> None:
        person = self._one()
        if person is None:
            return
        others = [p.display_name for p in self.db.persons() if p.id != person.id]
        target, ok = QInputDialog.getItem(
            self, "Merge", f"{person.display_name} is the same person as:", others, 0, False)
        if ok and (keep := self.db.find_person(target)):
            self.db.merge_persons(keep.id, person.id)
            self.reload(target)

    def _consent(self) -> None:
        person = self._one()
        if person is None:
            return
        if person.consent_at:
            n = len(self.db.voiceprint_info(person.id))
            if QMessageBox.question(
                    self, "Withdraw consent",
                    f"Withdraw {person.display_name}'s consent? This deletes all {n} "
                    "voiceprint(s); the person is no longer recognised.") \
                    == QMessageBox.StandardButton.Yes:
                self.db.delete_voiceprints_of(person.id)
                self.db.set_consent(person.id, False)
        elif self._ask_consent(person.display_name):
            self.db.set_consent(person.id, True)
        self.reload(person.display_name)

    def _notes(self) -> None:
        person = self._one()
        if person is None:
            return
        notes, ok = QInputDialog.getMultiLineText(self, "Notes", person.display_name,
                                                  person.notes)
        if ok:
            self.db.set_notes(person.id, notes.strip())
            self.reload(person.display_name)

    def _delete(self) -> None:
        persons = self.selected()
        if not persons:
            return
        names = ", ".join(p.display_name for p in persons)
        if QMessageBox.question(self, "Delete", f"Delete {names} and all their voiceprints?"
                                ) == QMessageBox.StandardButton.Yes:
            for p in persons:
                self.db.delete_person(p.id)
            self.reload()

    def _export(self) -> None:
        persons = self.selected()
        path, _ = QFileDialog.getSaveFileName(self, "Export voiceprints",
                                              str(Path.home() / "voiceprints.json"),
                                              "JSON (*.json)")
        if not path:
            return
        try:
            n = self.db.export_json(Path(path),
                                    [p.display_name for p in persons] if persons else None)
        except RuntimeError as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        self.status.setText(f"Exported {n} voiceprint(s) to {path} - the file is not "
                            "encrypted, keep it safe.")

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import voiceprints", str(Path.home()),
                                              "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            persons, vps = self.db.import_json(Path(path))
        except (ValueError, KeyError, OSError) as e:
            QMessageBox.warning(self, "Import", f"Cannot import {path}: {e}")
            return
        self.status.setText(f"Imported {vps} voiceprint(s) of {persons} person(s).")
        self.reload()

    def done(self, result: int) -> None:
        if self._job is not None:
            QMessageBox.information(self, "Voiceprints", "Please wait until enrollment is "
                                    "finished.")
            return
        self.db.close()
        super().done(result)
