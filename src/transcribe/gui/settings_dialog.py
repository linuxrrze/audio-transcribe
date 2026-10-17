"""Settings (tabbed, all of config.toml that matters to users) and the vocabulary editor."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from transcribe.core.config import Config, config_dir
from transcribe.core.vocabulary import MAX_HOTWORD_CHARS, Vocabulary

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
SAME_AS_LIVE = "(same as live)"


class VocabularyDialog(QDialog):
    """Edit the custom vocabulary: one term per line, 'wrong -> right' corrections."""

    def __init__(self, vocabulary: Vocabulary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vocabulary")
        self.resize(560, 520)
        intro = QLabel(
            "Names, acronyms and jargon Whisper should recognise - one per line (or "
            "comma-separated). <b>wrong -> right</b> corrects a word the model still "
            "mishears. <b>#</b> starts a comment.")
        intro.setWordWrap(True)
        self.editor = QPlainTextEdit(vocabulary.to_text())
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.editor.setPlaceholderText("# cast\nJustus, Peter, Bob\nFentriss\nDSGVO\n"
                                       "Fanfress -> Fentriss")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.editor.textChanged.connect(self._update)
        import_btn = QPushButton("Import…")
        import_btn.setToolTip("Add the entries of a vocabulary file")
        import_btn.clicked.connect(self._import)
        export_btn = QPushButton("Export…")
        export_btn.clicked.connect(self._export)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.editor.clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row = QHBoxLayout()
        for b in (import_btn, export_btn, clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.status)
        layout.addLayout(row)
        self._update()

    def vocabulary(self) -> Vocabulary:
        return Vocabulary.parse(self.editor.toPlainText())

    def _update(self) -> None:
        v = self.vocabulary()
        hint = v.hotwords() or ""
        targets = list(dict.fromkeys(v.terms + list(v.corrections.values())))
        used = len(hint.split(", ")) if hint else 0
        text = f"{len(v.terms)} terms, {len(v.corrections)} corrections."
        if used < len(targets):
            text += (f" <span style='color:#c60;'>Only the first {used} of {len(targets)} "
                     f"hint words fit Whisper's prompt ({MAX_HOTWORD_CHARS} characters); "
                     "put the important ones first. Corrections always apply.</span>")
        self.status.setText(text)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import vocabulary", str(Path.home()),
                                              "Text files (*.txt);;All files (*)")
        if path:
            merged = self.vocabulary().merged(Vocabulary.load(Path(path)))
            self.editor.setPlainText(merged.to_text())

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export vocabulary",
                                              str(Path.home() / "vocabulary.txt"))
        if path:
            Path(path).write_text(self.vocabulary().to_text() + "\n", encoding="utf-8")


def _spin(value: float, lo: float, hi: float, step: float, decimals: int = 2,
          suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def _int_spin(value: int, lo: int, hi: int, special: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    if special:
        box.setSpecialValueText(special)
    box.setValue(value)
    return box


def _path_row(edit: QLineEdit, pick) -> QWidget:
    browse = QPushButton("Browse…")
    browse.clicked.connect(pick)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    row.addWidget(browse)
    w = QWidget()
    w.setLayout(row)
    return w


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config
        self.vocab = Vocabulary.from_config(config.asr.vocabulary, config.asr.corrections)
        tabs = QTabWidget()
        tabs.addTab(self._asr_tab(), "Speech recognition")
        tabs.addTab(self._speakers_tab(), "Speakers")
        tabs.addTab(self._live_tab(), "Live")
        tabs.addTab(self._voiceprints_tab(), "Voiceprints")
        tabs.addTab(self._storage_tab(), "Storage")
        self.tabs = tabs
        where = QLabel(f"Saved in {config_dir() / 'config.toml'}")
        where.setStyleSheet("color: #888;")
        where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(where)
        layout.addWidget(buttons)

    # --- tabs ---------------------------------------------------------------------------
    def _asr_tab(self) -> QWidget:
        a = self.config.asr
        self.engine = QComboBox()
        self.engine.addItems(["faster-whisper", "whisper.cpp"])
        self.engine.setCurrentText(a.engine)
        self.engine.setToolTip("faster-whisper: CPU or NVIDIA GPU. whisper.cpp: also AMD/Intel "
                               "GPUs (Vulkan) and Apple GPUs (Metal); needs whisper-server, "
                               "see scripts/build_whispercpp.sh.")
        self.model = QComboBox()
        self.model.addItems(WHISPER_MODELS)
        self.model.setEditable(True)
        self.model.setCurrentText(a.model)
        self.refine_model = QComboBox()
        self.refine_model.addItems([SAME_AS_LIVE, *WHISPER_MODELS])
        self.refine_model.setCurrentText(self.config.live.refine_model or SAME_AS_LIVE)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "gpu"])
        self.device.setCurrentText(a.device)
        self.device.setToolTip("cuda: faster-whisper on NVIDIA; gpu: whisper.cpp on any GPU")
        self.languages = QLineEdit(",".join(a.languages))
        self.languages.setToolTip("Allowed languages, comma-separated (e.g. de,en)")
        self.beam_size = _int_spin(a.beam_size, 1, 10)
        self.beam_size.setToolTip("Beam search width for files and refinement (live: Live tab)")
        self.cpu_threads = _int_spin(a.cpu_threads, 0, 64, "auto")
        self.style_prompt = QCheckBox("Punctuated example prompt (recommended)")
        self.style_prompt.setChecked(a.style_prompt)
        self.style_prompt.setToolTip("Keeps punctuation and capitals; off gives more verbatim "
                                     "text (see docs/BENCHMARKS.md)")
        self.vocab_label = QLabel()
        vocab_btn = QPushButton("Edit vocabulary…")
        vocab_btn.clicked.connect(self._edit_vocabulary)
        vocab_row = QHBoxLayout()
        vocab_row.addWidget(self.vocab_label, 1)
        vocab_row.addWidget(vocab_btn)
        vocab_w = QWidget()
        vocab_w.setLayout(vocab_row)
        vocab_row.setContentsMargins(0, 0, 0, 0)
        self.wcpp_server = QLineEdit(a.whispercpp_server)
        self.wcpp_server.setPlaceholderText("own build or PATH")
        self.wcpp_dtw = QCheckBox("Accurate word times (DTW; needed for speaker attribution)")
        self.wcpp_dtw.setChecked(a.whispercpp_dtw)
        self.engine.currentTextChanged.connect(self._engine_changed)
        form = QFormLayout()
        form.addRow("Engine:", self.engine)
        form.addRow("Model (live and files):", self.model)
        form.addRow("Model after a live session:", self.refine_model)
        form.addRow("Compute device:", self.device)
        form.addRow("Languages:", self.languages)
        form.addRow("Beam size:", self.beam_size)
        form.addRow("CPU threads:", self.cpu_threads)
        form.addRow(self.style_prompt)
        form.addRow("Vocabulary:", vocab_w)
        form.addRow(QLabel("<b>whisper.cpp</b>"))
        form.addRow("whisper-server:", _path_row(self.wcpp_server, self._pick_server))
        form.addRow(self.wcpp_dtw)
        self._update_vocab_label()
        self._engine_changed(a.engine)
        return _page(form)

    def _speakers_tab(self) -> QWidget:
        s = self.config.speakers
        self.threshold = _spin(s.diarize_threshold, 0.05, 0.95, 0.05)
        self.threshold.setToolTip("Clustering distance of the speaker detection: lower finds "
                                  "more speakers")
        self.resegment = QCheckBox("Second pass over sentences (recommended)")
        self.resegment.setChecked(s.resegment)
        self.resegment.setToolTip("Re-clusters transcript phrases: fixes merged turns in fast "
                                  "dialogue and far too many speakers")
        self.speaker_count = QComboBox()
        self.speaker_count.addItem("Estimate (may split a person; merge afterwards)", "estimate")
        self.speaker_count.addItem("Silhouette (distinct voices, e.g. meetings)", "silhouette")
        self.speaker_count.setCurrentIndex(max(0, self.speaker_count.findData(s.speaker_count)))
        self.num_threads = _int_spin(s.num_threads, 0, 64, "auto")
        self.live_threshold = _spin(s.live_threshold, 0.1, 0.9, 0.05)
        self.live_threshold.setToolTip("Live labels: similarity at which a sentence joins a "
                                       "known speaker; higher finds more speakers")
        form = QFormLayout()
        form.addRow("Speaker threshold:", self.threshold)
        form.addRow(self.resegment)
        form.addRow("Number of speakers:", self.speaker_count)
        form.addRow("Threads:", self.num_threads)
        form.addRow("Live speaker threshold:", self.live_threshold)
        note = QLabel("If you know how many people speak, give the number when transcribing "
                      "(Sessions tab: Refine with a speaker count) - the most reliable setting.")
        note.setWordWrap(True)
        form.addRow(note)
        return _page(form)

    def _live_tab(self) -> QWidget:
        lv = self.config.live
        self.refine = QCheckBox("Refine the transcript when a session stops")
        self.refine.setChecked(lv.refine)
        self.step = _spin(lv.step, 0.25, 5.0, 0.25, suffix=" s")
        self.step.setToolTip("New audio between two decodes; smaller = faster text, more CPU")
        self.end_silence = _spin(lv.end_silence, 0.1, 3.0, 0.1, suffix=" s")
        self.end_silence.setToolTip("Pause that ends an utterance")
        self.max_buffer = _spin(lv.max_buffer, 4.0, 30.0, 1.0, 1, " s")
        self.live_beam = _int_spin(lv.beam_size, 1, 10)
        self.live_beam.setToolTip("1 = fastest; larger is slightly more accurate")
        form = QFormLayout()
        form.addRow(self.refine)
        form.addRow("Decode every:", self.step)
        form.addRow("Utterance ends after:", self.end_silence)
        form.addRow("Longest utterance:", self.max_buffer)
        form.addRow("Beam size (live):", self.live_beam)
        return _page(form)

    def _voiceprints_tab(self) -> QWidget:
        s = self.config.speakers
        self.identify = QCheckBox("Recognise known people")
        self.identify.setChecked(s.identify_known)
        self.store = QCheckBox("Store voiceprints (needed to learn voices)")
        self.store.setChecked(s.store_voiceprints)
        self.encrypt = QCheckBox("Encrypt voiceprints (key in the OS keyring)")
        self.encrypt.setChecked(s.encrypt_voiceprints)
        self.high = _spin(s.match_threshold_high, 0.3, 0.99, 0.01)
        self.high.setToolTip("Similarity at which a known person is labelled automatically")
        self.low = _spin(s.match_threshold_low, 0.2, 0.99, 0.01)
        self.low.setToolTip("Similarity at which the app asks 'Is this X?'")
        self.margin = _spin(s.match_margin, 0.0, 0.5, 0.01)
        self.margin.setToolTip("Required lead over the second-best person")
        self.max_exemplars = _int_spin(s.max_exemplars_per_person, 1, 200)
        manage = QPushButton("Manage voiceprints…")
        manage.clicked.connect(self._manage_voiceprints)
        info = QLabel("Voiceprints are biometric data: stored only on this computer, only "
                      "with the person's consent, deletable at any time.")
        info.setWordWrap(True)
        form = QFormLayout()
        form.addRow(info)
        form.addRow(self.identify)
        form.addRow(self.store)
        form.addRow(self.encrypt)
        form.addRow("Label automatically from:", self.high)
        form.addRow("Suggest from:", self.low)
        form.addRow("Margin:", self.margin)
        form.addRow("Voiceprints per person:", self.max_exemplars)
        form.addRow(manage)
        return _page(form)

    def _storage_tab(self) -> QWidget:
        from transcribe.core.models import models_dir

        self.sessions_dir = QLineEdit(self.config.sessions_dir)
        self.sessions_dir.setPlaceholderText(str(self.config.resolved_sessions_dir()))
        models = QLabel(str(models_dir()))
        models.setToolTip("Change with the TRANSCRIBE_MODELS_DIR environment variable")
        form = QFormLayout()
        form.addRow("Sessions:", _path_row(self.sessions_dir, self._pick_sessions_dir))
        form.addRow("Models:", models)
        return _page(form)

    # --- actions ------------------------------------------------------------------------
    def _engine_changed(self, engine: str) -> None:
        for w in (self.wcpp_server, self.wcpp_dtw):
            w.setEnabled(engine == "whisper.cpp")

    def _update_vocab_label(self) -> None:
        v = self.vocab
        self.vocab_label.setText(f"{len(v.terms)} terms, {len(v.corrections)} corrections"
                                 if v else "none")

    def _edit_vocabulary(self) -> None:
        dlg = VocabularyDialog(self.vocab, self)
        if dlg.exec():
            self.vocab = dlg.vocabulary()
            self._update_vocab_label()

    def _manage_voiceprints(self) -> None:
        from transcribe.gui.voiceprints_dialog import VoiceprintsDialog

        try:
            # the settings as edited (e.g. encryption), without saving them yet
            VoiceprintsDialog(self.apply_to(copy.deepcopy(self.config)), self).exec()
        except RuntimeError as e:  # e.g. encrypted DB without key
            QMessageBox.warning(self, "Voiceprints", str(e))

    def _pick_server(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "whisper-server binary",
                                              self.wcpp_server.text() or str(Path.home()))
        if path:
            self.wcpp_server.setText(path)

    def _pick_sessions_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Sessions folder", self.sessions_dir.text()
            or str(self.config.resolved_sessions_dir()))
        if path:
            self.sessions_dir.setText(path)

    def apply(self) -> Config:
        return self.apply_to(self.config)

    def apply_to(self, c: Config) -> Config:
        a, s, lv = c.asr, c.speakers, c.live
        a.engine = self.engine.currentText()
        a.model = self.model.currentText().strip() or a.model
        refine = self.refine_model.currentText()
        lv.refine_model = "" if refine == SAME_AS_LIVE else refine
        a.device = self.device.currentText()
        a.languages = [x.strip() for x in self.languages.text().split(",") if x.strip()]
        a.beam_size = self.beam_size.value()
        a.cpu_threads = self.cpu_threads.value()
        a.style_prompt = self.style_prompt.isChecked()
        a.vocabulary, a.corrections = list(self.vocab.terms), dict(self.vocab.corrections)
        a.whispercpp_server = self.wcpp_server.text().strip()
        a.whispercpp_dtw = self.wcpp_dtw.isChecked()
        s.diarize_threshold = self.threshold.value()
        s.resegment = self.resegment.isChecked()
        s.speaker_count = self.speaker_count.currentData()
        s.num_threads = self.num_threads.value()
        s.live_threshold = self.live_threshold.value()
        s.identify_known = self.identify.isChecked()
        s.store_voiceprints = self.store.isChecked()
        s.encrypt_voiceprints = self.encrypt.isChecked()
        s.match_threshold_high = self.high.value()
        s.match_threshold_low = min(self.low.value(), self.high.value())
        s.match_margin = self.margin.value()
        s.max_exemplars_per_person = self.max_exemplars.value()
        lv.refine = self.refine.isChecked()
        lv.step = self.step.value()
        lv.end_silence = self.end_silence.value()
        lv.max_buffer = self.max_buffer.value()
        lv.beam_size = self.live_beam.value()
        c.sessions_dir = self.sessions_dir.text().strip()
        return c

    def _save(self) -> None:
        self.apply().save()
        self.accept()


def _page(form: QFormLayout) -> QWidget:
    w = QWidget()
    w.setLayout(form)
    return w
