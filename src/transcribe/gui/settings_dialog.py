"""Settings (tabbed, all of config.toml that matters to users) and the vocabulary editor."""

from __future__ import annotations

import copy
from pathlib import Path

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
from transcribe.gui.widgets import compact_combo, path_text
from transcribe.i18n import LANGUAGES, _

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]


class VocabularyDialog(QDialog):
    """Edit the custom vocabulary: one term per line, 'wrong -> right' corrections."""

    def __init__(self, vocabulary: Vocabulary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Vocabulary"))
        self.resize(560, 520)
        intro = QLabel(_(
            "Names, acronyms and jargon Whisper should recognise - one per line (or "
            "comma-separated). <b>wrong -> right</b> corrects a word the model still "
            "mishears. <b>#</b> starts a comment."))
        intro.setWordWrap(True)
        self.editor = QPlainTextEdit(vocabulary.to_text())
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.editor.setPlaceholderText("# cast\nJustus, Peter, Bob\nFentriss\nDSGVO\n"
                                       "Fanfress -> Fentriss")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.editor.textChanged.connect(self._update)
        import_btn = QPushButton(_("Import…"))
        import_btn.setToolTip(_("Add the entries of a vocabulary file"))
        import_btn.clicked.connect(self._import)
        export_btn = QPushButton(_("Export…"))
        export_btn.clicked.connect(self._export)
        clear_btn = QPushButton(_("Clear"))
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
        text = _("{terms} terms, {corrections} corrections.").format(
            terms=len(v.terms), corrections=len(v.corrections))
        if used < len(targets):
            warning = _("Only the first {used} of {total} hint words fit Whisper's prompt "
                        "({chars} characters); put the important ones first. Corrections "
                        "always apply.").format(used=used, total=len(targets),
                                                chars=MAX_HOTWORD_CHARS)
            text += f" <span style='color:#c60;'>{warning}</span>"
        self.status.setText(text)

    def _import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Import vocabulary"), str(Path.home()),
            _("Text files") + " (*.txt);;" + _("All files") + " (*)")
        if path:
            merged = self.vocabulary().merged(Vocabulary.load(Path(path)))
            self.editor.setPlainText(merged.to_text())

    def _export(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, _("Export vocabulary"),
                                                    str(Path.home() / "vocabulary.txt"))
        if path:
            Path(path).write_text(self.vocabulary().to_text() + "\n", encoding="utf-8")


class _DecimalSpinBox(QDoubleSpinBox):
    """Takes both "." and "," as the decimal point, in every language. Qt reads the other
    one as a thousands separator and drops it: "0.5" gave 5 in German, "0,5" 5 in English
    (no setting here needs thousands)."""

    def _fix(self, text: str) -> str:
        point = self.locale().decimalPoint()
        return text.replace(".", point).replace(",", point)

    def validate(self, text: str, pos: int):  # noqa: ANN201 - Qt returns a tuple
        return super().validate(self._fix(text), pos)

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt API
        return super().valueFromText(self._fix(text))


def _spin(value: float, lo: float, hi: float, step: float, decimals: int = 2,
          suffix: str = "") -> QDoubleSpinBox:
    box = _DecimalSpinBox()
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
    browse = QPushButton(_("Browse…"))
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
        self.setWindowTitle(_("Settings"))
        self.config = config
        self.vocab = Vocabulary.from_config(config.asr.vocabulary, config.asr.corrections)
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(False)  # all tabs visible: at least as wide as their titles
        tabs.addTab(self._asr_tab(), _("Speech recognition"))
        tabs.addTab(self._speakers_tab(), _("Speakers"))
        tabs.addTab(self._live_tab(), _("Live"))
        tabs.addTab(self._voiceprints_tab(), _("Voiceprints"))
        tabs.addTab(self._general_tab(), _("General"))
        self.tabs = tabs
        where = path_text(_("Saved in {path}").format(path=config_dir() / "config.toml"))
        where.setStyleSheet("QLineEdit { background: transparent; color: #888; }")
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
        self.engine.setToolTip(_("faster-whisper: CPU or NVIDIA GPU. whisper.cpp: also "
                                 "AMD/Intel GPUs (Vulkan) and Apple GPUs (Metal); needs "
                                 "whisper-server, see scripts/build_whispercpp.sh."))
        self.model = QComboBox()
        self.model.addItems(WHISPER_MODELS)
        self.model.setEditable(True)
        self.model.setCurrentText(a.model)
        self.refine_model = QComboBox()
        self.refine_model.addItem(_("(same as live)"), "")  # data = stored value
        for m in WHISPER_MODELS:
            self.refine_model.addItem(m, m)
        index = self.refine_model.findData(self.config.live.refine_model)
        if index < 0:  # a model not in the list (e.g. a path)
            self.refine_model.addItem(self.config.live.refine_model,
                                      self.config.live.refine_model)
            index = self.refine_model.count() - 1
        self.refine_model.setCurrentIndex(index)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda", "gpu"])
        self.device.setCurrentText(a.device)
        self.device.setToolTip(_("cuda: faster-whisper on NVIDIA; gpu: whisper.cpp on any "
                                 "GPU"))
        self.languages = QLineEdit(",".join(a.languages))
        self.languages.setToolTip(_("Allowed languages, comma-separated (e.g. de,en)"))
        self.beam_size = _int_spin(a.beam_size, 1, 10)
        self.beam_size.setToolTip(_("Beam search width for files and refinement (live: Live "
                                    "tab)"))
        self.cpu_threads = _int_spin(a.cpu_threads, 0, 64, _("auto"))
        self.style_prompt = QCheckBox(_("Punctuated example prompt (recommended)"))
        self.style_prompt.setChecked(a.style_prompt)
        self.style_prompt.setToolTip(_("Keeps punctuation and capitals; off gives more "
                                       "verbatim text (see docs/BENCHMARKS.md)"))
        self.vocab_label = QLabel()
        vocab_btn = QPushButton(_("Edit vocabulary…"))
        vocab_btn.clicked.connect(self._edit_vocabulary)
        vocab_row = QHBoxLayout()
        vocab_row.addWidget(self.vocab_label, 1)
        vocab_row.addWidget(vocab_btn)
        vocab_w = QWidget()
        vocab_w.setLayout(vocab_row)
        vocab_row.setContentsMargins(0, 0, 0, 0)
        self.wcpp_server = QLineEdit(a.whispercpp_server)
        self.wcpp_server.setPlaceholderText(_("own build or PATH"))
        self.wcpp_dtw = QCheckBox(_("Accurate word times (DTW; needed for speaker "
                                    "attribution)"))
        self.wcpp_dtw.setChecked(a.whispercpp_dtw)
        self.engine.currentTextChanged.connect(self._engine_changed)
        form = QFormLayout()
        form.addRow(_("Engine:"), self.engine)
        form.addRow(_("Model (live and files):"), self.model)
        form.addRow(_("Model after a live session:"), self.refine_model)
        form.addRow(_("Compute device:"), self.device)
        form.addRow(_("Languages:"), self.languages)
        form.addRow(_("Beam size:"), self.beam_size)
        form.addRow(_("CPU threads:"), self.cpu_threads)
        form.addRow(self.style_prompt)
        form.addRow(_("Vocabulary:"), vocab_w)
        form.addRow(QLabel("<b>whisper.cpp</b>"))
        form.addRow("whisper-server:", _path_row(self.wcpp_server, self._pick_server))
        form.addRow(self.wcpp_dtw)
        self._update_vocab_label()
        self._engine_changed(a.engine)
        return _page(form)

    def _speakers_tab(self) -> QWidget:
        s = self.config.speakers
        self.threshold = _spin(s.diarize_threshold, 0.05, 0.95, 0.05)
        self.threshold.setToolTip(_("Clustering distance of the speaker detection: lower "
                                    "finds more speakers"))
        self.resegment = QCheckBox(_("Second pass over sentences (recommended)"))
        self.resegment.setChecked(s.resegment)
        self.resegment.setToolTip(_("Re-clusters transcript phrases: fixes merged turns in "
                                    "fast dialogue and far too many speakers"))
        self.speaker_count = QComboBox()
        self.speaker_count.addItem(_("Estimate (may split a person; merge afterwards)"),
                                   "estimate")
        self.speaker_count.addItem(_("Silhouette (distinct voices, e.g. meetings)"),
                                   "silhouette")
        self.speaker_count.setCurrentIndex(max(0, self.speaker_count.findData(s.speaker_count)))
        compact_combo(self.speaker_count)  # long entries (in German) widened the dialog
        self.num_threads = _int_spin(s.num_threads, 0, 64, _("auto"))
        self.live_threshold = _spin(s.live_threshold, 0.1, 0.9, 0.05)
        self.live_threshold.setToolTip(_("Live labels: similarity at which a sentence joins a "
                                         "known speaker; higher finds more speakers"))
        self.live_threshold_nb = _spin(s.live_threshold_narrowband, 0.1, 0.9, 0.05)
        self.live_threshold_nb.setToolTip(_("The same for telephone-band audio (phone and "
                                            "some call audio), where voices sound more alike"))
        form = QFormLayout()
        form.addRow(_("Speaker threshold:"), self.threshold)
        form.addRow(self.resegment)
        form.addRow(_("Number of speakers:"), self.speaker_count)
        form.addRow(_("Threads:"), self.num_threads)
        form.addRow(_("Live speaker threshold:"), self.live_threshold)
        form.addRow(_("… for telephone audio:"), self.live_threshold_nb)
        note = QLabel(_("If you know how many people speak, give the number when "
                        "transcribing (Sessions tab: Refine with a speaker count) - the most "
                        "reliable setting."))
        note.setWordWrap(True)
        form.addRow(note)
        return _page(form)

    def _live_tab(self) -> QWidget:
        lv = self.config.live
        self.refine = QCheckBox(_("Refine the transcript when a session stops"))
        self.refine.setChecked(lv.refine)
        self.step = _spin(lv.step, 0.25, 5.0, 0.25, suffix=" s")
        self.step.setToolTip(_("New audio between two decodes; smaller = faster text, more "
                               "CPU"))
        self.end_silence = _spin(lv.end_silence, 0.1, 3.0, 0.1, suffix=" s")
        self.end_silence.setToolTip(_("Pause that ends an utterance"))
        self.max_buffer = _spin(lv.max_buffer, 4.0, 30.0, 1.0, 1, " s")
        self.live_beam = _int_spin(lv.beam_size, 1, 10)
        self.live_beam.setToolTip(_("1 = fastest; larger is slightly more accurate"))
        form = QFormLayout()
        form.addRow(self.refine)
        form.addRow(_("Decode every:"), self.step)
        form.addRow(_("Utterance ends after:"), self.end_silence)
        form.addRow(_("Longest utterance:"), self.max_buffer)
        form.addRow(_("Beam size (live):"), self.live_beam)
        return _page(form)

    def _voiceprints_tab(self) -> QWidget:
        s = self.config.speakers
        self.identify = QCheckBox(_("Recognise known people"))
        self.identify.setChecked(s.identify_known)
        self.store = QCheckBox(_("Store voiceprints (needed to learn voices)"))
        self.store.setChecked(s.store_voiceprints)
        self.encrypt = QCheckBox(_("Encrypt voiceprints (key in the OS keyring)"))
        self.encrypt.setChecked(s.encrypt_voiceprints)
        self.high = _spin(s.match_threshold_high, 0.3, 0.99, 0.01)
        self.high.setToolTip(_("Similarity at which a known person is labelled "
                               "automatically"))
        self.low = _spin(s.match_threshold_low, 0.2, 0.99, 0.01)
        self.low.setToolTip(_("Similarity at which the app asks 'Is this X?'"))
        self.margin = _spin(s.match_margin, 0.0, 0.5, 0.01)
        self.margin.setToolTip(_("Required lead over the second-best person"))
        self.max_exemplars = _int_spin(s.max_exemplars_per_person, 1, 200)
        manage = QPushButton(_("Manage voiceprints…"))
        manage.clicked.connect(self._manage_voiceprints)
        info = QLabel(_("Voiceprints are biometric data: stored only on this computer, only "
                        "with the person's consent, deletable at any time."))
        info.setWordWrap(True)
        form = QFormLayout()
        form.addRow(info)
        form.addRow(self.identify)
        form.addRow(self.store)
        form.addRow(self.encrypt)
        form.addRow(_("Label automatically from:"), self.high)
        form.addRow(_("Suggest from:"), self.low)
        form.addRow(_("Margin:"), self.margin)
        form.addRow(_("Voiceprints per person:"), self.max_exemplars)
        form.addRow(manage)
        return _page(form)

    def _general_tab(self) -> QWidget:
        from transcribe.core.models import models_dir

        self.ui_language = QComboBox()
        self.ui_language.addItem(_("Automatic (system language)"), "auto")
        for code, name in LANGUAGES.items():  # native names, not translated
            self.ui_language.addItem(name, code)
        self.ui_language.setCurrentIndex(
            max(0, self.ui_language.findData(self.config.ui.language)))
        self.ui_language.setToolTip(_("Takes effect after restarting the app"))
        self.sessions_dir = QLineEdit(self.config.sessions_dir)
        self.sessions_dir.setPlaceholderText(str(self.config.resolved_sessions_dir()))
        models = path_text(str(models_dir()))
        models.setToolTip(str(models_dir()) + "\n"
                          + _("Change with the TRANSCRIBE_MODELS_DIR environment variable"))
        form = QFormLayout()
        form.addRow(_("Language:"), self.ui_language)
        form.addRow(_("Sessions:"), _path_row(self.sessions_dir, self._pick_sessions_dir))
        form.addRow(_("Models:"), models)
        return _page(form)

    # --- actions ------------------------------------------------------------------------
    def _engine_changed(self, engine: str) -> None:
        for w in (self.wcpp_server, self.wcpp_dtw):
            w.setEnabled(engine == "whisper.cpp")

    def _update_vocab_label(self) -> None:
        v = self.vocab
        self.vocab_label.setText(_("{terms} terms, {corrections} corrections").format(
            terms=len(v.terms), corrections=len(v.corrections)) if v else _("none"))

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
            QMessageBox.warning(self, _("Voiceprints"), str(e))

    def _pick_server(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, _("whisper-server binary"), self.wcpp_server.text() or str(Path.home()))
        if path:
            self.wcpp_server.setText(path)

    def _pick_sessions_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, _("Sessions folder"), self.sessions_dir.text()
            or str(self.config.resolved_sessions_dir()))
        if path:
            self.sessions_dir.setText(path)

    def apply(self) -> Config:
        return self.apply_to(self.config)

    def apply_to(self, c: Config) -> Config:
        a, s, lv = c.asr, c.speakers, c.live
        a.engine = self.engine.currentText()
        a.model = self.model.currentText().strip() or a.model
        lv.refine_model = self.refine_model.currentData() or ""
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
        s.live_threshold_narrowband = self.live_threshold_nb.value()
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
        c.ui.language = self.ui_language.currentData()
        return c

    def _save(self) -> None:
        self.apply().save()
        self.accept()


def _page(form: QFormLayout) -> QWidget:
    w = QWidget()
    w.setLayout(form)
    return w
