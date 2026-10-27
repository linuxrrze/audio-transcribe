"""Setup wizard: shown at the first start (no config.toml yet, or models missing) and
from File → Download models…

Three pages: model for this hardware, languages and voiceprint privacy, download with
byte progress. Choices are saved when leaving the second page; everything stays
editable in File → Settings. The logic lives in ``core/setup.py`` (shared with
`tx setup`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from transcribe.core import models
from transcribe.core import setup as core_setup
from transcribe.core.config import Config, required_models
from transcribe.core.setup import Hardware, Profile, format_size, hardware_text
from transcribe.gui.bridge import Job
from transcribe.i18n import LANGUAGES, _

# (names, progress(name, done, total), cancel) - models.download_all, or a fake in tests
DownloadFn = Callable[[list[str], Callable[[str, int, int], None], threading.Event], None]
_STEPS = 1000  # progress bar resolution


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(placeholder-text);")
    return label


class SetupWizard(QDialog):
    progressed = Signal(str, int, int)  # from the download thread: (name, done, total)

    def __init__(self, config: Config, parent=None, hardware: Hardware | None = None,
                 download: DownloadFn | None = None, path: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Set up Transcribe"))
        self.config = config
        self.path = path or core_setup.config_path()
        self.hardware = hardware or core_setup.detect_hardware()
        self.download_fn: DownloadFn = download or models.download_all
        self._names: list[str] = []
        self._fraction: dict[str, float] = {}
        self._cancel = threading.Event()
        self._retry = False  # the last download failed or was cancelled
        self._job: Job | None = None
        self.progressed.connect(self._on_progress)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._model_page())
        self.pages.addWidget(self._privacy_page())
        self.pages.addWidget(self._download_page())
        self.later_btn = QPushButton(_("Later"))
        self.later_btn.setToolTip(_("Close; models download on first use, or later via "
                                    "File → Download models…"))
        self.later_btn.clicked.connect(self.reject)
        self.back_btn = QPushButton(_("Back"))
        self.back_btn.clicked.connect(self._back)
        self.cancel_btn = QPushButton(_("Cancel download"))
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.next_btn = QPushButton(_("Next"))
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(self._next)
        row = QHBoxLayout()
        row.addWidget(self.later_btn)
        row.addStretch(1)
        for b in (self.back_btn, self.cancel_btn, self.next_btn):
            row.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addWidget(self.pages, 1)
        layout.addLayout(row)
        self.setMinimumWidth(560)
        self._update_buttons()

    # --- page 1: model -------------------------------------------------------------------
    def _model_page(self) -> QWidget:
        intro = QLabel(_("Transcribe runs entirely on this computer. Choose the speech "
                         "recognition model; you can change it later in File → Settings."))
        intro.setWordWrap(True)
        hw = self.hardware
        hw_label = QLabel(f"<b>{hardware_text(hw)}</b>")
        hw_label.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addWidget(hw_label)
        if not hw.has_gpu:
            layout.addWidget(_hint(_(core_setup.NO_GPU_HINT)))
        self.profile_group = QButtonGroup(self)
        self.profile_buttons: dict[str, QRadioButton] = {}
        recommended = core_setup.recommend(hw)
        choices: list[Profile | None] = list(core_setup.profiles_for(hw))
        current = core_setup.matching_profile(self.config)
        if self.path.exists() and current is None:
            choices.insert(0, None)  # hand-made settings: keep them by default
        for profile in choices:
            if profile is None:
                text = _("Keep the current settings ({model})").format(
                    model=self.config.asr.model)
                key = ""
                size = sum(core_setup.size_mb(n) for n in core_setup.needed_models(self.config))
                detail = ""
            else:
                text = _(profile.label)
                if profile is recommended:
                    text += " " + _("(recommended)")
                key, detail = profile.key, _(profile.speed)
                size = core_setup.profile_size_mb(profile, self.config)
            button = QRadioButton(f"{text} - {format_size(size)}")
            self.profile_group.addButton(button)
            self.profile_buttons[key] = button
            layout.addWidget(button)
            if detail:
                hint = _hint(detail)
                hint.setContentsMargins(24, 0, 0, 4)
                layout.addWidget(hint)
        if self.path.exists():
            selected = "" if current is None else current.key
        else:  # first run: the recommendation
            selected = recommended.key
        self.profile_buttons.get(selected, self.profile_buttons[recommended.key]).setChecked(True)
        layout.addStretch(1)
        page = QWidget()
        page.setLayout(layout)
        return page

    def selected_profile(self) -> Profile | None:
        """The chosen profile; None = keep the current model settings."""
        for key, button in self.profile_buttons.items():
            if button.isChecked():
                return core_setup.PROFILES.get(key)
        return None

    # --- page 2: languages and privacy ---------------------------------------------------
    def _privacy_page(self) -> QWidget:
        self.ui_language = QComboBox()
        self.ui_language.addItem(_("Automatic (system language)"), "auto")
        for code, name in LANGUAGES.items():
            self.ui_language.addItem(name, code)
        self.ui_language.setCurrentIndex(
            max(0, self.ui_language.findData(self.config.ui.language)))
        self.languages = QComboBox()
        for value, label in core_setup.LANGUAGE_CHOICES.items():
            self.languages.addItem(_(label), value)
        current = ",".join(self.config.asr.languages)
        if self.languages.findData(current) < 0:
            self.languages.addItem(current, current)
        self.languages.setCurrentIndex(self.languages.findData(current))
        self.identify = QCheckBox(_("Recognise people whose voice was learned before"))
        self.identify.setChecked(self.config.speakers.identify_known)
        self.store = QCheckBox(_("Store voiceprints when I name a speaker (with the person's "
                                 "consent)"))
        self.store.setChecked(self.config.speakers.store_voiceprints)
        form = QFormLayout()
        form.addRow(_("App language:"), self.ui_language)
        form.addRow("", _hint(_("Takes effect after restarting the app")))
        form.addRow(_("Spoken languages:"), self.languages)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(QLabel(f"<b>{_('Voiceprints')}</b>"))
        layout.addWidget(self.identify)
        layout.addWidget(self.store)
        layout.addWidget(_hint(_(core_setup.VOICEPRINT_NOTE)))
        layout.addStretch(1)
        page = QWidget()
        page.setLayout(layout)
        return page

    def apply(self) -> Config:
        """Write the choices of the first two pages into the configuration."""
        return core_setup.apply_choices(
            self.config, self.selected_profile(), self.ui_language.currentData(),
            self.languages.currentData().split(","), self.store.isChecked(),
            self.identify.isChecked())

    # --- page 3: download -----------------------------------------------------------------
    def _download_page(self) -> QWidget:
        self.download_intro = QLabel()
        self.download_intro.setWordWrap(True)
        self.rows = QGridLayout()
        self.bars: dict[str, QProgressBar] = {}
        self.total = QProgressBar()
        self.total.setRange(0, _STEPS)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(self.download_intro)
        layout.addLayout(self.rows)
        layout.addSpacing(8)
        layout.addWidget(QLabel(_("Total:")))
        layout.addWidget(self.total)
        layout.addWidget(self.status)
        layout.addStretch(1)
        page = QWidget()
        page.setLayout(layout)
        return page

    def _prepare_download(self) -> None:
        self._names = models.missing_models(*required_models(self.config))
        while self.rows.count():
            item = self.rows.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self.bars = {}
        for i, name in enumerate(self._names):
            bar = QProgressBar()
            bar.setRange(0, _STEPS)
            bar.setValue(0)
            self.rows.addWidget(QLabel(name), i, 0)
            self.rows.addWidget(QLabel(format_size(core_setup.size_mb(name))), i, 1)
            self.rows.addWidget(bar, i, 2)
            self.bars[name] = bar
        self._fraction = dict.fromkeys(self._names, 0.0)
        total = sum(core_setup.size_mb(n) for n in self._names)
        if self._names:
            self.download_intro.setText(_("Settings saved. These models are needed once "
                                          "({size}); they are stored in {path}.").format(
                size=format_size(total), path=models.models_dir()))
            self.status.setText("")
        else:
            self.download_intro.setText(_("Settings saved. All models are already "
                                          "downloaded."))
        self.total.setValue(_STEPS if not self._names else 0)

    def start_download(self) -> None:
        names = [n for n in self._names if self._fraction.get(n, 0.0) < 1.0]
        self._fraction.update(dict.fromkeys(names, 0.0))  # a retry starts these again
        self._cancel = threading.Event()
        self._retry = False
        cancel = self._cancel
        self.status.setText(_("Downloading…"))
        self._job = Job(lambda: self.download_fn(names, self.progressed.emit, cancel), self)
        self._job.done.connect(self._download_done)
        self._job.failed.connect(self._download_failed)
        self._job.start()
        self._update_buttons()

    def cancel_download(self) -> None:
        self._cancel.set()
        self.status.setText(_("Cancelling…"))

    @property
    def downloading(self) -> bool:
        return self._job is not None

    def _on_progress(self, name: str, done: int, total: int) -> None:
        bar = self.bars.get(name)
        if bar is None:
            return
        if total > 0:
            self._fraction[name] = min(1.0, done / total)
            bar.setRange(0, _STEPS)
            bar.setValue(int(self._fraction[name] * _STEPS))
            if done < total:
                bar.setFormat(f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
            else:
                bar.setFormat("%p%")
        else:
            bar.setRange(0, 0)  # size unknown: busy indicator
        # total: files weighted by their (approximate) size
        weights = {n: core_setup.size_mb(n) or 1.0 for n in self._names}
        frac = sum(self._fraction[n] * w for n, w in weights.items()) / sum(weights.values())
        self.total.setValue(int(frac * _STEPS))

    def _download_done(self, _result: object) -> None:
        self._job = None
        for name in self._names:
            self._on_progress(name, 1, 1)
        self.status.setText(_("All models downloaded."))
        self._update_buttons()

    def _download_failed(self, message: str) -> None:
        self._job = None
        self._retry = True
        if self._cancel.is_set():
            self.status.setText(_("Download cancelled. Retry, or close and download later."))
        else:
            self.status.setText(_("Download failed: {error}").format(error=message))
        self._update_buttons()

    # --- navigation ------------------------------------------------------------------------
    @property
    def finished_downloading(self) -> bool:
        return all(f >= 1.0 for f in self._fraction.values())

    def _next(self) -> None:
        page = self.pages.currentIndex()
        if page == 0:
            self.pages.setCurrentIndex(1)
        elif page == 1:
            self.apply().save(self.path)
            self._prepare_download()
            self.pages.setCurrentIndex(2)
        elif self.finished_downloading:
            self.accept()
        else:
            self.start_download()
        self._update_buttons()

    def _back(self) -> None:
        self.pages.setCurrentIndex(max(0, self.pages.currentIndex() - 1))
        self._update_buttons()

    def _update_buttons(self) -> None:
        page = self.pages.currentIndex()
        busy = self.downloading
        self.back_btn.setVisible(page > 0)
        self.back_btn.setEnabled(not busy)
        self.cancel_btn.setVisible(page == 2 and busy)
        self.next_btn.setEnabled(not busy)
        if page < 2:
            self.next_btn.setText(_("Next"))
        elif self.finished_downloading:
            self.next_btn.setText(_("Finish"))
        elif self._retry:
            self.next_btn.setText(_("Retry"))
        else:
            self.next_btn.setText(_("Download"))
        self.later_btn.setVisible(not (page == 2 and self.finished_downloading))

    def reject(self) -> None:
        self._cancel.set()  # "Later" or closing the window stops a running download
        super().reject()
