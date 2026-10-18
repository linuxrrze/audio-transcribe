"""Speaker list with rename / merge / accept-or-decline-suggestion actions.

The panel only emits requests; the owning tab applies them (live session in memory, or
a stored session on disk).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from transcribe.gui.colors import speaker_color
from transcribe.i18n import _, display_label
from transcribe.storage.transcript_model import Speaker


def _swatch(color: str) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(QColor(color))
    return QIcon(pix)


class SpeakerPanel(QWidget):
    rename_requested = Signal(str)  # speaker id
    merge_requested = Signal(str, str)  # keep, drop
    accept_requested = Signal(str)  # take the suggestion as the name
    decline_requested = Signal(str)
    link_activated = Signal(str)  # a link in the note (see set_note)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.list = QListWidget()
        # a long name suggestion is cut with "…" (full text in the tooltip) instead of
        # making the narrow list scroll sideways
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.list.currentItemChanged.connect(self._update_buttons)
        self.rename_btn = QPushButton(_("Name…"))
        self.merge_btn = QPushButton(_("Merge into…"))
        self.accept_btn = QPushButton(_("Yes"))
        self.decline_btn = QPushButton(_("No"))
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setVisible(False)
        self.note.linkActivated.connect(self.link_activated)
        self._actions = {"rename": True, "merge": True, "suggestions": True}
        self.rename_btn.clicked.connect(lambda: self._emit(self.rename_requested))
        self.merge_btn.clicked.connect(self._ask_merge)
        self.accept_btn.clicked.connect(lambda: self._emit(self.accept_requested))
        self.decline_btn.clicked.connect(lambda: self._emit(self.decline_requested))
        # below each other: side by side they set the window's minimum width in German
        # ("Zusammenführen mit…")
        buttons = QVBoxLayout()
        buttons.addWidget(self.rename_btn)
        buttons.addWidget(self.merge_btn)
        # the question gets a line of its own: next to the buttons a long name (or a
        # longer translation) wrapped into a narrow column
        suggest = QHBoxLayout()
        suggest.addStretch(1)
        suggest.addWidget(self.accept_btn)
        suggest.addWidget(self.decline_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{_('Speakers')}</b>"))
        layout.addWidget(self.list, 1)
        layout.addWidget(self.hint)
        layout.addLayout(suggest)
        layout.addLayout(buttons)
        layout.addWidget(self.note)
        self._speakers: dict[str, Speaker] = {}
        self._update_buttons()

    def set_speakers(self, speakers: dict[str, Speaker]) -> None:
        current = self.selected()
        self._speakers = dict(speakers)
        self.list.clear()
        for sid, spk in sorted(speakers.items(),
                               key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0):
            text = display_label(spk.label) + (f"  — {spk.suggestion}?" if spk.suggestion
                                                else "")
            item = QListWidgetItem(_swatch(speaker_color(sid)), text)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            score = _(", score {score}").format(score=f"{spk.score:.2f}") if spk.score else ""
            item.setToolTip(f"{text}\n{sid}{score}")  # the full text: the list may cut it
            self.list.addItem(item)
            if sid == current:
                self.list.setCurrentItem(item)
        self._update_buttons()

    def set_actions(self, rename: bool = True, merge: bool = True,
                    suggestions: bool = True) -> None:
        """Show only the actions the owner can carry out in its current state."""
        self._actions = {"rename": rename, "merge": merge, "suggestions": suggestions}
        self.rename_btn.setVisible(rename)
        self.merge_btn.setVisible(merge)
        self._update_buttons()

    def set_note(self, html: str) -> None:
        """A short note under the list (e.g. where to edit instead); "" hides it."""
        self.note.setText(html)
        self.note.setVisible(bool(html))

    def update_speaker(self, speaker: Speaker) -> None:
        self._speakers[speaker.id] = speaker
        self.set_speakers(self._speakers)

    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def select(self, speaker_id: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == speaker_id:
                self.list.setCurrentRow(i)

    def _emit(self, signal) -> None:
        if sid := self.selected():
            signal.emit(sid)

    def _ask_merge(self) -> None:
        drop = self.selected()
        if drop is None:
            return
        others = [f"{display_label(s.label)} ({sid})" for sid, s in self._speakers.items()
                  if sid != drop]
        if not others:
            return
        choice, ok = QInputDialog.getItem(
            self, _("Merge speaker"),
            _("Merge {speaker} into:").format(speaker=display_label(self._speakers[drop].label)),
            others, 0, False)
        if ok:
            self.merge_requested.emit(choice.rsplit("(", 1)[1].rstrip(")"), drop)

    def _update_buttons(self, *_args) -> None:
        sid = self.selected()
        spk = self._speakers.get(sid) if sid else None
        self.rename_btn.setEnabled(spk is not None)
        self.merge_btn.setEnabled(spk is not None and len(self._speakers) > 1)
        has_suggestion = bool(spk and spk.suggestion) and self._actions["suggestions"]
        self.hint.setText(_("Is this <b>{name}</b>?").format(name=spk.suggestion)
                          if has_suggestion and spk else "")
        for w in (self.hint, self.accept_btn, self.decline_btn):
            w.setVisible(has_suggestion)
