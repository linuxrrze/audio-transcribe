"""Transcript display: final segments, live partials, click-to-seek, playback highlight.

With ``editable`` (stored sessions), a right-click offers per-line corrections (speaker,
text) and a double-click edits a line's text; the view only emits requests.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QTimer, QUrl, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMenu, QTextBrowser

from transcribe.export.base import fmt_clock
from transcribe.gui.colors import speaker_color
from transcribe.storage.transcript_model import Speaker, Transcript


@dataclass
class _Seg:
    id: str
    t0: float
    t1: float
    speaker: str | None
    text: str
    channel: str


class TranscriptView(QTextBrowser):
    """Re-renders (throttled) on changes; keeps following the end unless scrolled up."""

    seek_requested = Signal(float)  # user clicked a timestamp
    assign_requested = Signal(list, str)  # segment ids, speaker id ("" = new speaker)
    edit_requested = Signal(str)  # segment id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.editable = False
        self._order: list[str] = []  # segment id per text block, as rendered
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_anchor)
        self._segs: dict[str, _Seg] = {}
        self._speakers: dict[str, Speaker] = {}
        self._partials: dict[str, tuple[str, str]] = {}
        self._current: str | None = None
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._render_if_dirty)
        self._timer.start()

    # --- model -------------------------------------------------------------------------
    def clear_all(self) -> None:
        self._segs.clear()
        self._speakers.clear()
        self._partials.clear()
        self._current = None
        self._mark()

    def set_transcript(self, transcript: Transcript, keep_position: bool = False) -> None:
        position = self.verticalScrollBar().value()
        self.clear_all()
        self._speakers = dict(transcript.speakers)
        for i, s in enumerate(transcript.segments):
            sid = s.id or f"seg-{i}"
            self._segs[sid] = _Seg(sid, s.t0, s.t1, s.speaker_id, s.text, s.channel_id)
        self._render()
        if keep_position:  # after an edit: stay where the user was working
            self.verticalScrollBar().setValue(position)
        else:
            self.moveCursor(QTextCursor.MoveOperation.Start)

    def add_segment(self, seg_id: str, t0: float, t1: float, speaker: str | None, text: str,
                    channel: str = "") -> None:
        self._segs[seg_id] = _Seg(seg_id, t0, t1, speaker, text, channel)
        self._mark()

    def remove_segment(self, seg_id: str) -> None:
        if self._segs.pop(seg_id, None):
            self._mark()

    def relabel_segment(self, seg_id: str, speaker: str) -> None:
        if seg := self._segs.get(seg_id):
            seg.speaker = speaker
            self._mark()

    def set_speaker(self, speaker: Speaker) -> None:
        self._speakers[speaker.id] = speaker
        self._mark()

    def set_partial(self, channel: str, committed: str, pending: str) -> None:
        self._partials[channel] = (committed, pending)
        self._mark()

    def highlight_at(self, t: float) -> None:
        """Highlight the segment playing at time ``t`` (seconds)."""
        current = next((s.id for s in self._segs.values() if s.t0 <= t < s.t1 + 0.3), None)
        if current != self._current:
            self._current = current
            self._mark()

    def segment_count(self) -> int:
        return len(self._segs)

    def label(self, speaker_id: str | None) -> str:
        if speaker_id is None:
            return ""
        spk = self._speakers.get(speaker_id)
        return spk.label if spk else f"Speaker {speaker_id[1:]}"

    # --- rendering ---------------------------------------------------------------------
    def _mark(self) -> None:
        self._dirty = True

    def _render_if_dirty(self) -> None:
        if self._dirty:
            self._render()

    def _render(self) -> None:
        self._dirty = False
        bar = self.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        position = bar.value()
        multi = len({s.channel for s in self._segs.values()} | set(self._partials)) > 1
        parts = []
        self._order = []
        for seg in sorted(self._segs.values(), key=lambda s: s.t0):
            self._order.append(seg.id)
            color = speaker_color(seg.speaker)
            bg = ' style="background-color: rgba(255, 200, 0, 0.25);"' \
                if seg.id == self._current else ""
            channel = (f'<span style="color:#888;">[{html.escape(seg.channel)}]</span> '
                       if multi and seg.channel else "")
            speaker = self.label(seg.speaker)
            spk = (f'<b style="color:{color};">{html.escape(speaker)}:</b> ' if speaker else "")
            parts.append(
                f'<p{bg}><a href="seek:{seg.t0:.3f}" style="color:#888;text-decoration:none;">'
                f'[{fmt_clock(seg.t0)}]</a> {channel}{spk}{html.escape(seg.text)}</p>')
        for channel, (committed, pending) in self._partials.items():
            if committed or pending:
                tag = f"[{html.escape(channel)}] " if multi else ""
                parts.append(f'<p style="color:#888;">… {tag}{html.escape(committed)} '
                             f'<i>{html.escape(pending)}</i></p>')
        self.setHtml("".join(parts))
        bar.setValue(bar.maximum() if follow else position)

    # --- corrections (editable) ---------------------------------------------------------
    def segments_at(self, pos: QPoint) -> list[str]:
        """Segments of the selected lines if the selection includes ``pos`` (viewport
        coordinates), else the line under ``pos``."""
        clicked = self.cursorForPosition(pos).blockNumber()
        cursor = self.textCursor()
        blocks = [clicked]
        if cursor.hasSelection():
            doc = self.document()
            first = doc.findBlock(cursor.selectionStart()).blockNumber()
            last = doc.findBlock(cursor.selectionEnd()).blockNumber()
            if first <= clicked <= last:
                blocks = list(range(first, last + 1))
        return [self._order[b] for b in blocks if 0 <= b < len(self._order)]

    def contextMenuEvent(self, event) -> None:
        seg_ids = self.segments_at(event.pos()) if self.editable else []
        if not seg_ids:
            super().contextMenuEvent(event)
            return
        menu = QMenu(self)
        n = len(seg_ids)
        title = "this line" if n == 1 else f"{n} lines"
        speakers = menu.addMenu(f"Speaker of {title}")
        for sid, spk in sorted(self._speakers.items(), key=lambda kv: _num(kv[0])):
            action = speakers.addAction(spk.label)
            action.setCheckable(True)
            action.setChecked(all(self._segs[i].speaker == sid for i in seg_ids))
            action.triggered.connect(lambda _=False, sid=sid: self.assign_requested.emit(
                seg_ids, sid))
        speakers.addSeparator()
        speakers.addAction("New speaker…").triggered.connect(
            lambda: self.assign_requested.emit(seg_ids, ""))
        edit = menu.addAction("Edit text… (double-click)")
        edit.setEnabled(n == 1)
        edit.triggered.connect(lambda: self.edit_requested.emit(seg_ids[0]))
        menu.addAction("Play from here").triggered.connect(
            lambda: self.seek_requested.emit(self._segs[seg_ids[0]].t0))
        menu.exec(event.globalPos())

    def mouseDoubleClickEvent(self, event) -> None:
        seg_ids = self.segments_at(event.pos()) if self.editable else []
        if seg_ids:
            self.edit_requested.emit(seg_ids[-1])
        else:
            super().mouseDoubleClickEvent(event)

    def segment_text(self, seg_id: str) -> str:
        return self._segs[seg_id].text

    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() == "seek":
            self.seek_requested.emit(float(url.path()))


def _num(speaker_id: str) -> int:
    return int(speaker_id[1:]) if speaker_id[1:].isdigit() else 0
