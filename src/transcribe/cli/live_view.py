"""Rich terminal view for live sessions, fed by session events."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.text import Text

from transcribe.core.events import (
    AudioLevel,
    Event,
    FinalSegment,
    PartialText,
    SegmentRelabeled,
    StatusChanged,
)
from transcribe.export.base import fmt_clock

COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta"]


@dataclass
class _Line:
    t0: float
    speaker: str | None
    text: str


@dataclass
class LiveView:
    lines: dict[str, _Line] = field(default_factory=dict)
    committed: str = ""
    pending: str = ""
    level_db: float = -100.0
    status: str = "starting"
    detail: str = ""
    progress: float | None = None
    now: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, event: Event) -> None:
        with self._lock:
            self.now = max(self.now, event.t)
            match event:
                case PartialText():
                    self.committed, self.pending = event.committed, event.pending
                case FinalSegment(segment=seg):
                    self.lines[seg.id] = _Line(seg.t0, seg.speaker_id, seg.text)
                case SegmentRelabeled():
                    if line := self.lines.get(event.segment_id):
                        line.speaker = event.speaker_id
                case AudioLevel():
                    self.level_db = event.rms_db
                case StatusChanged():
                    self.status, self.detail, self.progress = (event.status, event.detail,
                                                               event.progress)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with self._lock:
            height = max(4, (options.height or console.height) - 4)
            body = [self._line(line) for line in list(self.lines.values())[-height:]]
            partial = Text("  … ", style="dim")
            partial.append(self.committed)
            if self.pending:
                partial.append((" " if self.committed else "") + self.pending, style="dim italic")
            yield Group(*body, partial, Text(""), self._status_bar(options.max_width))

    def _line(self, line: _Line) -> Text:
        t = Text(f"[{fmt_clock(line.t0)}] ", style="dim")
        if line.speaker:
            idx = int(line.speaker[1:]) - 1 if line.speaker[1:].isdigit() else 0
            t.append(f"Speaker {line.speaker[1:]}: ", style=f"bold {COLORS[idx % len(COLORS)]}")
        t.append(line.text)
        return t

    def _status_bar(self, width: int) -> Text:
        bars = max(0, min(20, int((self.level_db + 60) / 3)))
        t = Text(f" {self.status.upper()} ", style="reverse bold")
        t.append(f" {fmt_clock(self.now)} ")
        t.append("▮" * bars, style="green" if bars < 16 else "red")
        t.append("▯" * (20 - bars), style="dim")
        if self.status == "refining" and self.progress is not None:
            t.append(f"  {self.detail} {self.progress:.0%}")
        else:
            t.append("  Ctrl+C to stop", style="dim")
        t.truncate(width)
        return t
