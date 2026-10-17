"""Rich terminal view for live sessions, fed by session events."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.text import Text

from transcribe.audio.levels import Level
from transcribe.cli.levels import meter
from transcribe.core.events import (
    AudioLevel,
    Event,
    FinalSegment,
    PartialText,
    SegmentRelabeled,
    SegmentRemoved,
    SpeakerIdentified,
    StatusChanged,
)
from transcribe.export.base import fmt_clock
from transcribe.i18n import N_, _, display_label

# Status codes of live sessions and stage codes of the offline pipeline -> display names
STATUS_LABELS = {"starting": N_("starting"), "running": N_("running"),
                 "identifying": N_("identifying"), "refining": N_("refining"),
                 "stopped": N_("stopped"), "error": N_("error")}
STAGE_LABELS = {"diarize": N_("diarize"), "language": N_("language"),
                "transcribe": N_("transcribe"), "speakers": N_("speakers")}


def stage_label(stage: str) -> str:
    """Display name of a pipeline stage; a channel prefix ("mic diarize") is kept."""
    prefix, _sep, code = stage.rpartition(" ")
    label = _(STAGE_LABELS[code]) if code in STAGE_LABELS else code
    return f"{prefix} {label}" if prefix else label


COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta"]


@dataclass
class _Line:
    t0: float
    speaker: str | None
    text: str
    channel: str = ""


@dataclass
class LiveView:
    lines: dict[str, _Line] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)  # speaker id -> display name
    partials: dict[str, tuple[str, str]] = field(default_factory=dict)  # channel -> text
    levels: dict[str, float] = field(default_factory=dict)  # channel -> RMS dBFS
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
                    self.partials[event.channel_id] = (event.committed, event.pending)
                case FinalSegment(segment=seg):
                    self.lines[seg.id] = _Line(seg.t0, seg.speaker_id, seg.text, seg.channel_id)
                case SegmentRelabeled():
                    if line := self.lines.get(event.segment_id):
                        line.speaker = event.speaker_id
                case SegmentRemoved():
                    self.lines.pop(event.segment_id, None)
                case SpeakerIdentified():
                    label = display_label(event.label)
                    if event.suggestion:
                        label += f" ({event.suggestion}?)"
                    self.labels[event.speaker_id] = label
                case AudioLevel():
                    self.levels[event.channel_id] = event.rms_db
                case StatusChanged():
                    self.status, self.detail, self.progress = (event.status, event.detail,
                                                               event.progress)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with self._lock:
            height = max(4, (options.height or console.height) - 4)
            ordered = sorted(self.lines.values(), key=lambda line: line.t0)
            body = [self._line(line) for line in ordered[-height:]]
            multi = len(self.partials) > 1
            partials = []
            for channel, (committed, pending) in self.partials.items():
                partial = Text(f"  … {'[' + channel + '] ' if multi else ''}", style="dim")
                partial.append(committed)
                if pending:
                    partial.append((" " if committed else "") + pending, style="dim italic")
                partials.append(partial)
            yield Group(*body, *partials, Text(""), self._status_bar(options.max_width))

    def _line(self, line: _Line) -> Text:
        t = Text(f"[{fmt_clock(line.t0)}] ", style="dim")
        if line.speaker:
            idx = int(line.speaker[1:]) - 1 if line.speaker[1:].isdigit() else 0
            label = self.labels.get(line.speaker,
                                    display_label(f"Speaker {line.speaker[1:]}"))
            t.append(f"{label}: ", style=f"bold {COLORS[idx % len(COLORS)]}")
        t.append(line.text)
        return t

    def _status_bar(self, width: int) -> Text:
        status = _(STATUS_LABELS[self.status]) if self.status in STATUS_LABELS else self.status
        t = Text(f" {status.upper()} ", style="reverse bold")
        t.append(f" {fmt_clock(self.now)} ")
        for channel, db in self.levels.items():  # one meter per channel (mic, system, ...)
            t.append(f" {channel} ")
            t.append_text(meter(Level(rms_db=db, receiving=True), 12))
        if self.status == "refining" and self.progress is not None:
            t.append(f"  {stage_label(self.detail)} {self.progress:.0%}")
        elif self.status == "running":
            t.append("  " + _("Ctrl+C to stop"), style="dim")
        t.truncate(width)
        return t
