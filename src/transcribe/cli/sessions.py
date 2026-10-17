"""`tx sessions ...`: list sessions and correct their transcripts (speakers, text)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from transcribe import sessions
from transcribe.core.config import Config
from transcribe.export.base import fmt_clock
from transcribe.i18n import _, display_label, ngettext
from transcribe.storage import editing
from transcribe.storage.transcript_model import Transcript

sessions_app = typer.Typer(help=_("List sessions and correct their transcripts."),
                           no_args_is_help=True)
console = Console()
err = Console(stderr=True)

SessionArg = Annotated[str, typer.Argument(
    help=_("Session folder, its name in the sessions directory, or 'latest'."))]
TimeArg = Annotated[str, typer.Argument(
    help=_("Line start as shown in the transcript (24:10, 00:24:10, 1450.5) or a range "
           "(20:22-24:27)."))]


def resolve(session: str, config: Config | None = None) -> Path:
    path = Path(session).expanduser()
    if path.is_dir():
        return path
    root = (config or Config.load()).resolved_sessions_dir()
    if session in ("latest", "last"):
        found = sessions.list_sessions(root)
        if not found:
            raise _fail(_("No sessions in {path}").format(path=root))
        return found[0].path
    if (root / session).is_dir():
        return root / session
    matches = [p for p in root.iterdir() if p.is_dir() and session in p.name] \
        if root.exists() else []
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise _fail(ngettext("No session {name} (ambiguous: {n} match)",
                             "No session {name} (ambiguous: {n} matches)",
                             len(matches)).format(name=repr(session), n=len(matches)))
    raise _fail(_("No session {name} in {path}").format(name=repr(session), path=root))


def parse_time(text: str) -> float:
    """'1:02:03', '24:10', '00:24:10' or seconds ('1450.5')."""
    parts = text.strip().split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise _fail(_("Not a time: {text}").format(text=repr(text))) from e
    seconds = 0.0
    for v in values:
        seconds = seconds * 60 + v
    return seconds


def select(t: Transcript, when: str, only: str | None = None) -> list[str]:
    """Segment ids for a time or range. A time selects the lines starting in that second
    (as displayed); if none starts there, the line running at that time."""
    if m := re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", when):
        a, b = parse_time(m[1]), parse_time(m[2])
        segs = [s for s in t.segments if a <= s.t0 < int(b) + 1]
    else:
        at = parse_time(when)
        segs = [s for s in t.segments if int(s.t0) == int(at)] or \
            [s for s in t.segments if s.t0 <= at < s.t1]
    if only:
        sid = _speaker(t, only)
        segs = [s for s in segs if s.speaker_id == sid]
    if not segs:
        raise _fail(_("No line at {time} spoken by {speaker}").format(time=when, speaker=only)
                    if only else _("No line at {time}").format(time=when))
    return [s.id for s in segs]


def _speaker(t: Transcript, name: str) -> str:
    sid = editing.find_speaker(t, name)
    if sid is None:
        raise _fail(_("No speaker {name}; speakers: {speakers}").format(
            name=repr(name), speakers=", ".join(f"{k} ({display_label(v.label)})"
                                                for k, v in t.speakers.items())))
    return sid


def _fail(msg: str) -> typer.Exit:
    err.print(f"[red]{msg}[/]")
    return typer.Exit(1)


def _show(t: Transcript, seg_ids: list[str]) -> None:
    by_id = {s.id: s for s in t.segments}
    for sid in seg_ids:
        if seg := by_id.get(sid):
            console.print(f"[dim][{fmt_clock(seg.t0)}][/] "
                          f"[bold]{t.speaker_label(seg.speaker_id)}:[/] {seg.text}",
                          highlight=False)


@sessions_app.command("list", help=_("List stored sessions, newest first."))
def sessions_list() -> None:
    table = Table(_("Session"), _("Created"), _("Duration"), _("Speakers"), "")
    for info in sessions.list_sessions(Config.load().resolved_sessions_dir()):
        flags = [_("refined")] if info.refined else []
        flags += [_("live")] if info.live else []
        table.add_row(info.path.name, f"{info.created:%Y-%m-%d %H:%M}",
                      fmt_clock(info.duration_s), str(info.speakers), " ".join(flags))
    console.print(table)


@sessions_app.command(
    "speakers", help=_("A session's speakers: id, name, lines, speaking time and first line."))
def sessions_speakers(session: SessionArg) -> None:
    t = sessions.load_numbered(resolve(session))
    table = Table(_("Id"), _("Name"), _("Lines"), _("Words"), _("Time"), _("First line"))
    for sid, spk in sorted(t.speakers.items(), key=lambda kv: int(kv[0][1:])
                           if kv[0][1:].isdigit() else 0):
        segs = [s for s in t.segments if s.speaker_id == sid]
        first = segs[0] if segs else None
        table.add_row(sid, display_label(spk.label)
                      + (f" ({spk.suggestion}?)" if spk.suggestion else ""),
                      str(len(segs)), str(sum(len(s.words) for s in segs)),
                      fmt_clock(sum(s.t1 - s.t0 for s in segs)),
                      f"[{fmt_clock(first.t0)}] {first.text[:50]}" if first else "-")
    console.print(table)


@sessions_app.command("rename", help=_(
    "Rename a speaker (no voiceprint; see `tx speakers name` to also learn the voice)."))
def sessions_rename(session: SessionArg,
                    speaker: Annotated[str, typer.Argument(
                        help=_("Id (S3) or current name."))],
                    name: Annotated[str, typer.Argument(help=_("New name."))]) -> None:
    path = resolve(session)
    sid = _speaker(sessions.load_numbered(path), speaker)
    sessions.rename_speaker(path, sid, name)
    console.print(_("{speaker} is now {name}").format(speaker=sid, name=name))


@sessions_app.command("merge", help=_(
    "Merge speakers that are the same person (a split by the speaker detection)."))
def sessions_merge(session: SessionArg,
                   keep: Annotated[str, typer.Argument(
                       help=_("Speaker to keep (id or name)."))],
                   drop: Annotated[list[str], typer.Argument(
                       help=_("Speaker(s) merged into KEEP and removed."))]) -> None:
    path = resolve(session)
    config = Config.load()
    t = sessions.load_numbered(path)
    keep_id = _speaker(t, keep)
    for name in drop:
        sid = _speaker(t, name)
        sessions.merge_speakers(path, keep_id, sid, config)
        console.print(_("{speaker} merged into {keep} ({name})").format(
            speaker=sid, keep=keep_id, name=t.speaker_label(keep_id)))


@sessions_app.command("assign", help=_(
    "Attribute lines to another (or a new) speaker.\n\n"
    "Example: tx sessions assign latest 20:22-24:27 Carlos --from Justus"))
def sessions_assign(
    session: SessionArg,
    when: TimeArg,
    speaker: Annotated[str, typer.Argument(
        help=_("Speaker id or name; an unknown name creates a new speaker."))],
    only: Annotated[str | None, typer.Option(
        "--from", help=_("Only lines currently attributed to this speaker."))] = None,
) -> None:
    path = resolve(session)
    t = sessions.load_numbered(path)
    seg_ids = select(t, when, only)
    sid = editing.find_speaker(t, speaker)
    sid = sessions.assign_segments(path, seg_ids, sid or "", None if sid else speaker,
                                   Config.load())
    if editing.find_speaker(t, speaker) is None:
        console.print(_("New speaker {id}: {name}").format(id=sid, name=speaker))
    _show(sessions.load_numbered(path), seg_ids)


@sessions_app.command("edit", help=_(
    "Correct a line's text (word timings are kept where the words are unchanged)."))
def sessions_edit(
    session: SessionArg,
    when: Annotated[str, typer.Argument(
        help=_("Line start (24:10); must select one line."))],
    text: Annotated[str, typer.Argument(
        help=_("New text. '|' splits the line into several (then assign the parts); "
               "empty text deletes the line."))],
) -> None:
    path = resolve(session)
    t = sessions.load_numbered(path)
    seg_ids = select(t, when)
    if len(seg_ids) > 1 and "." in when:  # precise time: the nearest line start
        at = parse_time(when)
        starts = {s.id: s.t0 for s in t.segments}
        seg_ids = [min(seg_ids, key=lambda i: abs(starts[i] - at))]
    if len(seg_ids) != 1:
        _show(t, seg_ids)
        options = ", ".join(f"{s.t0:.2f}" for s in t.segments if s.id in seg_ids)
        raise _fail(_("{n} lines start at {time}; pick one by its exact start in seconds: "
                      "{options}").format(n=len(seg_ids), time=when, options=options))
    new_ids = sessions.edit_text(path, seg_ids[0], text.replace("|", "\n"), Config.load())
    if new_ids:
        _show(sessions.load_numbered(path), new_ids)
    else:
        console.print(_("Line deleted."))
