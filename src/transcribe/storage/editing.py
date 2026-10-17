"""Manual corrections of a transcript: speakers per line, merged speakers, edited text.

Pure functions on a ``Transcript`` (the session wrappers in ``transcribe.sessions`` load,
save and keep the voiceprint exemplars in step). Segments are addressed by id; call
``Transcript.number_segments()`` first for transcripts written before ids were assigned.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


@dataclass
class EditResult:
    """What an edit did to segments, so derived data (exemplars) can follow."""

    moved: dict[str, str] = field(default_factory=dict)  # segment id -> new speaker id
    renamed_speakers: dict[str, str] = field(default_factory=dict)  # dropped -> kept
    changed_audio: set[str] = field(default_factory=set)  # segment ids now covering other audio
    removed_speakers: set[str] = field(default_factory=set)
    new_segments: list[str] = field(default_factory=list)


def segment(t: Transcript, seg_id: str) -> Segment:
    for seg in t.segments:
        if seg.id == seg_id:
            return seg
    raise KeyError(f"No segment {seg_id!r}")


def add_speaker(t: Transcript, label: str) -> str:
    """Create a speaker named ``label`` (marked as set by the user); returns its id."""
    n = max((int(s[1:]) for s in t.speakers if s[1:].isdigit()), default=0) + 1
    sid = f"S{n}"
    t.speakers[sid] = Speaker(sid, label, confirmed=True)
    return sid


def find_speaker(t: Transcript, name: str) -> str | None:
    """Speaker id for an id ("S3"), a label ("Justus", case-insensitive) or "Speaker 3"."""
    if name in t.speakers:
        return name
    for sid, spk in t.speakers.items():
        if spk.label.casefold() == name.casefold():
            return sid
    return None


def assign(t: Transcript, seg_ids: list[str], speaker_id: str) -> EditResult:
    """Attribute whole segments (and their words) to ``speaker_id``."""
    if speaker_id not in t.speakers:
        raise KeyError(f"Unknown speaker {speaker_id}")
    result = EditResult()
    for seg_id in seg_ids:
        seg = segment(t, seg_id)
        seg.speaker_id = speaker_id
        for w in seg.words:
            w.speaker_id = speaker_id
        result.moved[seg_id] = speaker_id
    result.removed_speakers = drop_unused_speakers(t)
    return result


def merge(t: Transcript, keep: str, drop: str) -> EditResult:
    """All of ``drop``'s speech becomes ``keep``'s (fixes a diarization split)."""
    if keep not in t.speakers or drop not in t.speakers:
        raise KeyError(f"Unknown speaker {keep if keep not in t.speakers else drop}")
    if keep == drop:
        raise ValueError("Cannot merge a speaker into itself")
    for seg in t.segments:
        if seg.speaker_id == drop:
            seg.speaker_id = keep
        for w in seg.words:
            if w.speaker_id == drop:
                w.speaker_id = keep
    del t.speakers[drop]
    return EditResult(renamed_speakers={drop: keep}, removed_speakers={drop})


def drop_unused_speakers(t: Transcript) -> set[str]:
    """Remove speakers left without any segment, unless the user named them."""
    used = {s.speaker_id for s in t.segments}
    gone = {sid for sid, spk in t.speakers.items() if sid not in used and not spk.confirmed}
    for sid in gone:
        del t.speakers[sid]
    return gone


# --- text ---------------------------------------------------------------------------------
_TOKEN = re.compile(r"\S+")


def _norm(text: str) -> str:
    return text.strip().casefold()


def retime(old: list[Word], tokens: list[str], t0: float, t1: float) -> list[Word]:
    """Words for ``tokens`` with times taken from the old words they replace.

    Unchanged words keep their timing; a replaced run shares the old run's time span
    (by character length); inserted words get the gap around them. ``t0``/``t1`` bound
    the segment when there are no old words to borrow from.
    """
    sm = difflib.SequenceMatcher(None, [_norm(w.text) for w in old],
                                 [_norm(x) for x in tokens], autojunk=False)
    speaker = old[0].speaker_id if old else None
    out: list[Word] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                w = old[i1 + k]
                out.append(Word(" " + tokens[j1 + k], w.t0, w.t1, w.confidence, w.speaker_id,
                                w.overlap))
            continue
        if op == "delete":
            continue
        if op == "replace":
            a, b = old[i1].t0, old[i2 - 1].t1
        else:  # insert: the gap between neighbours
            a = old[i1 - 1].t1 if i1 > 0 else t0
            b = old[i1].t0 if i1 < len(old) else t1
            if out:
                a = max(a, out[-1].t1)
        b = max(b, a)
        new = tokens[j1:j2]
        total = sum(len(x) for x in new)
        pos = a
        for x in new:
            end = pos + (b - a) * len(x) / total
            out.append(Word(" " + x, pos, max(end, pos + 0.01), 1.0, speaker))
            pos = end
    return out


def set_text(t: Transcript, seg_id: str, text: str) -> EditResult:
    """Replace a segment's text, keeping word timings where the words are unchanged.

    Each line of ``text`` becomes its own segment (split a line where the speaker
    changes, then assign the parts); empty text deletes the segment.
    """
    seg = segment(t, seg_id)
    index = t.segments.index(seg)
    lines = [_TOKEN.findall(line) for line in text.splitlines()]
    lines = [x for x in lines if x]
    result = EditResult()
    if not lines:
        t.segments.remove(seg)
        result.changed_audio.add(seg_id)
        result.removed_speakers = drop_unused_speakers(t)
        return result
    words = retime(seg.words, [x for line in lines for x in line], seg.t0, seg.t1)
    parts, start = [], 0
    for line in lines:
        parts.append(words[start:start + len(line)])
        start += len(line)
    taken = {s.id for s in t.segments}
    new_segments = []
    for k, part in enumerate(parts):
        new_id = seg_id if k == 0 else _free_id(f"{seg_id}.{k}", taken)
        taken.add(new_id)
        new_segments.append(Segment(part[0].t0, part[-1].t1, part, seg.speaker_id, seg.lang,
                                    seg.channel_id, new_id))
    t.segments[index:index + 1] = new_segments
    if len(new_segments) > 1:  # exemplars of the old segment may now span two speakers
        result.changed_audio.add(seg_id)
        result.new_segments = [s.id for s in new_segments[1:]]
    return result


def _free_id(candidate: str, taken: set[str]) -> str:
    n = 1
    out = candidate
    while out in taken:
        n += 1
        out = f"{candidate}-{n}"
    return out
