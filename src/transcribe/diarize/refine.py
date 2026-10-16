"""Map refined (offline) speaker ids onto live ids so labels stay consistent."""

from __future__ import annotations

from collections import defaultdict

from transcribe.storage.transcript_model import Speaker, Transcript


def _speaker_time(transcript: Transcript) -> list[tuple[float, float, str]]:
    return [(w.t0, w.t1, s.speaker_id) for s in transcript.segments for w in s.words
            if s.speaker_id]


def map_to_reference(refined: Transcript, reference: Transcript,
                     reserved: set[str] | None = None) -> dict[str, str]:
    """Greedy one-to-one mapping refined id -> reference id by overlapping speech time.

    Refined speakers without a counterpart get fresh ids after the reference's ids and
    any ``reserved`` ids (e.g. speakers of other channels).
    """
    ref_spans = _speaker_time(reference)
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    for s in refined.segments:
        if not s.speaker_id:
            continue
        for a0, a1, rid in ref_spans:
            d = min(s.t1, a1) - max(s.t0, a0)
            if d > 0:
                overlap[(s.speaker_id, rid)] += d
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (new, old), _ in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
        if new not in mapping and old not in used:
            mapping[new] = old
            used.add(old)
    taken = [int(x[1:]) for x in set(reference.speakers) | used | (reserved or set())
             if x[1:].isdigit()]
    next_id = max(taken, default=0) + 1
    for new in sorted(refined.speakers, key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
        if new not in mapping:
            mapping[new] = f"S{next_id}"
            next_id += 1
    return mapping


def relabel(refined: Transcript, reference: Transcript,
            reserved: set[str] | None = None) -> Transcript:
    """Apply ``map_to_reference`` and carry over labels/person links from the reference."""
    mapping = map_to_reference(refined, reference, reserved)
    for seg in refined.segments:
        if seg.speaker_id:
            seg.speaker_id = mapping[seg.speaker_id]
        for w in seg.words:
            if w.speaker_id:
                w.speaker_id = mapping[w.speaker_id]
    speakers = {}
    for old_id in refined.speakers:
        sid = mapping[old_id]
        ref = reference.speakers.get(sid)
        speakers[sid] = (Speaker(sid, ref.label, ref.person_id, ref.confirmed) if ref
                         else Speaker(sid, f"Speaker {sid[1:]}"))
    refined.speakers = dict(sorted(speakers.items(), key=lambda kv: int(kv[0][1:])))
    return refined
