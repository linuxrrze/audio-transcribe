"""Assign speakers to ASR words using diarization turns."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from transcribe.storage.transcript_model import Segment, Word


@dataclass(frozen=True)
class Turn:
    t0: float
    t1: float
    speaker_id: str


def assign_speakers(words: list[Word], turns: list[Turn], max_gap: float = 0.5) -> None:
    """Set ``speaker_id`` (in place) to the speaker with the largest overlap.

    Words with no overlapping turn take the nearest turn within ``max_gap`` seconds.
    Words overlapped by more than one speaker are flagged ``overlap=True``.
    """
    turns = sorted(turns, key=lambda t: t.t0)
    for word in words:
        overlap: dict[str, float] = defaultdict(float)
        for turn in turns:
            if turn.t0 >= word.t1:
                break
            d = min(word.t1, turn.t1) - max(word.t0, turn.t0)
            if d > 0:
                overlap[turn.speaker_id] += d
        if overlap:
            word.speaker_id = max(overlap, key=overlap.__getitem__)
            word.overlap = len(overlap) > 1
            continue
        nearest = min(
            turns,
            key=lambda t: min(abs(t.t0 - word.t1), abs(word.t0 - t.t1)),
            default=None,
        )
        if nearest and min(abs(nearest.t0 - word.t1), abs(word.t0 - nearest.t1)) <= max_gap:
            word.speaker_id = nearest.speaker_id


def words_to_segments(words: list[Word], max_pause: float = 1.5, max_duration: float = 30.0,
                      lang: str | None = None, channel_id: str = "default") -> list[Segment]:
    """Group consecutive words into segments.

    Splits on speaker change, on pauses longer than ``max_pause`` and, once a segment is
    longer than ``max_duration``, at the next sentence end (hard split at twice that).
    """
    segments: list[Segment] = []
    for word in words:
        cur = segments[-1] if segments else None
        if cur is None or _should_split(cur, word, max_pause, max_duration):
            cur = Segment(t0=word.t0, t1=word.t1, speaker_id=word.speaker_id, lang=lang,
                          channel_id=channel_id)
            segments.append(cur)
        cur.words.append(word)
        cur.t1 = word.t1
    return segments


def _should_split(cur: Segment, word: Word, max_pause: float, max_duration: float) -> bool:
    if cur.speaker_id != word.speaker_id or word.t0 - cur.t1 > max_pause:
        return True
    length = word.t1 - cur.t0
    sentence_end = cur.words[-1].text.rstrip().endswith((".", "?", "!"))
    return length > 2 * max_duration or (length > max_duration and sentence_end)
