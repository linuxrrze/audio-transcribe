"""Echo suppression across channels.

Without a headset, the microphone also picks up what the speakers play, so remote
participants would be transcribed twice: once from system audio (clean, digital) and
once from the microphone (echo). A mic segment that overlaps a segment of another
channel in time and has similar text is treated as echo and dropped.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from transcribe.storage.transcript_model import Segment

ECHO_CHANNEL = "mic"
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def is_echo(mic: Segment, other: Segment, min_overlap: float = 0.5,
            min_similarity: float = 0.6) -> bool:
    overlap = min(mic.t1, other.t1) - max(mic.t0, other.t0)
    shorter = min(mic.t1 - mic.t0, other.t1 - other.t0)
    if shorter <= 0 or overlap / shorter < min_overlap:
        return False
    a, b = _tokens(mic.text), _tokens(other.text)
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= min_similarity


def echo_of(seg: Segment, others: list[Segment], echo_channel: str = ECHO_CHANNEL
            ) -> list[Segment]:
    """Segments that are echoes in the pair (``seg`` vs ``others``).

    If ``seg`` is on the echo channel, returns ``[seg]`` when it duplicates any other
    channel's segment; otherwise returns the echo-channel segments it duplicates.
    """
    if seg.channel_id == echo_channel:
        dup = any(o.channel_id != echo_channel and is_echo(seg, o) for o in others)
        return [seg] if dup else []
    return [o for o in others if o.channel_id == echo_channel and is_echo(o, seg)]


def suppress_echo(segments: list[Segment], echo_channel: str = ECHO_CHANNEL
                  ) -> tuple[list[Segment], list[Segment]]:
    """Split ``segments`` into (kept, dropped echoes)."""
    others = [s for s in segments if s.channel_id != echo_channel]
    dropped = [s for s in segments if s.channel_id == echo_channel
               and any(is_echo(s, o) for o in others)]
    dropped_ids = {id(s) for s in dropped}
    return [s for s in segments if id(s) not in dropped_ids], dropped
