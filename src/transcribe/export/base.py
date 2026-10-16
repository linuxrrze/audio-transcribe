"""Exporter plugin protocol."""

from __future__ import annotations

from typing import Protocol

from transcribe.storage.transcript_model import Transcript


class Exporter(Protocol):
    extension: str

    def render(self, transcript: Transcript) -> str: ...


def fmt_clock(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
