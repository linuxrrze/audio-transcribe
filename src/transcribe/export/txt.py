from __future__ import annotations

from transcribe.export.base import fmt_clock
from transcribe.storage.transcript_model import Transcript


class TxtExporter:
    extension = "txt"

    def render(self, transcript: Transcript) -> str:
        lines = []
        for s in transcript.segments:
            who = f"{transcript.speaker_label(s.speaker_id)}: " if s.speaker_id else ""
            lines.append(f"[{fmt_clock(s.t0)}] {who}{s.text}")
        return "\n".join(lines) + "\n"
