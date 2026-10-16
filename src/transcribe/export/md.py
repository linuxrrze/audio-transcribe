from __future__ import annotations

from transcribe.export.base import fmt_clock
from transcribe.storage.transcript_model import Transcript


class MarkdownExporter:
    extension = "md"

    def render(self, transcript: Transcript) -> str:
        title = transcript.meta.get("title", "Transcript")
        out = [f"# {title}", ""]
        for s in transcript.segments:
            label = f"**{transcript.speaker_label(s.speaker_id)}** " if s.speaker_id else ""
            out.append(f"{label}_{fmt_clock(s.t0)}_  ")
            out.append(s.text)
            out.append("")
        return "\n".join(out)
