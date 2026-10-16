from __future__ import annotations

from transcribe.storage.transcript_model import Transcript


def _ts(t: float, sep: str) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class SrtExporter:
    extension = "srt"

    def render(self, transcript: Transcript) -> str:
        blocks = []
        for i, s in enumerate(transcript.segments, 1):
            who = f"[{transcript.speaker_label(s.speaker_id)}] " if s.speaker_id else ""
            blocks.append(f"{i}\n{_ts(s.t0, ',')} --> {_ts(s.t1, ',')}\n{who}{s.text}\n")
        return "\n".join(blocks)


class VttExporter:
    extension = "vtt"

    def render(self, transcript: Transcript) -> str:
        blocks = ["WEBVTT\n"]
        for s in transcript.segments:
            who = f"<v {transcript.speaker_label(s.speaker_id)}>" if s.speaker_id else ""
            blocks.append(f"{_ts(s.t0, '.')} --> {_ts(s.t1, '.')}\n{who}{s.text}\n")
        return "\n".join(blocks)
