from __future__ import annotations

import json

from transcribe.storage.transcript_model import Transcript


class JsonExporter:
    extension = "json"

    def render(self, transcript: Transcript) -> str:
        return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)
