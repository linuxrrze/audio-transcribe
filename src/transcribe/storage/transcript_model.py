"""Canonical transcript data model, serialised as ``transcript.json`` in a session folder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Word:
    text: str
    t0: float
    t1: float
    confidence: float = 1.0
    speaker_id: str | None = None
    overlap: bool = False


@dataclass
class Segment:
    t0: float
    t1: float
    words: list[Word] = field(default_factory=list)
    speaker_id: str | None = None
    lang: str | None = None
    channel_id: str = "default"
    id: str = ""

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()


@dataclass
class Speaker:
    id: str
    label: str
    person_id: int | None = None
    confirmed: bool = False  # identity set by the user; automatic identification keeps it
    suggestion: str | None = None  # "Is this <name>?" from voiceprint matching
    score: float | None = None


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    speakers: dict[str, Speaker] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def number_segments(self, force: bool = False) -> None:
        """Give every segment a unique id ("seg-<index>"), as edits and voiceprint
        exemplars refer to segments by id. Keeps existing ids unless some are missing or
        duplicated (or ``force``)."""
        ids = [s.id for s in self.segments]
        if force or "" in ids or len(set(ids)) != len(ids):
            for i, seg in enumerate(self.segments):
                seg.id = f"seg-{i}"

    def speaker_label(self, speaker_id: str | None) -> str:
        if speaker_id is None:
            return "Unknown"
        spk = self.speakers.get(speaker_id)
        return spk.label if spk else speaker_id

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": self.meta,
            "speakers": [asdict(s) for s in self.speakers.values()],
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported transcript schema: {data.get('schema_version')}")
        speakers = {s["id"]: Speaker(**s) for s in data.get("speakers", [])}
        segments = [
            Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]})
            for s in data.get("segments", [])
        ]
        return cls(segments=segments, speakers=speakers, meta=dict(data.get("meta", {})))

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Transcript:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
