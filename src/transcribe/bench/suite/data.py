"""Benchmark data in one format: a dataset is a folder with audio files and a manifest.

    <bench dir>/datasets/<dataset>/recordings.jsonl   one recording per line
    <bench dir>/datasets/<dataset>/audio/<id>.flac    (or a path to the original file)

A recording has reference speaker turns (for DER) and, where the dataset has them,
the reference text per turn (for WER and cpWER). ``split`` is "dev" (for tuning) or
"test" (only for reporting) - keep them apart, or the numbers are optimistic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from transcribe.core import models


def bench_dir() -> Path:
    """Where benchmark data and caches live (next to the model cache)."""
    return models.models_dir().parent / "bench"


@dataclass
class RefTurn:
    t0: float
    t1: float
    speaker: str
    text: str = ""  # empty if the dataset has no transcripts (e.g. VoxConverse)


@dataclass
class Recording:
    id: str
    dataset: str
    split: str  # dev | test
    audio: str  # path relative to the dataset folder, or absolute
    language: str  # "en", "de", or "" (unknown / mixed: detected)
    turns: list[RefTurn]
    has_text: bool = True
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    @property
    def duration(self) -> float:
        value = self.meta.get("duration_s")
        return float(value) if isinstance(value, int | float) else \
            max((t.t1 for t in self.turns), default=0.0)

    def ref_text(self) -> str:
        return " ".join(t.text for t in sorted(self.turns, key=lambda t: t.t0))

    def ref_by_speaker(self) -> dict[str, str]:
        out: dict[str, list[str]] = {}
        for t in sorted(self.turns, key=lambda t: t.t0):
            out.setdefault(t.speaker, []).append(t.text)
        return {s: " ".join(v) for s, v in out.items()}

    def ref_turns(self) -> list[tuple[float, float, str]]:
        return [(t.t0, t.t1, t.speaker) for t in self.turns]


class Dataset:
    def __init__(self, name: str, root: Path | None = None) -> None:
        self.name = name
        self.dir = (root or bench_dir() / "datasets") / name
        self.manifest = self.dir / "recordings.jsonl"

    def exists(self) -> bool:
        return self.manifest.exists()

    def recordings(self, split: str | None = None) -> Iterator[Recording]:
        if not self.manifest.exists():
            return
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rec = Recording(**{**d, "turns": [RefTurn(**t) for t in d["turns"]]})
                if split in (None, "all", rec.split):
                    yield rec

    def audio_path(self, rec: Recording) -> Path:
        p = Path(rec.audio)
        return p if p.is_absolute() else self.dir / p

    def write(self, recordings: list[Recording]) -> None:
        """Add or replace recordings (by id)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        keep = {r.id: r for r in self.recordings()}
        keep.update({r.id: r for r in recordings})
        lines = [json.dumps(asdict(r), ensure_ascii=False) for r in keep.values()]
        tmp = self.manifest.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.manifest)


def available() -> list[str]:
    root = bench_dir() / "datasets"
    return sorted(p.name for p in root.iterdir() if (p / "recordings.jsonl").exists()) \
        if root.exists() else []
