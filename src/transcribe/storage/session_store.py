"""Session folders: ``<sessions_dir>/<timestamp>_<name>/{transcript.json, session.toml}``."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import tomli_w

from transcribe.storage.transcript_model import Transcript

TRANSCRIPT = "transcript.json"
SESSION_META = "session.toml"


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "-", name).strip("-")[:60] or "session"


def new_session_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = root / f"{stamp}_{_slug(name)}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{stamp}_{_slug(name)}-{suffix}"
    path.mkdir(parents=True)
    return path


def save_session(path: Path, transcript: Transcript, meta: dict[str, object]) -> None:
    transcript.save(path / TRANSCRIPT)
    clean = {k: v for k, v in meta.items() if v is not None}
    (path / SESSION_META).write_text(tomli_w.dumps(clean), encoding="utf-8")


def resolve_transcript(path: Path) -> Path:
    return path / TRANSCRIPT if path.is_dir() else path
