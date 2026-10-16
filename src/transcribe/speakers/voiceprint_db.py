"""Local voiceprint store (SQLite).

Voiceprints are biometric data (GDPR Art. 9): they stay local, consent is recorded per
person, and deleting a person removes every vector belonging to them.
Embeddings are only comparable within one embedding model, so every vector carries a
``model_id`` and queries are always scoped to one model.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS person (
    id INTEGER PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    consent_at TEXT,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS voiceprint (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    quality REAL NOT NULL DEFAULT 1.0,
    duration_s REAL NOT NULL DEFAULT 0.0,
    channel_type TEXT NOT NULL DEFAULT 'unknown',
    source_session TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS voiceprint_person_model ON voiceprint(person_id, model_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Person:
    id: int
    display_name: str
    created_at: str
    consent_at: str | None
    notes: str


@dataclass(frozen=True)
class Voiceprint:
    id: int
    person_id: int
    model_id: str
    vector: np.ndarray
    quality: float
    duration_s: float
    channel_type: str


class VoiceprintDB:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- persons -------------------------------------------------------------------------
    def add_person(self, name: str, consent: bool = False, notes: str = "") -> Person:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO person (display_name, created_at, consent_at, notes) "
                "VALUES (?, ?, ?, ?)",
                (name, _now(), _now() if consent else None, notes),
            )
        return self.get_person(cur.lastrowid)  # type: ignore[arg-type]

    def get_person(self, person_id: int) -> Person:
        row = self._conn.execute("SELECT * FROM person WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            raise KeyError(person_id)
        return Person(**dict(row))

    def find_person(self, name: str) -> Person | None:
        row = self._conn.execute(
            "SELECT * FROM person WHERE display_name = ?", (name,)
        ).fetchone()
        return Person(**dict(row)) if row else None

    def persons(self) -> list[Person]:
        rows = self._conn.execute("SELECT * FROM person ORDER BY display_name").fetchall()
        return [Person(**dict(r)) for r in rows]

    def rename_person(self, person_id: int, new_name: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE person SET display_name = ? WHERE id = ?", (new_name, person_id)
            )

    def merge_persons(self, keep_id: int, drop_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE voiceprint SET person_id = ? WHERE person_id = ?", (keep_id, drop_id)
            )
            self._conn.execute("DELETE FROM person WHERE id = ?", (drop_id,))

    def delete_person(self, person_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM person WHERE id = ?", (person_id,))

    # --- voiceprints ---------------------------------------------------------------------
    def add_voiceprint(
        self,
        person_id: int,
        model_id: str,
        vector: np.ndarray,
        quality: float = 1.0,
        duration_s: float = 0.0,
        channel_type: str = "unknown",
        source_session: str | None = None,
    ) -> int:
        v = np.ascontiguousarray(vector, dtype=np.float32)
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO voiceprint (person_id, model_id, dim, vector, quality, duration_s,"
                " channel_type, source_session, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (person_id, model_id, v.shape[0], v.tobytes(), quality, duration_s,
                 channel_type, source_session, _now()),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def voiceprints(self, model_id: str, person_id: int | None = None) -> list[Voiceprint]:
        sql = "SELECT * FROM voiceprint WHERE model_id = ?"
        args: tuple = (model_id,)
        if person_id is not None:
            sql += " AND person_id = ?"
            args += (person_id,)
        rows = self._conn.execute(sql, args).fetchall()
        return [
            Voiceprint(
                id=r["id"], person_id=r["person_id"], model_id=r["model_id"],
                vector=np.frombuffer(r["vector"], dtype=np.float32, count=r["dim"]).copy(),
                quality=r["quality"], duration_s=r["duration_s"],
                channel_type=r["channel_type"],
            )
            for r in rows
        ]

    def delete_voiceprint(self, voiceprint_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM voiceprint WHERE id = ?", (voiceprint_id,))
