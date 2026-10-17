"""Local voiceprint store (SQLite).

Voiceprints are biometric data (GDPR Art. 9): they stay local, consent is recorded per
person, and deleting a person removes every vector belonging to them.
Embeddings are only comparable within one embedding model, so every vector carries a
``model_id`` and queries are always scoped to one model.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from transcribe.speakers.crypto import Cipher, NullCipher

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
    created_at TEXT NOT NULL,
    encrypted INTEGER NOT NULL DEFAULT 0
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


@dataclass(frozen=True)
class VoiceprintInfo:
    id: int
    model_id: str
    duration_s: float
    channel_type: str
    source_session: str | None
    created_at: str


@dataclass(frozen=True)
class PersonStats:
    voiceprints: int
    seconds: float
    models: list[str]
    sources: list[str]


EXPORT_FORMAT = "transcribe-voiceprints/1"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class VoiceprintDB:
    def __init__(self, path: Path | str, cipher: Cipher | None = None) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher: Cipher = cipher or NullCipher()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(voiceprint)")}
        if "encrypted" not in cols:  # databases created before encryption support
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE voiceprint ADD COLUMN encrypted INTEGER NOT NULL DEFAULT 0")

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

    def set_consent(self, person_id: int, consent: bool) -> None:
        with self._conn:
            self._conn.execute("UPDATE person SET consent_at = ? WHERE id = ?",
                               (_now() if consent else None, person_id))

    def voiceprint_counts(self, model_id: str | None = None) -> dict[int, int]:
        sql = "SELECT person_id, COUNT(*) AS n FROM voiceprint"
        args: tuple = ()
        if model_id:
            sql += " WHERE model_id = ?"
            args = (model_id,)
        rows = self._conn.execute(sql + " GROUP BY person_id", args).fetchall()
        return {r["person_id"]: r["n"] for r in rows}

    def person_stats(self) -> dict[int, PersonStats]:
        """Per person: number of voiceprints, seconds of speech, models and sources."""
        rows = self._conn.execute(
            "SELECT person_id, COUNT(*) AS n, SUM(duration_s) AS seconds, "
            "GROUP_CONCAT(DISTINCT model_id) AS models, "
            "GROUP_CONCAT(DISTINCT source_session) AS sources "
            "FROM voiceprint GROUP BY person_id").fetchall()
        return {r["person_id"]: PersonStats(
            r["n"], r["seconds"] or 0.0, sorted((r["models"] or "").split(",")),
            sorted(x for x in (r["sources"] or "").split(",") if x)) for r in rows}

    def voiceprint_info(self, person_id: int) -> list[VoiceprintInfo]:
        """A person's voiceprints without their vectors (no decryption needed)."""
        rows = self._conn.execute(
            "SELECT id, model_id, duration_s, channel_type, source_session, created_at "
            "FROM voiceprint WHERE person_id = ? ORDER BY created_at, id",
            (person_id,)).fetchall()
        return [VoiceprintInfo(**dict(r)) for r in rows]

    def set_notes(self, person_id: int, notes: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE person SET notes = ? WHERE id = ?", (notes, person_id))

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
        blob = self.cipher.encrypt(v.tobytes())
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO voiceprint (person_id, model_id, dim, vector, quality, duration_s,"
                " channel_type, source_session, created_at, encrypted)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (person_id, model_id, v.shape[0], blob, quality, duration_s,
                 channel_type, source_session, _now(), int(self.cipher.encrypted)),
            )
        return int(cur.lastrowid)  # type: ignore[arg-type]

    def add_exemplars(self, person_id: int, model_id: str, vectors: list[np.ndarray],
                      durations: list[float], channel_type: str = "unknown",
                      source_session: str | None = None, max_per_person: int = 30,
                      duplicate_sim: float = 0.97) -> int:
        """Add exemplars, skipping near-duplicates and keeping at most ``max_per_person``.

        When over the cap, the most redundant exemplar (highest mean similarity to the
        others) is removed, so the stored set stays diverse (mics, rooms, codecs).
        Returns the number of exemplars added.
        """
        existing = self.voiceprints(model_id, person_id)
        kept = [_unit(vp.vector) for vp in existing]
        added = 0
        for vec, dur in zip(vectors, durations, strict=True):
            u = _unit(np.asarray(vec, dtype=np.float32))
            if kept and max(float(k @ u) for k in kept) >= duplicate_sim:
                continue
            self.add_voiceprint(person_id, model_id, vec, duration_s=dur,
                                channel_type=channel_type, source_session=source_session)
            kept.append(u)
            added += 1
        self._prune(person_id, model_id, max_per_person)
        return added

    def _prune(self, person_id: int, model_id: str, cap: int) -> None:
        vps = self.voiceprints(model_id, person_id)
        while len(vps) > cap:
            m = np.stack([_unit(v.vector) for v in vps])
            sims = m @ m.T
            np.fill_diagonal(sims, 0.0)
            worst = int(np.argmax(sims.sum(axis=1)))
            self.delete_voiceprint(vps[worst].id)
            vps.pop(worst)

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
                vector=np.frombuffer(self._decode(r), dtype=np.float32, count=r["dim"]).copy(),
                quality=r["quality"], duration_s=r["duration_s"],
                channel_type=r["channel_type"],
            )
            for r in rows
        ]

    def _decode(self, row: sqlite3.Row) -> bytes:
        if row["encrypted"] and not self.cipher.encrypted:
            raise RuntimeError("Voiceprints are encrypted; enable speakers.encrypt_voiceprints "
                               "(and provide the key) to read them")
        return self.cipher.decrypt(row["vector"]) if row["encrypted"] else row["vector"]

    # --- export / import ------------------------------------------------------------------
    def export_json(self, path: Path, names: list[str] | None = None) -> int:
        """Write persons and (decrypted) voiceprints as JSON. Returns #voiceprints."""
        persons = [p for p in self.persons() if names is None or p.display_name in names]
        out, n = [], 0
        for p in persons:
            rows = self._conn.execute("SELECT * FROM voiceprint WHERE person_id = ?",
                                      (p.id,)).fetchall()
            vps = [{"model_id": r["model_id"], "dim": r["dim"],
                    "vector": base64.b64encode(self._decode(r)).decode(),
                    "duration_s": r["duration_s"], "channel_type": r["channel_type"],
                    "created_at": r["created_at"]} for r in rows]
            n += len(vps)
            out.append({"name": p.display_name, "consent_at": p.consent_at, "notes": p.notes,
                        "voiceprints": vps})
        path.write_text(json.dumps({"format": EXPORT_FORMAT, "persons": out}, indent=1),
                        encoding="utf-8")
        return n

    def import_json(self, path: Path) -> tuple[int, int]:
        """Merge an export into this DB (persons matched by name). Returns (#persons, #vps)."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != EXPORT_FORMAT:
            raise ValueError(f"Not a voiceprint export: {path}")
        n_vp = 0
        for entry in data["persons"]:
            person = self.find_person(entry["name"]) or self.add_person(
                entry["name"], consent=bool(entry.get("consent_at")), notes=entry.get("notes", ""))
            by_model: dict[str, tuple[list[np.ndarray], list[float], list[str]]] = {}
            for vp in entry["voiceprints"]:
                vec = np.frombuffer(base64.b64decode(vp["vector"]), dtype=np.float32,
                                    count=vp["dim"])
                vecs, durs, chans = by_model.setdefault(vp["model_id"], ([], [], []))
                vecs.append(vec)
                durs.append(vp.get("duration_s", 0.0))
                chans.append(vp.get("channel_type", "unknown"))
            for model_id, (vecs, durs, chans) in by_model.items():
                for vec, dur, chan in zip(vecs, durs, chans, strict=True):
                    n_vp += self.add_exemplars(person.id, model_id, [vec], [dur], chan)
        return len(data["persons"]), n_vp

    def delete_voiceprints_of(self, person_id: int) -> None:
        """Remove all voiceprints (every model) of a person, e.g. on withdrawn consent."""
        with self._conn:
            self._conn.execute("DELETE FROM voiceprint WHERE person_id = ?", (person_id,))

    def delete_voiceprint(self, voiceprint_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM voiceprint WHERE id = ?", (voiceprint_id,))
