"""Open-set speaker identification against the voiceprint DB."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from transcribe.speakers.online_cluster import l2norm
from transcribe.speakers.voiceprint_db import VoiceprintDB


class Decision(Enum):
    KNOWN = "known"  # score >= high threshold: label automatically
    SUGGEST = "suggest"  # between thresholds: ask the user ("Is this Alice?")
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Match:
    person_id: int | None
    name: str | None
    score: float
    decision: Decision


class SpeakerMatcher:
    """Score = mean of cosine to the person's centroid and to their best-k exemplars."""

    def __init__(self, db: VoiceprintDB, model_id: str, high: float = 0.65, low: float = 0.50,
                 top_k: int = 3, channel_type: str | None = None) -> None:
        if low > high:
            raise ValueError("low threshold must not exceed high threshold")
        self.db, self.model_id = db, model_id
        self.high, self.low, self.top_k = high, low, top_k
        self.channel_type = channel_type
        self.reload()

    def reload(self) -> None:
        by_person: dict[int, list[np.ndarray]] = defaultdict(list)
        preferred: dict[int, list[np.ndarray]] = defaultdict(list)
        for vp in self.db.voiceprints(self.model_id):
            v = l2norm(vp.vector)
            by_person[vp.person_id].append(v)
            if self.channel_type and vp.channel_type == self.channel_type:
                preferred[vp.person_id].append(v)
        # Prefer exemplars recorded on the same channel type when available.
        self._exemplars = {
            pid: np.stack(preferred.get(pid) or vecs) for pid, vecs in by_person.items()
        }
        self._centroids = {pid: l2norm(m.mean(axis=0)) for pid, m in self._exemplars.items()}
        self._names = {p.id: p.display_name for p in self.db.persons()}

    def score_all(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        e = l2norm(np.asarray(embedding, dtype=np.float32))
        scores = []
        for pid, ex in self._exemplars.items():
            sims = np.sort(ex @ e)[::-1][: self.top_k]
            scores.append((pid, 0.5 * float(self._centroids[pid] @ e) + 0.5 * float(sims.mean())))
        return sorted(scores, key=lambda s: s[1], reverse=True)

    def identify(self, embedding: np.ndarray) -> Match:
        scores = self.score_all(embedding)
        if not scores:
            return Match(None, None, 0.0, Decision.UNKNOWN)
        pid, score = scores[0]
        if score >= self.high:
            decision = Decision.KNOWN
        elif score >= self.low:
            decision = Decision.SUGGEST
        else:
            return Match(None, None, score, Decision.UNKNOWN)
        return Match(pid, self._names.get(pid), score, decision)
