"""Open-set speaker identification against the voiceprint DB."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from transcribe.speakers.voiceprint_db import VoiceprintDB


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


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
    """Open-set identification against the voiceprint DB.

    Score = mean of the cosine to a person's centroid and to their best-k exemplars.
    Raw speaker-embedding cosines are inflated by shared channel/room characteristics
    (different people in one recording can score ~0.8), so a decision needs both an
    absolute threshold and a ``margin`` over the best competing candidate.
    """

    def __init__(self, db: VoiceprintDB, model_id: str, high: float = 0.75, low: float = 0.60,
                 margin: float = 0.10, top_k: int = 3, channel_type: str | None = None) -> None:
        if low > high:
            raise ValueError("low threshold must not exceed high threshold")
        self.db, self.model_id = db, model_id
        self.high, self.low, self.margin, self.top_k = high, low, margin, top_k
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

    @property
    def empty(self) -> bool:
        return not self._exemplars

    def score_all(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        e = l2norm(np.asarray(embedding, dtype=np.float32))
        scores = []
        for pid, ex in self._exemplars.items():
            sims = np.sort(ex @ e)[::-1][: self.top_k]
            scores.append((pid, 0.5 * float(self._centroids[pid] @ e) + 0.5 * float(sims.mean())))
        return sorted(scores, key=lambda s: s[1], reverse=True)

    def _decide(self, pid: int, score: float, competitor: float) -> Match:
        if score >= self.high and score - competitor >= self.margin:
            return Match(pid, self._names.get(pid), score, Decision.KNOWN)
        if score >= self.low:
            return Match(pid, self._names.get(pid), score, Decision.SUGGEST)
        return Match(None, None, score, Decision.UNKNOWN)

    def identify(self, embedding: np.ndarray) -> Match:
        scores = self.score_all(embedding)
        if not scores:
            return Match(None, None, 0.0, Decision.UNKNOWN)
        pid, score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else -1.0
        return self._decide(pid, score, runner_up)

    def identify_session(self, speakers: dict[str, list[np.ndarray]]) -> dict[str, Match]:
        """Identify all speakers of a session jointly, one person per speaker.

        Each speaker is scored by the centroid of its exemplars. Pairs are assigned
        greedily by score; a pair's margin is measured against the best alternative for
        both the speaker (other persons) and the person (other speakers).
        """
        unknown = Match(None, None, 0.0, Decision.UNKNOWN)
        if not speakers or self.empty:
            return dict.fromkeys(speakers, unknown)
        sids = list(speakers)
        pids = list(self._exemplars)
        table = np.full((len(sids), len(pids)), -1.0)
        for i, sid in enumerate(sids):
            if speakers[sid]:
                centroid = l2norm(np.mean([l2norm(e) for e in speakers[sid]], axis=0))
                row = dict(self.score_all(centroid))
                table[i] = [row[p] for p in pids]
        result = dict.fromkeys(sids, unknown)
        free_s, free_p = set(range(len(sids))), set(range(len(pids)))
        for flat in np.argsort(-table, axis=None):
            i, j = divmod(int(flat), len(pids))
            if i not in free_s or j not in free_p or table[i, j] < 0:
                continue
            others = [table[i, k] for k in range(len(pids)) if k != j]
            others += [table[k, j] for k in range(len(sids)) if k != i]
            result[sids[i]] = self._decide(pids[j], float(table[i, j]),
                                           max(others, default=-1.0))
            free_s.discard(i)
            free_p.discard(j)
        return result
