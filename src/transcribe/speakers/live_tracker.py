"""Live speaker labels: online assignment of speech pieces to speaker centroids.

Each finished piece (a sentence of an utterance, see live/session.py) is embedded and
assigned to the most similar speaker if the cosine similarity to that speaker's centroid
reaches ``threshold``; otherwise it starts a new speaker. Pieces shorter than
``min_new`` seconds never start a speaker - their embeddings are unreliable - they join
the most similar one (or the previous speaker). Labels are final when shown: unlike
re-clustering everything after each piece, nothing is relabelled later.

Measured with ``python -m transcribe.bench.live_speakers`` (docs/BENCHMARKS.md): with
TitaNet-small embeddings and sentence pieces, 81-91 % of speech labelled correctly vs
29-73 % for per-utterance re-clustering. The end-of-session refinement pass (full offline
diarization) remains authoritative.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.4  # cosine similarity; tuned on AMI + synthetic radio play
MAX_WEIGHT_S = 3.0  # one piece counts at most this many seconds in a centroid


class LiveSpeakerTracker:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, num_speakers: int | None = None,
                 min_new: float = 1.0) -> None:
        self.threshold = threshold
        self.num_speakers = num_speakers
        self.min_new = min_new
        # duration-weighted sums of unit vectors; None until a speaker's first embedding
        self._centroids: dict[str, np.ndarray | None] = {}
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self.last_speaker: str | None = None

    @property
    def speakers(self) -> list[str]:
        return list(self._centroids)

    def embeddings_by_speaker(self) -> dict[str, list[np.ndarray]]:
        return {sid: list(v) for sid, v in self._embeddings.items()}

    def assign(self, key: str, embedding: np.ndarray | None,
               duration: float) -> tuple[str, dict[str, str]]:
        """Label piece ``key``. Returns (speaker, {}) - earlier labels never change (the
        second value keeps the interface of relabelling trackers)."""
        if embedding is None or not len(embedding):
            return self._previous(), {}
        v = embedding / (np.linalg.norm(embedding) or 1.0)
        best, sim = self._nearest(v)
        full = self.num_speakers is not None and len(self._centroids) >= self.num_speakers
        if best is not None and (sim >= self.threshold or duration < self.min_new or full):
            speaker = best
        elif best is None:  # first voice: a speaker without embedding yet takes it
            speaker = next((k for k, c in self._centroids.items() if c is None), None) \
                or self._new_id()
        else:
            speaker = self._new_id()
        weight = min(duration, MAX_WEIGHT_S)
        old = self._centroids.get(speaker)
        self._centroids[speaker] = weight * v if old is None else old + weight * v
        self._embeddings.setdefault(speaker, []).append(v)
        self.last_speaker = speaker
        return speaker, {}

    def _nearest(self, v: np.ndarray) -> tuple[str | None, float]:
        best, sim = None, -1.0
        for sid, c in self._centroids.items():
            if c is None:
                continue
            s = float(c @ v) / (float(np.linalg.norm(c)) or 1.0)
            if s > sim:
                best, sim = sid, s
        return best, sim

    def _previous(self) -> str:
        if self.last_speaker is None:
            self.last_speaker = self._new_id()
            self._centroids[self.last_speaker] = None
        return self.last_speaker

    def _new_id(self) -> str:
        n = 1
        while f"S{n}" in self._centroids:
            n += 1
        return f"S{n}"
