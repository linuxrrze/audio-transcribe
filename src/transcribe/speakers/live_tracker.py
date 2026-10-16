"""Live (provisional) speaker labels.

Every finalized utterance is embedded; all embeddings so far are re-clustered after each
new one, and the new clustering is mapped onto the previous labels so that ids stay
stable. When re-clustering moves earlier utterances to another speaker, the tracker
reports them so frontends can relabel. Labels are provisional: the end-of-session
refinement pass (full offline diarization) is authoritative.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

ClusterFn = Callable[[np.ndarray, int | None], Sequence[int]]


def sherpa_cluster(threshold: float) -> ClusterFn:
    import sherpa_onnx

    def cluster(features: np.ndarray, num_clusters: int | None) -> Sequence[int]:
        cfg = sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters or -1,
                                               threshold=threshold)
        return sherpa_onnx.FastClustering(cfg)(np.ascontiguousarray(features, np.float32))

    return cluster


@dataclass
class _Item:
    key: str
    embedding: np.ndarray
    speaker: str


class LiveSpeakerTracker:
    def __init__(self, cluster: ClusterFn, num_speakers: int | None = None,
                 min_duration: float = 1.0) -> None:
        self._cluster = cluster
        self.num_speakers = num_speakers
        self.min_duration = min_duration
        self._items: list[_Item] = []
        self._next_id = 1
        self.last_speaker: str | None = None

    @property
    def speakers(self) -> list[str]:
        return sorted({i.speaker for i in self._items}, key=lambda s: int(s[1:]))

    def embeddings_by_speaker(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for item in self._items:
            out.setdefault(item.speaker, []).append(item.embedding)
        return out

    def assign(self, key: str, embedding: np.ndarray | None,
               duration: float) -> tuple[str, dict[str, str]]:
        """Label utterance ``key``. Returns (speaker, {earlier_key: new_speaker} changes).

        Utterances shorter than ``min_duration`` (or without embedding) inherit the
        previous speaker; their embeddings are too unreliable to cluster.
        """
        if embedding is None or duration < self.min_duration:
            if self.last_speaker is None:
                self.last_speaker = self._new_id()
            return self.last_speaker, {}
        v = embedding / (np.linalg.norm(embedding) or 1.0)
        self._items.append(_Item(key, v, ""))
        changes = self._recluster()
        speaker = self._items[-1].speaker
        changes.pop(key, None)
        self.last_speaker = speaker
        return speaker, changes

    def _recluster(self) -> dict[str, str]:
        if len(self._items) == 1:
            self._items[0].speaker = self.last_speaker or self._new_id()
            return {}
        k = min(self.num_speakers, len(self._items)) if self.num_speakers else None
        labels = list(self._cluster(np.stack([i.embedding for i in self._items]), k))
        mapping = self._stable_mapping(labels)
        changes = {}
        for item, label in zip(self._items, labels, strict=True):
            new = mapping[label]
            if item.speaker and item.speaker != new:
                changes[item.key] = new
            item.speaker = new
        return changes

    def _stable_mapping(self, labels: list[int]) -> dict[int, str]:
        """Map cluster labels to existing speaker ids by maximum overlap (greedy)."""
        overlap = Counter((lab, item.speaker) for lab, item in zip(labels, self._items,
                                                                    strict=True)
                          if item.speaker)
        mapping: dict[int, str] = {}
        used: set[str] = set()
        for (lab, spk), _ in overlap.most_common():
            if lab not in mapping and spk not in used:
                mapping[lab] = spk
                used.add(spk)
        for lab in sorted(set(labels)):
            if lab not in mapping:
                mapping[lab] = self._new_id()
        return mapping

    def _new_id(self) -> str:
        sid = f"S{self._next_id}"
        self._next_id += 1
        return sid
