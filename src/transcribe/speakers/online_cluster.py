"""Incremental speaker clustering for the live (provisional) diarization tier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class Cluster:
    id: str
    centroid: np.ndarray
    count: int = 1
    members: list[int] = field(default_factory=list)  # indices of assigned embeddings


class OnlineClusterer:
    """Assign each embedding to the closest centroid by cosine similarity, or open a new one.

    Centroids are updated as running means (with optional cap to adapt over time).
    ``merge()`` folds clusters whose centroids have drifted together.
    """

    def __init__(self, threshold: float = 0.55, max_weight: int = 50, prefix: str = "S") -> None:
        self.threshold = threshold
        self.max_weight = max_weight
        self.prefix = prefix
        self.clusters: list[Cluster] = []
        self._seen = 0
        self._next_id = 1

    def add(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return (cluster_id, similarity). Similarity is 1.0 for a newly created cluster."""
        e = l2norm(np.asarray(embedding, dtype=np.float32))
        idx = self._seen
        self._seen += 1
        best, sim = self._best(e)
        if best is not None and sim >= self.threshold:
            w = min(best.count, self.max_weight)
            best.centroid = l2norm(best.centroid * w + e)
            best.count += 1
            best.members.append(idx)
            return best.id, sim
        cluster = Cluster(id=f"{self.prefix}{self._next_id}", centroid=e, members=[idx])
        self._next_id += 1
        self.clusters.append(cluster)
        return cluster.id, 1.0

    def merge(self) -> dict[str, str]:
        """Merge clusters with centroid similarity >= threshold. Returns {old_id: new_id}."""
        remap: dict[str, str] = {}
        merged = True
        while merged:
            merged = False
            for i, a in enumerate(self.clusters):
                for b in self.clusters[i + 1 :]:
                    if float(a.centroid @ b.centroid) >= self.threshold:
                        keep, drop = (a, b) if a.count >= b.count else (b, a)
                        keep.centroid = l2norm(keep.centroid * keep.count
                                               + drop.centroid * drop.count)
                        keep.count += drop.count
                        keep.members.extend(drop.members)
                        self.clusters.remove(drop)
                        remap = {k: (keep.id if v == drop.id else v) for k, v in remap.items()}
                        remap[drop.id] = keep.id
                        merged = True
                        break
                if merged:
                    break
        return remap

    def _best(self, e: np.ndarray) -> tuple[Cluster | None, float]:
        if not self.clusters:
            return None, -1.0
        sims = np.array([c.centroid @ e for c in self.clusters])
        i = int(np.argmax(sims))
        return self.clusters[i], float(sims[i])
