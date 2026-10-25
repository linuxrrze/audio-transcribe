"""Second diarization pass on transcript phrases.

The diarizer merges turns of different speakers when the pause between them is short
(fast dialogue, radio plays), mixes voices within clusters, and over-counts speakers
(55 clusters for a 4-person AMI meeting). The transcript knows where turns can change:
at sentence ends and longer pauses. So the words are cut into phrases there, each
phrase long enough is embedded, and the phrase embeddings are clustered again.

k is the known speaker count (``--speakers``); otherwise, by default, the number of
diarizer speakers with a meaningful amount of speech ("estimate"), or optionally the
k of best silhouette score up to that number ("silhouette").

Word speaker accuracy (words with exactly one reference speaker; labels after the pass):
                              diarizer only   estimate (default)   silhouette
  AMI ES2004a (4 people)      45.5 % (55)       82.1 % (15)        97.3 % (4)
  AMI IS1009a (4 people)      62.5 % (15)       83.2 % (10)        82.7 % (6)
  synthetic play (6 voices)   71.2 % (6)        98.8 % (6)         92.6 % (5)
  radio play, ~10 roles       21 speakers       13 speakers         3 speakers (!)
Silhouette is best for distinct voices (meetings) but collapsed a radio play with
similar young voices into 3 speakers. Too many speakers can be merged by the user;
merged characters cannot be separated - hence "estimate" is the default. Parameters were
tuned on these recordings; treat them as a starting point, not a guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.diarize.align import is_sentence_end
from transcribe.speakers.embedder import Embedder
from transcribe.storage.transcript_model import Word

ClusterFn = Callable[[np.ndarray, int], Sequence[int]]

PHRASE_GAP = 0.5  # phrases also end at pauses this long (shorter splits single sentences)
K_TOLERANCE = 0.02  # silhouette scores within this of the best count as a tie
MAX_SPEAKERS = 15
MIN_PHRASE = 0.6  # shorter phrases are not embedded (unreliable); they inherit a label
# Diarizer speakers count towards k only with at least this much speech, absolute and
# as a share of all speech. On a 50 min radio play the diarizer found 20 speakers: 8
# large ones, a few with 1-2 % (minor characters) and crumbs below 1 %, several of them
# a single turn - spurious clusters. The diarizer also splits real people more the longer
# the recording (~3 clusters per minute on meetings); 3 % keeps that in check (AMI cpWER
# 76 -> 58 %, docs/BENCHMARKS.md), at the cost of very quiet participants of long
# recordings sharing another's label. k is only an upper bound for the second pass.
MIN_SPEAKER_S = 3.0
MIN_SPEAKER_SHARE = 0.03
PAD = 0.05


def silhouette(x: np.ndarray, labels: Sequence[int]) -> float:
    """Mean cosine silhouette of ``labels`` for unit vectors ``x`` (vectorized)."""
    lab = np.asarray(labels)
    ids = np.unique(lab)
    if len(ids) < 2:
        return -1.0
    dist = 1.0 - x @ x.T
    onehot = (lab[:, None] == ids[None, :]).astype(np.float32)
    counts = onehot.sum(axis=0)
    mean_to = dist @ onehot  # (n, k) summed distances to each cluster
    own = np.searchsorted(ids, lab)
    n_own = counts[own] - 1
    a = np.where(n_own > 0, mean_to[np.arange(len(x)), own] / np.maximum(n_own, 1), 0.0)
    mean_to = mean_to / counts
    mean_to[np.arange(len(x)), own] = np.inf
    b = mean_to.min(axis=1)
    s = np.where(n_own > 0, (b - a) / np.maximum(np.maximum(a, b), 1e-9), 0.0)
    return float(s.mean())


def choose_k(features: np.ndarray, k_max: int, cluster: ClusterFn,
             tolerance: float = K_TOLERANCE) -> list[int]:
    """Cluster with the k in 2..k_max of best silhouette (larger k wins near-ties)."""
    x = features / np.linalg.norm(features, axis=1, keepdims=True)
    scored = []
    for k in range(2, max(2, min(k_max, len(x) // 4)) + 1):
        labels = list(cluster(features, k))
        scored.append((silhouette(x, labels), k, labels))
    best = max(score for score, _, _ in scored)
    return max((c for c in scored if c[0] >= best - tolerance), key=lambda c: c[1])[2]


def sherpa_cluster(features: np.ndarray, k: int) -> Sequence[int]:
    import sherpa_onnx

    cfg = sherpa_onnx.FastClusteringConfig(num_clusters=k)
    return sherpa_onnx.FastClustering(cfg)(np.ascontiguousarray(features, np.float32))


def phrases(words: list[Word], gap: float = PHRASE_GAP) -> list[list[Word]]:
    """Split at sentence ends and at pauses of at least ``gap`` seconds."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (w.t0 - cur[-1].t1 >= gap or is_sentence_end(cur[-1].text)):
            out.append(cur)
            cur = []
        cur.append(w)
    return out + ([cur] if cur else [])


def _majority(phrase: list[Word]) -> str | None:
    weight: dict[str, float] = defaultdict(float)
    for w in phrase:
        if w.speaker_id:
            weight[w.speaker_id] += w.t1 - w.t0
    return max(weight, key=weight.__getitem__) if weight else None


def speech_per_speaker(spans: Sequence[tuple[float, float, str | None]]) -> dict[str, float]:
    speech: dict[str, float] = defaultdict(float)
    for t0, t1, sid in spans:
        if sid:
            speech[sid] += t1 - t0
    return dict(speech)


def speaker_count(speech: dict[str, float], min_speech: float = MIN_SPEAKER_S,
                  min_share: float = MIN_SPEAKER_SHARE) -> int:
    """Speakers with a meaningful amount of speech (pass the diarizer's turn durations:
    the evidence for a speaker; word durations understate it)."""
    total = sum(speech.values())
    return max(1, sum(1 for s in speech.values() if s >= max(min_speech, min_share * total)))


def resegment(words: list[Word], audio: np.ndarray, embedder: Embedder,
              num_speakers: int | None = None, max_speakers: int | None = None,
              count: str = "estimate", cluster: ClusterFn = sherpa_cluster,
              gap: float = PHRASE_GAP, min_phrase: float = MIN_PHRASE) -> int:
    """Reassign ``speaker_id`` of ``words`` (in place). Returns #phrases relabelled.

    ``num_speakers``: exact k. Otherwise ``max_speakers`` (callers with diarizer turns
    pass ``speaker_count(speech_per_speaker(turns))``; default: estimated from the
    words) is k itself (``count="estimate"``) or the upper bound for the silhouette
    choice (``count="silhouette"``). Words keep their diarizer labels when there is too
    little material to cluster.
    """
    if count not in ("estimate", "silhouette"):
        raise ValueError(f"count must be 'estimate' or 'silhouette', not {count!r}")
    k_max = num_speakers or min(MAX_SPEAKERS, max_speakers or speaker_count(
        speech_per_speaker([(w.t0, w.t1, w.speaker_id) for w in words])))
    exact = num_speakers or (k_max if count == "estimate" else None)
    groups = [p for p in phrases(words, gap) if _majority(p)]
    long = [i for i, p in enumerate(groups) if p[-1].t1 - p[0].t0 >= min_phrase]
    if k_max < 2 or len(long) < 2 * (exact or 2):
        return 0
    vectors = []
    for i in long:
        t0, t1 = groups[i][0].t0 - PAD, groups[i][-1].t1 + PAD
        vectors.append(embedder.embed(audio[int(max(0.0, t0) * SAMPLE_RATE):
                                            int(t1 * SAMPLE_RATE)]))
    features = np.stack(vectors).astype(np.float32)
    labels = (list(cluster(features, exact)) if exact
              else choose_k(features, k_max, cluster))

    # Name clusters after the diarizer speaker they overlap most (one-to-one, by speech
    # time), so ids stay stable; leftovers get fresh ids.
    overlap: dict[tuple[int, str], float] = defaultdict(float)
    for i, lab in zip(long, labels, strict=True):
        overlap[(lab, _majority(groups[i]) or "")] += groups[i][-1].t1 - groups[i][0].t0
    names: dict[int, str] = {}
    used: set[str] = set()
    for (lab, spk), _ in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
        if lab not in names and spk not in used:
            names[lab] = spk
            used.add(spk)
    taken = {w.speaker_id for w in words if w.speaker_id} | used
    next_id = max((int(s[1:]) for s in taken if s[1:].isdigit()), default=0) + 1
    for lab in sorted(set(labels)):
        if lab not in names:
            names[lab] = f"S{next_id}"
            next_id += 1

    # Short phrases follow the nearest (in time) long phrase that the diarizer gave the
    # same speaker - local context beats a global mapping when diarizer clusters are
    # mixed. If that diarizer speaker has no long phrase at all (typically a spurious
    # mini-cluster), the nearest long phrase of any speaker decides; keeping the old
    # label would leave a phantom speaker in the transcript.
    new_label: dict[int, str] = {i: names[lab] for i, lab in zip(long, labels, strict=True)}
    by_diarizer: dict[str, list[int]] = defaultdict(list)
    for i in long:
        by_diarizer[_majority(groups[i]) or ""].append(i)
    for i, p in enumerate(groups):
        if i in new_label:
            continue
        candidates = by_diarizer[_majority(p) or ""] or long
        j = min(candidates, key=lambda k: abs(groups[k][0].t0 - p[0].t0))
        new_label[i] = new_label[j]
    changed = 0
    for i, p in enumerate(groups):
        if any(w.speaker_id != new_label[i] for w in p):
            changed += 1
        for w in p:
            w.speaker_id = new_label[i]
    return changed
