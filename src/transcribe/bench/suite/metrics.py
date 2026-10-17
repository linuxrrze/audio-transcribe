"""Standard metrics for "who said what": WER, cpWER, DER - dependency-free.

- **WER**: word errors / reference words after normalisation (lower case, no punctuation,
  filler words removed, German and English).
- **cpWER** (concatenated minimum-permutation WER, CHiME-6): each reference speaker's
  words concatenated, each hypothesis speaker's likewise; speakers matched one-to-one so
  that the total edit distance is minimal; unmatched speakers count as all deletions or
  insertions. One number for text *and* attribution.
- **DER** (diarization error rate, NIST): (missed speech + false alarm + speaker
  confusion) / reference speech, frame-based (10 ms), overlapping speech included,
  optional no-score collar around reference boundaries, optimal one-to-one speaker
  mapping.

Counts are returned (not ratios), so results add up over recordings (micro average).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FRAME = 0.01
_TOKEN = re.compile(r"[\w']+", re.UNICODE)
FILLERS = {
    # English (AMI annotates them, Whisper mostly drops them)
    "um", "uh", "hmm", "mm", "mhm", "uhm", "erm", "er", "ah", "oh", "hm", "uh-huh",
    "mm-hmm", "huh",
    # German
    "äh", "ähm", "öh", "öhm", "na",
}

Turn = tuple[float, float, str]  # (t0, t1, speaker)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in FILLERS]


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance between token sequences; numpy row updates, O(n*m) but fast
    enough for hour-long recordings (thousands of words per speaker)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    vocab: dict[str, int] = {}
    h = np.array([vocab.setdefault(t, len(vocab)) for t in hyp])
    r = [vocab.setdefault(t, len(vocab)) for t in ref]
    j = np.arange(len(h) + 1)
    prev = j.copy()
    for i, tok in enumerate(r, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        # deletion (prev[j] + 1) and substitution (prev[j-1] + cost); insertions then
        # propagate left to right: cur[j] = min_k<=j (tmp[k] + j - k)
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (h != tok))
        prev = np.minimum.accumulate(cur - j) + j
    return int(prev[-1])


@dataclass
class Counts:
    errors: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.errors + other.errors, self.total + other.total)


def wer(ref: str, hyp: str) -> Counts:
    r = tokens(ref)
    return Counts(edit_distance(r, tokens(hyp)), len(r))


def cpwer(ref: dict[str, str], hyp: dict[str, str]) -> Counts:
    """``ref``/``hyp``: speaker -> all their words in time order."""
    rt = {s: tokens(t) for s, t in ref.items()}
    ht = {s: tokens(t) for s, t in hyp.items()}
    rs, hs = list(rt), list(ht)
    n = max(len(rs), len(hs))
    cost = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a < len(rs) and b < len(hs):
                cost[a, b] = edit_distance(rt[rs[a]], ht[hs[b]])
            elif a < len(rs):
                cost[a, b] = len(rt[rs[a]])  # speaker not found: all deleted
            elif b < len(hs):
                cost[a, b] = len(ht[hs[b]])  # extra speaker: all inserted
    assignment = hungarian(cost)
    return Counts(int(sum(cost[a, b] for a, b in enumerate(assignment))),
                  sum(len(t) for t in rt.values()))


@dataclass
class DerCounts:
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    total: float = 0.0  # scored reference speech (speaker-seconds)

    @property
    def rate(self) -> float:
        return (self.missed + self.false_alarm + self.confusion) / self.total \
            if self.total else 0.0

    def __add__(self, o: DerCounts) -> DerCounts:
        return DerCounts(self.missed + o.missed, self.false_alarm + o.false_alarm,
                         self.confusion + o.confusion, self.total + o.total)


def _activity(turns: Sequence[Turn], n: int) -> tuple[np.ndarray, list[str]]:
    names = sorted({s for *_, s in turns})
    act = np.zeros((n, len(names)), bool)
    for t0, t1, s in turns:
        act[int(round(t0 / FRAME)):int(round(t1 / FRAME)), names.index(s)] = True
    return act, names


def der(ref: Sequence[Turn], hyp: Sequence[Turn], collar: float = 0.0,
        duration: float | None = None) -> DerCounts:
    """Diarization error counts (seconds). ``collar``: seconds around each reference
    boundary that are not scored (0.25 is the common choice)."""
    end = max([t1 for _, t1, _ in [*ref, *hyp]] + [duration or 0.0])
    n = int(np.ceil(end / FRAME)) + 1
    r, _ = _activity(ref, n)
    h, _ = _activity(hyp, n)
    scored = np.ones(n, bool)
    if collar > 0:
        c = int(round(collar / FRAME))
        for t0, t1, _ in ref:
            for edge in (int(round(t0 / FRAME)), int(round(t1 / FRAME))):
                scored[max(0, edge - c):edge + c] = False
    r, h = r[scored], h[scored]
    n_ref, n_hyp = r.sum(axis=1), h.sum(axis=1)
    co = r.T.astype(np.int64) @ h.astype(np.int64)  # frames both active, ref x hyp
    size = max(co.shape)
    pad = np.zeros((size, size))
    pad[:co.shape[0], :co.shape[1]] = co
    assignment = hungarian(-pad)
    mapped = np.zeros(len(r), np.int64)
    for a, b in enumerate(assignment):
        if a < co.shape[0] and b < co.shape[1]:
            mapped += r[:, a] & h[:, b]
    return DerCounts(
        missed=float(np.maximum(n_ref - n_hyp, 0).sum()) * FRAME,
        false_alarm=float(np.maximum(n_hyp - n_ref, 0).sum()) * FRAME,
        confusion=float((np.minimum(n_ref, n_hyp) - mapped).sum()) * FRAME,
        total=float(n_ref.sum()) * FRAME)


def purity(ref: Sequence[Turn], hyp: Sequence[Turn]) -> Counts:
    """Frames (with one reference speaker) whose hypothesis label's majority person is
    that speaker - extra labels are fine, mixed voices are not."""
    end = max([t1 for _, t1, _ in [*ref, *hyp]] + [0.0])
    n = int(np.ceil(end / FRAME)) + 1
    r, _ = _activity(ref, n)
    h, _ = _activity(hyp, n)
    single = (r.sum(axis=1) == 1) & (h.sum(axis=1) == 1)
    co = r[single].T.astype(np.int64) @ h[single].astype(np.int64)
    return Counts(int(co.sum() - co.max(axis=0).sum()), int(co.sum()))


def hungarian(cost: np.ndarray) -> list[int]:
    """Minimum-cost assignment for a square matrix: row i -> column result[i].
    (Kuhn-Munkres with potentials, O(n^3).)"""
    n = cost.shape[0]
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p = np.zeros(n + 1, int)  # p[j]: row matched to column j (1-based, 0 = none)
    way = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result
