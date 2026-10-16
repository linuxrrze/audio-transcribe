"""LocalAgreement-n streaming policy (cf. Macháček et al., "Turning Whisper into
Real-Time Transcription System", 2023).

The audio window is re-decoded repeatedly; a word is committed once ``n`` consecutive
hypotheses agree on it. Committed words are never revised.
"""

from __future__ import annotations

import re

from transcribe.storage.transcript_model import Word

_NORMALISE = re.compile(r"[^\w]+", re.UNICODE)


def _norm(text: str) -> str:
    return _NORMALISE.sub("", text.lower())


class LocalAgreement:
    def __init__(self, n: int = 2, time_tolerance: float = 0.3) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.time_tolerance = time_tolerance
        self._history: list[list[Word]] = []
        self.committed: list[Word] = []

    @property
    def committed_until(self) -> float:
        return self.committed[-1].t1 if self.committed else 0.0

    def insert(self, hypothesis: list[Word]) -> tuple[list[Word], list[Word]]:
        """Feed a new hypothesis (absolute timestamps).

        Returns ``(newly_committed, pending)`` where ``pending`` is the uncommitted tail of
        the latest hypothesis, suitable for display as partial text.
        """
        fresh = self._strip_committed(hypothesis)
        self._history.append(fresh)
        self._history = self._history[-self.n :]

        newly: list[Word] = []
        if len(self._history) == self.n:
            for words in zip(*self._history, strict=False):
                if len({_norm(w.text) for w in words}) != 1:
                    break
                newly.append(words[-1])
        self.committed.extend(newly)
        k = len(newly)
        self._history = [h[k:] for h in self._history]
        return newly, self._history[-1]

    def flush(self) -> list[Word]:
        """Commit the remaining tail of the latest hypothesis (end of utterance/stream)."""
        tail = self._history[-1] if self._history else []
        self.committed.extend(tail)
        self._history = []
        return tail

    def _strip_committed(self, hypothesis: list[Word]) -> list[Word]:
        """Drop words already committed (re-decoded audio overlaps committed text)."""
        horizon = self.committed_until - self.time_tolerance
        words = [w for w in hypothesis if w.t1 > horizon]
        # Remove an n-gram overlap between committed tail and hypothesis head.
        tail = [_norm(w.text) for w in self.committed[-5:]]
        for size in range(min(len(tail), len(words)), 0, -1):
            if tail[-size:] == [_norm(w.text) for w in words[:size]]:
                return words[size:]
        return words
