"""ASR engine plugin protocol.

Language policy (which languages are allowed, per-speaker priors) lives in the pipeline;
engines only score languages and transcribe with a given language.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from transcribe.storage.transcript_model import Word

ProgressFn = Callable[[float], None]  # seconds of audio processed


@runtime_checkable
class AsrEngine(Protocol):
    name: str

    def detect_language(self, audio: np.ndarray, languages: Sequence[str]) -> dict[str, float]:
        """Probabilities over ``languages`` (renormalised to sum to 1)."""
        ...

    def transcribe(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
        language: str | None = None,
        prompt: str | None = None,
        progress: ProgressFn | None = None,
    ) -> list[Word]:
        """Return words with absolute timestamps (audio start = ``offset``)."""
        ...
