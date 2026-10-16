"""Offline diarizer plugin protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from transcribe.diarize.align import Turn

ProgressFn = Callable[[float], None]  # fraction done, 0..1


@runtime_checkable
class Diarizer(Protocol):
    name: str

    def diarize(self, audio: np.ndarray, num_speakers: int | None = None,
                progress: ProgressFn | None = None) -> list[Turn]:
        """Return speaker turns (may overlap) with arbitrary but consistent speaker ids."""
        ...
