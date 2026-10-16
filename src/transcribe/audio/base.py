"""Audio source plugin protocol.

Every source delivers 16 kHz mono float32 frames tagged with a channel id. Composite
sources (mixer, future meeting integrations) emit several channels and may attach a
speaker hint when the channel is known to carry a single, named participant.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray  # float32, mono, SAMPLE_RATE
    t_start: float  # session time in seconds
    channel_id: str = "default"
    speaker_hint: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@dataclass(frozen=True)
class SourceInfo:
    name: str
    channels: list[str] = field(default_factory=lambda: ["default"])
    single_speaker_channels: bool = False  # True: diarizer may skip clustering per channel
    realtime: bool = True


@runtime_checkable
class AudioSource(Protocol):
    def describe(self) -> SourceInfo: ...
    def open(self) -> None: ...
    def frames(self) -> Iterator[AudioFrame]: ...
    def close(self) -> None: ...
