"""Telephone-band detection: does audio lack everything above ~3.4 kHz?

Ratio of energy above 4 kHz to the speech band (300-3400 Hz): ~1e-7 on a phone line,
~1e-2 for wideband speech - also with noise, reverb, MP3 or Opus (benchmark suite).
Single short pieces can come close (1e-4), so decisions use the median of several.
Narrowband audio needs other thresholds: the VAD finds less speech, and voices sound
more alike to the speaker embedding.
"""

from __future__ import annotations

import numpy as np

from transcribe.audio.base import SAMPLE_RATE

CUTOFF = 1e-4
_MIN_LEVEL = 1e-3  # windows quieter than this (RMS) are silence: no evidence


def high_band_ratio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float | None:
    """Energy above 4 kHz relative to 300-3400 Hz, or None without enough signal."""
    if len(audio) < sample_rate // 10 or float(np.sqrt(np.mean(audio ** 2))) < _MIN_LEVEL:
        return None
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    f = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    speech = float(spectrum[(f > 300) & (f < 3400)].sum())
    return float(spectrum[(f > 4000) & (f < 7500)].sum()) / speech if speech > 0 else None


def is_narrowband(audio: np.ndarray, window: float = 1.0) -> bool:
    """Median decision over ``window``-second pieces of a recording or buffer."""
    n = int(window * SAMPLE_RATE)
    ratios = [r for i in range(0, max(1, len(audio) - n // 2), n)
              if (r := high_band_ratio(audio[i:i + n])) is not None]
    return bool(ratios) and float(np.median(ratios)) < CUTOFF
