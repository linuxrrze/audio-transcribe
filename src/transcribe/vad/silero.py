"""Silero VAD (the ONNX copy bundled with faster-whisper)."""

from __future__ import annotations

import numpy as np

from transcribe.audio.base import SAMPLE_RATE

THRESHOLD = 0.5  # speech probability (Silero's default)
# Silero's probabilities are lower on telephone-band audio, for speech and non-speech:
# at 0.5 it found only 73 % of the speech in band-limited AMI meetings.
# See docs/BENCHMARKS.md, "First results", item 4 ("VAD on telephone-band audio").
THRESHOLD_NARROWBAND = 0.25


def threshold_for(audio: np.ndarray, wideband: float = THRESHOLD,
                  narrowband: float = THRESHOLD_NARROWBAND) -> float:
    from transcribe.audio.bandwidth import is_narrowband

    return narrowband if is_narrowband(audio) else wideband


def speech_regions(audio: np.ndarray, min_silence_ms: int = 500, pad_ms: int = 200,
                   threshold: float | None = None) -> list[tuple[float, float]]:
    """Return speech regions as (t0, t1) seconds. ``threshold`` None: chosen from the
    audio (lower for telephone-band audio)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold_for(audio) if threshold is None else threshold,
                      min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms)
    return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in get_speech_timestamps(audio, opts)]
