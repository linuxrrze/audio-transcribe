"""Silero VAD (the ONNX copy bundled with faster-whisper)."""

from __future__ import annotations

import numpy as np

from transcribe.audio.base import SAMPLE_RATE


def speech_regions(audio: np.ndarray, min_silence_ms: int = 500,
                   pad_ms: int = 200) -> list[tuple[float, float]]:
    """Return speech regions as (t0, t1) seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms)
    return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in get_speech_timestamps(audio, opts)]
