"""Telephone-band detection and the speech-detection threshold that depends on it."""

import numpy as np

from transcribe.audio.bandwidth import high_band_ratio, is_narrowband
from transcribe.bench.suite.augment import phone
from transcribe.vad import silero


def _speechlike(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)


def test_narrowband_detection():
    wide = _speechlike()
    narrow = phone(wide, np.random.default_rng(1))
    assert high_band_ratio(wide) > 1e-2 and high_band_ratio(narrow) < 1e-4
    assert not is_narrowband(wide) and is_narrowband(narrow)
    assert not is_narrowband(np.zeros(32000, np.float32))  # silence: no evidence
    assert high_band_ratio(wide[:100]) is None  # too short


def test_vad_threshold_follows_the_band(monkeypatch):
    wide = _speechlike()
    narrow = phone(wide, np.random.default_rng(1))
    assert silero.threshold_for(wide) == silero.THRESHOLD
    assert silero.threshold_for(narrow) == silero.THRESHOLD_NARROWBAND
    assert silero.threshold_for(narrow, 0.6, 0.3) == 0.3
    seen = []
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps",
                        lambda audio, opts: seen.append(opts.threshold) or [])
    silero.speech_regions(narrow)
    silero.speech_regions(wide, threshold=0.42)
    assert seen == [silero.THRESHOLD_NARROWBAND, 0.42]
