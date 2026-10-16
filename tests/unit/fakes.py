"""Scripted fakes for the live pipeline.

A script is a timeline of words with languages. ``ScriptedWorld.audio()`` renders it to a
signal whose amplitude encodes the language (0 = silence), so the fake VAD and language
detector work on the real samples the code under test passes them, while the fake ASR
returns the scripted words inside the decoded window (absolute times via ``offset``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.storage.transcript_model import Word

LEVEL = {"de": 0.1, "en": 0.2, "fr": 0.3}


@dataclass(frozen=True)
class W:
    text: str
    t0: float
    t1: float
    lang: str


def utterance(text: str, start: float, lang: str, word_dur: float = 0.4) -> list[W]:
    return [W(f" {tok}", start + i * word_dur, start + (i + 1) * word_dur - 0.05, lang)
            for i, tok in enumerate(text.split())]


def fake_vad(audio: np.ndarray, merge_gap: float = 0.3) -> list[tuple[float, float]]:
    voiced = np.flatnonzero(audio > 0)
    if not len(voiced):
        return []
    regions: list[list[float]] = []
    for i in voiced[np.r_[0, np.flatnonzero(np.diff(voiced) > 1) + 1]]:
        j = i
        while j < len(audio) and audio[j] > 0:
            j += 1
        a, b = i / SAMPLE_RATE, j / SAMPLE_RATE
        if regions and a - regions[-1][1] < merge_gap:
            regions[-1][1] = b
        else:
            regions.append([a, b])
    return [(a, b) for a, b in regions]


class ScriptedWorld:
    """Fake ASR. Decoding with the wrong language mimics Whisper translating: '???'."""

    name = "scripted"

    def __init__(self, words: list[W]) -> None:
        self.words = words
        self.decode_calls: list[tuple[float, float, str | None]] = []
        self.detect_calls = 0

    def audio(self, duration: float) -> np.ndarray:
        out = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        for w in self.words:
            out[int(w.t0 * SAMPLE_RATE) : int(w.t1 * SAMPLE_RATE)] = LEVEL[w.lang]
        return out

    def detect_language(self, audio, languages):
        self.detect_calls += 1
        voiced = audio[audio > 0]
        level = float(np.median(voiced)) if len(voiced) else 0.0
        lang = min(LEVEL, key=lambda k: abs(LEVEL[k] - level))
        return {x: (0.9 if x == lang else 0.1 / max(1, len(languages) - 1)) for x in languages}

    def transcribe(self, audio, offset=0.0, language=None, prompt=None, progress=None):
        end = offset + len(audio) / SAMPLE_RATE
        self.decode_calls.append((round(offset, 2), round(end, 2), language))
        return [Word(w.text if language in (None, w.lang) else " ???", w.t0, w.t1)
                for w in self.words if w.t0 >= offset and w.t1 <= end]


class FakeEmbedder:
    """Embedding = one-hot of the language level, i.e. one 'voice' per language."""

    model_id = "fake"

    def embed(self, audio):
        voiced = audio[audio > 0]
        level = float(np.median(voiced)) if len(voiced) else 0.0
        v = np.zeros(4, dtype=np.float32)
        v[min(3, round(level * 10))] = 1.0
        return v
