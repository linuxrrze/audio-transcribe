import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.offline import Chunk, plan_chunks, renumber_speakers, transcribe_audio
from transcribe.diarize.align import Turn
from transcribe.storage.transcript_model import Word

# Synthetic "audio": amplitude encodes the language (0.1 = de, 0.2 = en).
LEVEL = {"de": 0.1, "en": 0.2}


def make_audio(spans, duration):
    audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
    for t0, t1, lang in spans:
        audio[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)] = LEVEL[lang]
    return audio


class FakeAsr:
    name = "fake"

    def __init__(self):
        self.detect_calls = 0
        self.calls = []

    def detect_language(self, audio, languages):
        self.detect_calls += 1
        de = float(np.mean(audio[audio > 0])) < 0.15
        return {"de": 0.99, "en": 0.01} if de else {"de": 0.01, "en": 0.99}

    def transcribe(self, audio, offset=0.0, language=None, prompt=None, progress=None):
        self.calls.append((round(offset, 2), round(offset + len(audio) / SAMPLE_RATE, 2),
                           language))
        words, t = [], 0.0
        while t + 0.5 <= len(audio) / SAMPLE_RATE:
            if audio[int((t + 0.25) * SAMPLE_RATE)] > 0:
                words.append(Word(f" {language}", offset + t, offset + t + 0.5))
            t += 0.5
        if progress:
            progress(len(audio) / SAMPLE_RATE)
        return words


class FakeDiarizer:
    name = "fake-dia"

    def __init__(self, turns):
        self.turns = turns

    def diarize(self, audio, num_speakers=None, progress=None):
        return self.turns


def test_renumber_by_first_appearance():
    turns = renumber_speakers([Turn(5, 6, "b"), Turn(0, 1, "z"), Turn(2, 3, "b")])
    assert [(t.t0, t.speaker_id) for t in turns] == [(5, "S2"), (0, "S1"), (2, "S2")]


def test_plan_chunks_cuts_in_gaps_and_merges_same_language():
    turns = [Turn(0, 2, "S1"), Turn(3, 5, "S2"), Turn(5.5, 6, "S3"), Turn(7, 9, "S1")]
    assert plan_chunks(turns, ["de", "en", "en", "de"], 10.0) == [
        Chunk(0.0, 2.5, "de"), Chunk(2.5, 6.5, "en"), Chunk(6.5, 10.0, "de")]


def test_mixed_language_pipeline():
    spans = [(0, 2, "de"), (3, 5, "en"), (6, 8, "de")]
    audio = make_audio(spans, 9.0)
    dia = FakeDiarizer([Turn(0, 2, "B"), Turn(3, 5, "A"), Turn(6, 8, "B")])
    asr = FakeAsr()
    stages = []
    tr = transcribe_audio(audio, asr, dia, ["de", "en"], progress=lambda s, f: stages.append(s))
    assert asr.detect_calls == 2  # one per speaker
    assert [c[2] for c in asr.calls] == ["de", "en", "de"]
    assert [(s.speaker_id, s.lang, s.text.split()[0]) for s in tr.segments] == [
        ("S1", "de", "de"), ("S2", "en", "en"), ("S1", "de", "de")]
    assert {k: v.label for k, v in tr.speakers.items()} == {"S1": "Speaker 1", "S2": "Speaker 2"}
    assert tr.meta["speaker_languages"] == {"S1": "de", "S2": "en"}
    assert {"diarize", "language", "transcribe"} <= set(stages)


def test_mixed_language_speaker_gets_per_turn_detection():
    # Diarization merged a German and an English voice into one speaker.
    spans = [(0, 3, "de"), (4, 7, "en"), (8, 11, "de")]
    audio = make_audio(spans, 12.0)
    dia = FakeDiarizer([Turn(0, 3, "A"), Turn(4, 7, "A"), Turn(8, 11, "A")])

    class UndecidedAsr(FakeAsr):
        def detect_language(self, audio, languages):
            voiced = audio[audio > 0]
            if np.ptp(voiced) > 0.05:  # probe mixes both languages: prior undecided
                self.detect_calls += 1
                return {"de": 0.6, "en": 0.4}
            return super().detect_language(audio, languages)

    asr = UndecidedAsr()
    tr = transcribe_audio(audio, asr, dia, ["de", "en"])
    assert [c[2] for c in asr.calls] == ["de", "en", "de"]
    assert tr.meta["per_turn_language"] == ["S1"]
    assert {s.text.split()[0] for s in tr.segments if s.lang == "en"} == {"en"}


def test_single_language_skips_detection():
    audio = make_audio([(0, 2, "en"), (3, 5, "en")], 6.0)
    asr = FakeAsr()
    transcribe_audio(audio, asr, FakeDiarizer([Turn(0, 2, "a"), Turn(3, 5, "b")]), ["de"])
    assert asr.detect_calls == 0
    assert asr.calls == [(0.0, 6.0, "de")]  # same language everywhere -> one chunk


def test_without_diarizer_detects_language_per_vad_region():
    spans = [(0, 3, "de"), (4, 7, "en"), (7.5, 8.5, "en"), (9, 12, "de")]
    audio = make_audio(spans, 13.0)
    asr = FakeAsr()
    tr = transcribe_audio(audio, asr, None, ["de", "en"],
                          vad=lambda a: [(t0, t1) for t0, t1, _ in spans])
    assert asr.detect_calls == 3  # the 1 s region inherits from its predecessor
    assert [c[2] for c in asr.calls] == ["de", "en", "de"]
    assert tr.speakers == {} and all(s.speaker_id is None for s in tr.segments)
    assert [s.lang for s in tr.segments] == ["de", "en", "de"]
