"""Offline (file) pipeline: diarize -> per-speaker language prior -> chunked ASR -> align.

Whisper picks one language per call and cannot be restricted to a language subset. We
therefore detect a language prior per speaker (restricted to the configured languages),
group consecutive turns with the same language into chunks and transcribe each chunk with
its language pinned. A monolingual recording becomes a single chunk.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from transcribe.asr.base import AsrEngine
from transcribe.audio.base import SAMPLE_RATE
from transcribe.diarize.align import Turn, assign_speakers, words_to_segments
from transcribe.diarize.base import Diarizer
from transcribe.storage.transcript_model import Speaker, Transcript, Word

StageProgress = Callable[[str, float], None]  # (stage, fraction 0..1)
VadFn = Callable[[np.ndarray], list[tuple[float, float]]]

LANG_PROBE_SECONDS = 30.0  # audio per speaker used for language detection
MIN_REGION_FOR_LID = 2.0  # shorter VAD regions inherit a neighbour's language


@dataclass(frozen=True)
class Chunk:
    t0: float
    t1: float
    language: str | None


def renumber_speakers(turns: list[Turn]) -> list[Turn]:
    """Map arbitrary speaker ids to S1, S2, ... in order of first appearance."""
    mapping: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t.t0):
        mapping.setdefault(t.speaker_id, f"S{len(mapping) + 1}")
    return [Turn(t.t0, t.t1, mapping[t.speaker_id]) for t in turns]


def speaker_probe(audio: np.ndarray, turns: list[Turn], seconds: float) -> np.ndarray:
    """Concatenate a speaker's longest turns (up to ``seconds``) for language detection."""
    parts, total = [], 0.0
    for t in sorted(turns, key=lambda t: t.t1 - t.t0, reverse=True):
        take = min(t.t1 - t.t0, seconds - total)
        if take <= 0:
            break
        a = int(t.t0 * SAMPLE_RATE)
        parts.append(audio[a : a + int(take * SAMPLE_RATE)])
        total += take
    return np.concatenate(parts) if parts else audio[: int(seconds * SAMPLE_RATE)]


def plan_chunks(turns: list[Turn], speaker_lang: dict[str, str], duration: float) -> list[Chunk]:
    """Split the timeline at language changes; boundaries fall in the gap between turns."""
    turns = sorted(turns, key=lambda t: t.t0)
    if not turns:
        return [Chunk(0.0, duration, None)]
    chunks: list[Chunk] = []
    start, lang, prev_end = 0.0, speaker_lang[turns[0].speaker_id], turns[0].t1
    for t in turns[1:]:
        tl = speaker_lang[t.speaker_id]
        if tl != lang:
            cut = (prev_end + t.t0) / 2 if t.t0 > prev_end else t.t0
            chunks.append(Chunk(start, cut, lang))
            start, lang = cut, tl
        prev_end = max(prev_end, t.t1)
    chunks.append(Chunk(start, duration, lang))
    return chunks


def region_languages(audio: np.ndarray, regions: list[tuple[float, float]], asr: AsrEngine,
                     languages: list[str]) -> list[str]:
    """Detect a language per VAD region; short regions inherit from the previous one."""
    detected: list[str | None] = []
    for t0, t1 in regions:
        if t1 - t0 < MIN_REGION_FOR_LID:
            detected.append(None)
            continue
        probs = asr.detect_language(audio[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)],
                                    languages)
        detected.append(max(probs, key=probs.__getitem__))
    fallback = next((x for x in detected if x), None)
    if fallback is None:  # only short regions: detect on the whole recording
        probs = asr.detect_language(audio[: int(LANG_PROBE_SECONDS * SAMPLE_RATE)], languages)
        fallback = max(probs, key=probs.__getitem__)
    out, prev = [], fallback
    for lang in detected:
        prev = lang or prev
        out.append(prev)
    return out


def transcribe_audio(
    audio: np.ndarray,
    asr: AsrEngine,
    diarizer: Diarizer | None,
    languages: list[str],
    num_speakers: int | None = None,
    progress: StageProgress | None = None,
    vad: VadFn | None = None,
) -> Transcript:
    duration = len(audio) / SAMPLE_RATE
    report = progress or (lambda stage, frac: None)
    timings: dict[str, float] = {}

    turns: list[Turn] = []
    if diarizer is not None:
        t = time.perf_counter()
        turns = renumber_speakers(
            diarizer.diarize(audio, num_speakers, lambda f: report("diarize", f)))
        timings["diarize_s"] = time.perf_counter() - t
    report("diarize", 1.0)

    t = time.perf_counter()
    by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        by_speaker[turn.speaker_id].append(turn)
    speaker_lang: dict[str, str] = {}
    lang_probs: dict[str, dict[str, float]] = {}
    chunks: list[Chunk]
    if len(languages) == 1:
        speaker_lang = dict.fromkeys(by_speaker, languages[0])
        chunks = (plan_chunks(turns, speaker_lang, duration) if turns
                  else [Chunk(0.0, duration, languages[0])])
    elif turns:
        for spk, spk_turns in by_speaker.items():
            probs = asr.detect_language(speaker_probe(audio, spk_turns, LANG_PROBE_SECONDS),
                                        languages)
            lang_probs[spk] = probs
            speaker_lang[spk] = max(probs, key=probs.__getitem__)
        chunks = plan_chunks(turns, speaker_lang, duration)
    else:
        # No speakers to anchor a language prior: use VAD regions as language units.
        if vad is None:
            from transcribe.vad.silero import speech_regions

            vad = speech_regions
        regions = vad(audio) or [(0.0, duration)]
        langs = region_languages(audio, regions, asr, languages)
        pseudo = [Turn(t0, t1, f"R{i}") for i, (t0, t1) in enumerate(regions)]
        region_lang = {t.speaker_id: lang for t, lang in zip(pseudo, langs, strict=True)}
        chunks = plan_chunks(pseudo, region_lang, duration)
    timings["language_s"] = time.perf_counter() - t
    report("language", 1.0)

    t = time.perf_counter()
    words: list[Word] = []
    chunk_langs: list[tuple[float, float, str | None]] = []
    for chunk in chunks:
        a, b = int(chunk.t0 * SAMPLE_RATE), int(chunk.t1 * SAMPLE_RATE)

        def chunk_progress(seconds: float, start: float = chunk.t0) -> None:
            report("transcribe", min(1.0, (start + seconds) / duration))

        chunk_words = asr.transcribe(audio[a:b], offset=chunk.t0, language=chunk.language,
                                     progress=chunk_progress)
        words.extend(chunk_words)
        chunk_langs.append((chunk.t0, chunk.t1, chunk.language))
    timings["transcribe_s"] = time.perf_counter() - t
    report("transcribe", 1.0)

    if turns:
        assign_speakers(words, turns)
    segments = []
    for t0, t1, lang in chunk_langs:
        in_chunk = [w for w in words if t0 <= w.t0 < t1]
        segments.extend(words_to_segments(in_chunk, lang=lang))

    speakers = {
        spk: Speaker(id=spk, label=f"Speaker {spk[1:]}")
        for spk in sorted(by_speaker, key=lambda s: int(s[1:]))
    }
    meta: dict[str, object] = {
        "duration_s": round(duration, 3),
        "languages": languages,
        "speaker_languages": speaker_lang,
        "speaker_language_probs": {s: {k: round(v, 4) for k, v in p.items()}
                                   for s, p in lang_probs.items()},
        "asr_engine": asr.name,
        "diarizer": diarizer.name if diarizer else None,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    return Transcript(segments=segments, speakers=speakers, meta=meta)


def load_engines(config, diarize: bool) -> tuple[AsrEngine, Diarizer | None]:
    """Instantiate the configured ASR engine and diarizer in parallel (both load models)."""
    from transcribe.core import registry

    asr_cls = registry.load("transcribe.asr", config.asr.engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        asr_f = pool.submit(asr_cls.from_config, config)
        dia_f = None
        if diarize:
            dia_cls = registry.load("transcribe.diarizers", config.speakers.diarizer)
            dia_f = pool.submit(dia_cls.from_config, config)
        return asr_f.result(), dia_f.result() if dia_f else None
