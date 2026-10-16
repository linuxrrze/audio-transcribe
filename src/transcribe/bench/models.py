"""Compare Whisper models: accuracy, speaker attribution and speed (CPU or GPU).

    python -m transcribe.bench.models [--models small medium large-v3-turbo]
        [--meetings ES2004a IS1009a] [--dialogue] [--latency]
        [--audio FILE --lang de --check-words Name1,Name2] [--json results.json]

- AMI meetings (English, real speech): WER against the manual transcript, word speaker
  accuracy (diarizer alone / with the second pass) and real-time factor of the ASR.
  Diarization runs once per meeting (it does not depend on the Whisper model).
- --dialogue: the synthetic German/English two-speaker dialogue, full pipeline:
  WER per language and speaker accuracy.
- --latency: live mode, median/max delay from end of speech to final text.
- --audio: any recording without reference (e.g. a radio play): speakers found,
  segments starting mid-sentence, and how often the --check-words appear.

WER uses lower-cased word tokens without punctuation; filler words (um, uh, hmm, ...)
are removed from reference and hypothesis because the AMI transcripts annotate them
and Whisper mostly drops them - otherwise every model would be penalised equally.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.audio.file import load_audio
from transcribe.bench.ami import DEFAULT_MEETINGS, fetch, load_rttm, reference_text, word_accuracy
from transcribe.bench.metrics import edit_distance, normalize
from transcribe.core.config import Config
from transcribe.diarize.align import Turn, assign_speakers, words_to_segments
from transcribe.diarize.resegment import resegment, speaker_count, speech_per_speaker
from transcribe.diarize.sherpa_offline import SherpaDiarizer
from transcribe.storage.transcript_model import Word

FILLERS = {"um", "uh", "hmm", "mm", "mhm", "uhm", "erm", "er", "ah", "oh", "hm", "uh-huh",
           "mm-hmm", "huh"}


def wer_without_fillers(ref: str, hyp: str) -> float:
    r = [t for t in normalize(ref) if t not in FILLERS]
    h = [t for t in normalize(hyp) if t not in FILLERS]
    return edit_distance(r, h) / max(1, len(r))


@dataclass
class Row:
    model: str
    data: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)


def _engine(model: str, config: Config):
    from transcribe.asr.faster_whisper import FasterWhisperEngine

    c = copy.deepcopy(config)
    c.asr.model = model
    return FasterWhisperEngine.from_config(c)


def _timed_transcribe(engine, audio: np.ndarray, language: str) -> tuple[list[Word], float]:
    t = time.perf_counter()
    words = engine.transcribe(audio, language=language)
    return words, time.perf_counter() - t


def bench_meeting(meeting: str, engines: dict, config: Config, embedder) -> list[Row]:
    wav, rttm = fetch(meeting)
    audio, ref_spk, ref_text = load_audio(wav), load_rttm(rttm), reference_text(meeting)
    duration = len(audio) / SAMPLE_RATE
    turns: list[Turn] = SherpaDiarizer.from_config(config).diarize(audio)
    k_max = speaker_count(speech_per_speaker([(t.t0, t.t1, t.speaker_id) for t in turns]))
    rows = []
    for model, engine in engines.items():
        words, seconds = _timed_transcribe(engine, audio, "en")
        assign_speakers(words, turns)
        base = word_accuracy(words, ref_spk)
        resegment(words, audio, embedder, max_speakers=k_max)
        rows.append(Row(model, meeting, {
            "wer": round(wer_without_fillers(ref_text, "".join(w.text for w in words)), 4),
            "speaker_acc_diarizer": round(base, 4),
            "speaker_acc_second_pass": round(word_accuracy(words, ref_spk), 4),
            "speakers": len({w.speaker_id for w in words if w.speaker_id}),
            "asr_s": round(seconds, 1), "rtf": round(seconds / duration, 3),
        }))
        print(f"  {meeting} {model}: {rows[-1].metrics}", flush=True)
    return rows


def bench_dialogue(models: list[str], config: Config) -> list[Row]:
    from transcribe.bench.dialogue import synthesize
    from transcribe.bench.metrics import evaluate
    from transcribe.core.offline import transcribe_audio

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "dialogue.wav"
        ref = synthesize(wav)
        audio = load_audio(wav)
        diarizer = SherpaDiarizer.from_config(config)
        for model in models:
            engine = _engine(model, config)
            t = time.perf_counter()
            transcript = transcribe_audio(audio, engine, diarizer, ["de", "en"])
            seconds = time.perf_counter() - t
            r = evaluate(transcript, ref)
            rows.append(Row(model, "dialogue de/en", {
                "wer_de": round(r.wer_by_language["de"], 4),
                "wer_en": round(r.wer_by_language["en"], 4),
                "speaker_acc": round(r.speaker_accuracy, 4),
                "pipeline_s": round(seconds, 1)}))
            print(f"  dialogue {model}: {rows[-1].metrics}", flush=True)
    return rows


def bench_latency(models: list[str], config: Config) -> list[Row]:
    from transcribe.bench.dialogue import synthesize
    from transcribe.bench.latency import measure

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "dialogue.wav"
        synthesize(wav)
        for model in models:
            c = copy.deepcopy(config)
            c.asr.model = model
            c.live.refine = False
            m = measure(wav, c)
            metrics: dict[str, float | int | str] = dict(m)
            rows.append(Row(model, "live latency", metrics))
            print(f"  latency {model}: {m}", flush=True)
    return rows


def bench_audio(path: Path, language: str, check_words: list[str], engines: dict,
                config: Config, embedder) -> list[Row]:
    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    turns = SherpaDiarizer.from_config(config).diarize(audio)
    k_max = speaker_count(speech_per_speaker([(t.t0, t.t1, t.speaker_id) for t in turns]))
    rows = []
    for model, engine in engines.items():
        words, seconds = _timed_transcribe(engine, audio, language)
        assign_speakers(words, turns)
        resegment(words, audio, embedder, max_speakers=k_max)
        segs = words_to_segments(words, lang=language)
        text = " ".join(s.text for s in segs)
        metrics: dict[str, float | int | str] = {
            "speakers": len({w.speaker_id for w in words if w.speaker_id}),
            "segments": len(segs),
            "midsentence_starts": round(sum(s.text[:1].islower() for s in segs) / len(segs), 4),
            "asr_s": round(seconds, 1), "rtf": round(seconds / duration, 3)}
        for word in check_words:
            metrics[f"'{word}'"] = len(re.findall(rf"\b{re.escape(word)}\b", text, re.I))
        rows.append(Row(model, path.name, metrics))
        print(f"  {path.name} {model}: {metrics}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["small", "medium", "large-v3-turbo"])
    ap.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--dialogue", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--lang", default="de")
    ap.add_argument("--check-words", default="")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from transcribe.speakers.embedder import SherpaEmbedder

    config = Config()
    embedder = SherpaEmbedder(config.speakers.embedding_model)
    rows: list[Row] = []
    if args.meetings or args.audio:
        engines = {m: _engine(m, config) for m in args.models}
        print(f"device: {next(iter(engines.values())).device}", flush=True)
        for meeting in args.meetings:
            rows += bench_meeting(meeting, engines, config, embedder)
        if args.audio:
            words = [w.strip() for w in args.check_words.split(",") if w.strip()]
            rows += bench_audio(args.audio, args.lang, words, engines, config, embedder)
        del engines
    if args.dialogue:
        rows += bench_dialogue(args.models, config)
    if args.latency:
        rows += bench_latency(args.models, config)
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=1))


if __name__ == "__main__":
    main()
