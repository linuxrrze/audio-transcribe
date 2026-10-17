"""Run the pipeline over benchmark datasets and score it (see __main__.py for the CLI).

Per recording and condition the expensive steps are cached under
``<bench dir>/cache``: Whisper words per ASR setting, speaker turns per diarizer
setting. Sweeping speaker settings (thresholds, second pass, live labels) then only
re-runs the cheap steps.

"fast" pipeline (default): Whisper on the whole recording (language from the dataset, or
detected), the diarizer, words assigned to its turns, then the second pass - the
building blocks of ``core.offline.transcribe_audio`` without its per-speaker language
routing, so that ASR and speakers can be cached separately. ``--full`` runs
``transcribe_audio`` itself (cached per complete configuration).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.bench.suite import augment, metrics
from transcribe.bench.suite.data import Dataset, Recording, bench_dir
from transcribe.core.config import Config
from transcribe.storage.transcript_model import Word

ProgressFn = Callable[[str], None]


@dataclass
class Result:
    dataset: str
    recording: str
    split: str
    condition: str
    config: str  # "default" or "key=value,..." of the swept settings
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)


# --- configuration grid -----------------------------------------------------------------
def parse_grid(settings: list[str]) -> list[dict[str, str]]:
    """["speakers.diarize_threshold=0.4,0.5", "asr.model=small"] -> cartesian product."""
    axes = []
    for s in settings:
        key, _, values = s.partition("=")
        axes.append([(key.strip(), v.strip()) for v in values.split(",")])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


def configure(base: Config, overrides: dict[str, str]) -> Config:
    c = copy.deepcopy(base)
    for key, raw in overrides.items():
        section, _, name = key.partition(".")
        target = getattr(c, section)
        old = getattr(target, name)
        value: object = raw
        if isinstance(old, bool):
            value = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(old, int):
            value = int(raw)
        elif isinstance(old, float):
            value = float(raw)
        elif isinstance(old, list):
            value = [x for x in raw.split("+") if x]
        setattr(target, name, value)
    return c


def label(overrides: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides.items()) or "default"


# --- caching ----------------------------------------------------------------------------
def _key(*parts: object) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _cached(path: Path, compute: Callable[[], dict]) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    return result


def _asr_key(c: Config, language: str, vad_threshold: float | None = None) -> str:
    """``vad_threshold``: the one that applies to this audio (wideband or telephone band),
    so a telephone-band setting does not invalidate wideband results."""
    a = c.asr
    return _key("asr", a.engine, a.model, a.device, a.beam_size, a.style_prompt,
                a.vocabulary, a.corrections, a.whispercpp_dtw, language, "vad",
                vad_threshold)


def _dia_key(c: Config) -> str:
    s = c.speakers
    return _key("dia", s.segmentation_model, s.embedding_model, s.diarize_threshold)


class Engines:
    """Models loaded once per setting and reused across recordings."""

    def __init__(self) -> None:
        self._asr: dict[str, object] = {}
        self._dia: dict[str, object] = {}
        self._emb: dict[str, object] = {}

    def asr(self, c: Config):
        from transcribe.core import registry

        k = _asr_key(c, "")
        if k not in self._asr:
            self._asr[k] = registry.load("transcribe.asr", c.asr.engine).from_config(c)
        engine = copy.copy(self._asr[k])  # same loaded model, this setting's VAD thresholds
        if hasattr(engine, "vad_thresholds"):
            engine.vad_thresholds = (c.asr.vad_threshold, c.asr.vad_threshold_narrowband)
        return engine

    def diarizer(self, c: Config):
        from transcribe.diarize.sherpa_offline import SherpaDiarizer

        k = _dia_key(c)
        if k not in self._dia:
            self._dia[k] = SherpaDiarizer.from_config(c)
        return self._dia[k]

    def embedder(self, c: Config):
        from transcribe.speakers.embedder import SherpaEmbedder

        m = c.speakers.embedding_model
        if m not in self._emb:
            self._emb[m] = SherpaEmbedder.from_config(c)
        return self._emb[m]


# --- one recording ----------------------------------------------------------------------
def load_audio_for(ds: Dataset, rec: Recording, condition: str) -> np.ndarray:
    from transcribe.audio.file import load_audio

    audio = load_audio(ds.audio_path(rec))
    seed = zlib.crc32(f"{rec.dataset}/{rec.id}".encode())
    return augment.apply(audio, condition, seed)


def evaluate(ds: Dataset, rec: Recording, condition: str, c: Config, overrides: dict,
             engines: Engines, live: bool = False, full: bool = False) -> Result:
    from transcribe.diarize.align import assign_speakers, words_to_segments
    from transcribe.diarize.resegment import resegment, speaker_count, speech_per_speaker

    cache = bench_dir() / "cache" / rec.dataset / rec.id / condition.replace(":", "_")
    audio_box: list[np.ndarray] = []

    def audio() -> np.ndarray:
        if not audio_box:
            audio_box.append(load_audio_for(ds, rec, condition))
        return audio_box[0]

    m: dict[str, float] = {}
    if full:
        hyp_words, m = _full(cache, rec, audio, c, engines)
    else:
        from transcribe.vad.silero import threshold_for

        vad = threshold_for(audio(), c.asr.vad_threshold, c.asr.vad_threshold_narrowband)
        asr = _cached(cache / f"asr-{_asr_key(c, rec.language, vad)}.json", lambda: _asr(
            engines.asr(c), audio(), rec.language))
        dia = _cached(cache / f"dia-{_dia_key(c)}.json",
                      lambda: _diarize(engines.diarizer(c), audio()))
        from transcribe.diarize.align import Turn

        hyp_words = [Word(t, a, b, p) for t, a, b, p in asr["words"]]
        turns = [Turn(a, b, s) for a, b, s in dia["turns"]]
        assign_speakers(hyp_words, turns)
        m["diarizer_der"] = metrics.der(rec.ref_turns(), [(t.t0, t.t1, t.speaker_id)
                                                          for t in turns], 0.25).rate
        if c.speakers.resegment and turns:
            k = speaker_count(speech_per_speaker([(t.t0, t.t1, t.speaker_id) for t in turns]))
            resegment(hyp_words, audio(), engines.embedder(c), max_speakers=k,
                      count=c.speakers.speaker_count)
        m["rtf_asr"] = asr["seconds"] / max(rec.duration, 1e-9)
        m["rtf_diarize"] = dia["seconds"] / max(rec.duration, 1e-9)
    segs = words_to_segments(hyp_words, max_pause=1.0)
    hyp_turns = [(s.t0, s.t1, s.speaker_id or "?") for s in segs]
    _score(rec, hyp_words, hyp_turns, m)
    if live:
        live_turns = _live(rec, audio(), hyp_words, c, engines)
        d = metrics.der(rec.ref_turns(), live_turns, 0.25)
        m.update(live_der=d.rate, live_der_errors=d.missed + d.false_alarm + d.confusion,
                 live_der_total=d.total, live_speakers=len({s for *_, s in live_turns}))
        p = metrics.purity(rec.ref_turns(), live_turns)
        m.update(live_purity_errors=p.errors, live_purity_total=p.total)
    return Result(rec.dataset, rec.id, rec.split, condition, label(overrides), rec.duration,
                  {k: round(float(v), 5) for k, v in m.items()})


def _asr(asr, audio: np.ndarray, language: str) -> dict:
    t = time.perf_counter()
    words = asr.transcribe(audio, language=language or None)
    return {"words": [(w.text, w.t0, w.t1, w.confidence) for w in words],
            "seconds": time.perf_counter() - t}


def _diarize(diarizer, audio: np.ndarray) -> dict:
    t = time.perf_counter()
    turns = diarizer.diarize(audio)
    return {"turns": [(x.t0, x.t1, x.speaker_id) for x in turns],
            "seconds": time.perf_counter() - t}


def _full(cache: Path, rec: Recording, audio: Callable[[], np.ndarray], c: Config,
          engines: Engines) -> tuple[list[Word], dict[str, float]]:
    from transcribe.core.config import vocabulary
    from transcribe.core.offline import transcribe_audio

    def run() -> dict:
        t = time.perf_counter()
        tr = transcribe_audio(audio(), engines.asr(c), engines.diarizer(c),
                              [rec.language] if rec.language else c.asr.languages,
                              vocabulary=vocabulary(c))
        return {"words": [(w.text, w.t0, w.t1, w.speaker_id) for s in tr.segments
                          for w in s.words], "seconds": time.perf_counter() - t}

    key = _key(_asr_key(c, rec.language), _dia_key(c), asdict(c.speakers))
    r = _cached(cache / f"full-{key}.json", run)
    words = [Word(t, a, b, speaker_id=s) for t, a, b, s in r["words"]]
    return words, {"rtf_total": r["seconds"] / max(rec.duration, 1e-9)}


def _score(rec: Recording, words: list[Word], hyp_turns: list, m: dict[str, float]) -> None:
    by_speaker: dict[str, list[str]] = {}
    for w in words:
        by_speaker.setdefault(w.speaker_id or "?", []).append(w.text)
    if rec.has_text:
        wr = metrics.wer(rec.ref_text(), "".join(x.text for x in words))
        cp = metrics.cpwer(rec.ref_by_speaker(), {s: " ".join(t) for s, t in
                                                  by_speaker.items()})
        m.update(wer_errors=wr.errors, wer_total=wr.total, cpwer_errors=cp.errors,
                 cpwer_total=cp.total)
    ref = rec.ref_turns()
    for collar in (0.0, 0.25):
        d = metrics.der(ref, hyp_turns, collar, rec.duration)
        tag = "der" if collar else "der0"
        m.update({f"{tag}_missed": d.missed, f"{tag}_fa": d.false_alarm,
                  f"{tag}_confusion": d.confusion, f"{tag}_total": d.total})
    p = metrics.purity(ref, hyp_turns)
    m.update(purity_errors=p.errors, purity_total=p.total,
             ref_speakers=len(rec.speakers), hyp_speakers=len({s for *_, s in hyp_turns}))


def _live(rec: Recording, audio: np.ndarray, words: list[Word], c: Config,
          engines: Engines) -> list[tuple[float, float, str]]:
    """Live labels as live mode would show them (utterances, sentence pieces, tracker)."""
    from transcribe.bench.live_speakers import utterances
    from transcribe.live.session import MIN_EMBED_S, sentence_pieces
    from transcribe.speakers.live_tracker import BandwidthEstimator, LiveSpeakerTracker
    from transcribe.storage.transcript_model import Segment

    tracker = LiveSpeakerTracker(c.speakers.live_threshold,
                                 narrowband_threshold=c.speakers.live_threshold_narrowband)
    bandwidth = BandwidthEstimator()
    plain = [Word(w.text, w.t0, w.t1, w.confidence) for w in words]
    out: list[tuple[float, float, str]] = []
    for a, b in utterances(audio):
        inside = [w for w in plain if a <= (w.t0 + w.t1) / 2 < b]
        if not inside:
            continue
        seg = Segment(a, b, inside, id=f"u{len(out)}")
        for piece, clip in sentence_pieces(seg, audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]):
            emb = engines.embedder(c).embed(clip) if len(clip) >= MIN_EMBED_S * SAMPLE_RATE \
                else None
            speaker, _ = tracker.assign(piece.id, emb, piece.t1 - piece.t0,
                                        bandwidth.update(clip))
            out.append((piece.t0, piece.t1, speaker))
    return out


# --- many recordings --------------------------------------------------------------------
def run(datasets: list[str], split: str, conditions: list[str], grid: list[dict[str, str]],
        base: Config, out: Path, limit: int | None = None, live: bool = False,
        full: bool = False, progress: ProgressFn = print) -> list[Result]:
    """Evaluate every recording x condition x setting; results appended to
    ``out/results.jsonl`` (already present combinations are skipped: resumable)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.jsonl"
    done = {(r.dataset, r.recording, r.condition, r.config) for r in load_results(path)}
    engines = Engines()
    results = []
    for name in datasets:
        ds = Dataset(name)
        if not ds.exists():
            raise FileNotFoundError(f"Dataset {name!r} not built; see `build --help`")
        recs = list(ds.recordings(split))[:limit]
        for rec, condition, overrides in itertools.product(recs, conditions, grid):
            key = (rec.dataset, rec.id, condition, label(overrides))
            if key in done:
                continue
            progress(f"{name}/{rec.id} [{condition}] {label(overrides)}")
            r = evaluate(ds, rec, condition, configure(base, overrides), overrides, engines,
                         live, full)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(r)) + "\n")
            results.append(r)
    return results


def load_results(path: Path) -> list[Result]:
    if not path.exists():
        return []
    return [Result(**json.loads(x)) for x in path.read_text().splitlines() if x.strip()]


# --- summary ----------------------------------------------------------------------------
RATES = {  # summary metric -> (error fields, total field); lower is better
    "WER": (["wer_errors"], "wer_total"),
    "cpWER": (["cpwer_errors"], "cpwer_total"),
    "DER": (["der_missed", "der_fa", "der_confusion"], "der_total"),
    "DER (no collar)": (["der0_missed", "der0_fa", "der0_confusion"], "der0_total"),
    "impurity": (["purity_errors"], "purity_total"),
    "live DER": (["live_der_errors"], "live_der_total"),
    "live impurity": (["live_purity_errors"], "live_purity_total"),
}


Row = dict[str, Any]


def summarize(results: Iterable[Result], bootstrap: int = 1000, seed: int = 0) -> list[Row]:
    """One row per dataset x condition x setting: micro-averaged rates with 95 %
    bootstrap confidence intervals over recordings, speaker-count error, speed."""
    groups: dict[tuple[str, str, str], list[Result]] = {}
    for r in results:
        groups.setdefault((r.dataset, r.condition, r.config), []).append(r)
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, condition, config), rs in sorted(groups.items()):
        row: Row = {"dataset": dataset, "condition": condition,
                                  "config": config, "recordings": len(rs),
                                  "hours": round(sum(r.duration_s for r in rs) / 3600, 2)}
        for name, (errs, tot) in RATES.items():
            e = np.array([sum(r.metrics.get(x, 0.0) for x in errs) for r in rs])
            t = np.array([r.metrics.get(tot, 0.0) for r in rs])
            if t.sum() <= 0:
                continue
            row[name] = round(float(e.sum() / t.sum()), 4)
            if len(rs) > 1 and bootstrap:
                idx = rng.integers(0, len(rs), (bootstrap, len(rs)))
                boot = e[idx].sum(axis=1) / np.maximum(t[idx].sum(axis=1), 1e-9)
                row[f"{name} 95%"] = [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]
        counts = [(r.metrics["hyp_speakers"], r.metrics["ref_speakers"]) for r in rs
                  if "hyp_speakers" in r.metrics]
        if counts:
            row["speaker count error"] = round(float(np.mean([abs(h - r) for h, r in counts])), 2)
        for k in ("rtf_asr", "rtf_diarize", "rtf_total"):
            vals = [r.metrics[k] for r in rs if k in r.metrics]
            if vals:
                row[k] = round(float(np.mean(vals)), 3)
        rows.append(row)
    return rows


def markdown(rows: list[Row]) -> str:
    cols = ["WER", "cpWER", "DER", "DER (no collar)", "impurity", "live DER", "live impurity"]
    used = [c for c in cols if any(c in r for r in rows)]
    head = ["dataset", "condition", "config", "n", "h", *used, "spk err", "RTF"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["dataset"]), str(r["condition"]), str(r["config"]),
                 str(r["recordings"]), str(r["hours"])]
        for c in used:
            if c not in r:
                cells.append("-")
                continue
            ci = r.get(f"{c} 95%")
            cells.append(f"{100 * float(r[c]):.1f}" + (
                f" ({100 * ci[0]:.1f}-{100 * ci[1]:.1f})" if isinstance(ci, list) else ""))
        cells.append(str(r.get("speaker count error", "-")))
        rtf = [r.get(k) for k in ("rtf_total", "rtf_asr") if r.get(k) is not None]
        cells.append(f"{rtf[0]:.2f}" if rtf else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nRates in %, in brackets the 95 % confidence interval " \
        "over recordings; lower is better. DER with 0.25 s collar; impurity = share of " \
        "speech under a label that mostly belongs to someone else.\n"


def compare(rows: list[Row], baseline: list[Row], tolerance: float) -> list[str]:
    """Regressions: metrics more than ``tolerance`` (absolute) worse than the baseline."""
    base = {(b["dataset"], b["condition"], b["config"]): b for b in baseline}
    problems = []
    for r in rows:
        b = base.get((r["dataset"], r["condition"], r["config"]))
        if b is None:
            continue
        for name in RATES:
            if name in r and name in b and float(r[name]) > float(b[name]) + tolerance:
                problems.append(f"{r['dataset']} [{r['condition']}] {r['config']}: {name} "
                                f"{100 * float(r[name]):.1f} % (baseline "
                                f"{100 * float(b[name]):.1f} %)")
    return problems
