"""Calibrate voiceprint identification thresholds on synthetic speakers.

Each speaker is enrolled from one set of sentences ("session A") and tested on different
sentences ("session B"), optionally through a degraded channel (8 kHz band-limit + noise,
roughly a phone/codec) to mimic a different microphone. Reports genuine/impostor score
statistics, the equal error rate, thresholds for target false-accept rates, and the
outcome of the matcher's decision rule with the configured thresholds.

Synthetic voices are a stand-in for real recordings: use the numbers to sanity-check the
decision rule, then re-run on real enrollment data before trusting them.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.bench.dialogue import speak
from transcribe.core.config import Config
from transcribe.speakers.embedder import SherpaEmbedder
from transcribe.speakers.identity import make_matcher
from transcribe.speakers.voiceprint_db import VoiceprintDB

ENROLL_TEXTS = [
    "The quarterly report shows a steady increase in customer satisfaction.",
    "Please remember to send the updated slides before tomorrow's meeting.",
    "We should discuss the budget for the second half of the year in detail.",
]
TEST_TEXTS = [
    "I think the new design is much easier to use than the old version.",
    "Could you explain how the testing process works for this release?",
    "Let's schedule a follow up call with the whole team next Thursday.",
]


def _add_noise(audio: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = float(np.mean(audio**2)) or 1e-9
    noise = rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(audio))
    return (audio + noise).astype(np.float32)


def degrade(audio: np.ndarray, kind: str = "phone", seed: int = 0) -> np.ndarray:
    """Simulate another channel.

    ``phone``: band-limited through 8 kHz + 20 dB noise (worst case, e.g. dial-in).
    ``mic``: different wideband microphone - low-pass tilt (~4 kHz one-pole) + 25 dB noise.
    """
    import soxr

    if kind == "phone":
        narrow = soxr.resample(soxr.resample(audio, SAMPLE_RATE, 8000), 8000, SAMPLE_RATE)
        return _add_noise(narrow[: len(audio)], 20.0, seed)
    if kind == "mic":
        a = float(np.exp(-2 * np.pi * 4000 / SAMPLE_RATE))
        out = np.empty_like(audio)
        acc = 0.0
        for i, x in enumerate(audio):  # one-pole low-pass, mixed 50/50 with the original
            acc = (1 - a) * x + a * acc
            out[i] = acc
        return _add_noise(0.5 * audio + 0.5 * out, 25.0, seed)
    raise ValueError(kind)


@dataclass
class Trial:
    speaker: str
    enrolled: bool
    scores: list[tuple[str, float]]  # (person, score), best first


@dataclass
class Outcome:
    correct: int = 0  # enrolled speaker labelled with the right name
    suggested: int = 0  # "Is this X?" (right or wrong name: the user decides)
    unknown: int = 0
    wrong: int = 0  # confident wrong name, or a stranger labelled as someone


def decide(trials: list[Trial], high: float, low: float, margin: float) -> Outcome:
    """Apply the matcher's decision rule (single speaker: competitor = runner-up)."""
    out = Outcome()
    for t in trials:
        (person, score), runner = t.scores[0], (t.scores[1][1] if len(t.scores) > 1 else -1)
        if score >= high and score - runner >= margin:
            if t.enrolled and person == t.speaker:
                out.correct += 1
            else:
                out.wrong += 1
        elif score >= low:
            out.suggested += 1
        else:
            out.unknown += 1
    return out


def eer(trials: list[Trial]) -> float:
    gen = np.array([s for t in trials for p, s in t.scores if p == t.speaker])
    imp = np.array([s for t in trials for p, s in t.scores if p != t.speaker])
    return min(max(float(np.mean(gen < x)), float(np.mean(imp >= x)))
               for x in np.unique(np.r_[gen, imp]))


def run(n_speakers: int = 20, n_unknown: int = 5, channel: str | None = "mic",
        config: Config | None = None) -> list[Trial]:
    """Enroll ``n_speakers`` and score every test speaker (incl. strangers) on ``channel``."""
    config = config or Config()
    embedder = SherpaEmbedder.from_config(config)
    names = [f"vctk-{i * 5}" for i in range(n_speakers + n_unknown)]
    enrolled = set(names[:n_speakers])

    def emb(name: str, text: str, mismatch: bool) -> np.ndarray:
        audio = speak(name, text)
        return embedder.embed(degrade(audio, channel) if mismatch and channel else audio)

    trials = []
    with tempfile.TemporaryDirectory() as tmp:
        db = VoiceprintDB(Path(tmp) / "vp.sqlite")
        for name in names[:n_speakers]:
            person = db.add_person(name, consent=True)
            db.add_exemplars(person.id, embedder.model_id,
                             [emb(name, t, False) for t in ENROLL_TEXTS],
                             [3.0] * len(ENROLL_TEXTS))
        matcher = make_matcher(db, embedder.model_id, config)
        names_by_id = {p.id: p.display_name for p in db.persons()}
        for name in names:
            vecs = [emb(name, t, True) for t in TEST_TEXTS]
            centroid = np.mean([v / np.linalg.norm(v) for v in vecs], axis=0)
            trials.append(Trial(name, name in enrolled, [
                (names_by_id[p], s) for p, s in matcher.score_all(centroid)]))
        db.close()
    return trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speakers", type=int, default=20)
    ap.add_argument("--unknown", type=int, default=5, help="non-enrolled test speakers")
    ap.add_argument("--channels", nargs="+", default=["same", "mic", "phone"],
                    choices=["same", "mic", "phone"])
    ap.add_argument("--sweep", action="store_true", help="grid over high/margin")
    args = ap.parse_args()
    cfg = Config().speakers
    results = {ch: run(args.speakers, args.unknown, None if ch == "same" else ch)
               for ch in args.channels}
    for ch, trials in results.items():
        o = decide(trials, cfg.match_threshold_high, cfg.match_threshold_low, cfg.match_margin)
        gen = [s for t in trials for p, s in t.scores if p == t.speaker]
        imp = [s for t in trials for p, s in t.scores if p != t.speaker]
        print(f"{ch:5}: EER {eer(trials):5.1%} | correct {o.correct:2} suggest {o.suggested:2}"
              f" unknown {o.unknown:2} WRONG {o.wrong} | genuine min {min(gen):.2f} median "
              f"{np.median(gen):.2f} | impostor max {max(imp):.2f} p99 "
              f"{np.percentile(imp, 99):.2f}")
    if args.sweep:
        print("high  margin | " + " | ".join(f"{ch}: ok/sug/WRONG" for ch in results))
        for high in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for margin in (0.05, 0.10, 0.15):
                cells = []
                for trials in results.values():
                    o = decide(trials, high, cfg.match_threshold_low, margin)
                    cells.append(f"{o.correct:2}/{o.suggested:2}/{o.wrong}")
                print(f"{high:.2f}  {margin:.2f}   | " + " | ".join(
                    f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
