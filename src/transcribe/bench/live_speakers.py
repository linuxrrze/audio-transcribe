"""Live speaker labels: replay recordings with known speakers through the live labelling.

    python -m transcribe.bench.live_speakers [--data play ES2004a IS1009a]
        [--models nemo-titanet-small wespeaker-resnet34-lm] [--thresholds 0.4]
        [--modes sentences utterances] [--json out.json]

The audio is cut into utterances like live mode (VAD with the live settings; an
utterance ends at a pause of 0.4 s, at most 12 s), transcribed once with Whisper
``small`` (cached), then labelled piece by piece with the live code
(``live.session.sentence_pieces`` + ``speakers.live_tracker``). Mode ``utterances``
labels whole utterances (no sentence split) for comparison.

Scores are frame-based (10 ms) on speech with exactly one reference speaker: accuracy
under the best one-to-one mapping of labels to people, purity (each label counted as its
majority person) and the number of labels; "ceiling" labels every piece with its true
majority speaker - the best any method can do with these pieces.

Data: AMI meetings (real, 4 people; see bench.ami) and "play", a synthetic radio play
(6 Piper voices, 40 lines with 0.15-0.4 s gaps and interjections; generated once).
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.audio.file import load_audio
from transcribe.bench.ami import cache_dir, fetch, load_rttm
from transcribe.live.session import sentence_pieces
from transcribe.speakers.live_tracker import (
    DEFAULT_THRESHOLD,
    BandwidthEstimator,
    LiveSpeakerTracker,
)
from transcribe.storage.transcript_model import Segment, Word

FRAME = 0.01
END_SILENCE, MAX_UTTERANCE = 0.4, 12.0  # live defaults (config.live)

PLAY_LINES = [
    ("thorsten", "Hast du das gehört? Da hat jemand um Hilfe gerufen."),
    ("kerstin", "Ja, ich habe es auch gehört."),
    ("vctk-12", "Was?"),
    ("thorsten", "Es kam aus dem alten Haus am Ende der Straße."),
    ("vctk-40", "Dann sollten wir sofort nachsehen. Aber leise."),
    ("kerstin", "Moment. Ich hole noch schnell meine Taschenlampe."),
    ("vctk-77", "Halt! Wer seid ihr und was wollt ihr hier?"),
    ("thorsten", "Wir sind Detektive. Wir suchen einen verschwundenen Papagei."),
    ("vctk-77", "Einen Papagei? Hier gibt es keinen Papagei."),
    ("vctk-12", "Aber wir haben ihn doch gehört."),
    ("kerstin", "Genau."),
    ("vctk-95", "Ruhe jetzt! Alle setzen sich hin, und zwar sofort."),
    ("vctk-40", "Na gut, na gut. Wir setzen uns ja schon."),
    ("thorsten", "Darf ich Ihnen unsere Karte geben? Hier steht alles drauf."),
    ("vctk-95", "Die drei Detektive. Soso. Und wer hat euch geschickt?"),
    ("vctk-12", "Ein Freund von uns."),
    ("kerstin", "Er meinte, Sie könnten Hilfe brauchen."),
    ("vctk-77", "Hilfe? Ich brauche keine Hilfe."),
    ("thorsten", "Dann gehen wir wieder. Entschuldigen Sie die Störung."),
    ("vctk-95", "Nein. Ihr bleibt hier, bis ich weiß, was ihr gesehen habt."),
]


@dataclass
class Data:
    name: str
    audio: np.ndarray
    ref: list[tuple[float, float, str]]
    language: str


def make_play(out: Path) -> None:
    """Synthesize the radio play (needs the Piper voices; downloaded on first use)."""
    import soundfile as sf

    from transcribe.bench.dialogue import speak

    rng = random.Random(7)
    parts, ref, t = [np.zeros(8000, np.float32)], [], 0.5
    for spk, text in PLAY_LINES * 2:
        a = speak(spk, text)
        gap = rng.uniform(0.15, 0.4)
        ref.append({"speaker": spk, "text": text, "t0": t, "t1": t + len(a) / SAMPLE_RATE})
        parts += [a, np.zeros(int(gap * SAMPLE_RATE), np.float32)]
        t += len(a) / SAMPLE_RATE + gap
    sf.write(out / "play.wav", np.concatenate(parts), SAMPLE_RATE)
    (out / "play.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1))


def load_data(name: str) -> Data:
    if name == "play":
        out = cache_dir()
        if not (out / "play.json").exists():
            make_play(out)
        turns = json.loads((out / "play.json").read_text())
        return Data(name, load_audio(out / "play.wav"),
                    [(x["t0"], x["t1"], x["speaker"]) for x in turns], "de")
    wav, rttm = fetch(name)
    return Data(name, load_audio(wav), load_rttm(rttm), "en")


def words(data: Data) -> list[Word]:
    path = cache_dir() / f"{data.name}-small-words.json"
    if not path.exists():
        from transcribe.asr.faster_whisper import FasterWhisperEngine

        ws = FasterWhisperEngine("small").transcribe(data.audio, language=data.language)
        path.write_text(json.dumps([(w.text, w.t0, w.t1) for w in ws]))
    return [Word(w[0], w[1], w[2]) for w in json.loads(path.read_text())]


def utterances(audio: np.ndarray) -> list[tuple[float, float]]:
    """Live-mode utterances: VAD regions (live settings) joined across pauses shorter
    than END_SILENCE, cut into pieces of at most MAX_UTTERANCE seconds."""
    from transcribe.vad.silero import speech_regions

    joined: list[list[float]] = []
    for a, b in speech_regions(audio, min_silence_ms=300, pad_ms=100):
        if joined and a - joined[-1][1] < END_SILENCE:
            joined[-1][1] = b
        else:
            joined.append([a, b])
    out: list[tuple[float, float]] = []
    for a, b in joined:
        n = int(np.ceil((b - a) / MAX_UTTERANCE))
        out += [(a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]
    return out


def pieces(data: Data, ws: list[Word], mode: str) -> list[tuple[Segment, np.ndarray]]:
    out: list[tuple[Segment, np.ndarray]] = []
    for a, b in utterances(data.audio):
        inside = [w for w in ws if a <= (w.t0 + w.t1) / 2 < b]
        if not inside:  # live mode emits nothing without words
            continue
        seg = Segment(a, b, inside, id=f"u{len(out)}")
        audio = data.audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
        out += sentence_pieces(seg, audio) if mode == "sentences" else [(seg, audio)]
    return out


def label(items: list[tuple[Segment, np.ndarray]], embedder,
          threshold: float) -> list[tuple[float, float, str]]:
    tracker = LiveSpeakerTracker(threshold)
    bandwidth = BandwidthEstimator()
    out = []
    for seg, audio in items:
        emb = embedder.embed(audio) if len(audio) >= 0.3 * SAMPLE_RATE else None
        speaker, _ = tracker.assign(seg.id, emb, seg.t1 - seg.t0, bandwidth.update(audio))
        out.append((seg.t0, seg.t1, speaker))
    return out


def _frames(data: Data) -> np.ndarray:
    """Per 10 ms frame: index of the single active reference speaker, -1 otherwise."""
    n = int(len(data.audio) / SAMPLE_RATE / FRAME) + 1
    names = sorted({s for *_, s in data.ref})
    count, who = np.zeros(n, int), np.full(n, -1)
    for a, b, s in data.ref:
        count[int(a / FRAME):int(b / FRAME)] += 1
        who[int(a / FRAME):int(b / FRAME)] = names.index(s)
    who[count != 1] = -1
    return who


def score(data: Data, spans: Sequence[tuple[float, float, object]]) -> dict[str, float | int]:
    ref = _frames(data)
    hyp = np.full(len(ref), -1)
    labels: dict[object, int] = {}
    for a, b, s in spans:
        hyp[int(a / FRAME):int(b / FRAME)] = labels.setdefault(s, len(labels))
    total = int((ref >= 0).sum())
    both = (ref >= 0) & (hyp >= 0)
    m = np.zeros((max(1, len(labels)), int(ref.max()) + 1))
    np.add.at(m, (hyp[both], ref[both]), 1)
    return {"accuracy": round(_one_to_one(m) / total, 4),
            "purity": round(float(m.max(axis=1).sum()) / total, 4), "labels": len(labels)}


def _one_to_one(m: np.ndarray) -> float:
    """Best total of a one-to-one label -> person mapping (DP over subsets of people)."""
    cols = m.shape[1]
    best = np.full(1 << cols, -np.inf)
    best[0] = 0.0
    for row in m:
        new = best.copy()
        for mask in range(1 << cols):
            if best[mask] == -np.inf:
                continue
            for c in range(cols):
                if not mask & (1 << c):
                    new[mask | 1 << c] = max(new[mask | 1 << c], best[mask] + row[c])
        best = new
    return float(best.max())


def ceiling(data: Data, items: list[tuple[Segment, np.ndarray]]) -> float:
    ref = _frames(data)
    spans = []
    for seg, _ in items:
        r = ref[int(seg.t0 / FRAME):int(seg.t1 / FRAME)]
        r = r[r >= 0]
        if len(r):
            spans.append((seg.t0, seg.t1, int(np.bincount(r).argmax())))
    return float(score(data, spans)["accuracy"])


def main() -> None:
    from transcribe.speakers.embedder import SherpaEmbedder

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", default=["play", "IS1009a", "ES2004a"])
    ap.add_argument("--models", nargs="+", default=["nemo-titanet-small"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[DEFAULT_THRESHOLD])
    ap.add_argument("--modes", nargs="+", default=["sentences", "utterances"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = []
    embedders = {m: SherpaEmbedder(m) for m in args.models}
    for name in args.data:
        data = load_data(name)
        ws = words(data)
        for mode in args.modes:
            items = pieces(data, ws, mode)
            top = ceiling(data, items)
            for model, emb in embedders.items():
                for thr in args.thresholds:
                    s = score(data, label(items, emb, thr))
                    row = {"data": name, "mode": mode, "model": model, "threshold": thr,
                           "pieces": len(items), "ceiling": round(top, 4), **s}
                    rows.append(row)
                    print(row, flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
