"""Speaker-attribution benchmark on real meetings (AMI corpus, manual annotations).

    python -m transcribe.bench.diarization [MEETING ...]     (default: ES2004a IS1009a)

Downloads the meeting's headset mix and the reference RTTM (BUT's AMI diarization
setup, "only_words"), runs diarization and ASR once, and reports word speaker accuracy
(words with exactly one active reference speaker; one-to-one label mapping) for the
diarizer alone and with the second pass in both counting modes. Results of the
default pipeline are documented in transcribe.diarize.resegment.
"""

from __future__ import annotations

import argparse
import copy
import urllib.request
from collections import defaultdict
from pathlib import Path

from transcribe.audio.file import load_audio
from transcribe.core import models
from transcribe.core.config import Config
from transcribe.diarize.align import Turn, assign_speakers
from transcribe.diarize.resegment import resegment, speaker_count, speech_per_speaker
from transcribe.diarize.sherpa_offline import SherpaDiarizer
from transcribe.storage.transcript_model import Word

AMI_AUDIO = ("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{m}/audio/"
             "{m}.Mix-Headset.wav")
AMI_RTTM = ("https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
            "only_words/rttms/{split}/{m}.rttm")


def fetch(meeting: str, cache: Path) -> tuple[Path, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    wav, rttm = cache / f"{meeting}.wav", cache / f"{meeting}.rttm"
    if not wav.exists():
        urllib.request.urlretrieve(AMI_AUDIO.format(m=meeting), wav)
    if not rttm.exists():
        for split in ("test", "dev", "train"):
            try:
                urllib.request.urlretrieve(AMI_RTTM.format(m=meeting, split=split), rttm)
                break
            except OSError:
                continue
        else:
            raise FileNotFoundError(f"No reference RTTM for {meeting}")
    return wav, rttm


def load_rttm(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            out.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return out


def word_accuracy(words: list[Word], ref: list[tuple[float, float, str]]) -> float:
    """Share of words (with exactly one active reference speaker) labelled correctly
    under the best greedy one-to-one mapping of hypothesis to reference speakers."""
    pairs = []
    for w in words:
        mid = (w.t0 + w.t1) / 2
        active = {s for a, b, s in ref if a <= mid <= b}
        if len(active) == 1:
            pairs.append((active.pop(), w.speaker_id))
    co: dict[tuple[str | None, str], int] = defaultdict(int)
    for truth, hyp in pairs:
        co[(hyp, truth)] += 1
    mapping: dict[str | None, str] = {}
    used: set[str] = set()
    for (hyp, truth), _ in sorted(co.items(), key=lambda kv: -kv[1]):
        if hyp not in mapping and truth not in used:
            mapping[hyp] = truth
            used.add(truth)
    return sum(mapping.get(h) == t for t, h in pairs) / max(1, len(pairs))


def run(meeting: str, cache: Path, config: Config) -> dict[str, tuple[float, int]]:
    from transcribe.asr.faster_whisper import FasterWhisperEngine
    from transcribe.speakers.embedder import SherpaEmbedder

    wav, rttm = fetch(meeting, cache)
    audio, ref = load_audio(wav), load_rttm(rttm)
    turns: list[Turn] = SherpaDiarizer.from_config(config).diarize(audio)
    words = FasterWhisperEngine.from_config(config).transcribe(audio, language="en")
    assign_speakers(words, turns)
    k_max = speaker_count(speech_per_speaker([(t.t0, t.t1, t.speaker_id) for t in turns]))
    embedder = SherpaEmbedder(config.speakers.embedding_model)
    def n_speakers(ws: list[Word]) -> int:
        return len({w.speaker_id for w in ws if w.speaker_id})

    results = {"diarizer only": (word_accuracy(words, ref), n_speakers(words))}
    for mode in ("estimate", "silhouette"):
        w = [copy.copy(x) for x in words]
        resegment(w, audio, embedder, max_speakers=k_max, count=mode)
        results[f"second pass ({mode})"] = (word_accuracy(w, ref), n_speakers(w))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("meetings", nargs="*", default=["ES2004a", "IS1009a"])
    ap.add_argument("--cache", type=Path, default=models.models_dir().parent / "ami")
    args = ap.parse_args()
    config = Config()
    for meeting in args.meetings:
        res = run(meeting, args.cache, config)
        cells = " | ".join(f"{name}: {acc:.1%} ({n} speakers)" for name, (acc, n) in res.items())
        print(f"{meeting}: {cells}", flush=True)


if __name__ == "__main__":
    main()
