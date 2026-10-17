"""Speaker-attribution benchmark on real meetings (AMI corpus, manual annotations).

    python -m transcribe.bench.diarization [MEETING ...]     (default: ES2004a IS1009a)
        [--embedding-model nemo-titanet-small] [--threshold 0.5]

Downloads the meeting's headset mix and the reference RTTM (BUT's AMI diarization
setup, "only_words"), runs diarization and ASR once, and reports word speaker accuracy
(words with exactly one active reference speaker; one-to-one label mapping) for the
diarizer alone and with the second pass in both counting modes. Results of the
default pipeline are documented in transcribe.diarize.resegment.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from transcribe.audio.file import load_audio
from transcribe.bench.ami import (
    DEFAULT_MEETINGS,
    cache_dir,
    fetch,
    load_rttm,
    speaker_purity,
    word_accuracy,
)
from transcribe.core.config import Config
from transcribe.diarize.align import Turn, assign_speakers
from transcribe.diarize.resegment import resegment, speaker_count, speech_per_speaker
from transcribe.diarize.sherpa_offline import SherpaDiarizer
from transcribe.storage.transcript_model import Word


def run(meeting: str, cache: Path, config: Config) -> dict[str, tuple[float, float, int]]:
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

    results = {"diarizer only": (word_accuracy(words, ref), speaker_purity(words, ref),
                                 n_speakers(words))}
    for mode in ("estimate", "silhouette"):
        w = [copy.copy(x) for x in words]
        resegment(w, audio, embedder, max_speakers=k_max, count=mode)
        results[f"second pass ({mode})"] = (word_accuracy(w, ref), speaker_purity(w, ref),
                                            n_speakers(w))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--cache", type=Path, default=cache_dir())
    ap.add_argument("--embedding-model", help="speaker embedding model (default: config)")
    ap.add_argument("--threshold", type=float, help="diarizer clustering threshold")
    args = ap.parse_args()
    config = Config()
    if args.embedding_model:
        config.speakers.embedding_model = args.embedding_model
    if args.threshold is not None:
        config.speakers.diarize_threshold = args.threshold
    for meeting in args.meetings:
        res = run(meeting, args.cache, config)
        cells = " | ".join(f"{name}: {acc:.1%}, purity {pur:.1%} ({n} speakers)"
                           for name, (acc, pur, n) in res.items())
        print(f"{meeting}: {cells}", flush=True)


if __name__ == "__main__":
    main()
