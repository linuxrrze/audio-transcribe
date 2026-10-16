"""WER and speaker-attribution metrics against reference turns."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from transcribe.bench.dialogue import RefTurn
from transcribe.storage.transcript_model import Transcript, Word

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def normalize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = normalize(ref)
    return edit_distance(r, normalize(hyp)) / max(1, len(r))


@dataclass(frozen=True)
class Report:
    wer: float
    wer_by_language: dict[str, float]
    speaker_accuracy: float  # share of hypothesis words attributed to the right speaker
    num_ref_speakers: int
    num_hyp_speakers: int


def _words_in(words: list[Word], t0: float, t1: float, pad: float = 0.35) -> list[Word]:
    return [w for w in words if t0 - pad <= (w.t0 + w.t1) / 2 <= t1 + pad]


def true_speaker(word: Word, ref: list[RefTurn]) -> str:
    mid = (word.t0 + word.t1) / 2

    def distance(turn: RefTurn) -> float:
        return 0.0 if turn.t0 <= mid <= turn.t1 else min(abs(mid - turn.t0), abs(mid - turn.t1))

    return min(ref, key=distance).speaker


def evaluate(transcript: Transcript, ref: list[RefTurn]) -> Report:
    words = [w for s in transcript.segments for w in s.words]
    per_lang: dict[str, tuple[list[str], list[str]]] = {}
    for turn in ref:
        r, h = per_lang.setdefault(turn.language, ([], []))
        r.append(turn.text)
        h.append("".join(w.text for w in _words_in(words, turn.t0, turn.t1)))
    wer_by_language = {lang: wer(" ".join(r), " ".join(h)) for lang, (r, h) in per_lang.items()}
    all_ref = " ".join(t.text for t in ref)
    total_wer = wer(all_ref, "".join(w.text for w in words))

    # Each word belongs to exactly one reference turn (the one containing its midpoint,
    # else the nearest). Padding turns would count words near a speaker change twice -
    # once right, once wrong - which understates accuracy for fast dialogue.
    pairs = [(true_speaker(w, ref), w.speaker_id) for w in words]
    ref_spk = sorted({t.speaker for t in ref})
    hyp_spk = sorted({h for _, h in pairs if h is not None})
    best = 0
    for perm in itertools.permutations(hyp_spk, min(len(hyp_spk), len(ref_spk))):
        mapping = dict(zip(perm, ref_spk, strict=False))
        best = max(best, sum(mapping.get(h) == r for r, h in pairs))  # type: ignore[arg-type]
    return Report(
        wer=total_wer,
        wer_by_language=wer_by_language,
        speaker_accuracy=best / max(1, len(pairs)),
        num_ref_speakers=len(ref_spk),
        num_hyp_speakers=len(transcript.speakers),
    )
