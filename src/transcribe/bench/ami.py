"""AMI meeting corpus helpers: headset-mix audio, reference RTTMs, manual transcripts.

Audio: AMI corpus mirror (University of Edinburgh). RTTMs: BUT's AMI diarization setup
("only_words"). Words: AMI manual annotations 1.6.2 (one XML per speaker).
"""

from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

from transcribe.core import models
from transcribe.storage.transcript_model import Word

AMI_AUDIO = ("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{m}/audio/"
             "{m}.Mix-Headset.wav")
AMI_RTTM = ("https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
            "only_words/rttms/{split}/{m}.rttm")
AMI_MANUAL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
DEFAULT_MEETINGS = ["ES2004a", "IS1009a"]


def cache_dir() -> Path:
    return models.models_dir().parent / "ami"


def fetch(meeting: str, cache: Path | None = None) -> tuple[Path, Path]:
    """(audio .wav, reference .rttm), downloaded on first use."""
    cache = cache or cache_dir()
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


def reference_text(meeting: str, cache: Path | None = None) -> str:
    """Manual transcript of all speakers, ordered by time (punctuation tokens dropped)."""
    cache = cache or cache_dir()
    archive = cache / "ami_public_manual_1.6.2.zip"
    if not archive.exists():
        urllib.request.urlretrieve(AMI_MANUAL, archive)
    words: list[tuple[float, str]] = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if re.fullmatch(rf"words/{meeting}\.[A-Z]\.words\.xml", name):
                for el in ET.fromstring(z.read(name)).iter("w"):
                    if el.get("punc") == "true" or not el.text or el.get("starttime") is None:
                        continue
                    words.append((float(el.get("starttime", 0)), el.text))
    if not words:
        raise FileNotFoundError(f"No manual transcript for {meeting}")
    return " ".join(text for _, text in sorted(words))


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
