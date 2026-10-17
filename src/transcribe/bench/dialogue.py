"""Synthesise a German/English two-speaker dialogue with known ground truth (Piper TTS)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core import models

VOICES: dict[str, tuple[str, str, int]] = {  # name -> (asset, language, speaker id)
    "thorsten": ("piper-de-thorsten", "de", 0),
    "amy": ("piper-en-amy", "en", 0),
    "kerstin": ("piper-de-kerstin", "de", 0),
    # more German voices for the benchmark suite's synthetic conversations
    "karlsson": ("piper-de-karlsson", "de", 0),
    "eva": ("piper-de-evak", "de", 0),
    "ramona": ("piper-de-ramona", "de", 0),
    "pavoque": ("piper-de-pavoque", "de", 0),
    "miro": ("piper-de-miro", "de", 0),
    "dii": ("piper-de-dii", "de", 0),
}
VCTK_SPEAKERS = 109


def voice(name: str) -> tuple[str, str, int]:
    """Named voice, or ``vctk-<n>`` for speaker n of the VCTK multi-speaker model."""
    if name.startswith("vctk-"):
        return "piper-en-vctk", "en", int(name[5:]) % VCTK_SPEAKERS
    return VOICES[name]

SCRIPT: list[tuple[str, str]] = [
    ("thorsten", "Guten Morgen zusammen. Wir besprechen heute den Zeitplan für das neue "
                 "Projekt und die Verteilung der Aufgaben."),
    ("amy", "Good morning. I reviewed the draft yesterday, and I think the timeline for "
            "the first milestone is a little optimistic."),
    ("thorsten", "Das verstehe ich. Wie viele Wochen würdest du zusätzlich einplanen?"),
    ("amy", "Probably two more weeks. The testing phase needs more time than we expected."),
    ("thorsten", "Gut, dann verschieben wir den ersten Meilenstein auf Ende November."),
    ("amy", "That works for me. Who is going to update the project plan?"),
    ("thorsten", "Ich kümmere mich darum und schicke allen bis Freitag eine neue Version."),
    ("amy", "Perfect. Then let's meet again next week to discuss the budget."),
]


@dataclass(frozen=True)
class RefTurn:
    speaker: str
    language: str
    text: str
    t0: float
    t1: float


_ENGINES: dict[str, object] = {}


def _tts(asset: str):
    if asset in _ENGINES:
        return _ENGINES[asset]
    import sherpa_onnx

    model = models.ensure(asset)
    d = model.parent
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        # noise scales 0: deterministic output, so fixtures are identical on every run
        # (VITS sampling otherwise yields different audio each time -> flaky tests).
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data"),
            noise_scale=0.0, noise_scale_w=0.0),
        num_threads=2))
    _ENGINES[asset] = sherpa_onnx.OfflineTts(cfg)
    return _ENGINES[asset]


def speak(name: str, text: str) -> np.ndarray:
    """Synthesize ``text`` with voice ``name`` at 16 kHz (silence trimmed)."""
    import soxr

    asset, _, sid = voice(name)
    gen = _tts(asset).generate(text, sid=sid, speed=1.0)  # type: ignore[attr-defined]
    audio = soxr.resample(np.asarray(gen.samples, dtype=np.float32), gen.sample_rate,
                          SAMPLE_RATE)
    return _trim(audio)


def synthesize(out_wav: Path, pause: float = 0.7,
               script: list[tuple[str, str]] = SCRIPT) -> list[RefTurn]:
    """Write ``out_wav`` (16 kHz) and ``<out_wav>.json`` with the reference turns."""
    import soundfile as sf

    gap = np.zeros(int(pause * SAMPLE_RATE), dtype=np.float32)
    parts, turns, t = [gap], [], pause
    for speaker, text in script:
        audio = speak(speaker, text)
        turns.append(RefTurn(speaker, voice(speaker)[1], text, t, t + len(audio) / SAMPLE_RATE))
        parts += [audio, gap]
        t += len(audio) / SAMPLE_RATE + pause
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, np.concatenate(parts), SAMPLE_RATE, subtype="PCM_16")
    out_wav.with_suffix(".json").write_text(
        json.dumps([asdict(x) for x in turns], ensure_ascii=False, indent=2), encoding="utf-8")
    return turns


def load_reference(wav: Path) -> list[RefTurn]:
    return [RefTurn(**x) for x in json.loads(wav.with_suffix(".json").read_text("utf-8"))]


def _trim(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > threshold)
    return audio[idx[0] : idx[-1] + 1] if len(idx) else audio


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "dialogue_de_en.wav")
    for turn in synthesize(target):
        print(f"{turn.t0:6.2f}-{turn.t1:6.2f} {turn.speaker:9} {turn.text}")
