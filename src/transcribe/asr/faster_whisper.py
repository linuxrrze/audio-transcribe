"""faster-whisper (CTranslate2) ASR backend: int8 on CPU, float16 on CUDA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from transcribe.asr.base import ProgressFn
from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config
from transcribe.core.models import models_dir
from transcribe.storage.transcript_model import Word


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    import ctranslate2

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"faster-whisper supports devices cpu/cuda, not {device!r}")
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto",
                 beam_size: int = 5, cpu_threads: int = 0) -> None:
        from faster_whisper import WhisperModel

        self.device, self.compute_type = resolve_device(device, compute_type)
        self.model_name = model
        self.beam_size = beam_size
        self._model = WhisperModel(
            model, device=self.device, compute_type=self.compute_type,
            cpu_threads=cpu_threads, download_root=str(models_dir() / "whisper"),
        )

    @classmethod
    def from_config(cls, config: Config) -> FasterWhisperEngine:
        a = config.asr
        return cls(a.model, a.device, a.compute_type, a.beam_size, a.cpu_threads)

    def detect_language(self, audio: np.ndarray, languages: Sequence[str]) -> dict[str, float]:
        _, _, all_probs = self._model.detect_language(audio=audio)
        probs = {lang: p for lang, p in all_probs if lang in languages}
        total = sum(probs.values())
        if total <= 0:
            return {lang: 1.0 / len(languages) for lang in languages}
        return {lang: probs.get(lang, 0.0) / total for lang in languages}

    def transcribe(self, audio: np.ndarray, offset: float = 0.0, language: str | None = None,
                   prompt: str | None = None, progress: ProgressFn | None = None) -> list[Word]:
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=prompt,
            hallucination_silence_threshold=2.0,
        )
        words: list[Word] = []
        for seg in segments:  # generator: decoding happens while iterating
            for w in seg.words or []:
                words.append(Word(text=w.word, t0=offset + w.start, t1=offset + w.end,
                                  confidence=w.probability))
            if progress:
                progress(min(seg.end, len(audio) / SAMPLE_RATE))
        return words
