"""Speaker embedding extractors (plugin group ``transcribe.embedders``)."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core import models
from transcribe.core.config import Config


@runtime_checkable
class Embedder(Protocol):
    model_id: str  # embeddings are only comparable within one model id

    def embed(self, audio: np.ndarray) -> np.ndarray: ...


class SherpaEmbedder:
    def __init__(self, model: str = "nemo-titanet-small", num_threads: int = 0) -> None:
        import sherpa_onnx

        self.model_id = model
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(models.ensure(model)),
                num_threads=num_threads or max(1, (os.cpu_count() or 2) // 4),
            )
        )

    @classmethod
    def from_config(cls, config: Config) -> SherpaEmbedder:
        return cls(config.speakers.embedding_model, config.speakers.num_threads)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)
