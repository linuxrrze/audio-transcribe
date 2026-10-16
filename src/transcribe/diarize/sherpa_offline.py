"""Offline diarization with sherpa-onnx: pyannote segmentation + speaker embeddings +
clustering. Fully local, no Hugging Face token required."""

from __future__ import annotations

import os

import numpy as np

from transcribe.core import models
from transcribe.core.config import Config
from transcribe.diarize.align import Turn
from transcribe.diarize.base import ProgressFn


class SherpaDiarizer:
    name = "sherpa-onnx"

    def __init__(self, segmentation_model: str = "pyannote-segmentation-3.0",
                 embedding_model: str = "wespeaker-resnet34-lm", threshold: float = 0.5,
                 num_threads: int = 0) -> None:
        self.segmentation_path = models.ensure(segmentation_model)
        self.embedding_path = models.ensure(embedding_model)
        self.threshold = threshold
        self.num_threads = num_threads or max(1, (os.cpu_count() or 2) // 2)

    @classmethod
    def from_config(cls, config: Config) -> SherpaDiarizer:
        s = config.speakers
        return cls(s.segmentation_model, s.embedding_model, s.diarize_threshold, s.num_threads)

    def _build(self, num_speakers: int | None):
        import sherpa_onnx

        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_path)),
                num_threads=self.num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_path), num_threads=self.num_threads),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers or -1, threshold=self.threshold),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not cfg.validate():
            raise RuntimeError("Invalid sherpa-onnx diarization config")
        return sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def diarize(self, audio: np.ndarray, num_speakers: int | None = None,
                progress: ProgressFn | None = None) -> list[Turn]:
        sd = self._build(num_speakers)

        def callback(done: int, total: int) -> int:
            if progress and total:
                progress(done / total)
            return 0

        result = sd.process(np.ascontiguousarray(audio, dtype=np.float32), callback=callback)
        return [Turn(s.start, s.end, f"S{s.speaker + 1}") for s in result.sort_by_start_time()]
