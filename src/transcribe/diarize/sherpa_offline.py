"""Offline diarization with sherpa-onnx: pyannote segmentation + speaker embeddings +
clustering. Fully local, no Hugging Face token required.

sherpa-onnx holds the GIL for the whole segmentation pass (about 1.5 s per minute of
audio before any progress is reported), which would freeze progress displays and the
GUI. By default the computation therefore runs in a separate process.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from transcribe.core import models
from transcribe.core.config import Config
from transcribe.diarize.align import Turn
from transcribe.diarize.base import ProgressFn


def _build(segmentation: str, embedding: str, threshold: float, num_threads: int,
           num_speakers: int | None):
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding, num_threads=num_threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("Invalid sherpa-onnx diarization config")
    return sherpa_onnx.OfflineSpeakerDiarization(cfg)


def run_diarization(audio: np.ndarray | str, segmentation: str, embedding: str,
                    threshold: float, num_threads: int, num_speakers: int | None,
                    progress: ProgressFn | None = None) -> list[tuple[float, float, str]]:
    """Diarize ``audio`` (samples, or a path to a .npy file). Picklable in and out."""
    if isinstance(audio, str):
        audio = np.load(audio, mmap_mode="r")
    sd = _build(segmentation, embedding, threshold, num_threads, num_speakers)

    def callback(done: int, total: int) -> int:
        if progress and total:
            progress(done / total)
        return 0

    result = sd.process(np.ascontiguousarray(audio, dtype=np.float32), callback=callback)
    return [(s.start, s.end, f"S{s.speaker + 1}") for s in result.sort_by_start_time()]


class SherpaDiarizer:
    name = "sherpa-onnx"

    def __init__(self, segmentation_model: str = "pyannote-segmentation-3.0",
                 embedding_model: str = "nemo-titanet-small", threshold: float = 0.5,
                 num_threads: int = 0, isolated: bool = True,
                 resegment: bool = True, speaker_count: str = "estimate",
                 min_speaker_share: float = 0.03) -> None:
        self.segmentation_path = models.ensure(segmentation_model)
        self.embedding_model = embedding_model
        self.embedding_path = models.ensure(embedding_model)
        self.resegment = resegment
        self.speaker_count = speaker_count  # how the second pass chooses k (resegment.py)
        self.min_speaker_share = min_speaker_share  # for the "estimate" of k
        self._phrase_embedder = None
        self.threshold = threshold
        self.num_threads = num_threads or max(1, (os.cpu_count() or 2) // 2)
        self.isolated = isolated

    @classmethod
    def from_config(cls, config: Config) -> SherpaDiarizer:
        s = config.speakers
        return cls(s.segmentation_model, s.embedding_model, s.diarize_threshold, s.num_threads,
                   resegment=s.resegment, speaker_count=s.speaker_count,
                   min_speaker_share=s.min_speaker_share)

    def phrase_embedder(self):
        """Embedder for the second pass on transcript phrases, or None if disabled."""
        if not self.resegment:
            return None
        if self._phrase_embedder is None:
            from transcribe.speakers.embedder import SherpaEmbedder

            self._phrase_embedder = SherpaEmbedder(self.embedding_model, self.num_threads)
        return self._phrase_embedder

    def diarize(self, audio: np.ndarray, num_speakers: int | None = None,
                progress: ProgressFn | None = None) -> list[Turn]:
        args: dict[str, Any] = {
            "segmentation": str(self.segmentation_path), "embedding": str(self.embedding_path),
            "threshold": self.threshold, "num_threads": self.num_threads,
            "num_speakers": num_speakers}
        if not self.isolated:
            turns = run_diarization(audio, **args, progress=progress)
        else:
            from transcribe.diarize.isolated import WorkerStartupError, call_isolated

            # Hand the audio over as a file: an hour is ~230 MB, too much for a pipe.
            with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
                path = Path(tmp) / "audio.npy"
                np.save(path, np.ascontiguousarray(audio, dtype=np.float32))
                try:
                    turns = call_isolated(
                        "transcribe.diarize.sherpa_offline:run_diarization",
                        {"audio": str(path), **args}, progress)
                except WorkerStartupError as e:
                    warnings.warn(f"Diarization worker could not start ({e}); running "
                                  "in-process - progress display may pause.", stacklevel=2)
                    turns = run_diarization(audio, **args, progress=progress)
        return [Turn(t0, t1, sid) for t0, t1, sid in turns]
