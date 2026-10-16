"""Build a LiveSession from configuration (shared by CLI and GUI)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from transcribe.asr.base import AsrEngine
from transcribe.audio.base import AudioSource
from transcribe.core import registry
from transcribe.core.config import Config
from transcribe.live.session import LiveSession, StageProgress
from transcribe.live.streamer import StreamerConfig
from transcribe.speakers.live_tracker import LiveSpeakerTracker, sherpa_cluster
from transcribe.storage.transcript_model import Transcript


def make_refiner(config: Config, asr: AsrEngine, num_speakers: int | None):
    """Full offline pipeline over the recording; models load lazily at refine time."""

    def refine(audio: np.ndarray, progress: StageProgress) -> Transcript:
        from transcribe.core.offline import transcribe_audio

        dia_cls = registry.load("transcribe.diarizers", config.speakers.diarizer)
        diarizer = dia_cls.from_config(config)
        engine = asr
        if config.live.refine_model and config.live.refine_model != config.asr.model:
            asr_cls = registry.load("transcribe.asr", config.asr.engine)
            cfg = replace(config, asr=replace(config.asr, model=config.live.refine_model))
            engine = asr_cls.from_config(cfg)
        return transcribe_audio(audio, engine, diarizer, config.asr.languages, num_speakers,
                                progress)

    return refine


def build_live_session(source: AudioSource, config: Config, session_dir: Path,
                       num_speakers: int | None = None, diarize: bool = True,
                       meta: dict[str, object] | None = None) -> LiveSession:
    from transcribe.vad.silero import speech_regions

    asr_cls = registry.load("transcribe.asr", config.asr.engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        asr_f = pool.submit(asr_cls.from_config, config)
        emb_f = None
        if diarize:
            emb_cls = registry.load("transcribe.embedders", "sherpa-onnx")
            emb_f = pool.submit(emb_cls.from_config, config)
        asr = asr_f.result()
        embedder = emb_f.result() if emb_f else None

    with_options = getattr(asr, "with_options", None)
    live_asr = (with_options(config.live.beam_size, temperature_fallback=False)
                if with_options else asr)
    tracker = (LiveSpeakerTracker(sherpa_cluster(config.speakers.diarize_threshold),
                                  num_speakers) if diarize else None)
    identity = None
    if embedder is not None and (config.speakers.identify_known
                                 or config.speakers.store_voiceprints):
        from transcribe.speakers.identity import SessionIdentity

        identity = SessionIdentity(config, embedder)
    lc = config.live
    return LiveSession(
        source, live_asr, session_dir, config.asr.languages,
        partial(speech_regions, min_silence_ms=300, pad_ms=100),
        embedder=embedder, tracker=tracker,
        refine=make_refiner(config, asr, num_speakers) if lc.refine and diarize else None,
        streamer_config=StreamerConfig(step=lc.step, end_silence=lc.end_silence,
                                       max_buffer=lc.max_buffer),
        meta={"asr_engine": config.asr.engine, "asr_model": config.asr.model,
              "languages": config.asr.languages, **(meta or {})},
        identity=identity,
    )
