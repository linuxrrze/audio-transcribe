"""TOML configuration stored in the platform user config dir. CLI flags override it."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import platformdirs
import tomli_w

APP_NAME = "transcribe"


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME))


def data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME))


@dataclass
class AsrConfig:
    engine: str = "faster-whisper"  # faster-whisper | whisper.cpp (GPUs beyond NVIDIA)
    model: str = "small"
    device: str = "auto"  # auto | cpu | cuda (faster-whisper) | gpu (whisper.cpp)
    compute_type: str = "auto"  # auto | int8 | float16 | ...
    languages: list[str] = field(default_factory=lambda: ["de", "en"])
    beam_size: int = 5
    cpu_threads: int = 0  # 0 = library default
    style_prompt: bool = True  # punctuated example prompt (see asr/faster_whisper.py)
    # Custom vocabulary (core/vocabulary.py): hint words for names, acronyms, jargon ...
    vocabulary: list[str] = field(default_factory=list)
    # ... and corrections applied to the transcript, e.g. {"Fanfress" = "Fentriss"}
    corrections: dict[str, str] = field(default_factory=dict)
    # whisper.cpp engine (asr/whisper_cpp.py): whisper-server binary, "" = own build or PATH
    whispercpp_server: str = ""
    # DTW word timestamps (needed for accurate speaker attribution; disables flash attention)
    whispercpp_dtw: bool = True


@dataclass
class SpeakerConfig:
    diarizer: str = "sherpa-onnx"
    segmentation_model: str = "pyannote-segmentation-3.0"
    # TitaNet-small: half the speaker confusions of WeSpeaker ResNet34 on AMI, 2x faster
    # (docs/BENCHMARKS.md); voiceprints are only comparable within one model
    embedding_model: str = "nemo-titanet-small"
    # Clustering distance; lower -> more speakers. 0.5 for TitaNet: best purity with the
    # second pass on AMI and the synthetic radio play; 0.6 scored higher there but merged
    # two clearly different voices in a short dialogue (docs/BENCHMARKS.md)
    diarize_threshold: float = 0.5
    live_threshold: float = 0.4  # live labels: similarity to join a speaker; higher -> more
    live_threshold_narrowband: float = 0.6  # the same for telephone-band audio
    resegment: bool = True  # second pass: re-cluster transcript phrases (fast dialogue)
    # Without --speakers: "estimate" (never merges voices; may over-split - merge in the
    # app) or "silhouette" (best for meetings with distinct voices; can merge similar ones)
    speaker_count: str = "estimate"
    num_threads: int = 0  # 0 = half the CPU cores
    identify_known: bool = True  # label speakers that match stored voiceprints
    store_voiceprints: bool = True  # False: never persist embeddings (sessions or DB)
    encrypt_voiceprints: bool = False  # encrypt at rest (key in OS keyring); 'secure' extra
    # Calibrated with `python -m transcribe.bench.calibrate --sweep` (TitaNet-small,
    # synthetic VCTK voices): 0 wrong labels on same/other-mic/phone channels (strangers
    # score <= 0.66); a phone-like channel mostly yields suggestions instead of labels.
    match_threshold_high: float = 0.75  # auto-label at/above (and margin satisfied)
    match_threshold_low: float = 0.60  # suggest ("Is this X?") at/above
    match_margin: float = 0.10  # required lead over the best competing candidate
    max_exemplars_per_person: int = 30


@dataclass
class LiveConfig:
    step: float = 1.0  # seconds of new audio between decodes
    end_silence: float = 0.4  # gap between (padded) VAD regions that ends an utterance
    max_buffer: float = 12.0
    beam_size: int = 1  # live decoding favours latency; refinement uses asr.beam_size
    refine: bool = True  # re-run full offline pipeline when the session stops
    refine_model: str = ""  # Whisper model for refinement; "" = same as asr.model


@dataclass
class UiConfig:
    language: str = "auto"  # auto (system language) | en | de - applied at the next start


def vocabulary(config: Config):
    from transcribe.core.vocabulary import Vocabulary

    return Vocabulary.from_config(config.asr.vocabulary, config.asr.corrections)


def required_models(config: Config) -> tuple[list[str], list[str]]:
    """(Whisper models, other assets) the configuration needs.

    whisper.cpp models are listed as ``ggml:<name>`` (core/models.py downloads both kinds).
    """
    whisper = [config.asr.model] + ([config.live.refine_model] if config.live.refine_model
                                    else [])
    if config.asr.engine == "whisper.cpp":
        whisper = [m if m.endswith(".bin") else f"ggml:{m}" for m in whisper]
    return whisper, [config.speakers.segmentation_model, config.speakers.embedding_model]


@dataclass
class Config:
    asr: AsrConfig = field(default_factory=AsrConfig)
    speakers: SpeakerConfig = field(default_factory=SpeakerConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    sessions_dir: str = ""

    def resolved_sessions_dir(self) -> Path:
        return Path(self.sessions_dir) if self.sessions_dir else data_dir() / "sessions"

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or config_dir() / "config.toml"
        if not path.exists():
            return cls()
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(
            asr=_build(AsrConfig, raw.get("asr", {})),
            speakers=_build(SpeakerConfig, raw.get("speakers", {})),
            live=_build(LiveConfig, raw.get("live", {})),
            ui=_build(UiConfig, raw.get("ui", {})),
            sessions_dir=raw.get("sessions_dir", ""),
        )

    def save(self, path: Path | None = None) -> None:
        path = path or config_dir() / "config.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(asdict(self)), encoding="utf-8")


def _build[T](cls: type[T], raw: dict) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**raw)
