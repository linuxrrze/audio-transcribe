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
    engine: str = "faster-whisper"
    model: str = "small"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto | int8 | float16 | ...
    languages: list[str] = field(default_factory=lambda: ["de", "en"])
    beam_size: int = 5
    cpu_threads: int = 0  # 0 = library default


@dataclass
class SpeakerConfig:
    diarizer: str = "sherpa-onnx"
    segmentation_model: str = "pyannote-segmentation-3.0"
    embedding_model: str = "wespeaker-resnet34-lm"
    diarize_threshold: float = 0.5  # clustering distance; lower -> more speakers
    num_threads: int = 0  # 0 = half the CPU cores
    cluster_threshold: float = 0.55
    identify_known: bool = True
    store_voiceprints: bool = True
    match_threshold_high: float = 0.65
    match_threshold_low: float = 0.50


@dataclass
class Config:
    asr: AsrConfig = field(default_factory=AsrConfig)
    speakers: SpeakerConfig = field(default_factory=SpeakerConfig)
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
