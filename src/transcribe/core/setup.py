"""First-run setup, shared by `tx setup` and the app's setup wizard: detect the hardware,
recommend Whisper models for it, write the choices to config.toml.

Recommendations follow README/docs/BENCHMARKS.md: ``small`` (int8) on the CPU (~3 s live
latency on 16 cores), ``base`` live with ``small`` afterwards on smaller machines,
``large-v3-turbo`` on an NVIDIA GPU, and ``small`` live + ``large-v3-turbo`` afterwards
with the whisper.cpp engine on other GPUs (Vulkan/Metal).
"""

from __future__ import annotations

import copy
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from transcribe.core.config import Config, config_dir, required_models
from transcribe.i18n import N_, _

# Below this many (logical) cores, `small` falls behind live: base live, small afterwards
MIN_CORES_FOR_SMALL = 8

# Download sizes (MB) of the Whisper models, for the choice before downloading; the
# download itself reports the real byte counts. faster-whisper (CTranslate2) and ggml
# files are about the same size.
WHISPER_SIZES_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530,
                    "large-v3-turbo": 1620, "large-v3": 3090}


@dataclass(frozen=True)
class Hardware:
    cpu_cores: int
    cuda_devices: int = 0  # NVIDIA GPUs usable by faster-whisper (CTranslate2)
    gpu_build: str = ""  # whisper.cpp GPU build found: vulkan | metal | cuda | ""
    system: str = ""  # Linux | Windows | Darwin

    @property
    def has_gpu(self) -> bool:
        return self.cuda_devices > 0 or bool(self.gpu_build)


def detect_hardware() -> Hardware:
    """CPU cores, CUDA devices and whisper.cpp GPU builds; never raises."""
    try:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001 - missing CUDA libraries etc. mean "no CUDA"
        cuda = 0
    return Hardware(os.cpu_count() or 1, cuda, whispercpp_gpu_build(), platform.system())


def whispercpp_gpu_build() -> str:
    """The GPU backend of a whisper.cpp build from scripts/build_whispercpp.sh, if any."""
    from transcribe.asr.whisper_cpp import SERVER_NAME, build_dir

    exe = SERVER_NAME + (".exe" if os.name == "nt" else "")
    for backend in ("vulkan", "metal", "cuda"):
        base = build_dir() / f"build-{backend}" / "bin"
        if (base / exe).is_file() or (base / "Release" / exe).is_file():
            return backend
    return ""


@dataclass(frozen=True)
class Profile:
    """A model choice: engine, live model and the model for the pass after a session."""

    key: str
    label: str  # N_() - translate when shown
    speed: str  # N_() - expected speed, from docs/BENCHMARKS.md
    engine: str = "faster-whisper"
    model: str = "small"
    refine_model: str = ""  # "" = same as model

    def applied(self, config: Config) -> Config:
        """A copy of ``config`` with this profile's ASR settings. The device stays as it
        is: "auto" (the default) uses the GPU whenever the engine can."""
        c = copy.deepcopy(config)
        c.asr.engine, c.asr.model = self.engine, self.model
        c.live.refine_model = self.refine_model
        return c


PROFILES = {p.key: p for p in [
    Profile("fast", N_("Fast: base live, small for the final transcript"),
            N_("For computers with few cores: live text appears quickly; the final "
               "transcript after a session uses small."),
            model="base", refine_model="small"),
    Profile("balanced", N_("Balanced: small"),
            N_("Live text ~3 s after speech; a 1-hour file takes ~10 min (16 cores)."),
            model="small"),
    Profile("accurate", N_("Accurate: small live, large-v3-turbo for the final transcript"),
            N_("Live like Balanced; the final transcript is more accurate but takes "
               "~25 min per hour of audio on the CPU."),
            model="small", refine_model="large-v3-turbo"),
    Profile("cuda", N_("NVIDIA GPU: large-v3-turbo"),
            N_("Most accurate, fast live and for files (CUDA)."),
            model="large-v3-turbo"),
    Profile("gpu", N_("GPU via whisper.cpp: small live, large-v3-turbo for the final "
                      "transcript"),
            N_("Live text ~2 s after speech; a 1-hour file takes ~6 min (Radeon 780M)."),
            engine="whisper.cpp", model="small", refine_model="large-v3-turbo"),
]}


LANGUAGE_CHOICES = {  # transcription languages: config value -> label (N_)
    "de,en": N_("German and English (detected automatically)"),
    "de": N_("German only"),
    "en": N_("English only"),
}


def profiles_for(hw: Hardware) -> list[Profile]:
    """The choices that make sense on this hardware (CPU ones always)."""
    keys = ["fast", "balanced", "accurate"]
    if hw.cuda_devices:
        keys.append("cuda")
    if hw.gpu_build:
        keys.append("gpu")
    return [PROFILES[k] for k in keys]


def recommend(hw: Hardware) -> Profile:
    if hw.cuda_devices:  # faster-whisper on CUDA: turbo is fast enough for everything
        return PROFILES["cuda"]
    if hw.gpu_build:
        return PROFILES["gpu"]
    return PROFILES["balanced" if hw.cpu_cores >= MIN_CORES_FOR_SMALL else "fast"]


def matching_profile(config: Config) -> Profile | None:
    """The profile the configuration currently uses, if it is one of them."""
    a = config.asr
    for p in PROFILES.values():
        if (p.engine, p.model, p.refine_model) == (a.engine, a.model, config.live.refine_model):
            return p
    return None


def size_mb(name: str) -> float:
    """Approximate download size of ``whisper:<m>``, ``ggml:<m>``, ``<m>`` or an asset;
    0 if unknown (e.g. a model given as a path or Hugging Face repo)."""
    from transcribe.core.models import ASSETS

    if name in ASSETS:
        return ASSETS[name].size_mb
    return WHISPER_SIZES_MB.get(name.split(":", 1)[-1], 0.0)


def needed_models(config: Config) -> list[str]:
    """Everything ``config`` needs, as download names (present or not)."""
    whisper, assets = required_models(config)
    return [m if m.startswith("ggml:") else f"whisper:{m}"
            for m in dict.fromkeys(whisper) if m and not m.endswith(".bin")] + assets


def profile_size_mb(profile: Profile, config: Config) -> float:
    """Download size of everything the profile needs (speaker models included)."""
    return sum(size_mb(n) for n in needed_models(profile.applied(config)))


def apply_choices(config: Config, profile: Profile | None = None,
                  ui_language: str | None = None, languages: list[str] | None = None,
                  store_voiceprints: bool | None = None,
                  identify_known: bool | None = None) -> Config:
    """Write the setup choices into ``config`` (in place; ``None`` keeps a value)."""
    if profile is not None:
        p = profile.applied(config)
        config.asr, config.live = p.asr, p.live
    if ui_language is not None:
        config.ui.language = ui_language
    if languages:
        config.asr.languages = list(languages)
    if store_voiceprints is not None:
        config.speakers.store_voiceprints = store_voiceprints
    if identify_known is not None:
        config.speakers.identify_known = identify_known
    return config


def format_size(mb: float) -> str:
    if not mb:
        return "?"
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def hardware_text(hw: Hardware) -> str:
    if hw.cuda_devices:
        gpu = _("NVIDIA GPU (CUDA)")
    elif hw.gpu_build:
        gpu = _("GPU via whisper.cpp ({backend})").format(backend=hw.gpu_build)
    else:
        gpu = _("no usable GPU found")
    return _("This computer: {cores} CPU cores, {gpu}.").format(cores=hw.cpu_cores, gpu=gpu)


NO_GPU_HINT = N_("AMD, Intel and Apple GPUs work with the whisper.cpp engine once it is "
                 "built (scripts/build_whispercpp.sh).")


VOICEPRINT_NOTE = N_("Voiceprints are biometric data (GDPR Art. 9): they stay on this "
                     "computer, need the person's consent and can be deleted at any time.")


def config_path() -> Path:
    return config_dir() / "config.toml"


def needs_setup(config: Config, path: Path | None = None) -> bool:
    """First run: no config file yet, or models the configuration needs are missing."""
    from transcribe.core.models import missing_models

    if not (path or config_path()).exists():
        return True
    return bool(missing_models(*required_models(config)))
