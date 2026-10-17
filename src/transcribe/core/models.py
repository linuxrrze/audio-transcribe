"""Downloadable model assets (cached in the platform user cache dir).

Whisper models are fetched by faster-whisper from the Hugging Face hub into
``models_dir() / "whisper"``, ggml models for whisper.cpp into ``models_dir() /
"whisper.cpp"``; everything else is listed in ``ASSETS``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from transcribe.core.config import APP_NAME

_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    kind: str  # segmentation | embedding | tts
    url: str
    sha256: str
    main_file: str  # path of the primary file relative to the asset dir
    description: str = ""

    @property
    def is_archive(self) -> bool:
        return self.url.endswith(".tar.bz2")


ASSETS: dict[str, ModelAsset] = {
    a.name: a
    for a in [
        ModelAsset(
            "pyannote-segmentation-3.0", "segmentation",
            f"{_SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
            "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
            "pyannote segmentation 3.0 (MIT), ONNX export",
        ),
        ModelAsset(
            "wespeaker-resnet34-lm", "embedding",
            f"{_SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
            "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
            "wespeaker_en_voxceleb_resnet34_LM.onnx",
            "WeSpeaker ResNet34-LM, VoxCeleb (256-dim)",
        ),
        ModelAsset(
            "nemo-titanet-small", "embedding",
            f"{_SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx",
            "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
            "nemo_en_titanet_small.onnx",
            "NVIDIA NeMo TitaNet-small (CC-BY-4.0, 192-dim); default",
        ),
        ModelAsset(
            "piper-de-thorsten", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-thorsten-medium.tar.bz2",
            "50487d9c95fdf2191f31d2588569381063ba1591dcd4c7d4bdd30f12b2191714",
            "vits-piper-de_DE-thorsten-medium/de_DE-thorsten-medium.onnx",
            "Piper German voice (test fixtures only)",
        ),
        ModelAsset(
            "piper-en-amy", "tts",
            f"{_SHERPA}/tts-models/vits-piper-en_US-amy-low.tar.bz2",
            "c70f5284a09a7fd4ed203b39b2ff51cac1432b422b852eb647b481dade3cf639",
            "vits-piper-en_US-amy-low/en_US-amy-low.onnx",
            "Piper English voice (test fixtures only)",
        ),
        ModelAsset(
            "piper-de-kerstin", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-kerstin-low.tar.bz2",
            "4bfe121d5f690a3b87acecd7ae1e70b48f4589ce6ae9f45a1809e6eb19fad97b",
            "vits-piper-de_DE-kerstin-low/de_DE-kerstin-low.onnx",
            "Piper German voice (test fixtures only)",
        ),
        ModelAsset(
            "piper-en-vctk", "tts",
            f"{_SHERPA}/tts-models/vits-piper-en_GB-vctk-medium.tar.bz2",
            "abafd35bdab0a72a3c6b947228ae3cccdf3624db83313c77d1abb5cbc75e1f64",
            "vits-piper-en_GB-vctk-medium/en_GB-vctk-medium.onnx",
            "Piper English multi-speaker (109 VCTK voices; calibration only)",
        ),
    ]
}

ProgressFn = Callable[[int, int], None]  # (bytes_done, bytes_total)


def models_dir() -> Path:
    env = os.environ.get("TRANSCRIBE_MODELS_DIR")
    return Path(env) if env else Path(platformdirs.user_cache_dir(APP_NAME)) / "models"


def asset_dir(name: str) -> Path:
    return models_dir() / name


def is_present(name: str) -> bool:
    return (asset_dir(name) / ASSETS[name].main_file).exists()


def ensure(name: str, progress: ProgressFn | None = None) -> Path:
    """Return the path of the asset's main file, downloading and verifying it if needed."""
    asset = ASSETS[name]
    target = asset_dir(name)
    main = target / asset.main_file
    if main.exists():
        return main
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{name}.part"
    _download(asset.url, tmp, progress)
    digest = _sha256(tmp)
    if digest != asset.sha256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}: got {digest}")
    staging = target.parent / f".{name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    if asset.is_archive:
        with tarfile.open(tmp, "r:bz2") as tar:
            tar.extractall(staging, filter="data")
        tmp.unlink()
    else:
        tmp.replace(staging / asset.main_file)
    shutil.rmtree(target, ignore_errors=True)
    staging.replace(target)
    if not main.exists():
        raise RuntimeError(f"{name}: expected file {asset.main_file} missing after download")
    return main


def _download(url: str, dest: Path, progress: ProgressFn | None) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --- Whisper (Hugging Face hub, cached by faster-whisper) ---------------------------------
def whisper_dir() -> Path:
    return models_dir() / "whisper"


def whisper_present(model: str) -> bool:
    """True if the faster-whisper model is in the local cache (or ``model`` is a path)."""
    if Path(model).is_dir():
        return True
    from faster_whisper.utils import download_model

    try:
        download_model(model, local_files_only=True, cache_dir=str(whisper_dir()))
    except Exception:  # noqa: BLE001 - huggingface_hub raises various "not cached" errors
        return False
    return True


def ensure_whisper(model: str) -> Path:
    from faster_whisper.utils import download_model

    return Path(download_model(model, cache_dir=str(whisper_dir())))


# --- ggml models for whisper.cpp (Hugging Face, ggerganov/whisper.cpp) ------------------
GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"


def ggml_path(model: str) -> Path:
    return models_dir() / "whisper.cpp" / f"ggml-{model}.bin"


def ensure_ggml(model: str, progress: ProgressFn | None = None) -> Path:
    path = ggml_path(model)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            _download(GGML_URL.format(name=model), tmp, progress)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Cannot download whisper.cpp model {model!r}: {e}") from e
        tmp.replace(path)
    return path


def model_present(name: str) -> bool:
    """``whisper:<m>``/``<m>`` (faster-whisper), ``ggml:<m>`` (whisper.cpp) or a model file."""
    if name.startswith("ggml:"):
        return ggml_path(name[5:]).exists()
    if name.endswith(".bin"):
        return Path(name).exists()
    return whisper_present(name.removeprefix("whisper:"))


def missing_models(whisper_models: list[str], assets: list[str]) -> list[str]:
    """Names of models that still need downloading (``whisper:<name>``, ``ggml:<name>``
    or asset names)."""
    out = [m if m.startswith("ggml:") else f"whisper:{m}"
           for m in dict.fromkeys(whisper_models)
           if m and not m.endswith(".bin") and not model_present(m)]
    return out + [a for a in assets if not is_present(a)]


def download(name: str, progress: ProgressFn | None = None) -> Path:
    """Download ``whisper:<model>``, ``ggml:<model>`` or a named asset."""
    if name.startswith("whisper:"):
        return ensure_whisper(name.split(":", 1)[1])
    if name.startswith("ggml:"):
        return ensure_ggml(name.split(":", 1)[1], progress)
    return ensure(name, progress)
