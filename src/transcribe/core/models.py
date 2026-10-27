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
import threading
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
    size_mb: float = 0.0  # download size, shown before downloading (0 = unknown)

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
            "pyannote segmentation 3.0 (MIT), ONNX export", 7,
        ),
        ModelAsset(
            "wespeaker-resnet34-lm", "embedding",
            f"{_SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
            "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
            "wespeaker_en_voxceleb_resnet34_LM.onnx",
            "WeSpeaker ResNet34-LM, VoxCeleb (256-dim)", 27,
        ),
        ModelAsset(
            "nemo-titanet-small", "embedding",
            f"{_SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx",
            "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
            "nemo_en_titanet_small.onnx",
            "NVIDIA NeMo TitaNet-small (CC-BY-4.0, 192-dim); default", 40,
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
            "piper-de-karlsson", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-karlsson-low.tar.bz2",
            "5ba231af059746ac2052c9b7d7eaf116e07003dc6948066ab4f40951dca260af",
            "vits-piper-de_DE-karlsson-low/de_DE-karlsson-low.onnx",
            "Piper German voice, male (benchmarks only)",
        ),
        ModelAsset(
            "piper-de-evak", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-eva_k-x_low.tar.bz2",
            "daec8658456d8e5ec714972f2e5a71f8a84d58b974db396f4eaeace0df06455a",
            "vits-piper-de_DE-eva_k-x_low/de_DE-eva_k-x_low.onnx",
            "Piper German voice, female (benchmarks only)",
        ),
        ModelAsset(
            "piper-de-ramona", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-ramona-low.tar.bz2",
            "79f985c10f86c4d58205b519abc488aa65eb713d54665ff729880a02e2625a42",
            "vits-piper-de_DE-ramona-low/de_DE-ramona-low.onnx",
            "Piper German voice, female (benchmarks only)",
        ),
        ModelAsset(
            "piper-de-pavoque", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-pavoque-low.tar.bz2",
            "5fcd4dad1c6c8362027d7359e54ddf482b7e6ac4af6751b5df0ea1899b698f54",
            "vits-piper-de_DE-pavoque-low/de_DE-pavoque-low.onnx",
            "Piper German voice, male (benchmarks only)",
        ),
        ModelAsset(
            "piper-de-miro", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-miro-high.tar.bz2",
            "8226bb45d3684286981a55b3e0937e373f37673036a7eee27fc51def2d8c926d",
            "vits-piper-de_DE-miro-high/de_DE-miro-high.onnx",
            "Piper German voice, male (benchmarks only)",
        ),
        ModelAsset(
            "piper-de-dii", "tts",
            f"{_SHERPA}/tts-models/vits-piper-de_DE-dii-high.tar.bz2",
            "64d46d39afb0f53a49871f5415259ffe474f443696148930b34a003612b69110",
            "vits-piper-de_DE-dii-high/de_DE-dii-high.onnx",
            "Piper German voice, female (benchmarks only)",
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


class Cancelled(Exception):
    """Raised by a progress callback to abort a download (see ``download_all``)."""


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
    try:
        _download(asset.url, tmp, progress)
    except BaseException:  # failed or cancelled: no partial file left behind
        tmp.unlink(missing_ok=True)
        raise
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


WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json",
                 "vocabulary.*"]  # what faster-whisper's download_model fetches


def ensure_whisper(model: str, progress: ProgressFn | None = None) -> Path:
    from faster_whisper.utils import download_model

    if progress is None or Path(model).is_dir():
        return Path(download_model(model, cache_dir=str(whisper_dir())))
    # faster-whisper's download (same repo, files and cache layout), with byte progress
    import huggingface_hub
    from faster_whisper.utils import _MODELS

    repo = model if "/" in model else _MODELS.get(model)
    if repo is None:
        raise ValueError(f"Unknown Whisper model {model!r}")
    return Path(huggingface_hub.snapshot_download(
        repo, cache_dir=str(whisper_dir()), allow_patterns=WHISPER_FILES,
        tqdm_class=hub_progress(progress)))


def hub_progress(progress: ProgressFn) -> type:
    """A tqdm class for ``snapshot_download`` that reports bytes to ``progress``.

    huggingface_hub (1.x) sums the bytes of all files into bars of unit "B" made from
    this class; older versions only count files, so no bytes are reported (the total stays
    unknown). An exception from ``progress`` (``Cancelled``) aborts the download; the Xet
    backend also prints it as a traceback (``download_all`` checks again per file).
    """
    from tqdm.auto import tqdm  # type: ignore[import-untyped]

    state = {"done": 0, "total": 0}

    class _Bar(tqdm):  # type: ignore[type-arg]
        def __init__(self, *args, **kwargs) -> None:
            self._bytes = kwargs.get("unit") == "B"
            self._done = float(kwargs.get("initial") or 0)
            kwargs.pop("name", None)  # huggingface_hub's own tqdm option
            kwargs["disable"] = True  # no console output
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            self._done += n or 0
            if self._bytes:  # several byte bars (transfer, written): the furthest one
                state["done"] = max(state["done"], int(self._done))
                state["total"] = max(state["total"], int(self.total or 0))
                progress(state["done"], state["total"])
            return None

    return _Bar


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
        return ensure_whisper(name.split(":", 1)[1], progress)
    if name.startswith("ggml:"):
        return ensure_ggml(name.split(":", 1)[1], progress)
    return ensure(name, progress)


def download_all(names: list[str], progress: Callable[[str, int, int], None] | None = None,
                 cancel: threading.Event | None = None) -> None:
    """Download ``names`` one after the other; ``progress(name, done, total)`` reports
    bytes (``done == total`` once a file is complete). Setting ``cancel`` aborts with
    ``Cancelled``, during a file or before the next one."""
    for name in names:
        if cancel is not None and cancel.is_set():
            raise Cancelled(name)

        size = [0]

        def report(done: int, total: int, name: str = name, size: list[int] = size) -> None:
            if cancel is not None and cancel.is_set():
                raise Cancelled(name)
            size[0] = max(size[0], total, done)
            if progress:
                progress(name, done, total)

        download(name, report)
        if cancel is not None and cancel.is_set():  # e.g. Xet finished the file anyway
            raise Cancelled(name)
        if progress:  # complete (also when nothing was reported, e.g. already cached)
            progress(name, size[0] or 1, size[0] or 1)
