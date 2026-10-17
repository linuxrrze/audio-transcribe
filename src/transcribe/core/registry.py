"""Plugin registry backed by Python entry points.

Groups: transcribe.sources, transcribe.asr, transcribe.embedders,
transcribe.diarizers, transcribe.exporters.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

GROUPS = (
    "transcribe.sources",
    "transcribe.asr",
    "transcribe.embedders",
    "transcribe.diarizers",
    "transcribe.exporters",
)

# Built-in plugins ("module:attr"), also declared as entry points in pyproject.toml. Kept
# here so frozen (PyInstaller) builds work even when package metadata is not bundled.
BUILTINS: dict[str, dict[str, str]] = {
    "transcribe.sources": {"file": "transcribe.audio.file:FileSource",
                           "mic": "transcribe.audio.mic:MicSource"},
    "transcribe.asr": {"faster-whisper": "transcribe.asr.faster_whisper:FasterWhisperEngine",
                       "whisper.cpp": "transcribe.asr.whisper_cpp:WhisperCppEngine"},
    "transcribe.embedders": {"sherpa-onnx": "transcribe.speakers.embedder:SherpaEmbedder"},
    "transcribe.diarizers": {"sherpa-onnx": "transcribe.diarize.sherpa_offline:SherpaDiarizer"},
    "transcribe.exporters": {"txt": "transcribe.export.txt:TxtExporter",
                             "md": "transcribe.export.md:MarkdownExporter",
                             "srt": "transcribe.export.srt:SrtExporter",
                             "vtt": "transcribe.export.srt:VttExporter",
                             "json": "transcribe.export.json:JsonExporter"},
}

_overrides: dict[str, dict[str, Any]] = {g: {} for g in GROUPS}


def register(group: str, name: str, obj: Any) -> None:
    """Register a plugin at runtime (takes precedence over entry points)."""
    _check_group(group)
    _overrides[group][name] = obj


def available(group: str) -> list[str]:
    _check_group(group)
    names = ({ep.name for ep in entry_points(group=group)} | set(_overrides[group])
             | set(BUILTINS[group]))
    return sorted(names)


def load(group: str, name: str) -> Any:
    _check_group(group)
    if name in _overrides[group]:
        return _overrides[group][name]
    for ep in entry_points(group=group, name=name):
        return ep.load()
    if target := BUILTINS[group].get(name):
        module, attr = target.split(":")
        return getattr(import_module(module), attr)
    raise KeyError(f"No plugin {name!r} in {group}; available: {available(group)}")


def _check_group(group: str) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown plugin group {group!r}")
