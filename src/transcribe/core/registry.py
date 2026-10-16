"""Plugin registry backed by Python entry points.

Groups: transcribe.sources, transcribe.asr, transcribe.embedders,
transcribe.diarizers, transcribe.exporters.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

GROUPS = (
    "transcribe.sources",
    "transcribe.asr",
    "transcribe.embedders",
    "transcribe.diarizers",
    "transcribe.exporters",
)

_overrides: dict[str, dict[str, Any]] = {g: {} for g in GROUPS}


def register(group: str, name: str, obj: Any) -> None:
    """Register a plugin at runtime (takes precedence over entry points)."""
    _check_group(group)
    _overrides[group][name] = obj


def available(group: str) -> list[str]:
    _check_group(group)
    names = {ep.name for ep in entry_points(group=group)} | set(_overrides[group])
    return sorted(names)


def load(group: str, name: str) -> Any:
    _check_group(group)
    if name in _overrides[group]:
        return _overrides[group][name]
    for ep in entry_points(group=group, name=name):
        return ep.load()
    raise KeyError(f"No plugin {name!r} in {group}; available: {available(group)}")


def _check_group(group: str) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown plugin group {group!r}")
