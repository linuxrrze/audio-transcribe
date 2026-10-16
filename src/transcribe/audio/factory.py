"""Build the audio source for a live session (shared by CLI and GUI)."""

from __future__ import annotations

from pathlib import Path

from transcribe.audio.base import AudioSource

SOURCES = ("mic", "system", "both")


def build_source(kind: str = "mic", input_device: str | None = None,
                 output_device: str | None = None, mix: bool = False,
                 file: Path | None = None, realtime: bool = True) -> AudioSource:
    """``mic`` | ``system`` (what the computer plays) | ``both``.

    ``both`` records microphone and system audio as separate channels ("mic", "system":
    speaker separation for free, echo suppression), or summed into one channel with
    ``mix`` (one ASR stream - cheaper on CPU). ``file`` replays a recording instead.
    """
    if file is not None:
        from transcribe.audio.file import FileSource

        return FileSource(file, realtime=realtime)
    if kind not in SOURCES:
        raise ValueError(f"Unknown source {kind!r}; choose from {SOURCES}")
    from transcribe.audio.mic import MicSource
    from transcribe.audio.system import system_source

    if kind == "mic":
        return MicSource(input_device)
    if kind == "system":
        return system_source(output_device)
    from transcribe.audio.mixer import MixDownSource, MixerSource

    mixer = MixerSource([MicSource(input_device, channel_id="mic"),
                         system_source(output_device, channel_id="system")])
    return MixDownSource(mixer) if mix else mixer
