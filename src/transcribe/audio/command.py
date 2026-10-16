"""Audio from a helper process that writes raw 16 kHz mono float32 PCM to stdout.

Used for PulseAudio/PipeWire monitors (``parec``) and the macOS Core Audio tap helper;
keeping capture in a separate process isolates platform APIs and their permissions.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Iterator

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, SourceInfo


class CommandSource:
    def __init__(self, cmd: list[str], channel_id: str = "system", name: str | None = None,
                 frame_ms: int = 100) -> None:
        self.cmd = cmd
        self.channel_id = channel_id
        self.name = name or f"cmd:{cmd[0]}"
        self.frame_bytes = SAMPLE_RATE * frame_ms // 1000 * 4
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: list[str] = []

    def describe(self) -> SourceInfo:
        return SourceInfo(name=self.name, channels=[self.channel_id])

    def open(self) -> None:
        if shutil.which(self.cmd[0]) is None:
            raise RuntimeError(f"Capture helper not found: {self.cmd[0]}")
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr = (self._stderr + [line.decode(errors="replace").rstrip()])[-20:]

    def frames(self) -> Iterator[AudioFrame]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("open() must be called first")
        t = 0.0
        while chunk := proc.stdout.read(self.frame_bytes):
            usable = len(chunk) - len(chunk) % 4
            samples = np.frombuffer(chunk[:usable], dtype="<f4").astype(np.float32)
            yield AudioFrame(samples, t, self.channel_id)
            t += len(samples) / SAMPLE_RATE
        code = proc.wait()
        if code not in (0, -15) and self._proc is not None:  # -15: we terminated it
            raise RuntimeError(f"{self.cmd[0]} exited with {code}: "
                               + " | ".join(self._stderr[-3:]))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
