"""Audio of a single application (e.g. a meeting client) instead of a whole output device.

Selected as ``--output app:<name or pid>`` (and in the app's system-audio picker), so it
works with ``--source system`` and ``--source both`` (you on the mic, the meeting app's
participants on the "system" channel).

- Linux (PulseAudio, PipeWire via pipewire-pulse): the app's playback streams ("sink
  inputs", from ``pactl``) are recorded one by one with ``parec --monitor-stream=<index>``.
  That reads a copy of the stream and changes no routing: the user keeps hearing the
  app on whatever device it plays to, and a crash leaves nothing to restore (no null
  sink, no loopback, no moved streams that the server's stream-restore would remember).
- macOS 14.2+: the Core Audio tap helper with ``--pid`` (``--list`` lists processes
  with audio).
- Windows: needs the WASAPI process-loopback API (ActivateAudioInterfaceAsync with
  AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS) - not implemented yet; use the output device.

The app may not be playing yet, pause, restart its stream or quit mid-session: the
source keeps looking for it and delivers silence meanwhile, paced by the clock like a
real device.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, AudioSource, SourceInfo
from transcribe.i18n import _

log = logging.getLogger(__name__)

APP_PREFIX = "app:"  # --output app:<name or pid>


@dataclass(frozen=True)
class AudioApp:
    name: str
    pid: int | None
    playing: bool  # a stream is running (not paused/corked)

    @property
    def device(self) -> str:
        """The value for ``--output``."""
        return APP_PREFIX + self.name


# --- Linux: PulseAudio / PipeWire sink inputs --------------------------------------------
@dataclass(frozen=True)
class SinkInput:
    index: int
    app: str  # application.name (fallbacks: binary, media name)
    pid: int | None
    binary: str = ""
    corked: bool = False


def _sink_input(index: int, props: dict[str, str], corked: bool) -> SinkInput:
    binary = props.get("application.process.binary", "")
    name = props.get("application.name") or binary or props.get("media.name") or f"#{index}"
    pid = props.get("application.process.id", "")
    return SinkInput(index, name, int(pid) if pid.isdigit() else None, binary, corked)


def parse_pactl_json(text: str) -> list[SinkInput]:
    """``pactl -f json list sink-inputs`` (pactl 16+)."""
    out = []
    for item in json.loads(text or "[]"):
        props = {k: str(v) for k, v in (item.get("properties") or {}).items()}
        out.append(_sink_input(int(item["index"]), props, bool(item.get("corked", False))))
    return out


_HEADER = re.compile(r"^Sink Input #(\d+)")
_PROP = re.compile(r'^\s+([\w.\-]+) = "(.*)"$')


def parse_pactl_text(text: str) -> list[SinkInput]:
    """``pactl list sink-inputs`` in the C locale (older pactl without JSON output)."""
    out = []
    index: int | None = None
    props: dict[str, str] = {}
    corked = False
    for line in text.splitlines() + ["Sink Input #-1"]:  # sentinel flushes the last one
        if line.startswith("Sink Input #"):
            if index is not None:
                out.append(_sink_input(index, props, corked))
            m = _HEADER.match(line)
            index, props, corked = (int(m.group(1)) if m else None), {}, False
        elif line.strip().startswith("Corked:"):
            corked = line.split(":", 1)[1].strip() == "yes"
        elif m := _PROP.match(line):
            props[m.group(1)] = m.group(2).replace('\\"', '"')
    return out


def pulse_sink_inputs() -> list[SinkInput]:
    """Playback streams on the PulseAudio/PipeWire server; [] if there is none."""
    if shutil.which("pactl") is None:
        return []
    env = {**os.environ, "LC_ALL": "C"}  # the text output is translated otherwise
    try:
        res = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"],
                             capture_output=True, text=True, timeout=5, env=env)
        if res.returncode == 0:
            try:
                return parse_pactl_json(res.stdout)
            except (ValueError, KeyError, TypeError):
                pass  # some versions print invalid JSON (e.g. empty values): use the text
        res = subprocess.run(["pactl", "list", "sink-inputs"], capture_output=True,
                             text=True, timeout=5, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_pactl_text(res.stdout) if res.returncode == 0 else []


def match_app(spec: str, apps: list[SinkInput]) -> list[SinkInput]:
    """Streams of the app ``spec``: a pid, else the name or binary (case-insensitive;
    exact matches win over substrings, so "Chrome" does not pick "Chromium" as well)."""
    spec = spec.strip()
    if spec.isdigit():
        return [a for a in apps if a.pid == int(spec)]
    low = spec.lower()
    exact = [a for a in apps if low in (a.app.lower(), a.binary.lower())]
    return exact or [a for a in apps if low in a.app.lower() or low in a.binary.lower()]


def group_apps(inputs: list[SinkInput]) -> list[AudioApp]:
    """One entry per application (a browser can have several streams)."""
    apps: dict[str, AudioApp] = {}
    for s in inputs:
        prev = apps.get(s.app)
        apps[s.app] = AudioApp(s.app, s.pid if prev is None else prev.pid,
                               (not s.corked) or (prev is not None and prev.playing))
    return sorted(apps.values(), key=lambda a: (not a.playing, a.name.lower()))


# --- macOS: processes with audio, via the tap helper --------------------------------------
def mac_audio_processes() -> list[AudioApp]:
    """Processes known to Core Audio (``transcribe-audiotap --list``, macOS 14.2+)."""
    from transcribe.audio.system import audiotap_path

    helper = audiotap_path()
    if helper is None:
        return []
    try:
        res = subprocess.run([str(helper), "--list"], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    names = _process_names([int(r["pid"]) for r in rows])
    apps = [AudioApp(names.get(int(r["pid"])) or r.get("bundle") or str(r["pid"]),
                     int(r["pid"]), bool(r.get("output"))) for r in rows]
    return sorted(apps, key=lambda a: (not a.playing, a.name.lower()))


def _process_names(pids: list[int]) -> dict[int, str]:
    if not pids:
        return {}
    try:
        res = subprocess.run(["ps", "-o", "pid=,comm=", "-p", ",".join(map(str, pids))],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out = {}
    for line in res.stdout.splitlines():
        pid, _sep, comm = line.strip().partition(" ")
        if pid.isdigit():  # "/Applications/zoom.us.app/Contents/MacOS/zoom.us" -> zoom.us
            out[int(pid)] = os.path.basename(comm.strip())
    return out


def list_audio_apps() -> list[AudioApp]:
    """Applications that currently have audio output (playing ones first)."""
    system = platform.system()
    if system == "Linux":
        return group_apps(pulse_sink_inputs())
    if system == "Darwin":
        return mac_audio_processes()
    return []  # Windows: per-app capture not implemented (see module docstring)


def app_available(spec: str) -> bool:
    """Whether the application ``spec`` (name or pid) currently has audio output."""
    spec = spec.strip()
    low = spec.lower()
    return any(a.pid == int(spec) if spec.isdigit() else low in a.name.lower()
               for a in list_audio_apps())


# --- capture -------------------------------------------------------------------------------
class _JitterBuffer:
    """Samples of one stream; read only once ``prebuffer`` samples have arrived, so a
    stream delivering in bursts is not chopped up with silence."""

    def __init__(self, prebuffer: int, max_backlog: int) -> None:
        self.parts: deque[np.ndarray] = deque()
        self.size = 0
        self.primed = False
        self.ended = False  # the stream is gone: play out what is left
        self.prebuffer, self.max_backlog = prebuffer, max_backlog

    def push(self, samples: np.ndarray) -> None:
        self.parts.append(samples)
        self.size += len(samples)
        while self.size > self.max_backlog:  # the reader fell behind: drop the oldest
            self.size -= len(self.parts.popleft())

    def take(self, n: int) -> np.ndarray | None:
        if not self.primed:
            if self.size < self.prebuffer and not (self.ended and self.size):
                return None
            self.primed = True
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        while filled < n and self.parts:
            part = self.parts.popleft()
            use = min(n - filled, len(part))
            out[filled:filled + use] = part[:use]
            if use < len(part):
                self.parts.appendleft(part[use:])
            filled += use
            self.size -= use
        if filled < n:  # underrun: wait for a new prebuffer
            self.primed = False
        return out


class AppAudioSource:
    """Captures the streams of one application and sums them into one channel.

    ``targets()`` returns the capture commands for the app's current streams, keyed by
    stream; it is polled, so streams that appear later (the app starts playing) are
    picked up and streams that end (paused, quit) drop out. Frames are paced by the
    clock and silent while nothing is captured.
    """

    def __init__(self, spec: str, targets: Callable[[], dict[str, list[str]]],
                 channel_id: str = "system", poll_s: float = 1.0, frame_ms: int = 100,
                 prebuffer_ms: int = 200, retry_s: float = 5.0) -> None:
        self.spec, self.targets, self.channel_id = spec, targets, channel_id
        self.poll_s, self.retry_s = poll_s, retry_s
        self.frame = SAMPLE_RATE * frame_ms // 1000
        self._prebuffer = SAMPLE_RATE * prebuffer_ms // 1000
        self._buffers: dict[str, _JitterBuffer] = {}
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._failed: dict[str, float] = {}  # stream -> time of the next attempt
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.captured: set[str] = set()  # streams that delivered audio (diagnostics)

    def describe(self) -> SourceInfo:
        return SourceInfo(name=APP_PREFIX + self.spec, channels=[self.channel_id])

    def open(self) -> None:
        self._closed.clear()
        atexit.register(self.close)  # no orphaned capture processes
        threading.Thread(target=self._watch, daemon=True, name=f"app:{self.spec}").start()

    def _watch(self) -> None:
        while not self._closed.is_set():
            try:
                wanted = self.targets()
            except Exception as e:  # noqa: BLE001 - keep looking; the app may come back
                log.warning("listing streams of %s failed: %s", self.spec, e)
                wanted = {}
            now = time.monotonic()
            for key, cmd in wanted.items():
                with self._lock:  # (a stream that ended is played out before a restart)
                    busy = (key in self._procs or key in self._buffers
                            or self._failed.get(key, 0) > now)
                if not busy and not self._closed.is_set():
                    self._start(key, cmd)
            self._closed.wait(self.poll_s)

    def _start(self, key: str, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            log.warning("cannot capture %s stream %s: %s", self.spec, key, e)
            self._failed[key] = time.monotonic() + self.retry_s
            return
        with self._lock:
            self._procs[key] = proc
            self._buffers[key] = _JitterBuffer(self._prebuffer, SAMPLE_RATE)
        threading.Thread(target=self._read, args=(key, proc), daemon=True).start()

    def _read(self, key: str, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        got = 0
        try:
            while chunk := proc.stdout.read(self.frame * 4):
                samples = np.frombuffer(chunk[: len(chunk) - len(chunk) % 4],
                                        dtype="<f4").astype(np.float32)
                got += len(samples)
                with self._lock:
                    self._buffers[key].push(samples)
        finally:
            code = proc.wait()
            with self._lock:
                self._procs.pop(key, None)
                self._buffers[key].ended = True  # mix() drops it once played out
            if got:
                self.captured.add(key)
            elif code != 0 and not self._closed.is_set():  # e.g. stream already gone
                err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
                log.info("capture of %s stream %s ended (%s): %s", self.spec, key, code, err)
                self._failed[key] = time.monotonic() + self.retry_s

    def mix(self) -> np.ndarray:
        """The next frame: the sum of all streams, silence where none delivers."""
        out = np.zeros(self.frame, dtype=np.float32)
        with self._lock:
            for key, buf in list(self._buffers.items()):
                part = buf.take(self.frame)
                if part is not None:
                    out += part
                if buf.ended and not buf.size:
                    del self._buffers[key]
        return out

    def frames(self) -> Iterator[AudioFrame]:
        step = self.frame / SAMPLE_RATE
        start, n = time.monotonic(), 0
        while not self._closed.is_set():
            delay = start + (n + 1) * step - time.monotonic()
            if delay > 0 and self._closed.wait(delay):
                break
            yield AudioFrame(self.mix(), n * step, self.channel_id)
            n += 1

    def close(self) -> None:
        self._closed.set()
        atexit.unregister(self.close)
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()


def pulse_targets(spec: str,
                  list_inputs: Callable[[], list[SinkInput]] = pulse_sink_inputs
                  ) -> Callable[[], dict[str, list[str]]]:
    def targets() -> dict[str, list[str]]:
        return {str(s.index): ["parec", f"--monitor-stream={s.index}", "--format=float32le",
                               f"--rate={SAMPLE_RATE}", "--channels=1", "--raw",
                               "--latency-msec=100"]
                for s in match_app(spec, list_inputs())}
    return targets


def mac_targets(spec: str, helper: str,
                list_apps: Callable[[], list[AudioApp]] = mac_audio_processes
                ) -> Callable[[], dict[str, list[str]]]:
    def targets() -> dict[str, list[str]]:
        if spec.isdigit():
            pids = [int(spec)]
        else:
            low = spec.lower()
            apps = list_apps()
            pids = [a.pid for a in apps if a.name.lower() == low and a.pid is not None] or [
                a.pid for a in apps if low in a.name.lower() and a.pid is not None]
        return {str(p): [helper, "--rate", str(SAMPLE_RATE), "--pid", str(p)] for p in pids}
    return targets


def app_source(spec: str, channel_id: str = "system") -> AudioSource:
    """Capture the application ``spec`` (name or pid) - see the module docstring."""
    spec = spec.strip()
    if not spec:
        raise ValueError(_("Give an application name or process id: app:<name or pid>"))
    system = platform.system()
    if system == "Linux":
        for tool in ("pactl", "parec"):
            if shutil.which(tool) is None:
                raise RuntimeError(_("Capturing an application on Linux needs `{tool}` "
                                     "(pulseaudio-utils; works with PipeWire as well)")
                                   .format(tool=tool))
        return AppAudioSource(spec, pulse_targets(spec), channel_id)
    if system == "Darwin":
        from transcribe.audio.system import audiotap_path

        helper = audiotap_path()
        if helper is None:
            raise RuntimeError(_("Capturing an application on macOS needs the "
                                 "transcribe-audiotap helper (macOS 14.2+)."))
        return AppAudioSource(spec, mac_targets(spec, str(helper)), channel_id,
                              poll_s=2.0)
    raise RuntimeError(_("Capturing a single application is not supported on {system} yet; "
                         "capture the whole output device instead (--output, see `tx "
                         "devices`).").format(system=system))
