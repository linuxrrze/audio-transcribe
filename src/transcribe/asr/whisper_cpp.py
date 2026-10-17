"""whisper.cpp ASR backend: GPU acceleration beyond NVIDIA (Vulkan for AMD/Intel, Metal).

The engine runs whisper.cpp's ``whisper-server`` as a child process, so the model stays
loaded between calls (live mode decodes every second), and talks to it over HTTP on
localhost. Build it with ``scripts/build_whispercpp.sh`` or point ``asr.whispercpp_server``
at any whisper-server binary (v1.8+). Models are ggml files from the whisper.cpp Hugging
Face repository, downloaded on first use.

VAD is done here, not by the server: with ``--vad`` whisper-server v1.9.5 returns word
timestamps on the shortened (silence-free) timeline. So the engine cuts the speech regions
out itself (Silero, same settings as faster-whisper's VAD filter), sends only those and
maps the times back. Without VAD Whisper invents text in silence (loops of one sentence on
AMI, see docs/BENCHMARKS.md).
"""

from __future__ import annotations

import atexit
import copy
import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
import weakref
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcribe.asr.base import ProgressFn
from transcribe.asr.faster_whisper import STYLE_PROMPTS
from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config
from transcribe.storage.transcript_model import Word

log = logging.getLogger(__name__)

SERVER_NAME = "whisper-server"
_PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")
# Speech regions as in faster-whisper's VAD filter defaults (merge pauses < 2 s, pad 0.4 s)
_VAD_MIN_SILENCE_MS = 2000
_VAD_PAD_MS = 400
_WINDOW_S = 28.0  # speech per request (Whisper's window is 30 s)
_MAX_PROMPT_TOKENS = 220  # whisper.cpp keeps the last 223 (n_text_ctx / 2 - 1)
_STARTUP_TIMEOUT = 300.0  # large models on slow disks; GPU shader compilation on first run


def build_dir() -> Path:
    """Where scripts/build_whispercpp.sh puts its builds."""
    return Path.home() / ".cache" / "transcribe" / "whisper.cpp"


def find_server(configured: str = "", device: str = "auto") -> Path:
    """The whisper-server binary: configured path, own builds (GPU first), then PATH."""
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"asr.whispercpp_server: {path} not found")
        return path
    exe = SERVER_NAME + (".exe" if os.name == "nt" else "")
    builds = ["build-cpu"] if device == "cpu" else ["build-vulkan", "build-metal",
                                                     "build-cuda", "build-cpu"]
    for build in builds:
        for path in (build_dir() / build / "bin" / exe,
                     build_dir() / build / "bin" / "Release" / exe):  # MSVC layout
            if path.is_file():
                return path
    if found := shutil.which(SERVER_NAME):
        return Path(found)
    raise FileNotFoundError(
        f"{SERVER_NAME} not found. Build it with scripts/build_whispercpp.sh or set "
        f"[asr] whispercpp_server = \"/path/to/{SERVER_NAME}\"")


def dtw_preset(model: str) -> str | None:
    """whisper.cpp's DTW alignment-head preset for a model name (None if unknown)."""
    name = Path(model).name.removeprefix("ggml-").removesuffix(".bin")
    name = re.sub(r"-q\d.*$", "", name).replace("-", ".")  # quantised: ggml-small-q5_1
    known = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
             "medium.en", "large.v1", "large.v2", "large.v3", "large.v3.turbo"}
    return name if name in known else None


# --- VAD: cut out speech, map times back ------------------------------------------------
@dataclass(frozen=True)
class _Piece:
    orig: float  # start in the original audio (s)
    cut: float  # start in the concatenated speech (s)
    length: float


def cut_speech(audio: np.ndarray,
               regions: list[tuple[float, float]]) -> tuple[np.ndarray, list[_Piece]]:
    """Concatenate the speech regions; return the audio and the time map."""
    parts: list[np.ndarray] = []
    pieces: list[_Piece] = []
    cut = 0.0
    for t0, t1 in regions:
        a, b = int(t0 * SAMPLE_RATE), min(len(audio), int(t1 * SAMPLE_RATE))
        if b <= a:
            continue
        parts.append(audio[a:b])
        pieces.append(_Piece(a / SAMPLE_RATE, cut, (b - a) / SAMPLE_RATE))
        cut += (b - a) / SAMPLE_RATE
    if not parts:
        return np.zeros(0, np.float32), []
    return np.concatenate(parts), pieces


def to_original(t: float, pieces: list[_Piece]) -> float:
    """Map a time on the concatenated timeline back to the original audio."""
    for p in pieces:
        if t < p.cut + p.length:
            return p.orig + max(0.0, t - p.cut)
    last = pieces[-1]
    return last.orig + last.length


# --- prompt length in tokens (whisper.cpp's own tokenizer) ------------------------------
_WORDS = re.compile(rb"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+"
                    rb"|\s+(?!\S)|\s+")


def read_vocab(model: Path) -> set[bytes]:
    """Token strings of a ggml Whisper model (empty set if the format is not recognised)."""
    import struct

    try:
        with open(model, "rb") as f:
            if f.read(4) != b"lmgg":  # magic 0x67676d6c, little endian
                return set()
            f.read(11 * 4)  # hyperparameters
            n_mel, n_fft = struct.unpack("<ii", f.read(8))
            f.seek(n_mel * n_fft * 4, os.SEEK_CUR)  # mel filters
            (n_vocab,) = struct.unpack("<i", f.read(4))
            vocab = set()
            for _ in range(n_vocab):
                (n,) = struct.unpack("<I", f.read(4))
                vocab.add(f.read(n))
            return vocab
    except (OSError, struct.error):
        return set()


def tokenize(text: str, vocab: set[bytes]) -> list[bytes]:
    """whisper.cpp's tokenizer: split into words, then greedy longest vocabulary match."""
    tokens = []
    for word in _WORDS.findall(text.encode("utf-8")):
        i = 0
        while i < len(word):
            for j in range(len(word), i, -1):
                if word[i:j] in vocab:
                    tokens.append(word[i:j])
                    i = j
                    break
            else:
                i += 1  # unknown byte: whisper.cpp skips it too
    return tokens


# --- whisper-server response -> words ---------------------------------------------------
def tokens_to_words(segments: list[dict], use_dtw: bool = False) -> list[Word]:
    """Merge verbose_json tokens (sub-words, times in s) into words.

    A token starting with a space opens a new word; other tokens (word pieces,
    punctuation) extend the current one. Confidence is the mean token probability, as in
    faster-whisper.

    Token timestamps (default) are rough (AMI: median start error 0.3 s), and the last
    token before a pause is stretched over it ("here." lasting 26 s), so durations are
    capped at twice the median. With ``use_dtw``, ``t_dtw`` (centiseconds) marks where a
    token *ends*: a word ends at its last spoken token and starts where the previous word
    ended, after a pause at most ``_dtw_length(word)`` before its end (AMI: median error
    0.06 s, like faster-whisper).
    """
    words: list[Word] = []
    ends: list[float] = []  # DTW end of each word's last spoken token
    text, t0, t1 = "", 0.0, 0.0
    probs: list[float] = []
    dtw_end = -1.0

    def flush() -> None:
        if text.strip() and not re.fullmatch(r"\s*\[[^\]]*\]\s*", text):  # [BLANK_AUDIO]
            words.append(Word(text=text, t0=t0, t1=max(t1, t0 + 0.01),
                              confidence=float(np.mean(probs))))
            ends.append(dtw_end)

    for seg in segments:
        for tok in seg.get("words") or []:
            piece = tok.get("word", "")
            if not piece:
                continue
            start, end = float(tok.get("start", 0.0)), float(tok.get("end", 0.0))
            spoken = re.search(r"\w", piece) is not None  # punctuation is not spoken
            if piece.startswith(" ") or not text:
                flush()
                text, t0, t1, probs, dtw_end = piece, start, end, [], -1.0
            else:
                text += piece
                if spoken:
                    t1 = max(t1, end)
            if spoken and tok.get("t_dtw", -1) >= 0:
                dtw_end = tok["t_dtw"] / 100.0
            probs.append(float(tok.get("probability", 0.0)))
    flush()
    if use_dtw and words and all(e >= 0 for e in ends):
        prev = 0.0
        for w, end in zip(words, ends, strict=True):
            w.t1 = max(end, prev + 0.01)
            w.t0 = min(max(prev, end - _dtw_length(w.text)), w.t1 - 0.01)
            prev = w.t1
        return words
    if words:
        limit = max(0.5, 2 * float(np.median([w.t1 - w.t0 for w in words])))
        for w in words:
            w.t1 = min(w.t1, w.t0 + limit)
    return words


def _dtw_length(text: str) -> float:
    """Upper bound for how long a word is spoken (start after a pause)."""
    return 0.15 + 0.06 * len(re.sub(r"\W", "", text))


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _multipart(fields: dict[str, str], wav: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for key, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                  .encode())
        out.write(value.encode("utf-8") + b"\r\n")
    out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
              f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    out.write(wav + f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


# --- the server process ---------------------------------------------------------------
class _Server:
    """One whisper-server child process; shared by an engine and its option views."""

    def __init__(self, cmd: list[str]) -> None:
        self.port = cmd[cmd.index("--port") + 1]
        self.log: deque[str] = deque(maxlen=200)
        self.lock = threading.Lock()  # one request at a time (the server serialises too)
        self.on_progress: ProgressFn | None = None
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._finalizer = weakref.finalize(self, _terminate, self._proc)
        threading.Thread(target=self._read_log, daemon=True, name="whisper-server-log").start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read_log(self) -> None:
        assert self._proc.stdout
        for line in self._proc.stdout:  # must be drained, or the server blocks on a full pipe
            self.log.append(line.rstrip())
            if (m := _PROGRESS.search(line)) and self.on_progress:
                self.on_progress(int(m.group(1)) / 100.0)

    def tail(self, n: int = 15) -> str:
        return "\n".join(list(self.log)[-n:])

    def wait_ready(self, timeout: float = _STARTUP_TIMEOUT) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"{SERVER_NAME} exited with code {self._proc.returncode}:"
                                   f"\n{self.tail()}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.2)
        self.close()
        raise RuntimeError(f"{SERVER_NAME} did not become ready in {timeout:.0f} s:\n"
                           f"{self.tail()}")

    def post(self, fields: dict[str, str], audio: np.ndarray,
             progress: ProgressFn | None = None) -> dict:
        body, ctype = _multipart(fields, _wav_bytes(audio))
        req = urllib.request.Request(self.url + "/inference", data=body,
                                     headers={"Content-Type": ctype})
        with self.lock:
            self.on_progress = progress
            try:
                with urllib.request.urlopen(req, timeout=None) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                raise RuntimeError(f"{SERVER_NAME}: {e.code} {detail}\n{self.tail()}") from e
            except (urllib.error.URLError, ConnectionError) as e:
                raise RuntimeError(f"{SERVER_NAME} not reachable ({e}):\n{self.tail()}") from e
            finally:
                self.on_progress = None

    def backend(self) -> str:
        """The compute device, as whisper.cpp logged it at startup."""
        text = "\n".join(self.log)
        if m := re.search(r"ggml_vulkan: \d+ = ([^|(\n]+)", text):
            return f"Vulkan ({m.group(1).strip()})"
        if m := re.search(r"using (\w+) backend", text):
            return m.group(1)
        return "CPU"

    def close(self) -> None:
        self._finalizer()


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _launcher(binary: Path) -> list[str]:
    """Command prefix that starts the server (tests substitute a fake server script)."""
    return [str(binary)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_servers: list[weakref.ref[_Server]] = []


@atexit.register
def _stop_all() -> None:
    for ref in _servers:
        if server := ref():
            server.close()


# --- the engine -----------------------------------------------------------------------
class WhisperCppEngine:
    name = "whisper.cpp"

    def __init__(self, model: str = "small", device: str = "auto", beam_size: int = 5,
                 cpu_threads: int = 0, hotwords: str | None = None, style_prompt: bool = True,
                 server: str = "", dtw: bool = True) -> None:
        from transcribe.core.models import ensure_ggml

        device = "gpu" if device == "cuda" else device  # either name means "use the GPU"
        if device not in ("auto", "cpu", "gpu"):
            raise ValueError(f"whisper.cpp supports devices auto/cpu/gpu, not {device!r}")
        self.model_name = model
        self.beam_size = beam_size
        self.temperature_fallback = True
        self.hotwords = hotwords
        self.style_prompt = style_prompt
        model_path = Path(model) if model.endswith(".bin") else ensure_ggml(model)
        self.model_path = model_path
        self._vocab: set[bytes] | None = None
        self.binary = find_server(server, device)
        threads = cpu_threads or min(8, os.cpu_count() or 4)
        cmd = [*_launcher(self.binary), "-m", str(model_path), "--host", "127.0.0.1",
               "--port", str(_free_port()), "-t", str(threads), "-nlp", "-pp", "-sns"]
        if device == "cpu":
            cmd.append("-ng")
        preset = dtw_preset(model) if dtw else None
        if dtw and not preset:
            log.warning("No DTW preset for %s; using token timestamps", model)
        if preset:  # whisper.cpp silently drops DTW when flash attention is on
            cmd += ["-dtw", preset, "-nfa"]
        self.dtw = preset is not None
        self._server = _Server(cmd)
        _servers.append(weakref.ref(self._server))
        self._server.wait_ready()
        self.device = self._server.backend()
        log.info("whisper.cpp %s on %s (%s)", model, self.device, self.binary)

    @classmethod
    def from_config(cls, config: Config) -> WhisperCppEngine:
        from transcribe.core.config import vocabulary

        a = config.asr
        return cls(a.model, a.device, a.beam_size, a.cpu_threads, vocabulary(config).hotwords(),
                   a.style_prompt, a.whispercpp_server, a.whispercpp_dtw)

    def with_options(self, beam_size: int, temperature_fallback: bool) -> WhisperCppEngine:
        """A view of this engine (sharing the server process) with other decoding options."""
        clone = copy.copy(self)
        clone.beam_size = beam_size
        clone.temperature_fallback = temperature_fallback
        return clone

    def close(self) -> None:
        self._server.close()

    def _count_tokens(self, text: str) -> int:
        if self._vocab is None:
            self._vocab = read_vocab(self.model_path)
        return len(tokenize(text, self._vocab)) if self._vocab else len(text.encode())

    def _prompt(self, language: str | None, prompt: str | None) -> str:
        """Hint words first, then the prompt - the order faster-whisper uses for hotwords."""
        text = prompt or (STYLE_PROMPTS.get(language or "") if self.style_prompt else None)
        return " ".join(p for p in (self.hotwords, text) if p)

    def detect_language(self, audio: np.ndarray, languages: Sequence[str]) -> dict[str, float]:
        # The server computes language_probabilities with its own encoder pass after
        # decoding. A fixed language and a 50 ms duration make the decode return at once,
        # so this costs one encoder pass instead of two (detect_language=true: detection,
        # then the probabilities again) - same probabilities, half the time.
        audio = audio[: 30 * SAMPLE_RATE]
        base = {"response_format": "verbose_json", "no_language_probabilities": "false"}
        result = self._server.post(base | {"language": languages[0], "duration": "50"}, audio)
        if not result.get("language_probabilities"):  # other server versions
            result = self._server.post(base | {"language": "auto", "detect_language": "true"},
                                       audio)
        all_probs = result.get("language_probabilities") or {}
        probs = {lang: float(all_probs.get(lang, 0.0)) for lang in languages}
        total = sum(probs.values())
        if total <= 0:
            return {lang: 1.0 / len(languages) for lang in languages}
        return {lang: p / total for lang, p in probs.items()}

    def transcribe(self, audio: np.ndarray, offset: float = 0.0, language: str | None = None,
                   prompt: str | None = None, progress: ProgressFn | None = None) -> list[Word]:
        """Decode window by window (<= ~28 s of speech each), the context under our control.

        whisper.cpp feeds the text decoded so far back as context. Once a window comes out
        in lower case without punctuation, every later one copies that (IS1009a: no full
        stop after minute 3); without any context it drops more speech (ES2004a: +6 WER).
        So each request is about one window, and its prompt is the previous window's text
        only if that looks well-formed - otherwise the style prompt. Within a request the
        context budget is exactly the prompt, leaving no room for whisper.cpp's own
        feedback.
        """
        from transcribe.vad.silero import speech_regions

        regions = speech_regions(audio, min_silence_ms=_VAD_MIN_SILENCE_MS, pad_ms=_VAD_PAD_MS)
        speech, pieces = cut_speech(audio, regions)
        if len(speech) < SAMPLE_RATE // 10:
            return []
        fields = {
            "response_format": "verbose_json",
            "language": language or "auto",
            "beam_size": str(self.beam_size),
            "temperature": "0.0",
            "temperature_inc": "0.2" if self.temperature_fallback else "0.0",
            "token_timestamps": "true",
            "suppress_nst": "true",
        }
        words: list[Word] = []
        context = prompt
        bounds = window_bounds(speech)
        for a, b in zip(bounds, bounds[1:], strict=False):
            start = a / SAMPLE_RATE
            req = dict(fields)
            text = self._prompt(language, context)
            if context and self._count_tokens(text) > _MAX_PROMPT_TOKENS:
                text = self._prompt(language, None)  # whisper.cpp would cut the hint words
            if text:
                req |= {"prompt": text, "carry_initial_prompt": "true",
                        "max_context": str(self._count_tokens(text) + 1)}
            on_progress = None
            if progress:
                def on_progress(frac: float, start: float = start, n: int = b - a) -> None:
                    progress(to_original(start + frac * n / SAMPLE_RATE, pieces))
            result = self._server.post(req, speech[a:b], on_progress)
            chunk = tokens_to_words(result.get("segments") or [], self.dtw)
            for w in chunk:
                w.t0 = offset + to_original(start + w.t0, pieces)
                w.t1 = max(w.t0 + 0.01, offset + to_original(start + w.t1, pieces))
            words += chunk
            context = previous_text(chunk)
        if progress:
            progress(len(audio) / SAMPLE_RATE)
        return words


def window_bounds(speech: np.ndarray, max_s: float = _WINDOW_S,
                  min_s: float = _WINDOW_S - 8.0) -> list[int]:
    """Sample indices that cut ``speech`` into windows of at most ``max_s`` seconds, each
    cut at the quietest 0.1 s between ``min_s`` and ``max_s`` (usually a VAD pause)."""
    frame = SAMPLE_RATE // 10
    bounds = [0]
    while len(speech) - bounds[-1] > max_s * SAMPLE_RATE:
        lo, hi = bounds[-1] + int(min_s * SAMPLE_RATE), bounds[-1] + int(max_s * SAMPLE_RATE)
        frames = speech[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        bounds.append(lo + int(np.argmin((frames ** 2).mean(axis=1))) * frame + frame // 2)
    return bounds + [len(speech)]


def previous_text(words: list[Word], max_chars: int = 200) -> str | None:
    """The end of a window's text as context for the next one, if it is well-formed
    (capitals and sentence punctuation) - otherwise None, so the style prompt is used."""
    text = "".join(w.text for w in words).strip()
    if not re.search(r"[.?!]", text) or not re.search(r"[A-ZÄÖÜ]", text):
        return None
    if len(text) > max_chars:  # whole sentences from the end
        cut = text[-max_chars:]
        m = re.search(r"[.?!]\s+(?=\S)", cut)
        text = cut[m.end():] if m else cut
    return text
