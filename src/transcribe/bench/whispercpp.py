"""Benchmark whisper.cpp builds (CPU, Vulkan, ...) on the AMI meetings.

    scripts/build_whispercpp.sh
    python -m transcribe.bench.whispercpp \\
        --binary cpu=~/.cache/transcribe/whisper.cpp/build-cpu/bin/whisper-cli \\
        --binary vulkan=~/.cache/transcribe/whisper.cpp/build-vulkan/bin/whisper-cli \\
        --models small large-v3-turbo [--threads 8] [--vad] [--out-dir runs/]
        [--json out.json]

Reports WER (same normalisation as transcribe.bench.models, fillers removed), the
real-time factor of the full whisper-cli run (wall clock, model load excluded via a
warm-up run on 2 s of audio) and ``repeated_lines`` (consecutive identical output lines,
a sign of a decoding loop). ggml models are downloaded from the whisper.cpp Hugging Face
repository on first use. A Vulkan build falls back to the CPU if no GPU is usable -
check the "backend" column, parsed from whisper-cli's log.

``--vad`` enables whisper.cpp's Silero VAD (skips silence, like faster-whisper's VAD
filter in the main pipeline - without it Whisper tends to invent text in silence).
``--out-dir`` keeps every transcript and log for inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.audio.file import load_audio
from transcribe.bench.ami import DEFAULT_MEETINGS, fetch, reference_text
from transcribe.bench.models import wer_without_fillers
from transcribe.core import models

VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def ggml_model(name: str) -> Path:
    return models.ensure_ggml(name)


def vad_model() -> Path:
    return _download(VAD_URL, models.models_dir() / "whisper.cpp" / "ggml-silero-v6.2.0.bin")


def repeated_lines(text: str) -> int:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return sum(a == b for a, b in zip(lines, lines[1:], strict=False))


def _backend(log: str) -> str:
    """Which device whisper.cpp used, from its log (e.g. 'Vulkan0 (AMD Radeon 680M)')."""
    for pattern in (r"ggml_vulkan: \d+ = ([^|(\n]+)", r"using (\w+\d*) backend",
                    r"whisper_backend_init_gpu: using (.+?) backend"):
        if m := re.search(pattern, log):
            return m.group(1).strip()
    if "no GPU found" in log or "use gpu    = 0" in log:
        return "CPU"
    return "unknown (see log)"


def run_cli(binary: Path, model: Path, wav: Path, language: str, threads: int,
            extra: list[str]) -> tuple[str, float, str]:
    """(transcript text, wall seconds, log) of one whisper-cli run."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        # no -np: it also silences the log lines that reveal the backend (GPU or CPU)
        cmd = [str(binary), "-m", str(model), "-f", str(wav), "-l", language, "-t", str(threads),
               "-otxt", "-of", str(out), *extra]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        seconds = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(f"{binary} failed: {proc.stderr[-500:]}")
        return (out.with_suffix(".txt").read_text(encoding="utf-8"), seconds,
                proc.stderr + proc.stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", action="append", required=True,
                    help="label=path/to/whisper-cli (repeatable)")
    ap.add_argument("--models", nargs="+", default=["small", "large-v3-turbo"])
    ap.add_argument("--meetings", nargs="+", default=DEFAULT_MEETINGS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extra", default="", help="extra whisper-cli arguments")
    ap.add_argument("--vad", action="store_true", help="use whisper.cpp's Silero VAD")
    ap.add_argument("--out-dir", type=Path, help="keep transcripts and logs here")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    extra = args.extra.split()
    if args.vad:
        extra += ["--vad", "-vm", str(vad_model())]
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    binaries = {}
    for item in args.binary:
        label, _, path = item.partition("=")
        binaries[label] = Path(path).expanduser()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        warm = Path(tmp) / "warmup.wav"
        sf.write(warm, np.zeros(2 * SAMPLE_RATE, np.float32), SAMPLE_RATE, subtype="PCM_16")
        for meeting in args.meetings:
            wav, _ = fetch(meeting)
            duration = len(load_audio(wav)) / SAMPLE_RATE
            ref = reference_text(meeting)
            for model_name in args.models:
                model = ggml_model(model_name)
                for label, binary in binaries.items():
                    _, load_s, _ = run_cli(binary, model, warm, "en", args.threads, extra)
                    text, seconds, log = run_cli(binary, model, wav, "en", args.threads, extra)
                    if args.out_dir:
                        stem = f"{meeting}-{model_name}-{label}{'-vad' if args.vad else ''}"
                        (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
                        (args.out_dir / f"{stem}.log").write_text(log, encoding="utf-8")
                    row = {"engine": f"whisper.cpp {label}", "model": model_name,
                           "data": meeting, "vad": args.vad, "backend": _backend(log),
                           "wer": round(wer_without_fillers(ref, text), 4),
                           "repeated_lines": repeated_lines(text),
                           "seconds": round(seconds, 1),
                           "rtf": round(max(0.0, seconds - load_s) / duration, 3)}
                    rows.append(row)
                    print(row, flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
