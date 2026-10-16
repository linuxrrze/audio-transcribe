"""Benchmark whisper.cpp builds (CPU, Vulkan, ...) on the AMI meetings.

    scripts/build_whispercpp.sh
    python -m transcribe.bench.whispercpp \\
        --binary cpu=~/.cache/transcribe/whisper.cpp/build-cpu/bin/whisper-cli \\
        --binary vulkan=~/.cache/transcribe/whisper.cpp/build-vulkan/bin/whisper-cli \\
        --models small large-v3-turbo [--threads 8] [--json out.json]

Reports WER (same normalisation as transcribe.bench.models, fillers removed) and the
real-time factor of the full whisper-cli run (wall clock, model load excluded via a
warm-up run on 2 s of audio). ggml models are downloaded from the whisper.cpp
Hugging Face repository on first use. A Vulkan build falls back to the CPU if no GPU is
usable - check the "backend" column, which is parsed from whisper-cli's log.
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

GGML_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"


def ggml_model(name: str) -> Path:
    path = models.models_dir() / "whisper.cpp" / f"ggml-{name}.bin"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(GGML_URL.format(name=name), tmp)
        tmp.replace(path)
    return path


def _backend(log: str) -> str:
    """Which device whisper.cpp used, from its log (e.g. 'Vulkan0 (AMD Radeon 680M)')."""
    for pattern in (r"whisper_backend_init_gpu: using (.+?) backend",
                    r"ggml_vulkan: \d+ = (.+?) \(", r"using (\w+) backend"):
        if m := re.search(pattern, log):
            return m.group(1).strip()
    return "CPU" if "no GPU found" in log or "use gpu    = 0" in log else "CPU?"


def run_cli(binary: Path, model: Path, wav: Path, language: str, threads: int,
            extra: list[str]) -> tuple[str, float, str]:
    """(transcript text, wall seconds, log) of one whisper-cli run."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        cmd = [str(binary), "-m", str(model), "-f", str(wav), "-l", language, "-t", str(threads),
               "-otxt", "-of", str(out), "-np", *extra]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        seconds = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(f"{binary} failed: {proc.stderr[-500:]}")
        return out.with_suffix(".txt").read_text(encoding="utf-8"), seconds, proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", action="append", required=True,
                    help="label=path/to/whisper-cli (repeatable)")
    ap.add_argument("--models", nargs="+", default=["small", "large-v3-turbo"])
    ap.add_argument("--meetings", nargs="+", default=DEFAULT_MEETINGS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extra", default="", help="extra whisper-cli arguments")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

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
                    _, load_s, _ = run_cli(binary, model, warm, "en", args.threads,
                                           args.extra.split())
                    text, seconds, log = run_cli(binary, model, wav, "en", args.threads,
                                                 args.extra.split())
                    row = {"engine": f"whisper.cpp {label}", "model": model_name,
                           "data": meeting, "backend": _backend(log),
                           "wer": round(wer_without_fillers(ref, text), 4),
                           "seconds": round(seconds, 1),
                           "rtf": round(max(0.0, seconds - load_s) / duration, 3)}
                    rows.append(row)
                    print(row, flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
