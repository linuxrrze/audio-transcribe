"""Measure live latency by replaying a file in real time through the threaded pipeline.

Latency of a segment = wall-clock time when its FinalSegment event arrives minus the
wall-clock time its last word was "spoken" (replay start + segment end).
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

from transcribe.audio.file import FileSource
from transcribe.core.config import Config
from transcribe.core.events import Event, FinalSegment
from transcribe.live.factory import build_live_session


def measure(wav: Path, config: Config | None = None) -> dict[str, float]:
    config = config or Config()
    with tempfile.TemporaryDirectory() as tmp:
        session = build_live_session(FileSource(wav, realtime=True), config, Path(tmp))
        finals: list[float] = []
        start = 0.0

        def on_event(e: Event) -> None:
            if isinstance(e, FinalSegment):
                finals.append(time.monotonic() - start - e.segment.t1)

        session.bus.subscribe(on_event)
        start = time.monotonic()
        session.start()
        session.wait()
        session.stop(refine=False)
    return {
        "segments": len(finals),
        "final_latency_median_s": round(float(statistics.median(finals)), 2) if finals else -1,
        "final_latency_max_s": round(float(max(finals)), 2) if finals else -1,
    }


if __name__ == "__main__":
    cfg = Config()
    if len(sys.argv) > 2:
        cfg.asr.model = sys.argv[2]
    print(measure(Path(sys.argv[1]), cfg))
