import os
import time

import pytest

from transcribe.diarize.isolated import call_isolated


def work(n, progress):
    for i in range(n):
        progress((i + 1) / n)
    return {"pid": os.getpid(), "n": n}


def fail(progress):
    raise ValueError("bad input")


def die(progress):
    os._exit(3)


def busy(seconds, progress):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass
    return "done"


def test_result_and_progress_come_back_from_child():
    seen = []
    out = call_isolated("tests.unit.test_isolated:work", {"n": 4}, seen.append)
    assert out["n"] == 4 and out["pid"] != os.getpid()
    assert seen == [0.25, 0.5, 0.75, 1.0]


def test_child_exception_is_reported():
    with pytest.raises(RuntimeError, match="ValueError: bad input"):
        call_isolated("tests.unit.test_isolated:fail", {})


def die_after_progress(progress):
    progress(0.5)
    time.sleep(0.2)  # let the progress message reach the parent
    os._exit(3)


def test_dead_child_is_detected():
    from transcribe.diarize.isolated import WorkerStartupError

    with pytest.raises(WorkerStartupError, match="exit code 3"):  # died silently
        call_isolated("tests.unit.test_isolated:die", {})
    with pytest.raises(RuntimeError, match="exit code 3") as info:  # died mid-way
        call_isolated("tests.unit.test_isolated:die_after_progress", {})
    assert not isinstance(info.value, WorkerStartupError)


def test_diarizer_falls_back_in_process_when_worker_cannot_start(monkeypatch):
    import numpy as np

    from transcribe.diarize import isolated, sherpa_offline

    def no_worker(*a, **k):
        raise isolated.WorkerStartupError("worker died (exit code 1)")

    monkeypatch.setattr(isolated, "call_isolated", no_worker)
    monkeypatch.setattr(sherpa_offline, "run_diarization",
                        lambda audio, progress=None, **kw: [(0.0, 1.0, "S1")])
    d = sherpa_offline.SherpaDiarizer.__new__(sherpa_offline.SherpaDiarizer)
    d.segmentation_path = d.embedding_path = "x"
    d.threshold, d.num_threads, d.isolated = 0.5, 1, True
    with pytest.warns(UserWarning, match="in-process"):
        turns = d.diarize(np.zeros(16000, np.float32))
    assert [(t.t0, t.speaker_id) for t in turns] == [(0.0, "S1")]


def test_parent_threads_keep_running_while_child_is_busy():
    import threading

    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.05)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        assert call_isolated("tests.unit.test_isolated:busy", {"seconds": 1.0}) == "done"
    finally:
        stop.set()
        t.join()
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    assert max(gaps) < 0.5
