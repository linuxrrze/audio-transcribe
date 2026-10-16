"""Run a long native computation in a child process, relaying progress.

Some native libraries (sherpa-onnx diarization) hold the GIL for many seconds, which
freezes every Python thread of the process: progress bars, timers, the GUI. Running the
call in a spawned child keeps the parent responsive; progress and the result come back
through a queue.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import traceback
from collections.abc import Callable
from typing import Any

ProgressFn = Callable[[float], None]


class WorkerStartupError(RuntimeError):
    """The child died before reporting anything (e.g. it could not import the parent's
    __main__, as with scripts piped through stdin). Callers may fall back to in-process."""


def _child(target: str, kwargs: dict[str, Any], out: Any) -> None:
    try:
        module, name = target.split(":")
        fn = getattr(importlib.import_module(module), name)
        result = fn(**kwargs, progress=lambda f: out.put(("progress", f)))
        out.put(("result", result))
    except BaseException as e:  # noqa: BLE001 - re-raised in the parent
        out.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def call_isolated(target: str, kwargs: dict[str, Any], progress: ProgressFn | None = None,
                  poll: float = 0.1) -> Any:
    """Call ``module:function(**kwargs, progress=...)`` in a spawned child process.

    ``kwargs`` and the return value must be picklable. Raises RuntimeError if the child
    fails or dies; terminates the child if the caller is interrupted.
    """
    ctx = mp.get_context("spawn")  # fork is unsafe with threads and native libraries
    out = ctx.Queue()
    proc = ctx.Process(target=_child, args=(target, kwargs, out), daemon=True)
    proc.start()
    heard = False
    try:
        while True:
            try:
                kind, value = out.get(timeout=poll)
            except queue.Empty:
                if not proc.is_alive():
                    error = RuntimeError if heard else WorkerStartupError
                    raise error(
                        f"{target} worker died (exit code {proc.exitcode})") from None
                continue
            heard = True
            if kind == "progress":
                if progress:
                    progress(value)
            elif kind == "result":
                return value
            else:
                raise RuntimeError(f"{target} failed in worker process: {value}")
    finally:
        if proc.is_alive():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        out.close()
