"""Thread bridges: core events and background jobs -> Qt GUI thread."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

from transcribe.core.events import Event, EventBus


class EventBridge(QObject):
    """Re-emits events from an EventBus (any thread) as a Qt signal in the GUI thread."""

    received = Signal(object)  # not "event": that would shadow QObject.event()

    def attach(self, bus: EventBus) -> Callable[[], None]:
        return bus.subscribe(self._forward)

    def _forward(self, event: Event) -> None:
        self.received.emit(event)  # queued to the receiver's (GUI) thread


class Job(QObject):
    """Run ``fn`` in a background thread; ``done``/``failed`` arrive in the GUI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn
        self.worker = threading.Thread(target=self._run, daemon=True)  # not QObject.thread

    def start(self) -> Job:
        self.worker.start()
        return self

    def _run(self) -> None:
        try:
            result = self._fn()
        except BaseException as e:  # noqa: BLE001 - reported to the GUI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.done.emit(result)


def run_job(fn: Callable[[], Any], on_done: Callable[[Any], None],
            on_error: Callable[[str], None], parent: QObject) -> Job:
    job = Job(fn, parent)
    job.done.connect(on_done)
    job.failed.connect(on_error)
    return job.start()
