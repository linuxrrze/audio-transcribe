"""Typed events emitted by a Session. Frontends (CLI, GUI, servers) consume only these."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from transcribe.storage.transcript_model import Segment


@dataclass(frozen=True)
class Event:
    t: float  # session time in seconds


@dataclass(frozen=True)
class PartialText(Event):
    """Uncommitted hypothesis; replaces the previous partial for the same channel."""

    text: str
    channel_id: str = "default"


@dataclass(frozen=True)
class FinalSegment(Event):
    segment: Segment


@dataclass(frozen=True)
class SpeakerDetected(Event):
    speaker_id: str
    label: str
    suggested_person: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class AudioLevel(Event):
    channel_id: str
    rms_db: float


@dataclass(frozen=True)
class StatusChanged(Event):
    status: str  # "starting" | "running" | "paused" | "refining" | "stopped" | "error"
    detail: str = ""


Listener = Callable[[Event], None]


class EventBus:
    """Thread-safe fan-out bus. Listeners run on a dedicated dispatch thread."""

    _STOP = object()

    def __init__(self, maxsize: int = 10_000) -> None:
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def publish(self, event: Event) -> None:
        if self._thread is None:
            self._dispatch(event)
        else:
            self._queue.put(event)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="event-bus", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(self._STOP)
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while (item := self._queue.get()) is not self._STOP:
            assert isinstance(item, Event)
            self._dispatch(item)

    def _dispatch(self, event: Event) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener(event)
