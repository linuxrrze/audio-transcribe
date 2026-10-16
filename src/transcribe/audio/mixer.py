"""Combine several sources: separate channels (``MixerSource``) or one sum (``MixDownSource``).

Separate channels keep "me" (microphone) and "them" (system audio) apart, which gives
speaker separation for free and allows echo suppression; the mix-down needs only one
ASR stream (cheaper on CPU).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np

from transcribe.audio.base import SAMPLE_RATE, AudioFrame, AudioSource, SourceInfo

_DONE = object()


class MixerSource:
    """Run sources in parallel; frames keep their source's channel id."""

    def __init__(self, sources: list[AudioSource]) -> None:
        ids = [c for s in sources for c in s.describe().channels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Channel ids must be unique, got {ids}")
        self.sources = sources
        self.ended: set[str] = set()  # channels whose source has finished
        self._queue: queue.Queue[object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []

    def describe(self) -> SourceInfo:
        infos = [s.describe() for s in self.sources]
        return SourceInfo(name="+".join(i.name for i in infos),
                          channels=[c for i in infos for c in i.channels],
                          realtime=all(i.realtime for i in infos))

    def open(self) -> None:
        for s in self.sources:
            s.open()
        self._threads = [threading.Thread(target=self._pump, args=(s,), daemon=True)
                         for s in self.sources]
        for t in self._threads:
            t.start()

    def _pump(self, source: AudioSource) -> None:
        try:
            for frame in source.frames():
                self._queue.put(frame)
        except BaseException as e:  # noqa: BLE001 - re-raised in frames()
            self._errors.append(e)
        finally:
            self._queue.put((_DONE, source.describe().channels))

    def frames(self) -> Iterator[AudioFrame]:
        running = len(self.sources)
        while running:
            item = self._queue.get()
            if isinstance(item, tuple) and item[0] is _DONE:
                running -= 1
                self.ended.update(item[1])
                continue
            assert isinstance(item, AudioFrame)
            yield item
        if self._errors:
            raise RuntimeError("audio source failed") from self._errors[0]

    def close(self) -> None:
        for s in self.sources:
            s.close()


class MixDownSource:
    """Sum all channels of a mixer into one channel, aligned by sample count.

    For real-time sources, a channel that falls more than ``max_lag`` behind (e.g.
    WASAPI loopback delivers nothing while the system is silent) is padded with silence
    instead of stalling. Non-real-time sources (files) are only padded once they end.
    """

    def __init__(self, mixer: MixerSource, channel_id: str = "mix",
                 frame_ms: int = 100, max_lag: float = 0.3) -> None:
        self.mixer, self.channel_id = mixer, channel_id
        self.frame = SAMPLE_RATE * frame_ms // 1000
        self.max_lag = int(max_lag * SAMPLE_RATE)

    def describe(self) -> SourceInfo:
        info = self.mixer.describe()
        return SourceInfo(name=f"mix({info.name})", channels=[self.channel_id],
                          realtime=info.realtime)

    def open(self) -> None:
        self.mixer.open()

    def frames(self) -> Iterator[AudioFrame]:
        info = self.mixer.describe()
        bufs = {c: np.zeros(0, np.float32) for c in info.channels}
        t = 0.0
        for frame in self.mixer.frames():
            bufs[frame.channel_id] = np.concatenate([bufs[frame.channel_id], frame.samples])
            while True:
                lead = max(len(b) for b in bufs.values())
                ready = min(len(b) for b in bufs.values())
                lagging = [c for c, b in bufs.items() if len(b) < self.frame]
                may_pad = (lead - ready > self.max_lag if info.realtime
                           else all(c in self.mixer.ended for c in lagging))
                if ready < self.frame and lead >= self.frame and may_pad:
                    for c, b in bufs.items():  # pad lagging channels with silence
                        if len(b) < self.frame:
                            bufs[c] = np.concatenate([b, np.zeros(self.frame - len(b),
                                                                  np.float32)])
                    ready = self.frame
                if ready < self.frame:
                    break
                mixed = np.sum([b[: self.frame] for b in bufs.values()], axis=0)
                for c in bufs:
                    bufs[c] = bufs[c][self.frame :]
                yield AudioFrame(np.clip(mixed, -1.0, 1.0).astype(np.float32), t,
                                 self.channel_id)
                t += self.frame / SAMPLE_RATE
        rest = max(len(b) for b in bufs.values())
        if rest:
            mixed = np.zeros(rest, np.float32)
            for b in bufs.values():
                mixed[: len(b)] += b
            yield AudioFrame(np.clip(mixed, -1.0, 1.0), t, self.channel_id)

    def close(self) -> None:
        self.mixer.close()
