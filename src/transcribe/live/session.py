"""Live session: capture -> record -> stream ASR -> live speaker labels -> refinement.

Threads: a capture thread pulls frames from the source into a queue; a processing thread
records, measures levels, feeds the channel streamer and assigns speakers. Events go
through the EventBus (own dispatch thread), so frontends never block processing.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from transcribe.asr.base import AsrEngine
from transcribe.audio.base import SAMPLE_RATE, AudioFrame, AudioSource
from transcribe.core.events import (
    AudioLevel,
    EventBus,
    FinalSegment,
    PartialText,
    SegmentRelabeled,
    SpeakerDetected,
    SpeakerIdentified,
    StatusChanged,
)
from transcribe.diarize.refine import relabel
from transcribe.live.streamer import ChannelStreamer, StreamerConfig, VadFn
from transcribe.speakers.embedder import Embedder
from transcribe.speakers.identity import SessionIdentity, apply_matches
from transcribe.speakers.live_tracker import LiveSpeakerTracker
from transcribe.storage import session_store
from transcribe.storage.recorder import AudioRecorder, load_recording
from transcribe.storage.transcript_model import Segment, Speaker, Transcript

LIVE_TRANSCRIPT = "transcript.live.json"
StageProgress = Callable[[str, float], None]
RefineFn = Callable[[np.ndarray, StageProgress], Transcript]


class LiveSession:
    def __init__(
        self,
        source: AudioSource,
        asr: AsrEngine,
        session_dir: Path,
        languages: list[str],
        vad: VadFn,
        embedder: Embedder | None = None,
        tracker: LiveSpeakerTracker | None = None,
        refine: RefineFn | None = None,
        streamer_config: StreamerConfig | None = None,
        meta: dict[str, object] | None = None,
        save_interval: float = 5.0,
        identity: SessionIdentity | None = None,
    ) -> None:
        self.source, self.session_dir = source, session_dir
        self.embedder, self.tracker, self.refine_fn = embedder, tracker, refine
        self.identity = identity
        self.bus = EventBus()
        self.transcript = Transcript(meta=dict(meta or {}))
        self.final_transcript: Transcript | None = None
        self.recorder = AudioRecorder(session_dir)
        self.streamer = ChannelStreamer(asr, languages, vad, self._on_partial, self._on_final,
                                        config=streamer_config)
        self._segments: dict[str, Segment] = {}
        self._queue: queue.Queue[AudioFrame | None] = queue.Queue()
        self._stop = threading.Event()
        # Completion is signalled with an Event, not Thread.join(): a KeyboardInterrupt
        # delivered inside join() can leave the thread marked finished while it still
        # runs, and finishing concurrently with it duplicates/loses utterances.
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._level_due = 0.0
        self._save_interval = save_interval
        self._last_save = 0.0
        self._lock = threading.Lock()

    # --- public API ---------------------------------------------------------------------
    def start(self) -> None:
        self.bus.start()
        self._emit_status("starting")
        self.source.open()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="process", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._emit_status("running")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the source is exhausted and processed (True) or timeout (False)."""
        if not self._threads:
            return True
        return self._done.wait(timeout)

    def stop(self, refine: bool = True) -> Transcript:
        """Stop capture, finish processing, optionally refine; returns the final transcript."""
        self._stop.set()
        self.source.close()
        if self._threads:
            self._done.wait()
        try:
            if self._error:
                raise RuntimeError("live processing failed") from self._error
            return self._finish(refine)
        finally:
            self.bus.stop()

    def run_sync(self, refine: bool = True) -> Transcript:
        """Process the whole source in the calling thread (deterministic replay)."""
        self.source.open()
        for frame in self.source.frames():
            self._handle(frame)
        self.source.close()
        return self._finish(refine)

    # --- threads ------------------------------------------------------------------------
    def _capture_loop(self) -> None:
        try:
            for frame in self.source.frames():
                if self._stop.is_set():
                    break
                self._queue.put(frame)
        except BaseException as e:  # noqa: BLE001 - surfaced via stop()
            self._error = e
        finally:
            self._queue.put(None)

    def _process_loop(self) -> None:
        try:
            self._process()
        finally:
            self._done.set()

    def _process(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    return
                batch = [frame]
                while True:  # drain: when ASR lags, decode more audio at once
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        self._queue.put(None)
                        break
                    batch.append(nxt)
                for f in batch:
                    self._ingest(f)
                self.streamer.update()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._emit_status("error", repr(e))

    # --- processing ---------------------------------------------------------------------
    def _handle(self, frame: AudioFrame) -> None:
        self._ingest(frame)
        self.streamer.update()

    def _ingest(self, frame: AudioFrame) -> None:
        self.recorder.write(frame.samples)
        self.streamer.feed(frame.samples)
        t = self.recorder.total_samples / SAMPLE_RATE
        if t >= self._level_due and len(frame.samples):
            rms = float(np.sqrt(np.mean(np.square(frame.samples))))
            self.bus.publish(AudioLevel(t, frame.channel_id, 20 * math.log10(max(rms, 1e-6))))
            self._level_due = t + 0.1

    def _on_partial(self, t: float, committed: str, pending: str) -> None:
        self.bus.publish(PartialText(t, committed, pending, self.streamer.channel_id))

    def _on_final(self, seg: Segment, audio: np.ndarray) -> None:
        if self.tracker is not None:
            emb = self.embedder.embed(audio) if self.embedder and len(audio) else None
            speaker, changes = self.tracker.assign(seg.id, emb, seg.t1 - seg.t0)
            self._set_speaker(seg, speaker)
            for seg_id, new in changes.items():
                old = self._segments.get(seg_id)
                if old is not None:
                    self._set_speaker(old, new)
                    self.bus.publish(SegmentRelabeled(seg.t1, seg_id, new))
            if self.identity is not None:
                self._identify(seg.t1)
        with self._lock:
            self._segments[seg.id] = seg
            self.transcript.segments.append(seg)
        self.bus.publish(FinalSegment(seg.t1, seg))
        if time.monotonic() - self._last_save >= self._save_interval:
            self._save_live()

    def _identify(self, t: float) -> None:
        assert self.tracker is not None and self.identity is not None
        matches = self.identity.live_matches(self.tracker.embeddings_by_speaker())
        for sid in apply_matches(self.transcript, matches):
            spk, m = self.transcript.speakers[sid], matches[sid]
            self.bus.publish(SpeakerIdentified(t, sid, spk.label, spk.person_id,
                                               m.decision.value, m.score, spk.suggestion))

    def _set_speaker(self, seg: Segment, speaker: str) -> None:
        seg.speaker_id = speaker
        for w in seg.words:
            w.speaker_id = speaker
        if speaker not in self.transcript.speakers:
            label = f"Speaker {speaker[1:]}"
            self.transcript.speakers[speaker] = Speaker(speaker, label)
            self.bus.publish(SpeakerDetected(seg.t1, speaker, label))

    def _save_live(self) -> None:
        with self._lock:
            self._prune_speakers(self.transcript)
            self.transcript.save(self.session_dir / LIVE_TRANSCRIPT)
        self._last_save = time.monotonic()

    @staticmethod
    def _prune_speakers(transcript: Transcript) -> None:
        used = {s.speaker_id for s in transcript.segments}
        transcript.speakers = {k: v for k, v in transcript.speakers.items() if k in used}

    def _finish(self, refine: bool) -> Transcript:
        self.streamer.finish()
        duration = self.recorder.total_samples / SAMPLE_RATE
        self.recorder.finalize()
        self.transcript.meta["duration_s"] = round(duration, 3)
        self._save_live()
        final = self.transcript
        audio: np.ndarray | None = None
        if refine and self.refine_fn is not None and duration > 0:
            audio = load_recording(self.session_dir)
            self._emit_status("refining", progress=0.0)

            def progress(stage: str, frac: float) -> None:
                self._emit_status("refining", stage, frac)

            refined = self.refine_fn(audio, progress)
            final = relabel(refined, self.transcript)
            final.meta = {**self.transcript.meta, **refined.meta, "refined": True}
        if self.identity is not None and final.speakers and duration > 0:
            audio = load_recording(self.session_dir) if audio is None else audio
            self._emit_status("identifying")
            self.identity.finalize(self.session_dir, audio, final)
        session_store.save_session(self.session_dir, final, {
            k: v for k, v in final.meta.items() if isinstance(v, str | int | float | bool)})
        self.final_transcript = final
        self._emit_status("stopped")
        return final

    def _emit_status(self, status: str, detail: str = "", progress: float | None = None) -> None:
        self.bus.publish(StatusChanged(self.streamer.now, status, detail, progress))
