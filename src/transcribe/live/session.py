"""Live session: capture -> record -> stream ASR -> live speaker labels -> refinement.

Threads: a capture thread pulls frames from the source into a queue; a processing thread
records, measures levels, feeds one streamer per channel and assigns speakers. Events go
through the EventBus (own dispatch thread), so frontends never block processing.

Multi-channel sources (e.g. microphone + system audio) get one streamer and recording
per channel, a shared speaker tracker, and echo suppression: a microphone segment that
repeats what the system channel played is dropped.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from transcribe.asr.base import AsrEngine
from transcribe.audio.base import SAMPLE_RATE, AudioFrame, AudioSource
from transcribe.audio.levels import level_db
from transcribe.core.events import (
    AudioLevel,
    EventBus,
    FinalSegment,
    PartialText,
    SegmentRelabeled,
    SegmentRemoved,
    SpeakerDetected,
    SpeakerIdentified,
    StatusChanged,
)
from transcribe.core.vocabulary import Vocabulary
from transcribe.diarize.align import is_sentence_end
from transcribe.diarize.refine import relabel
from transcribe.live.echo import echo_of, suppress_echo
from transcribe.live.streamer import ChannelStreamer, StreamerConfig, VadFn
from transcribe.speakers.embedder import Embedder
from transcribe.speakers.identity import SessionIdentity, apply_matches
from transcribe.speakers.live_tracker import BandwidthEstimator, LiveSpeakerTracker
from transcribe.storage import session_store
from transcribe.storage.recorder import AudioRecorder, load_recording
from transcribe.storage.transcript_model import Segment, Speaker, Transcript

LIVE_TRANSCRIPT = "transcript.live.json"
StageProgress = Callable[[str, float], None]
RefineFn = Callable[[np.ndarray, StageProgress], Transcript]


MIN_EMBED_S = 0.3  # shorter pieces get no embedding (the tracker keeps the speaker)


def sentence_pieces(seg: Segment, audio: np.ndarray) -> list[tuple[Segment, np.ndarray]]:
    """Split a finished segment at sentence ends; ``audio`` covers [seg.t0, seg.t1].

    Cuts lie halfway between a sentence's last word and the next word. The first piece
    keeps the segment id, later ones get "<id>.1", "<id>.2", ...
    """
    groups: list[list] = [[]]
    for w in seg.words:
        if groups[-1] and is_sentence_end(groups[-1][-1].text):
            groups.append([])
        groups[-1].append(w)
    if len(groups) < 2:
        return [(seg, audio)]
    out = []
    for k, words in enumerate(groups):
        t0 = seg.t0 if k == 0 else (groups[k - 1][-1].t1 + words[0].t0) / 2
        t1 = seg.t1 if k == len(groups) - 1 else (words[-1].t1 + groups[k + 1][0].t0) / 2
        piece = Segment(t0, t1, words, seg.speaker_id, seg.lang, seg.channel_id,
                        seg.id if k == 0 else f"{seg.id}.{k}")
        a, b = (max(0, int((x - seg.t0) * SAMPLE_RATE)) for x in (t0, t1))
        out.append((piece, audio[a:b]))
    return out


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
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self.source, self.session_dir = source, session_dir
        self.embedder, self.tracker, self.refine_fn = embedder, tracker, refine
        self.identity = identity
        self.vocabulary = vocabulary
        self.bus = EventBus()
        self.channels = source.describe().channels
        self.multi = len(self.channels) > 1
        self.transcript = Transcript(meta={**(meta or {}), "channels": self.channels})
        self.final_transcript: Transcript | None = None
        self.recorders = {ch: AudioRecorder(session_dir, channel=ch if self.multi else "")
                          for ch in self.channels}
        self.streamers = {ch: self._make_streamer(asr, languages, vad, ch, streamer_config)
                          for ch in self.channels}
        self._segments: dict[str, Segment] = {}
        self._bandwidth: dict[str, BandwidthEstimator] = {}  # per channel: telephone band?
        self._queue: queue.Queue[AudioFrame | None] = queue.Queue()
        self._stop = threading.Event()
        # Completion is signalled with an Event, not Thread.join(): a KeyboardInterrupt
        # delivered inside join() can leave the thread marked finished while it still
        # runs, and finishing concurrently with it duplicates/loses utterances.
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self._error: BaseException | None = None
        self._level_due = dict.fromkeys(self.channels, 0.0)
        self._save_interval = save_interval
        self._last_save = 0.0
        self._echo_dropped = 0
        self._lock = threading.Lock()

    def _make_streamer(self, asr: AsrEngine, languages: list[str], vad: VadFn, channel: str,
                       config: StreamerConfig | None) -> ChannelStreamer:
        def on_partial(t: float, committed: str, pending: str) -> None:
            self.bus.publish(PartialText(t, committed, pending, channel))

        return ChannelStreamer(asr, languages, vad, on_partial, self._on_final, channel,
                               config)

    @property
    def streamer(self) -> ChannelStreamer:
        """The first channel's streamer (the only one for single-channel sources)."""
        return self.streamers[self.channels[0]]

    @property
    def now(self) -> float:
        return max(s.now for s in self.streamers.values())

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
            self._ingest(frame)
            self.streamers[frame.channel_id].update()
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
                for ch in {f.channel_id for f in batch}:
                    self.streamers[ch].update()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._emit_status("error", repr(e))

    # --- processing ---------------------------------------------------------------------
    def _ingest(self, frame: AudioFrame) -> None:
        ch = frame.channel_id
        if ch not in self.streamers:
            raise ValueError(f"Frame for undeclared channel {ch!r}; source declares "
                             f"{self.channels}")
        recorder = self.recorders[ch]
        recorder.write(frame.samples)
        self.streamers[ch].feed(frame.samples)
        t = recorder.total_samples / SAMPLE_RATE
        if t >= self._level_due[ch] and len(frame.samples):
            self.bus.publish(AudioLevel(t, ch, level_db(frame.samples)[0]))
            self._level_due[ch] = t + 0.1

    def _on_final(self, seg: Segment, audio: np.ndarray) -> None:
        if self.vocabulary:
            seg.words = self.vocabulary.correct(seg.words)
        if self.multi and self._drop_echo(seg):
            return
        # In dialogue, people answer each other without the pause that ends an utterance:
        # label each sentence on its own (live speaker accuracy on a radio play 29 -> 91 %,
        # see speakers/live_tracker.py).
        pieces = sentence_pieces(seg, audio) if self.tracker is not None else [(seg, audio)]
        for piece, piece_audio in pieces:
            self._final_piece(piece, piece_audio)
        if time.monotonic() - self._last_save >= self._save_interval:
            self._save_live()

    def _final_piece(self, seg: Segment, audio: np.ndarray) -> None:
        if self.tracker is not None:
            emb = (self.embedder.embed(audio)
                   if self.embedder and len(audio) >= MIN_EMBED_S * SAMPLE_RATE else None)
            narrow = self._bandwidth.setdefault(seg.channel_id, BandwidthEstimator()).update(
                audio)
            speaker, changes = self.tracker.assign(seg.id, emb, seg.t1 - seg.t0, narrow)
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

    def _drop_echo(self, seg: Segment) -> bool:
        """Drop ``seg`` if it is mic echo; withdraw earlier mic segments it duplicates."""
        with self._lock:
            echoes = echo_of(seg, self.transcript.segments)
            if echoes == [seg]:
                self._echo_dropped += 1
                return True
            for old in echoes:
                self.transcript.segments.remove(old)
                self._segments.pop(old.id, None)
                self._echo_dropped += 1
        for old in echoes:
            self.bus.publish(SegmentRemoved(seg.t1, old.id, "echo"))
        return False

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

    # --- end of session -----------------------------------------------------------------
    def _finish(self, refine: bool) -> Transcript:
        for streamer in self.streamers.values():
            streamer.finish()
        duration = max(r.total_samples for r in self.recorders.values()) / SAMPLE_RATE
        for recorder in self.recorders.values():
            recorder.finalize()
        with self._lock:  # channels finalize independently: restore time order
            self.transcript.segments.sort(key=lambda s: s.t0)
        self.transcript.meta["duration_s"] = round(duration, 3)
        if self.multi:
            self.transcript.meta["echo_segments_dropped"] = self._echo_dropped
        self._save_live()
        final = self.transcript
        audio: dict[str, np.ndarray] = {}
        if refine and self.refine_fn is not None and duration > 0:
            audio = self._load_audio()
            final = self._refine(audio)
        if self.identity is not None and final.speakers and duration > 0:
            audio = audio or self._load_audio()
            self._emit_status("identifying")
            self.identity.finalize(self.session_dir, audio, final)
        session_store.save_session(self.session_dir, final, {
            k: v for k, v in final.meta.items() if isinstance(v, str | int | float | bool)})
        self.final_transcript = final
        self._emit_status("stopped")
        return final

    def _load_audio(self) -> dict[str, np.ndarray]:
        return {ch: load_recording(self.session_dir, ch if self.multi else "")
                for ch in self.channels}

    def _refine(self, audio: dict[str, np.ndarray]) -> Transcript:
        assert self.refine_fn is not None

        def progress(stage: str, frac: float) -> None:
            self._emit_status("refining", stage, frac)

        return refine_channels(audio, self.refine_fn, self.transcript, progress)

    def _emit_status(self, status: str, detail: str = "", progress: float | None = None) -> None:
        self.bus.publish(StatusChanged(self.now, status, detail, progress))


def refine_channels(audio: dict[str, np.ndarray], refine_fn: RefineFn, reference: Transcript,
                    progress: StageProgress | None = None) -> Transcript:
    """Refine each channel, map its speakers onto the reference ids, merge, drop echoes.

    ``audio`` maps channel id -> recording (a single entry for one-channel sessions).
    """
    report = progress or (lambda stage, frac: None)
    multi = len(audio) > 1
    segments: list[Segment] = []
    speakers: dict[str, Speaker] = {}
    meta: dict[str, object] = {}
    for ch, signal in audio.items():
        prefix = f"{ch} " if multi else ""

        def channel_progress(stage: str, frac: float, prefix: str = prefix) -> None:
            report(prefix + stage, frac)

        refined = refine_fn(signal, channel_progress)
        channel_ref = Transcript(
            segments=[s for s in reference.segments if not multi or s.channel_id == ch],
            speakers=reference.speakers)
        mapped = relabel(refined, channel_ref, reserved=set(speakers))
        for seg in mapped.segments:
            seg.channel_id = ch if multi else seg.channel_id
        segments += mapped.segments
        speakers.update(mapped.speakers)
        meta = meta or dict(refined.meta)
    dropped: list[Segment] = []
    if multi:
        segments, dropped = suppress_echo(segments)
    segments.sort(key=lambda s: s.t0)
    used = {s.speaker_id for s in segments}
    transcript = Transcript(
        segments=segments,
        speakers={k: v for k, v in sorted(speakers.items(), key=lambda kv: int(kv[0][1:]))
                  if k in used},
        meta={**reference.meta, **meta, "refined": True,
              **({"echo_segments_dropped": len(dropped)} if multi else {})})
    transcript.number_segments(force=True)  # channels were numbered separately
    return transcript
