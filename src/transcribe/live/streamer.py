"""Per-channel streaming transcription, driven by audio time (not wall clock).

The channel keeps a buffer that starts at the current utterance. Every ``step`` seconds
of new audio the buffer is re-decoded; LocalAgreement-2 turns the hypotheses into
committed and pending words. An utterance ends after ``end_silence`` seconds of
trailing silence (VAD); the buffer is force-trimmed when it exceeds ``max_buffer``.
Because decisions depend only on audio time, replaying a file synchronously is
deterministic; when compute is slow, more audio accumulates per decode instead of
audio being dropped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from transcribe.asr.base import AsrEngine
from transcribe.asr.streaming import LocalAgreement
from transcribe.audio.base import SAMPLE_RATE
from transcribe.diarize.align import words_to_segments
from transcribe.storage.transcript_model import Segment, Word

VadFn = Callable[[np.ndarray], list[tuple[float, float]]]
PartialFn = Callable[[float, str, str], None]  # (session time, committed, pending)
FinalFn = Callable[[Segment, np.ndarray], None]  # segment + its audio


@dataclass
class StreamerConfig:
    step: float = 1.0  # seconds of new audio between decodes
    end_silence: float = 0.4  # gap between (padded) VAD regions that ends an utterance
    max_buffer: float = 12.0  # force-finalize committed words beyond this buffer length
    min_lid_speech: float = 1.5  # speech needed before detecting the utterance language
    idle_keep: float = 0.5  # audio kept when the buffer holds no speech
    prompt_chars: int = 200  # previous text passed as Whisper prompt


def _text(words: list[Word]) -> str:
    return "".join(w.text for w in words).strip()


class ChannelStreamer:
    def __init__(self, asr: AsrEngine, languages: list[str], vad: VadFn, on_partial: PartialFn,
                 on_final: FinalFn, channel_id: str = "default",
                 config: StreamerConfig | None = None) -> None:
        self.asr, self.languages, self.vad = asr, languages, vad
        self.on_partial, self.on_final = on_partial, on_final
        self.channel_id = channel_id
        self.cfg = config or StreamerConfig()
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0.0  # session time of _buf[0]
        self._since_decode = 0
        self._la = LocalAgreement(n=2)
        self._utt_words: list[Word] = []
        self._lang: str | None = None
        self._last_lang = languages[0]
        self._history: dict[str, str] = {}
        self._segments = 0
        self.decodes = 0

    @property
    def now(self) -> float:
        return self._buf_start + len(self._buf) / SAMPLE_RATE

    def feed(self, samples: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, samples.astype(np.float32, copy=False)])
        self._since_decode += len(samples)

    def update(self) -> None:
        """Decode if at least ``step`` seconds of new audio have arrived."""
        if self._since_decode >= self.cfg.step * SAMPLE_RATE:
            self._decode(final=False)

    def finish(self) -> None:
        """End of stream: decode the remainder and finalize the open utterance."""
        if len(self._buf):
            self._decode(final=True)

    # ------------------------------------------------------------------------------------
    def _decode(self, final: bool) -> None:
        self._since_decode = 0
        while True:  # the buffer may hold several complete utterances (when ASR lagged)
            regions = self.vad(self._buf)
            if not regions:
                if self._utt_words:
                    self._finalize(flush=True, cut=self.now)
                self._trim(self.now - self.cfg.idle_keep)
                return
            buf_len = len(self._buf) / SAMPLE_RATE
            cut = self._utterance_end(regions, buf_len, final)
            utt_regions = [r for r in regions if cut is None or r[1] <= cut] or regions
            language = self._utterance_language(utt_regions, ended=cut is not None)
            if language is None:
                return  # language not yet decidable; decoding now would risk translation
            audio = self._buf if cut is None else self._buf[: int(cut * SAMPLE_RATE)]
            self.decodes += 1
            hyp = self.asr.transcribe(audio, offset=self._buf_start, language=language,
                                      prompt=self._prompt(language))
            committed, pending = self._la.insert(hyp)
            self._utt_words.extend(committed)
            if cut is not None:
                self._finalize(flush=True, cut=self._buf_start + cut)
                continue
            if buf_len > self.cfg.max_buffer and self._utt_words:
                self._finalize(flush=False, cut=self._utt_words[-1].t1)
            else:
                self.on_partial(self.now, _text(self._utt_words), _text(pending))
            return

    def _utterance_end(self, regions: list[tuple[float, float]], buf_len: float,
                       final: bool) -> float | None:
        """Buffer-relative cut after the first utterance, or None if it is still open.

        An utterance ends at the first VAD gap of at least ``end_silence`` - a gap inside
        the buffer counts too, because the next speaker often starts before we decode.
        """
        for (_, a1), (b0, _) in zip(regions, regions[1:], strict=False):
            if b0 - a1 >= self.cfg.end_silence:
                return (a1 + b0) / 2
        if final:
            return buf_len
        if buf_len - regions[-1][1] >= self.cfg.end_silence:
            return regions[-1][1] + self.cfg.end_silence / 2
        return None

    def _utterance_language(self, regions: list[tuple[float, float]],
                            ended: bool) -> str | None:
        """Language of the current utterance, or None while too little speech is buffered.

        Whisper decoding with the wrong pinned language translates instead of transcribing,
        so with several allowed languages we wait for enough speech (or the utterance end).
        """
        if len(self.languages) == 1:
            return self.languages[0]
        if self._lang is None:
            speech = sum(b - a for a, b in regions)
            if speech < self.cfg.min_lid_speech and not ended:
                return None
            audio = np.concatenate([self._slice_rel(a, b) for a, b in regions])
            probs = self.asr.detect_language(audio, self.languages)
            self._lang = max(probs, key=probs.__getitem__)
            self._last_lang = self._lang
        return self._lang

    def _prompt(self, language: str) -> str | None:
        return self._history.get(language) or None

    def _finalize(self, flush: bool, cut: float) -> None:
        words = self._utt_words + (self._la.flush() if flush else [])
        # Words past the cut belong to the next utterance, which is re-decoded from scratch.
        words = [w for w in words if w.t0 < cut]
        self._utt_words = []
        language = self._lang or self._last_lang
        if words:
            for seg in words_to_segments(words, max_pause=1.0, lang=language,
                                         channel_id=self.channel_id):
                self._segments += 1
                seg.id = f"{self.channel_id}-{self._segments}"
                self.on_final(seg, self._slice_abs(seg.t0, seg.t1))
            text = (self._history.get(language, "") + " " + _text(words)).strip()
            self._history[language] = text[-self.cfg.prompt_chars :]
            cut = max(cut, words[-1].t1)
        if flush:
            self._la = LocalAgreement(n=2)  # fresh state: the next utterance starts clean
        self._lang = None  # re-detect: the next utterance may be another speaker/language
        self.on_partial(self.now, "", "")
        self._trim(cut)

    def _trim(self, t: float) -> None:
        drop = int((t - self._buf_start) * SAMPLE_RATE)
        if drop > 0:
            drop = min(drop, len(self._buf))
            self._buf = self._buf[drop:]
            self._buf_start += drop / SAMPLE_RATE

    def _slice_rel(self, a: float, b: float) -> np.ndarray:
        return self._buf[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)]

    def _slice_abs(self, t0: float, t1: float) -> np.ndarray:
        return self._slice_rel(max(0.0, t0 - self._buf_start), max(0.0, t1 - self._buf_start))
