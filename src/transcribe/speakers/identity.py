"""Cross-session speaker identity: exemplars, identification and enrollment.

Per session, the clean speech of each speaker is embedded into a few *exemplars*
(stored next to the transcript, encrypted if configured). Identification matches them
against the voiceprint DB; naming a speaker enrolls its exemplars for that person.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config, data_dir
from transcribe.speakers.crypto import Cipher, NullCipher, make_cipher
from transcribe.speakers.embedder import Embedder
from transcribe.speakers.matching import Decision, Match, SpeakerMatcher
from transcribe.speakers.voiceprint_db import VoiceprintDB
from transcribe.storage.transcript_model import Speaker, Transcript

EXEMPLARS = "exemplars.npz"
EXEMPLARS_ENC = "exemplars.npz.enc"
MAX_EXEMPLAR_SECONDS = 20.0
MIN_CONSISTENT_FRACTION = 0.75  # share of exemplars that must agree to enroll a speaker
PIECE_GAP = 0.5  # exemplar pieces end at pauses at least this long


@dataclass(frozen=True)
class Exemplar:
    speaker_id: str
    vector: np.ndarray
    duration: float
    segment_id: str


def open_db(config: Config, path: Path | None = None) -> VoiceprintDB:
    return VoiceprintDB(path or data_dir() / "voiceprints.sqlite",
                        make_cipher(config.speakers.encrypt_voiceprints))


def make_matcher(db: VoiceprintDB, model_id: str, config: Config,
                 channel_type: str | None = None) -> SpeakerMatcher:
    s = config.speakers
    return SpeakerMatcher(db, model_id, s.match_threshold_high, s.match_threshold_low,
                          s.match_margin, channel_type=channel_type)


# --- exemplars ---------------------------------------------------------------------------
def extract_exemplars(audio: np.ndarray, transcript: Transcript, embedder: Embedder,
                      min_duration: float = 3.0, window: float = 6.0,
                      per_speaker: int = 8) -> list[Exemplar]:
    """Embed each speaker's clean speech (no overlapping words) in ~``window`` s pieces.

    Several exemplars per speaker allow a consistency check before enrollment (a
    diarization error that merged two voices shows up as dissimilar exemplars). A
    speaker with only short segments gets one exemplar from their concatenation.
    """
    by_speaker: dict[str, list] = defaultdict(list)
    for seg in transcript.segments:
        if seg.speaker_id and not any(w.overlap for w in seg.words):
            by_speaker[seg.speaker_id].append(seg)

    def clip(t0: float, t1: float) -> np.ndarray:
        return audio[int(t0 * SAMPLE_RATE) : int(t1 * SAMPLE_RATE)]

    out: list[Exemplar] = []
    for sid, segs in by_speaker.items():
        pieces = []
        for s in segs:
            for r0, r1 in _runs(s.words, PIECE_GAP):  # never span a pause: a pause may be
                t = r0                                # an undetected change of speaker
                while r1 - t >= min_duration:
                    end = r1 if r1 - t < window + min_duration else t + window
                    pieces.append((t, end, s.id))
                    t = end
        pieces.sort(key=lambda p: p[1] - p[0], reverse=True)
        for t0, t1, seg_id in pieces[:per_speaker]:
            out.append(Exemplar(sid, embedder.embed(clip(t0, t1)), t1 - t0, seg_id))
        if not pieces and sum(s.t1 - s.t0 for s in segs) >= min_duration:
            joined = np.concatenate([clip(s.t0, s.t1) for s in segs])[
                : int(MAX_EXEMPLAR_SECONDS * SAMPLE_RATE)]
            out.append(Exemplar(sid, embedder.embed(joined), len(joined) / SAMPLE_RATE, ""))
    return out


def _runs(words: list, gap: float) -> list[tuple[float, float]]:
    """Spans of consecutive words separated by pauses shorter than ``gap``."""
    runs: list[list[float]] = []
    for w in words:
        if runs and w.t0 - runs[-1][1] < gap:
            runs[-1][1] = w.t1
        else:
            runs.append([w.t0, w.t1])
    return [(a, b) for a, b in runs]


def consistent_subset(vectors: list[np.ndarray], min_sim: float = 0.70
                      ) -> tuple[list[int], float]:
    """Indices of exemplars agreeing with the medoid, and the fraction kept.

    Measured on >= 2.5 s pieces (WeSpeaker, 5 Piper voices, 2.5-4 s): same voice p5 0.76,
    different voices p95 0.65 (max 0.73); shorter pieces overlap, hence the 3 s minimum
    piece length. A cluster mixing two people splits into groups and only part of it
    agrees with the medoid.
    """
    if len(vectors) <= 1:
        return list(range(len(vectors))), 1.0
    m = np.stack([v / (np.linalg.norm(v) or 1.0) for v in vectors])
    sims = m @ m.T
    medoid = int(np.argmax(sims.sum(axis=1)))
    keep = [i for i in range(len(vectors)) if sims[medoid, i] >= min_sim]
    return keep, len(keep) / len(vectors)


def save_exemplars(session_dir: Path, exemplars: list[Exemplar], model_id: str,
                   cipher: Cipher | None = None) -> Path:
    cipher = cipher or NullCipher()
    meta = {"model_id": model_id, "items": [
        {"speaker_id": e.speaker_id, "duration": e.duration, "segment_id": e.segment_id}
        for e in exemplars]}
    buf = io.BytesIO()
    vectors = (np.stack([e.vector for e in exemplars]).astype(np.float32) if exemplars
               else np.zeros((0, 0), np.float32))
    np.savez(buf, vectors=vectors, meta=np.array(json.dumps(meta)))
    for stale in (EXEMPLARS, EXEMPLARS_ENC):
        (session_dir / stale).unlink(missing_ok=True)
    path = session_dir / (EXEMPLARS_ENC if cipher.encrypted else EXEMPLARS)
    path.write_bytes(cipher.encrypt(buf.getvalue()))
    return path


def load_exemplars(session_dir: Path, cipher: Cipher | None = None
                   ) -> tuple[str, list[Exemplar]]:
    """Return (model_id, exemplars); raises FileNotFoundError if the session has none."""
    cipher = cipher or NullCipher()
    if (session_dir / EXEMPLARS_ENC).exists():
        if not cipher.encrypted:
            raise RuntimeError("Session exemplars are encrypted; enable encrypt_voiceprints")
        raw = cipher.decrypt((session_dir / EXEMPLARS_ENC).read_bytes())
    else:
        raw = (session_dir / EXEMPLARS).read_bytes()
    with np.load(io.BytesIO(raw)) as data:
        meta = json.loads(str(data["meta"]))
        vectors = data["vectors"]
    return meta["model_id"], [
        Exemplar(m["speaker_id"], vectors[i], m["duration"], m["segment_id"])
        for i, m in enumerate(meta["items"])]


def weak_speakers(exemplars: list[Exemplar]) -> set[str]:
    """Speakers represented only by a concatenation of short segments."""
    solid = {e.speaker_id for e in exemplars if e.segment_id}
    return {e.speaker_id for e in exemplars} - solid


def group(exemplars: list[Exemplar]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for e in exemplars:
        out[e.speaker_id].append(e.vector)
    return dict(out)


# --- identification ----------------------------------------------------------------------
def match_speakers(matcher: SpeakerMatcher, speakers: dict[str, list[np.ndarray]],
                   weak: set[str] | None = None) -> dict[str, Match]:
    """Identify session speakers using only each speaker's self-consistent exemplars.

    A speaker whose exemplars do not agree (likely two voices merged by diarization),
    or whose consistency cannot be checked (``weak``: e.g. only one exemplar built from
    concatenated short segments), can at most receive a suggestion.
    """
    subsets, mixed = {}, set(weak or ())
    for sid, vecs in speakers.items():
        keep, fraction = consistent_subset(vecs)
        subsets[sid] = [vecs[i] for i in keep]
        if fraction < MIN_CONSISTENT_FRACTION:
            mixed.add(sid)
    matches = matcher.identify_session(subsets)
    for sid in mixed:
        m = matches[sid]
        if m.decision is Decision.KNOWN:
            matches[sid] = Match(m.person_id, m.name, m.score, Decision.SUGGEST)
    return matches


def apply_matches(transcript: Transcript, matches: dict[str, Match]) -> list[str]:
    """Update speaker labels/suggestions in place. Returns ids whose identity changed.

    Speakers whose identity the user confirmed are left alone.
    """
    changed = []
    for sid, match in matches.items():
        spk = transcript.speakers.get(sid)
        if spk is None or spk.confirmed:
            continue
        default = f"Speaker {sid[1:]}"
        before = (spk.label, spk.person_id, spk.suggestion)
        if match.decision is Decision.KNOWN:
            spk.label, spk.person_id, spk.suggestion = match.name or default, match.person_id, None
        elif match.decision is Decision.SUGGEST:
            spk.label, spk.person_id, spk.suggestion = default, None, match.name
        else:
            spk.label, spk.person_id, spk.suggestion = default, None, None
        spk.score = round(match.score, 4)
        if (spk.label, spk.person_id, spk.suggestion) != before:
            changed.append(sid)
    return changed


def process_session(session_dir: Path | None, audio: np.ndarray, transcript: Transcript,
                    embedder: Embedder, config: Config, db: VoiceprintDB | None = None,
                    channel_type: str | None = None) -> dict[str, Match]:
    """Extract (and, if allowed, store) exemplars; identify speakers if enabled."""
    s = config.speakers
    exemplars = extract_exemplars(audio, transcript, embedder)
    if s.store_voiceprints and session_dir is not None:
        save_exemplars(session_dir, exemplars, embedder.model_id,
                       make_cipher(s.encrypt_voiceprints))
    if not s.identify_known or not exemplars:
        return {}
    own_db = db is None
    db = db or open_db(config)
    try:
        matches = match_speakers(make_matcher(db, embedder.model_id, config, channel_type),
                                 group(exemplars), weak_speakers(exemplars))
    finally:
        if own_db:
            db.close()
    apply_matches(transcript, matches)
    return matches


# --- enrollment --------------------------------------------------------------------------
def enroll_vectors(db: VoiceprintDB, name: str, model_id: str, vectors: list[np.ndarray],
                   durations: list[float], consent: bool, config: Config,
                   channel_type: str = "unknown", source: str | None = None) -> tuple[int, int]:
    """Add exemplars for ``name`` (created if needed). Returns (person_id, #added)."""
    if not config.speakers.store_voiceprints:
        raise PermissionError("Voiceprint storage is disabled (speakers.store_voiceprints)")
    person = db.find_person(name)
    if person is None:
        if not consent:
            raise PermissionError(f"Storing a voiceprint for {name} requires their consent")
        person = db.add_person(name, consent=True)
    elif consent and not person.consent_at:
        db.set_consent(person.id, True)
    elif not person.consent_at:
        raise PermissionError(f"No consent recorded for {name}")
    added = db.add_exemplars(person.id, model_id, vectors, durations, channel_type, source,
                             max_per_person=config.speakers.max_exemplars_per_person)
    return person.id, added


def name_speaker(session_dir: Path, transcript_path: Path, speaker_id: str, name: str,
                 config: Config, consent: bool, enroll: bool = True,
                 db: VoiceprintDB | None = None) -> int:
    """Label a session speaker as ``name``; with ``enroll`` also store its voiceprint.

    Returns the number of exemplars added to the DB.
    """
    transcript = Transcript.load(transcript_path)
    if speaker_id not in transcript.speakers:
        raise KeyError(f"No speaker {speaker_id} in {transcript_path}")
    added, person_id = 0, None
    if enroll:
        model_id, exemplars = load_exemplars(session_dir,
                                             make_cipher(config.speakers.encrypt_voiceprints))
        mine = [e for e in exemplars if e.speaker_id == speaker_id]
        if not mine:
            raise ValueError(f"{speaker_id} has no clean speech of >= 3 s to enroll")
        keep, fraction = consistent_subset([e.vector for e in mine])
        if fraction < MIN_CONSISTENT_FRACTION:
            raise ValueError(
                f"{speaker_id}'s speech does not sound like a single person (only "
                f"{fraction:.0%} consistent) - probably a diarization error merged voices. "
                f"Not enrolled; use `--no-enroll` to only rename, or `tx refine -n N`.")
        mine = [mine[i] for i in keep]
        own_db = db is None
        db = db or open_db(config)
        try:
            person_id, added = enroll_vectors(
                db, name, model_id, [e.vector for e in mine], [e.duration for e in mine],
                consent, config, source=session_dir.name)
        finally:
            if own_db:
                db.close()
    transcript.speakers[speaker_id] = Speaker(speaker_id, name, person_id, confirmed=True)
    transcript.save(transcript_path)
    return added


class SessionIdentity:
    """Identification for one session: live matching and end-of-session processing."""

    def __init__(self, config: Config, embedder: Embedder, db: VoiceprintDB | None = None,
                 channel_type: str | None = None) -> None:
        self.config, self.embedder, self.channel_type = config, embedder, channel_type
        self.db = db or open_db(config)
        self.matcher = (make_matcher(self.db, embedder.model_id, config, channel_type)
                        if config.speakers.identify_known else None)

    def live_matches(self, embeddings: dict[str, list[np.ndarray]]) -> dict[str, Match]:
        if self.matcher is None or self.matcher.empty:
            return {}
        return match_speakers(self.matcher, embeddings)

    def finalize(self, session_dir: Path, audio: np.ndarray,
                 transcript: Transcript) -> dict[str, Match]:
        return process_session(session_dir, audio, transcript, self.embedder, self.config,
                               self.db, self.channel_type)
