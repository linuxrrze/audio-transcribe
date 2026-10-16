import numpy as np
import pytest
from cryptography.fernet import Fernet

from tests.unit.fakes import FakeEmbedder, ScriptedWorld, fake_vad, utterance
from tests.unit.test_live_session import ArraySource
from tests.unit.test_live_tracker import threshold_cluster
from transcribe.audio.base import SAMPLE_RATE
from transcribe.core.config import Config
from transcribe.core.events import SpeakerIdentified
from transcribe.live.session import LiveSession
from transcribe.live.streamer import StreamerConfig
from transcribe.speakers import identity
from transcribe.speakers.crypto import ENV_KEY, FernetCipher, NullCipher, make_cipher
from transcribe.speakers.identity import (
    Exemplar,
    SessionIdentity,
    apply_matches,
    enroll_vectors,
    extract_exemplars,
    load_exemplars,
    name_speaker,
    save_exemplars,
)
from transcribe.speakers.live_tracker import LiveSpeakerTracker
from transcribe.speakers.matching import Decision, Match, SpeakerMatcher
from transcribe.speakers.voiceprint_db import VoiceprintDB
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word

DIM = 16
rng = np.random.default_rng(1)


@pytest.fixture(autouse=True)
def _reseed():
    """Each test gets the same random voices regardless of test order."""
    global rng
    rng = np.random.default_rng(1)


def voice():
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v, noise=0.05):
    return v + noise * rng.normal(size=DIM).astype(np.float32)


@pytest.fixture
def db():
    store = VoiceprintDB(":memory:")
    yield store
    store.close()


@pytest.fixture
def key(monkeypatch):
    k = Fernet.generate_key()
    monkeypatch.setenv(ENV_KEY, k.decode())
    return k


# --- storage -----------------------------------------------------------------------------
def test_encrypted_vectors_need_the_key(tmp_path, key):
    path = tmp_path / "vp.sqlite"
    enc = VoiceprintDB(path, FernetCipher(key))
    p = enc.add_person("Ann", consent=True)
    v = voice()
    enc.add_voiceprint(p.id, "m", v)
    raw = enc._conn.execute("SELECT vector FROM voiceprint").fetchone()[0]
    assert v.tobytes() not in raw  # not stored in clear
    assert np.allclose(enc.voiceprints("m")[0].vector, v)
    enc.close()
    with pytest.raises(RuntimeError, match="encrypted"):
        VoiceprintDB(path, NullCipher()).voiceprints("m")
    assert make_cipher(True).encrypted


def test_add_exemplars_skips_duplicates_and_caps(db):
    p = db.add_person("Ann", consent=True)
    base = voice()
    assert db.add_exemplars(p.id, "m", [base, base * 1.0001], [3, 3]) == 1
    db.add_exemplars(p.id, "m", [voice() for _ in range(10)], [3] * 10, max_per_person=5)
    assert len(db.voiceprints("m", p.id)) == 5


def test_export_import_roundtrip(tmp_path, db):
    p = db.add_person("Ann", consent=True)
    db.add_exemplars(p.id, "m", [voice(), voice()], [3, 4])
    out = tmp_path / "export.json"
    assert db.export_json(out) == 2
    other = VoiceprintDB(":memory:")
    assert other.import_json(out) == (1, 2)
    assert other.import_json(out) == (1, 0)  # idempotent: duplicates skipped
    assert other.find_person("Ann").consent_at is not None


# --- matching ----------------------------------------------------------------------------
def make_db_with(db, people):
    for name, v in people.items():
        p = db.add_person(name, consent=True)
        db.add_exemplars(p.id, "m", [near(v) for _ in range(3)], [3, 3, 3])


def test_identify_session_is_one_to_one(db):
    ann, bob = voice(), voice()
    make_db_with(db, {"Ann": ann, "Bob": bob})
    m = SpeakerMatcher(db, "m")
    res = m.identify_session({"S1": [near(ann)], "S2": [near(bob)], "S3": [near(ann)]})
    names = {sid: (r.name, r.decision) for sid, r in res.items()}
    assert names["S2"] == ("Bob", Decision.KNOWN)
    # Two session speakers sound like Ann: only one can be Ann, and with a competitor this
    # close neither gets a confident label.
    ann_like = [names["S1"], names["S3"]]
    assert sum(n == "Ann" for n, _ in ann_like) == 1
    assert all(d is not Decision.KNOWN for _, d in ann_like)


def test_margin_turns_ambiguous_match_into_suggestion(db):
    ann = voice()
    twin = ann + 0.02 * voice()
    make_db_with(db, {"Ann": ann, "Twin": twin})
    res = SpeakerMatcher(db, "m").identify(near(ann, 0.01))
    assert res.decision is Decision.SUGGEST


def test_empty_db_returns_unknown(db):
    assert SpeakerMatcher(db, "m").identify_session({"S1": [voice()]})["S1"].decision \
        is Decision.UNKNOWN


# --- exemplars & identities --------------------------------------------------------------
def transcript_with(segments):
    return Transcript(segments=segments, speakers={
        s.speaker_id: Speaker(s.speaker_id, f"Speaker {s.speaker_id[1:]}")
        for s in segments})


def test_extract_exemplars_gates_quality():
    audio = np.full(20 * SAMPLE_RATE, 0.1, np.float32)
    overlap_word = Word(" x", 6, 9, speaker_id="S2", overlap=True)
    segs = [Segment(0, 4, [Word(" a", 0, 4)], "S1", id="1"),
            Segment(4.5, 5, [Word(" b", 4.5, 5)], "S1", id="2"),
            Segment(6, 9, [overlap_word], "S2", id="3"),
            Segment(10, 11.8, [Word(" c", 10, 11.8)], "S3", id="4"),
            Segment(12, 13.5, [Word(" d", 12, 13.5)], "S3", id="5")]
    ex = extract_exemplars(audio, transcript_with(segs), FakeEmbedder())
    assert [(e.speaker_id, e.segment_id) for e in ex] == [("S1", "1"), ("S3", "")]
    assert ex[1].duration == pytest.approx(3.3)  # S3: concatenated short segments


@pytest.mark.parametrize("encrypted", [False, True])
def test_exemplars_roundtrip(tmp_path, key, encrypted):
    cipher = make_cipher(encrypted)
    ex = [Exemplar("S1", voice(), 3.0, "a"), Exemplar("S2", voice(), 2.5, "b")]
    path = save_exemplars(tmp_path, ex, "m", cipher)
    assert path.name.endswith(".enc") == encrypted
    model, back = load_exemplars(tmp_path, cipher)
    assert model == "m" and [e.speaker_id for e in back] == ["S1", "S2"]
    assert np.allclose(back[0].vector, ex[0].vector)


def test_apply_matches_respects_confirmed():
    t = Transcript(speakers={"S1": Speaker("S1", "Speaker 1"),
                             "S2": Speaker("S2", "Speaker 2"),
                             "S3": Speaker("S3", "Carla", 7, confirmed=True)})
    changed = apply_matches(t, {
        "S1": Match(1, "Ann", 0.9, Decision.KNOWN),
        "S2": Match(2, "Bob", 0.7, Decision.SUGGEST),
        "S3": Match(1, "Ann", 0.95, Decision.KNOWN)})
    assert changed == ["S1", "S2"]
    assert (t.speakers["S1"].label, t.speakers["S1"].person_id) == ("Ann", 1)
    assert (t.speakers["S2"].label, t.speakers["S2"].suggestion) == ("Speaker 2", "Bob")
    assert t.speakers["S3"].label == "Carla"


def test_enroll_requires_consent_and_storage(db):
    cfg = Config()
    with pytest.raises(PermissionError, match="consent"):
        enroll_vectors(db, "Ann", "m", [voice()], [3], consent=False, config=cfg)
    assert enroll_vectors(db, "Ann", "m", [voice()], [3], consent=True, config=cfg)[1] == 1
    cfg.speakers.store_voiceprints = False
    with pytest.raises(PermissionError, match="disabled"):
        enroll_vectors(db, "Ann", "m", [voice()], [3], consent=True, config=cfg)


def test_name_speaker_enrolls_and_confirms(tmp_path, db):
    t = transcript_with([Segment(0, 3, [Word(" hi", 0, 3)], "S2", id="1")])
    t.save(tmp_path / "transcript.json")
    save_exemplars(tmp_path, [Exemplar("S2", voice(), 3.0, "1")], "m")
    added = name_speaker(tmp_path, tmp_path / "transcript.json", "S2", "Ann", Config(),
                         consent=True, db=db)
    assert added == 1
    spk = Transcript.load(tmp_path / "transcript.json").speakers["S2"]
    assert (spk.label, spk.confirmed, spk.person_id) == ("Ann", True, db.find_person("Ann").id)


# --- live --------------------------------------------------------------------------------
def test_live_session_identifies_enrolled_speaker(tmp_path, db, monkeypatch):
    # FakeEmbedder gives one "voice" per language; enroll the German voice as Anna.
    german = FakeEmbedder().embed(np.full(SAMPLE_RATE, 0.1, np.float32))
    p = db.add_person("Anna", consent=True)
    db.add_exemplars(p.id, "fake", [german], [3.0])
    monkeypatch.setattr(identity, "open_db", lambda config, path=None: db)
    words = (utterance("hallo zusammen wie geht es", 0.3, "de")
             + utterance("fine thanks and you", 3.0, "en"))
    world = ScriptedWorld(words)
    session = LiveSession(
        ArraySource(world.audio(5.5)), world, tmp_path, ["de", "en"], fake_vad,
        embedder=FakeEmbedder(), tracker=LiveSpeakerTracker(threshold_cluster),
        streamer_config=StreamerConfig(step=0.5),
        identity=SessionIdentity(Config(), FakeEmbedder(), db))
    events = []
    session.bus.subscribe(events.append)
    final = session.run_sync()
    ids = [e for e in events if isinstance(e, SpeakerIdentified)]
    assert [(e.speaker_id, e.label, e.decision) for e in ids] == [("S1", "Anna", "known")]
    assert final.speakers["S1"].label == "Anna" and final.speakers["S2"].label == "Speaker 2"
    assert (tmp_path / "exemplars.npz").exists()


def test_weak_speaker_is_only_suggested(db):
    from transcribe.speakers.identity import match_speakers

    ann = voice()
    make_db_with(db, {"Ann": ann})
    res = match_speakers(SpeakerMatcher(db, "m"), {"S1": [near(ann)]}, weak={"S1"})
    assert (res["S1"].name, res["S1"].decision) == ("Ann", Decision.SUGGEST)


def test_mixed_speaker_is_only_suggested(db):
    from transcribe.speakers.identity import match_speakers

    ann, bob = voice(), voice()
    make_db_with(db, {"Ann": ann})
    m = SpeakerMatcher(db, "m")
    clean = match_speakers(m, {"S1": [near(ann) for _ in range(4)]})
    assert clean["S1"].decision is Decision.KNOWN
    mixed = match_speakers(m, {"S1": [near(ann), near(ann), near(bob), near(bob)]})
    assert mixed["S1"].decision is Decision.SUGGEST
