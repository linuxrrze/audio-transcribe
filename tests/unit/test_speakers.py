import numpy as np
import pytest

from transcribe.speakers.matching import Decision, SpeakerMatcher
from transcribe.speakers.online_cluster import OnlineClusterer
from transcribe.speakers.voiceprint_db import VoiceprintDB

rng = np.random.default_rng(0)
DIM = 192


def voice():
    return rng.normal(size=DIM).astype(np.float32)


def sample(base, noise=0.3):
    return base + noise * rng.normal(size=DIM).astype(np.float32)


def test_online_clusterer_separates_speakers():
    a, b = voice(), voice()
    c = OnlineClusterer(threshold=0.5)
    ids = [c.add(sample(x))[0] for x in [a, b, a, a, b, b, a]]
    assert ids == ["S1", "S2", "S1", "S1", "S2", "S2", "S1"]


def test_merge_combines_split_cluster():
    a = voice()
    c = OnlineClusterer(threshold=0.99)
    c.add(sample(a, 0.2))
    c.add(sample(a, 0.2))
    assert len(c.clusters) == 2
    c.threshold = 0.8
    remap = c.merge()
    assert len(c.clusters) == 1 and len(remap) == 1


@pytest.fixture
def db():
    store = VoiceprintDB(":memory:")
    yield store
    store.close()


def test_voiceprint_db_roundtrip_and_cascade(db):
    p = db.add_person("Alice", consent=True)
    v = voice()
    db.add_voiceprint(p.id, "m1", v, duration_s=4.0)
    db.add_voiceprint(p.id, "m2", voice())
    got = db.voiceprints("m1")
    assert len(got) == 1 and np.allclose(got[0].vector, v)
    assert db.get_person(p.id).consent_at is not None
    db.delete_person(p.id)
    assert db.voiceprints("m1") == [] and db.voiceprints("m2") == []


def test_merge_persons_moves_voiceprints(db):
    a, b = db.add_person("A"), db.add_person("B")
    db.add_voiceprint(b.id, "m", voice())
    db.merge_persons(a.id, b.id)
    assert [vp.person_id for vp in db.voiceprints("m")] == [a.id]
    assert db.find_person("B") is None


def test_matcher_open_set(db):
    alice, bob = voice(), voice()
    pa, pb = db.add_person("Alice"), db.add_person("Bob")
    for _ in range(4):
        db.add_voiceprint(pa.id, "m", sample(alice))
        db.add_voiceprint(pb.id, "m", sample(bob))
    m = SpeakerMatcher(db, "m", high=0.7, low=0.4)
    hit = m.identify(sample(alice))
    assert hit.name == "Alice" and hit.decision is Decision.KNOWN
    assert m.identify(voice()).decision is Decision.UNKNOWN
    # Model scoping: other model id sees no voiceprints.
    assert SpeakerMatcher(db, "other").identify(sample(alice)).decision is Decision.UNKNOWN
