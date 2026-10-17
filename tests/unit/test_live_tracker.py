import numpy as np

from transcribe.speakers.live_tracker import LiveSpeakerTracker

A, B, C = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])


def test_ids_stable_and_in_order_of_appearance():
    tr = LiveSpeakerTracker()
    got = [tr.assign(f"u{i}", v, 2.0) for i, v in enumerate([A, B, A, B + 0.2 * A, B])]
    assert [s for s, _ in got] == ["S1", "S2", "S1", "S2", "S2"]
    assert all(changes == {} for _, changes in got)  # labels are final when shown
    assert tr.speakers == ["S1", "S2"]
    assert [len(v) for v in tr.embeddings_by_speaker().values()] == [2, 3]


def test_short_pieces_never_create_speakers():
    tr = LiveSpeakerTracker(min_new=1.0)
    tr.assign("u0", A, 2.0)
    tr.assign("u1", B, 2.0)
    assert tr.assign("u2", C, 0.4)[0] in ("S1", "S2")  # too short for a new speaker
    assert tr.assign("u3", None, 0.2)[0] == tr.last_speaker  # no embedding: previous
    assert tr.speakers == ["S1", "S2"]


def test_first_piece_without_embedding():
    tr = LiveSpeakerTracker()
    assert tr.assign("u0", None, 0.3)[0] == "S1"
    assert tr.assign("u1", A, 2.0)[0] == "S1"  # S1 gets its first voice
    assert tr.assign("u2", B, 2.0)[0] == "S2"


def test_known_speaker_count_caps_new_speakers():
    tr = LiveSpeakerTracker(num_speakers=2)
    labels = [tr.assign(f"u{i}", v, 2.0)[0] for i, v in enumerate([A, B, C])]
    assert labels[:2] == ["S1", "S2"] and labels[2] in ("S1", "S2")


def test_threshold():
    near = A + 0.9 * B  # cosine 0.74 to A
    assert LiveSpeakerTracker(threshold=0.7).assign("x", A, 2)[0] == "S1"
    tr = LiveSpeakerTracker(threshold=0.8)
    tr.assign("a", A, 2.0)
    assert tr.assign("b", near, 2.0)[0] == "S2"
    tr = LiveSpeakerTracker(threshold=0.7)
    tr.assign("a", A, 2.0)
    assert tr.assign("b", near, 2.0)[0] == "S1"


def test_narrowband_pieces_use_their_own_threshold():
    near = A + 0.9 * B  # cosine 0.74 to A
    tr = LiveSpeakerTracker(threshold=0.7, narrowband_threshold=0.8)
    tr.assign("a", A, 2.0)
    assert tr.assign("wide", near, 2.0)[0] == "S1"  # 0.74 >= 0.7
    tr = LiveSpeakerTracker(threshold=0.7, narrowband_threshold=0.8)
    tr.assign("a", A, 2.0, narrowband=True)
    assert tr.assign("phone", near, 2.0, narrowband=True)[0] == "S2"  # 0.74 < 0.8


def test_bandwidth_estimator_detects_telephone_band():
    from transcribe.bench.suite.augment import phone
    from transcribe.speakers.live_tracker import BandwidthEstimator

    rng = np.random.default_rng(0)
    pieces = [(rng.standard_normal(16000) * 0.1).astype(np.float32) for _ in range(5)]
    wide, narrow = BandwidthEstimator(), BandwidthEstimator()
    assert not any(wide.update(p) for p in pieces)
    assert all(narrow.update(phone(p, rng)) for p in pieces)
    assert not BandwidthEstimator().narrowband  # no evidence yet: wideband
    mixed = BandwidthEstimator()  # one odd piece does not flip a channel
    for p in pieces[:4]:
        mixed.update(phone(p, rng))
    assert mixed.update(pieces[4])
