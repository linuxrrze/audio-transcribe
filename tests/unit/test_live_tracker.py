import numpy as np

from transcribe.speakers.live_tracker import LiveSpeakerTracker


def threshold_cluster(features, num_clusters):
    """Tiny stand-in for FastClustering: greedy cosine clustering (threshold 0.5)."""
    centroids, labels = [], []
    for f in features:
        sims = [float(c @ f) for c in centroids]
        if sims and max(sims) > 0.5:
            labels.append(int(np.argmax(sims)))
        else:
            centroids.append(f)
            labels.append(len(centroids) - 1)
    return labels


A, B = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])


def test_ids_stable_and_in_order_of_appearance():
    tr = LiveSpeakerTracker(threshold_cluster)
    got = [tr.assign(f"u{i}", v, 2.0)[0] for i, v in enumerate([A, B, A, B, B])]
    assert got == ["S1", "S2", "S1", "S2", "S2"]
    assert tr.speakers == ["S1", "S2"]


def test_short_utterance_inherits_previous_speaker():
    tr = LiveSpeakerTracker(threshold_cluster, min_duration=1.0)
    tr.assign("u0", A, 2.0)
    tr.assign("u1", B, 2.0)
    assert tr.assign("u2", A, 0.4) == ("S2", {})  # too short: no embedding decision


def test_reclustering_reports_changed_earlier_utterances():
    calls = {"n": 0}

    def flip_cluster(features, k):
        calls["n"] += 1
        # First call: two clusters; afterwards everything collapses into one.
        return [0, 1] if len(features) == 2 else [0] * len(features)

    tr = LiveSpeakerTracker(flip_cluster)
    tr.assign("u0", A, 2.0)
    assert tr.assign("u1", B, 2.0)[0] == "S2"
    speaker, changes = tr.assign("u2", A, 2.0)
    assert speaker == "S1" and changes == {"u1": "S1"}


def test_num_speakers_is_capped_by_items():
    seen = []

    def spy(features, k):
        seen.append(k)
        return list(range(len(features)))

    tr = LiveSpeakerTracker(spy, num_speakers=3)
    for i in range(4):
        tr.assign(f"u{i}", np.eye(4)[i], 2.0)
    assert seen == [2, 3, 3]
