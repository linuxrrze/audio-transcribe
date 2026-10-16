from transcribe.diarize.refine import map_to_reference, relabel
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


def tr(spans, labels=None):
    segs = [Segment(a, b, [Word(" x", a, b, speaker_id=s)], speaker_id=s) for a, b, s in spans]
    ids = sorted({s for _, _, s in spans})
    return Transcript(segments=segs, speakers={
        i: Speaker(i, (labels or {}).get(i, f"Speaker {i[1:]}")) for i in ids})


def test_maps_by_overlap_and_keeps_live_labels():
    live = tr([(0, 5, "S1"), (5, 10, "S2"), (10, 12, "S1")], labels={"S2": "Alice"})
    refined = tr([(0, 5, "S2"), (5, 10, "S1"), (10, 12, "S3")])
    assert map_to_reference(refined, live) == {"S2": "S1", "S1": "S2", "S3": "S3"}
    out = relabel(refined, live)
    assert [s.speaker_id for s in out.segments] == ["S1", "S2", "S3"]
    assert out.speakers["S2"].label == "Alice"
    assert [w.speaker_id for s in out.segments for w in s.words] == ["S1", "S2", "S3"]
