from transcribe.diarize.align import Turn, assign_speakers, words_to_segments
from transcribe.storage.transcript_model import Word


def test_assign_by_max_overlap_and_flags_overlap():
    words = [Word(" a", 0.0, 1.0), Word(" b", 1.0, 2.0), Word(" c", 5.0, 5.2)]
    turns = [Turn(0.0, 1.3, "S1"), Turn(1.2, 3.0, "S2")]
    assign_speakers(words, turns)
    assert [x.speaker_id for x in words] == ["S1", "S2", None]
    assert words[1].overlap and not words[0].overlap


def test_nearest_turn_within_gap():
    words = [Word(" a", 3.2, 3.4)]
    assign_speakers(words, [Turn(0.0, 3.0, "S1")])
    assert words[0].speaker_id == "S1"


def test_segments_split_on_speaker_and_pause():
    words = [Word(" a", 0, 1, speaker_id="S1"), Word(" b", 1, 2, speaker_id="S1"),
             Word(" c", 2, 3, speaker_id="S2"), Word(" d", 9, 10, speaker_id="S2")]
    segs = words_to_segments(words)
    assert [(s.speaker_id, s.text) for s in segs] == [("S1", "a b"), ("S2", "c"), ("S2", "d")]


def test_long_monologue_splits_at_sentence_end():
    words = []
    for i in range(40):
        text = " word." if i % 10 == 9 else " word"
        words.append(Word(text, i * 1.0, i * 1.0 + 0.9, speaker_id="S1"))
    segs = words_to_segments(words, max_duration=15.0)
    assert [len(s.words) for s in segs] == [20, 20]
    assert all(s.words[-1].text.endswith(".") for s in segs)


def test_orphan_word_inherits_adjacent_speaker():
    words = [Word(" a", 0.0, 1.0), Word(" orphan", 2.6, 2.9), Word(" b", 3.0, 4.0),
             Word(" lonely", 20.0, 20.5)]
    assign_speakers(words, [Turn(0.0, 1.0, "S1"), Turn(3.0, 4.0, "S2")])
    assert [w.speaker_id for w in words] == ["S1", "S2", "S2", None]
