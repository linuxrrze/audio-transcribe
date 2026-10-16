from transcribe.bench.metrics import normalize, wer


def test_normalize_keeps_umlauts_and_drops_punctuation():
    assert normalize("Grüße, Welt! Let's go.") == ["grüße", "welt", "let's", "go"]


def test_wer():
    assert wer("a b c d", "a x c") == 0.5
    assert wer("same text", "Same, text.") == 0.0


def test_speaker_accuracy_counts_each_word_once():
    from transcribe.bench.dialogue import RefTurn
    from transcribe.bench.metrics import evaluate
    from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word

    ref = [RefTurn("a", "de", "eins zwei", 0.0, 1.0), RefTurn("b", "de", "drei", 1.15, 2.0)]
    words = [Word(" eins", 0.0, 0.5, speaker_id="S1"), Word(" zwei", 0.5, 0.98, speaker_id="S1"),
             Word(" drei", 1.17, 2.0, speaker_id="S2")]
    t = Transcript(segments=[Segment(0, 2, words)],
                   speakers={"S1": Speaker("S1", "1"), "S2": Speaker("S2", "2")})
    # the gap is only 0.15 s: padded turns would count "zwei"/"drei" twice
    assert evaluate(t, ref).speaker_accuracy == 1.0
