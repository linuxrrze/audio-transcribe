from transcribe.bench.metrics import normalize, wer


def test_normalize_keeps_umlauts_and_drops_punctuation():
    assert normalize("Grüße, Welt! Let's go.") == ["grüße", "welt", "let's", "go"]


def test_wer():
    assert wer("a b c d", "a x c") == 0.5
    assert wer("same text", "Same, text.") == 0.0
