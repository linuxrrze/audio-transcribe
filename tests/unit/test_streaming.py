from transcribe.asr.streaming import LocalAgreement
from transcribe.storage.transcript_model import Word


def w(text, t0):
    return Word(text=f" {text}", t0=t0, t1=t0 + 0.4)


def test_commits_only_agreed_prefix():
    la = LocalAgreement(n=2)
    new, pending = la.insert([w("hallo", 0.0), w("welt", 0.5)])
    assert new == [] and [x.text for x in pending] == [" hallo", " welt"]
    new, pending = la.insert([w("Hallo,", 0.0), w("Welt", 0.5), w("wie", 1.0)])
    assert [x.text for x in new] == [" Hallo,", " Welt"]
    assert [x.text for x in pending] == [" wie"]


def test_disagreement_blocks_commit():
    la = LocalAgreement(n=2)
    la.insert([w("I", 0.0), w("scream", 0.5)])
    new, _ = la.insert([w("ice", 0.0), w("cream", 0.5)])
    assert new == []


def test_redecoded_committed_words_are_not_duplicated():
    la = LocalAgreement(n=2)
    la.insert([w("a", 0.0), w("b", 0.5)])
    la.insert([w("a", 0.0), w("b", 0.5)])
    assert [x.text for x in la.committed] == [" a", " b"]
    # Window still contains committed audio: words must be stripped.
    new, pending = la.insert([w("a", 0.0), w("b", 0.5), w("c", 1.0)])
    assert new == [] and [x.text for x in pending] == [" c"]
    new, _ = la.insert([w("b", 0.5), w("c", 1.0), w("d", 1.5)])
    assert [x.text for x in new] == [" c"]


def test_flush_commits_tail():
    la = LocalAgreement(n=2)
    la.insert([w("x", 0.0)])
    assert [x.text for x in la.flush()] == [" x"]
    assert len(la.committed) == 1
