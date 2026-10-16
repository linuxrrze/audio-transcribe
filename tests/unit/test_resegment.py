import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.diarize.resegment import (
    phrases,
    resegment,
    speaker_count,
    speech_per_speaker,
)
from transcribe.storage.transcript_model import Word

# Synthetic audio: the amplitude identifies the true voice (0.1, 0.2, 0.3).
VOICE_LEVEL = {"A": 0.1, "B": 0.2, "C": 0.3}


class LevelEmbedder:
    model_id = "level"

    def embed(self, audio):
        v = np.zeros(3, np.float32)
        v[int(round(float(np.median(audio[audio > 0])) * 10)) - 1] = 1.0
        return v


def kmeans_like(features, k):
    """Exact clustering for one-hot test vectors."""
    keys = {tuple(f): i for i, f in enumerate(sorted({tuple(f) for f in features}))}
    return [keys[tuple(f)] for f in features]


def build(script, diarized):
    """script: [(voice, text, duration)], 0.2 s gaps. diarized: label per line (diarizer)."""
    words, t = [], 0.0
    audio = np.zeros(int(60 * SAMPLE_RATE), np.float32)
    for (voice, text, dur), label in zip(script, diarized, strict=True):
        toks = text.split()
        step = dur / len(toks)
        for i, tok in enumerate(toks):
            words.append(Word(" " + tok, t + i * step, t + (i + 1) * step - 0.02,
                              speaker_id=label))
        audio[int(t * SAMPLE_RATE):int((t + dur) * SAMPLE_RATE)] = VOICE_LEVEL[voice]
        t += dur + 0.2
    return words, audio


SCRIPT = [("A", "Hast du das gehört heute?", 1.5), ("B", "Ja natürlich habe ich das.", 1.5),
          ("C", "Was?", 0.4), ("A", "Da hat jemand gerufen hier.", 1.5),
          ("B", "Dann gehen wir jetzt hin.", 1.5), ("C", "Gut, ich komme sofort mit.", 1.5),
          ("A", "Leise, bitte ganz leise jetzt.", 1.5), ("B", "Ich bin doch ganz leise.", 1.5),
          ("C", "Ihr streitet ja schon wieder.", 1.5)]


def test_phrases_split_at_sentence_ends_and_pauses():
    w = [Word(" Ja.", 0, 0.3), Word(" Und", 0.35, 0.5), Word(" dann", 0.55, 0.7),
         Word(" weiter", 1.2, 1.4)]
    assert [[x.text for x in p] for p in phrases(w, gap=0.25)] == [
        [" Ja."], [" Und", " dann"], [" weiter"]]


def test_resegment_fixes_mislabelled_lines():
    # The diarizer put one of C's lines under A's speaker (S1).
    diarized = ["S1", "S2", "S2", "S1", "S2", "S1", "S1", "S2", "S3"]
    words, audio = build(SCRIPT, diarized)
    changed = resegment(words, audio, LevelEmbedder(), num_speakers=3, cluster=kmeans_like,
                        min_phrase=0.6)
    labels, i = [], 0
    for _, text, _ in SCRIPT:
        n = len(text.split())
        line = {w.speaker_id for w in words[i:i + n]}
        assert len(line) == 1  # a phrase is never split between speakers
        labels.append(line.pop())
        i += n
    assert changed == 1
    # every embeddable line now carries its true voice's label (ids stay the diarizer's)
    assert labels == ["S1", "S2", "S2", "S1", "S2", "S3", "S1", "S2", "S3"]
    # limitation: the 0.4 s "Was?" is too short to embed and keeps its diarizer speaker
    assert SCRIPT[2][2] < 0.6 and labels[2] == "S2"


def test_too_little_material_leaves_labels_alone():
    words, audio = build(SCRIPT[:2], ["S1", "S2"])
    assert resegment(words, audio, LevelEmbedder(), num_speakers=2,
                     cluster=kmeans_like) == 0
    assert [w.speaker_id for w in words][:1] == ["S1"]


def test_speaker_count_ignores_tiny_clusters():
    speech = speech_per_speaker([(0, 200, "S1"), (200, 400, "S2"),
                                 (400, 405, "S3"),   # 5 s, 1.2 % -> a minor character
                                 (405, 407, "S4"),   # < 3 s -> spurious
                                 (407, 410, "S5")])  # 3 s but < 1 % -> spurious
    assert speaker_count(speech) == 3


def _naive_silhouette(x, labels):
    labels = np.asarray(labels)
    dist = 1 - x @ x.T
    out = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            out.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == lab].mean() for lab in set(labels) if lab != labels[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


def _groups(n_groups, per_group, noise, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_groups, 16))
    x = np.concatenate([c + noise * rng.normal(size=(per_group, 16)) for c in centers])
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_vectorized_silhouette_matches_definition():
    from transcribe.diarize.resegment import silhouette

    x = _groups(3, 8, 0.6)
    labels = [i % 3 for i in range(len(x))]  # deliberately poor labelling
    assert abs(silhouette(x, labels) - _naive_silhouette(x, labels)) < 1e-5
    good = [i // 8 for i in range(len(x))]
    assert abs(silhouette(x, good) - _naive_silhouette(x, good)) < 1e-5


def test_choose_k_finds_the_number_of_groups():
    from transcribe.diarize.resegment import choose_k, sherpa_cluster

    for n in (3, 5):
        x = _groups(n, 12, 0.3, seed=n)
        assert len(set(choose_k(x, k_max=8, cluster=sherpa_cluster))) == n


def test_count_modes():
    import pytest

    words, audio = build(SCRIPT * 3, ["S1", "S2", "S2", "S1", "S2", "S1", "S1", "S2", "S3"] * 3)
    seen = []

    def spy(features, k):
        seen.append(k)
        return kmeans_like(features, k)

    resegment(words, audio, LevelEmbedder(), max_speakers=3, cluster=spy)
    assert seen == [3]  # "estimate": max_speakers is k
    seen.clear()
    resegment(words, audio, LevelEmbedder(), max_speakers=3, count="silhouette", cluster=spy)
    assert seen == [2, 3]  # silhouette tries 2..max
    with pytest.raises(ValueError):
        resegment(words, audio, LevelEmbedder(), count="guess")
