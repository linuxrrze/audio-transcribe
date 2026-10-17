"""Benchmark suite: metrics, conditions, data format, generator and runner (fake models)."""

import itertools
import json
import random

import numpy as np
import pytest
import soundfile as sf

from transcribe.bench.metrics import edit_distance as slow_edit_distance
from transcribe.bench.suite import augment, metrics, run, sources
from transcribe.bench.suite.data import Dataset, Recording, RefTurn
from transcribe.core.config import Config
from transcribe.diarize.align import Turn
from transcribe.storage.transcript_model import Word

SR = 16000


# --- metrics ------------------------------------------------------------------------------
def test_edit_distance_and_hungarian_match_brute_force():
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        b = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        assert metrics.edit_distance(a, b) == slow_edit_distance(a, b)
    for _ in range(50):
        n = rng.randint(1, 5)
        cost = np.array([[rng.randint(0, 9) for _ in range(n)] for _ in range(n)], float)
        best = min(sum(cost[i, p[i]] for i in range(n))
                   for p in itertools.permutations(range(n)))
        got = metrics.hungarian(cost)
        assert sum(cost[i, got[i]] for i in range(n)) == best


def test_wer_ignores_case_punctuation_and_fillers():
    c = metrics.wer("Ähm, hallo Welt!", "hallo, welt")
    assert (c.errors, c.total) == (0, 2)


def test_cpwer_matches_speakers_and_counts_unmatched():
    ref = {"A": "guten morgen", "B": "hallo welt"}
    assert metrics.cpwer(ref, {"x": "hallo welt", "y": "guten morgen"}).errors == 0
    swapped_word = metrics.cpwer(ref, {"x": "guten welt", "y": "hallo morgen"})
    assert (swapped_word.errors, swapped_word.total) == (2, 4)  # one word per speaker
    merged = metrics.cpwer(ref, {"x": "guten morgen hallo welt"})
    assert merged.errors == 4  # B missing (2 deletions) + 2 insertions for A's match
    extra = metrics.cpwer({"A": "eins zwei"}, {"x": "eins zwei", "y": "drei"})
    assert extra.errors == 1


def test_der_components():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    assert metrics.der(ref, [(0, 10, "x"), (10, 20, "y")]).rate == 0.0
    d = metrics.der(ref, [(0, 10, "x"), (10, 15, "y"), (15, 20, "x")])
    assert d.confusion == pytest.approx(5.0) and d.rate == pytest.approx(0.25)
    d = metrics.der(ref, [(0, 5, "x"), (10, 20, "y"), (25, 30, "z")])
    assert d.missed == pytest.approx(5.0) and d.false_alarm == pytest.approx(5.0)
    # overlap: two reference speakers, one hypothesis speaker -> one missed speaker
    d = metrics.der([(0, 10, "A"), (5, 10, "B")], [(0, 10, "x")])
    assert d.total == pytest.approx(15.0) and d.missed == pytest.approx(5.0)
    # collar: boundary errors within 0.25 s are not scored
    late = [(0.2, 10.2, "x"), (10.2, 20.0, "y")]
    assert metrics.der(ref, late).rate > 0 and metrics.der(ref, late, 0.25).rate == 0.0


def test_purity_penalises_mixing_not_splitting():
    ref = [(0, 10, "A"), (10, 20, "B")]
    assert metrics.purity(ref, [(0, 5, "x"), (5, 10, "y"), (10, 20, "z")]).errors == 0
    mixed = metrics.purity(ref, [(0, 20, "x")])
    assert mixed.errors == pytest.approx(1000, abs=2) and mixed.total == pytest.approx(2000, 2)


# --- conditions ---------------------------------------------------------------------------
@pytest.mark.parametrize("condition", augment.CONDITIONS)
def test_conditions_keep_length_and_are_deterministic(condition):
    t = np.arange(SR * 2) / SR
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a, b = augment.apply(x, condition, 7), augment.apply(x, condition, 7)
    assert a.shape == x.shape and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.abs(a).max() <= 1.0


def test_unknown_condition():
    with pytest.raises(ValueError, match="Unknown condition"):
        augment.apply(np.zeros(100, np.float32), "underwater")


# --- data + generator ---------------------------------------------------------------------
@pytest.fixture
def bench_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


def test_dataset_roundtrip_and_splits(bench_home):
    ds = Dataset("demo")
    turns = [RefTurn(0.0, 1.0, "A", "hallo"), RefTurn(1.2, 2.0, "B", "welt")]
    ds.write([Recording("r1", "demo", "dev", "audio/r1.flac", "de", turns),
              Recording("r2", "demo", "test", "audio/r2.flac", "de", turns)])
    ds.write([Recording("r1", "demo", "dev", "audio/r1.flac", "de", turns[:1])])  # replace
    assert [r.id for r in ds.recordings("test")] == ["r2"]
    r1 = next(ds.recordings("dev"))
    assert len(r1.turns) == 1 and r1.ref_by_speaker() == {"A": "hallo"}
    assert ds.audio_path(r1) == ds.dir / "audio" / "r1.flac"


def test_conversation_timing_and_overlap():
    def voice(seconds):
        return lambda text: (np.full(int(seconds * SR), 0.1, np.float32), text)

    rng = random.Random(3)
    audio, ref = sources.conversation(rng, {"a": voice(1.0), "b": voice(0.5)},
                                      ["eins", "zwei"], 40)
    assert len(ref) == 40
    assert all(x.speaker != y.speaker for x, y in zip(ref, ref[1:], strict=False))
    gaps = [b.t0 - a.t1 for a, b in zip(ref, ref[1:], strict=False)]
    assert min(gaps) < 0 < max(gaps)  # some interruptions, some pauses
    assert len(audio) >= int(ref[-1].t1 * SR)


def test_build_synth_seeds_differ_per_split(bench_home, monkeypatch):
    monkeypatch.setattr("transcribe.bench.dialogue.speak",
                        lambda name, text: np.full(int(0.3 * SR), 0.05, np.float32))
    dev = sources.build_synth("de", "dev", 2, (2, 3), (4, 6), progress=lambda m: None)
    test = sources.build_synth("de", "test", 2, (2, 3), (4, 6), progress=lambda m: None)
    assert dev.name == test.name == "synth-de-piper"  # one dataset, two splits
    recs = list(dev.recordings())
    assert {r.split for r in recs} == {"dev", "test"} and len(recs) == 4
    assert [r.meta["seed"] for r in recs if r.split == "dev"] != \
        [r.meta["seed"] for r in recs if r.split == "test"]


# --- runner -------------------------------------------------------------------------------
class FakeAsr:
    calls = 0

    def __init__(self, rec):
        self.rec = rec

    def transcribe(self, audio, language=None):
        FakeAsr.calls += 1
        out = []
        for t in self.rec.turns:
            ws = t.text.split()
            step = (t.t1 - t.t0) / len(ws)
            out += [Word(" " + w, t.t0 + i * step, t.t0 + (i + 1) * step)
                    for i, w in enumerate(ws)]
        return out


class FakeDiarizer:
    def __init__(self, rec, merge=False):
        self.rec, self.merge = rec, merge

    def diarize(self, audio):
        return [Turn(t.t0, t.t1, "S1" if self.merge else f"S{t.speaker}")
                for t in self.rec.turns]


@pytest.fixture
def demo(bench_home, monkeypatch):
    ds = Dataset("demo")
    (ds.dir / "audio").mkdir(parents=True)
    turns = [RefTurn(0.5, 2.0, "A", "guten morgen zusammen"),
             RefTurn(2.4, 3.6, "B", "hallo anna"),
             RefTurn(4.0, 5.5, "A", "wie geht es dir")]
    sf.write(ds.dir / "audio" / "r1.flac", np.zeros(6 * SR, np.float32), SR)
    rec = Recording("r1", "demo", "dev", "audio/r1.flac", "de", turns,
                    meta={"duration_s": 6.0})
    ds.write([rec])
    monkeypatch.setattr(run.Engines, "asr", lambda self, c: FakeAsr(rec))
    monkeypatch.setattr(run.Engines, "diarizer", lambda self, c: FakeDiarizer(
        rec, merge=c.speakers.diarize_threshold > 0.8))
    return ds


def test_run_scores_caches_and_sweeps(demo, bench_home):
    FakeAsr.calls = 0
    base = Config()
    base.speakers.resegment = False  # no embedder in this test
    grid = run.parse_grid(["speakers.diarize_threshold=0.5,0.9"])
    out = bench_home / "results"
    results = run.run(["demo"], "dev", ["clean", "noise:20"], grid, base, out,
                      progress=lambda m: None)
    assert len(results) == 4  # 2 conditions x 2 settings
    assert FakeAsr.calls == 2  # ASR cached per condition, shared by both settings
    by = {(r.condition, r.config): r.metrics for r in results}
    good = by[("clean", "speakers.diarize_threshold=0.5")]
    assert good["wer_errors"] == 0 and good["cpwer_errors"] == 0
    assert good["purity_errors"] == 0 and good["hyp_speakers"] == 2
    merged = by[("clean", "speakers.diarize_threshold=0.9")]
    assert merged["cpwer_errors"] > 0 and merged["purity_errors"] > 0  # B merged into A
    assert run.run(["demo"], "dev", ["clean"], grid, base, out,
                   progress=lambda m: None) == []  # resumable: nothing left to do
    rows = run.summarize(run.load_results(out / "results.jsonl"))
    assert len(rows) == 4 and {r["config"] for r in rows} == {
        "speakers.diarize_threshold=0.5", "speakers.diarize_threshold=0.9"}
    assert "| demo | clean |" in run.markdown(rows)
    baseline = [dict(r) for r in rows]
    assert run.compare(rows, baseline, 0.02) == []
    for b in baseline:
        if "cpWER" in b:
            b["cpWER"] = 0.0
    worse = run.compare(rows, baseline, 0.02)
    assert worse and all("cpWER" in w for w in worse)


def test_configure_types():
    c = run.configure(Config(), {"speakers.resegment": "false", "asr.beam_size": "3",
                                 "speakers.live_threshold": "0.45",
                                 "asr.languages": "de+en", "asr.model": "large-v3-turbo"})
    assert c.speakers.resegment is False and c.asr.beam_size == 3
    assert c.speakers.live_threshold == 0.45 and c.asr.languages == ["de", "en"]
    assert c.asr.model == "large-v3-turbo"
    assert run.parse_grid([]) == [{}]
    assert len(run.parse_grid(["a.b=1,2", "c.d=x,y,z"])) == 6


def test_cli_list_and_report(demo, bench_home, capsys):
    from transcribe.bench.suite.__main__ import main

    assert main(["list"]) == 0
    assert "demo" in capsys.readouterr().out
    out = bench_home / "bench" / "results" / "r"
    out.mkdir(parents=True)
    rec = run.Result("demo", "r1", "dev", "clean", "default", 6.0,
                     {"wer_errors": 1, "wer_total": 10})
    (out / "results.jsonl").write_text(json.dumps(rec.__dict__) + "\n")
    base = bench_home / "base.json"
    assert main(["report", "r", "--save-baseline", str(base)]) == 0
    rows = json.loads(base.read_text())
    rows[0]["WER"] = 0.0
    base.write_text(json.dumps(rows))
    assert main(["report", "r", "--baseline", str(base)]) == 1  # 10 % vs 0 %: regression
