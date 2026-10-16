import numpy as np

from transcribe.core.config import Config, vocabulary
from transcribe.core.vocabulary import Vocabulary
from transcribe.storage.transcript_model import Word

TEXT = """# cast of episode 1
Justus, Peter, Bob
Fentriss
DSGVO   # acronym
Fanfress -> Fentriss
Mr Fan Fress -> Mr. Fentriss
"""


def test_parse_lines_commas_comments_and_corrections():
    v = Vocabulary.parse(TEXT)
    assert v.terms == ["Justus", "Peter", "Bob", "Fentriss", "DSGVO"]
    assert v.corrections == {"Fanfress": "Fentriss", "Mr Fan Fress": "Mr. Fentriss"}
    assert Vocabulary.parse(v.to_text()) == v


def test_hotwords_dedupe_include_targets_and_truncate():
    v = Vocabulary.parse(TEXT)
    assert v.hotwords() == "Justus, Peter, Bob, Fentriss, DSGVO, Mr. Fentriss"
    assert Vocabulary().hotwords() is None
    long = Vocabulary([f"Term{i}" for i in range(500)])
    assert len(long.hotwords()) <= 600


def w(text, t0, t1, spk="S1", conf=0.9):
    return Word(text, t0, t1, conf, spk)


def test_correct_single_word_keeps_spacing_punctuation_and_timing():
    v = Vocabulary.parse("Fanfress -> Fentriss")
    out = v.correct([w(" Mr.", 0, 0.4), w(" fanfress.", 0.4, 1.0, conf=0.5), w(" Ja", 1.1, 1.3)])
    assert [x.text for x in out] == [" Mr.", " Fentriss.", " Ja"]
    assert (out[1].t0, out[1].t1, out[1].speaker_id, out[1].confidence) == (0.4, 1.0, "S1", 0.5)


def test_correct_multi_word_merges_and_prefers_longest_rule():
    v = Vocabulary.parse("Fan -> Fan\nMr Fan Fress -> Mr. Fentriss")
    out = v.correct([w(" Mr", 0, 0.3), w(" Fan", 0.3, 0.6, conf=0.4), w(" Fress?", 0.6, 1.0),
                     w(" Fan", 2.0, 2.3)])
    assert [(x.text, x.t0, x.t1) for x in out] == [
        (" Mr. Fentriss?", 0, 1.0), (" Fan", 2.0, 2.3)]
    assert out[0].confidence == 0.4


def test_no_rules_is_identity():
    words = [w(" a", 0, 1)]
    assert Vocabulary.parse("Justus").correct(words) == words


def test_config_roundtrip(tmp_path):
    c = Config()
    c.asr.vocabulary = ["Justus", "DSGVO"]
    c.asr.corrections = {"Fanfress": "Fentriss"}
    c.save(tmp_path / "config.toml")
    loaded = Config.load(tmp_path / "config.toml")
    assert vocabulary(loaded) == Vocabulary(["Justus", "DSGVO"], {"Fanfress": "Fentriss"})


def test_engine_passes_hotwords_to_whisper(monkeypatch):
    import faster_whisper

    from transcribe.asr import faster_whisper as fw

    calls = {}

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            calls.update(kwargs)
            return iter(()), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(fw, "resolve_device", lambda d, c: ("cpu", "int8"))
    c = Config()
    c.asr.vocabulary = ["Justus", "Fentriss"]
    c.asr.corrections = {"Fanfress": "Fentriss"}
    engine = fw.FasterWhisperEngine.from_config(c)
    engine.transcribe(np.zeros(16000, np.float32))
    assert calls["hotwords"] == "Justus, Fentriss"
    assert engine.with_options(1, False).hotwords == "Justus, Fentriss"


def test_cli_merges_vocab_options(tmp_path, monkeypatch):
    from transcribe.cli.main import _config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    f = tmp_path / "names.txt"
    f.write_text("Hitchcock\nHitschcock -> Hitchcock\n", encoding="utf-8")
    c = _config(vocab="Justus, Peter", vocab_file=f)
    assert c.asr.vocabulary == ["Justus", "Peter", "Hitchcock"]
    assert c.asr.corrections == {"Hitschcock": "Hitchcock"}


def test_offline_pipeline_applies_corrections():
    from tests.unit.test_offline import FakeAsr, make_audio
    from transcribe.core.offline import transcribe_audio

    class NameAsr(FakeAsr):
        def transcribe(self, audio, offset=0.0, language=None, prompt=None, progress=None):
            return [Word(" Hallo", offset, offset + 0.5), Word(" Fanfress.", offset + 0.5,
                                                              offset + 1.0)]

    t = transcribe_audio(make_audio([(0, 1, "de")], 2.0), NameAsr(), None, ["de"],
                         vocabulary=Vocabulary.parse("Fanfress -> Fentriss"))
    assert t.segments[0].text == "Hallo Fentriss."


def test_live_session_applies_corrections(tmp_path):
    from tests.unit.fakes import ScriptedWorld, fake_vad, utterance
    from tests.unit.test_live_session import ArraySource
    from transcribe.live.session import LiveSession
    from transcribe.live.streamer import StreamerConfig

    world = ScriptedWorld(utterance("hallo herr fanfress wie geht es", 0.3, "de"))
    session = LiveSession(ArraySource(world.audio(3.5)), world, tmp_path, ["de"], fake_vad,
                          streamer_config=StreamerConfig(step=0.5),
                          vocabulary=Vocabulary.parse("fanfress -> Fentriss"))
    final = session.run_sync()
    assert final.segments[0].text == "hallo herr Fentriss wie geht es"


def test_style_prompt_only_without_other_prompt(monkeypatch):
    import faster_whisper

    from transcribe.asr import faster_whisper as fw

    seen = []

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kwargs):
            seen.append(kwargs["initial_prompt"])
            return iter(()), None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr(fw, "resolve_device", lambda d, c: ("cpu", "int8"))
    engine = fw.FasterWhisperEngine("small")
    audio = np.zeros(16000, np.float32)
    engine.transcribe(audio, language="de")
    engine.transcribe(audio, language="en", prompt="previous text")
    engine.transcribe(audio, language=None)
    fw.FasterWhisperEngine("small", style_prompt=False).transcribe(audio, language="de")
    assert seen == [fw.STYLE_PROMPTS["de"], "previous text", None, None]
