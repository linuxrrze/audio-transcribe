import pytest

from tests.unit.fakes import ScriptedWorld, fake_vad, utterance
from transcribe.audio.base import SAMPLE_RATE
from transcribe.live.streamer import ChannelStreamer, StreamerConfig


def run(world, languages, duration, step=1.0, frame=0.1, **cfg):
    finals, partials = [], []
    st = ChannelStreamer(world, languages, fake_vad,
                         on_partial=lambda t, c, p: partials.append((t, c, p)),
                         on_final=lambda seg, audio: finals.append(seg),
                         config=StreamerConfig(step=step, **cfg))
    audio = world.audio(duration)
    n = int(frame * SAMPLE_RATE)
    for i in range(0, len(audio), n):
        st.feed(audio[i : i + n])
        st.update()
    st.finish()
    return st, finals, partials


def test_back_to_back_turns_are_split_at_internal_gap():
    # Regression: the next turn starts 0.7 s later, before any decode sees trailing
    # silence; the utterance must still end at the gap, each with its own language.
    words = utterance("guten morgen wir besprechen heute", 0.3, "de") + \
        utterance("good morning I reviewed the draft", 2.95, "en")
    world = ScriptedWorld(words)
    _, finals, _ = run(world, ["de", "en"], duration=6.0, step=1.3)
    assert [(s.lang, s.text) for s in finals] == [
        ("de", "guten morgen wir besprechen heute"),
        ("en", "good morning I reviewed the draft")]
    assert "???" not in " ".join(s.text for s in finals)  # never decoded in wrong language


def test_waits_for_language_before_decoding():
    world = ScriptedWorld(utterance("one two three four five six seven", 0.0, "en"))
    run(world, ["de", "en"], duration=4.0, step=0.5, min_lid_speech=1.5)
    first_decode_end = world.decode_calls[0][1]
    assert first_decode_end >= 1.5
    assert all(lang == "en" for _, _, lang in world.decode_calls)


def test_single_language_never_detects():
    world = ScriptedWorld(utterance("eins zwei drei", 0.0, "de"))
    _, finals, _ = run(world, ["de"], duration=3.0)
    assert world.detect_calls == 0 and finals[0].text == "eins zwei drei"


def test_long_monologue_is_force_finalized():
    text = " ".join(f"w{i}" for i in range(60))  # 24 s without pauses
    world = ScriptedWorld(utterance(text, 0.0, "de"))
    st, finals, _ = run(world, ["de"], duration=26.0, max_buffer=8.0)
    assert len(finals) >= 3
    assert " ".join(s.text for s in finals).split() == text.split()  # nothing lost/duplicated


def test_partials_grow_then_clear():
    world = ScriptedWorld(utterance("a b c d e f g h", 0.0, "en"))
    _, finals, partials = run(world, ["en"], duration=5.0, step=0.5)
    committed = [c for _, c, _ in partials if c]
    assert committed == sorted(committed, key=len)  # committed text only grows
    assert partials[-1][1:] == ("", "")  # cleared after finalization
    assert finals[0].text == "a b c d e f g h"


def test_finish_flushes_open_utterance():
    world = ScriptedWorld(utterance("still talking when stopped", 0.0, "en"))
    _, finals, _ = run(world, ["en"], duration=1.6, step=10.0)  # never decodes before finish
    assert [s.text for s in finals] == ["still talking when stopped"]


@pytest.mark.parametrize("step", [0.5, 1.0, 2.5])
def test_result_independent_of_decode_step(step):
    words = (utterance("erste aussage hier", 0.2, "de") + utterance("second one", 2.5, "en")
             + utterance("dritte", 4.0, "de"))
    world = ScriptedWorld(words)
    _, finals, _ = run(world, ["de", "en"], duration=6.0, step=step)
    assert [s.text for s in finals] == ["erste aussage hier", "second one", "dritte"]
