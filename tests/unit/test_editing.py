"""Manual transcript corrections: speakers per line, merges, text edits, exemplars."""

import numpy as np
import pytest
from typer.testing import CliRunner

from transcribe import sessions
from transcribe.core.config import Config
from transcribe.speakers.identity import Exemplar, load_exemplars, save_exemplars
from transcribe.storage import editing
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


def seg(seg_id, speaker, *words):
    ws = [Word(" " + text, t0, t1, 0.9, speaker) for text, t0, t1 in words]
    return Segment(ws[0].t0, ws[-1].t1, ws, speaker, "de", "default", seg_id)


def transcript():
    return Transcript(
        segments=[seg("a", "S1", ("Hey,", 0.0, 0.3), ("Peter.", 0.3, 0.8)),
                  seg("b", "S2", ("Ich", 1.0, 1.2), ("habs", 1.2, 1.5), ("gehört.", 1.5, 2.0)),
                  seg("c", "S1", ("Wer", 3.0, 3.2), ("war", 3.2, 3.4), ("das?", 3.4, 3.8))],
        speakers={"S1": Speaker("S1", "Speaker 1"), "S2": Speaker("S2", "Speaker 2")})


def test_number_segments_only_when_needed():
    t = transcript()
    t.number_segments()
    assert [s.id for s in t.segments] == ["a", "b", "c"]
    t.segments[1].id = "a"
    t.number_segments()
    assert [s.id for s in t.segments] == ["seg-0", "seg-1", "seg-2"]


def test_assign_moves_words_and_drops_empty_unnamed_speakers():
    t = transcript()
    result = editing.assign(t, ["b"], "S1")
    assert all(w.speaker_id == "S1" for w in t.segments[1].words)
    assert result.moved == {"b": "S1"} and result.removed_speakers == {"S2"}
    assert set(t.speakers) == {"S1"}
    new = editing.add_speaker(t, "Carlos")
    assert new == "S2" and t.speakers[new].confirmed
    editing.assign(t, ["a", "c", "b"], new)
    assert "S1" not in t.speakers  # unnamed and empty
    t.speakers["S9"] = Speaker("S9", "Justus", confirmed=True)
    editing.assign(t, ["a"], "S9")
    editing.assign(t, ["a"], new)
    assert "S9" in t.speakers  # named by the user: kept even without lines
    with pytest.raises(KeyError):
        editing.assign(t, ["a"], "S42")


def test_find_speaker_by_id_or_name():
    t = transcript()
    t.speakers["S2"].label = "Justus"
    assert editing.find_speaker(t, "S1") == "S1"
    assert editing.find_speaker(t, "justus") == "S2"
    assert editing.find_speaker(t, "Speaker 1") == "S1"
    assert editing.find_speaker(t, "Bob") is None


def test_merge():
    t = transcript()
    result = editing.merge(t, "S1", "S2")
    assert set(t.speakers) == {"S1"} and result.renamed_speakers == {"S2": "S1"}
    assert {w.speaker_id for s in t.segments for w in s.words} == {"S1"}
    with pytest.raises(ValueError):
        editing.merge(t, "S1", "S1")


def test_retime_keeps_unchanged_words_and_spreads_replacements():
    old = transcript().segments[1].words  # Ich 1.0-1.2, habs 1.2-1.5, gehört. 1.5-2.0
    words = editing.retime(old, ["Ich", "hab's", "es", "gehört."], 1.0, 2.0)
    assert [w.text for w in words] == [" Ich", " hab's", " es", " gehört."]
    assert (words[0].t0, words[0].t1) == (1.0, 1.2) and words[0].confidence == 0.9
    assert (words[-1].t0, words[-1].t1) == (1.5, 2.0)
    assert words[1].t0 == 1.2 and words[2].t1 == pytest.approx(1.5)  # share habs' span
    assert words[1].confidence == 1.0  # typed by the user
    assert all(a.t1 <= b.t0 + 1e-9 for a, b in zip(words, words[1:], strict=False))
    # case-only changes keep the timing; inserts at the end use the segment end
    words = editing.retime(old, ["ich", "habs", "gehört.", "Ja."], 1.0, 2.5)
    assert words[0].text == " ich" and words[0].t0 == 1.0
    assert words[-1].t0 >= 2.0 and words[-1].t1 <= 2.5 + 0.01


def test_set_text_splits_and_deletes():
    t = transcript()
    result = editing.set_text(t, "b", "Ich habs\ngehört.")
    assert [s.id for s in t.segments] == ["a", "b", "b.1", "c"]
    assert t.segments[1].text == "Ich habs" and t.segments[2].text == "gehört."
    assert t.segments[2].t0 == 1.5 and t.segments[2].speaker_id == "S2"
    assert result.changed_audio == {"b"} and result.new_segments == ["b.1"]
    # a plain correction does not touch exemplars
    assert editing.set_text(t, "a", "Hey, Peter!").changed_audio == set()
    editing.set_text(t, "b.1", "  \n")
    assert [s.id for s in t.segments] == ["a", "b", "c"]
    editing.set_text(t, "b", "")
    assert "S2" not in t.speakers


def make_session(root, exemplars):
    d = root / "2026-01-01_120000_test"
    d.mkdir(parents=True)
    t = transcript()
    for s in t.segments:
        s.id = ""  # written before segments had ids
    t.save(d / "transcript.json")
    vec = np.ones(4, np.float32)
    save_exemplars(d, [Exemplar(sid, vec, 5.0, seg_id) for sid, seg_id in exemplars], "m")
    return d


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    c = Config()
    c.sessions_dir = str(tmp_path / "sessions")
    c.save()
    return c


def test_session_edits_keep_exemplars_in_step(tmp_path, config):
    d = make_session(tmp_path / "sessions",
                     [("S1", "seg-0"), ("S1", "seg-2"), ("S2", "seg-1"), ("S1", ""),
                      ("S2", "")])
    sid = sessions.assign_segments(d, ["seg-2"], "", "Bob", config)
    assert sid == "S3"
    t = sessions.load(d)
    assert [s.id for s in t.segments] == ["seg-0", "seg-1", "seg-2"]  # numbered, saved
    items = [(e.speaker_id, e.segment_id) for e in load_exemplars(d)[1]]
    # seg-2's exemplar follows it; S1's concatenated one may contain seg-2: dropped
    assert items == [("S1", "seg-0"), ("S3", "seg-2"), ("S2", "seg-1"), ("S2", "")]
    sessions.merge_speakers(d, "S1", "S2", config)
    items = [(e.speaker_id, e.segment_id) for e in load_exemplars(d)[1]]
    assert items == [("S1", "seg-0"), ("S3", "seg-2"), ("S1", "seg-1"), ("S1", "")]
    assert sessions.edit_text(d, "seg-0", "Hey,\nPeter.", config) == ["seg-0", "seg-0.1"]
    assert ("S1", "seg-0") not in [(e.speaker_id, e.segment_id)
                                   for e in load_exemplars(d)[1]]


def test_cli_sessions_workflow(tmp_path, config):
    from transcribe.cli.main import app

    d = make_session(tmp_path / "sessions", [])
    run = CliRunner().invoke

    out = run(app, ["sessions", "speakers", "latest"])
    assert out.exit_code == 0 and "Speaker 2" in out.output
    out = run(app, ["sessions", "assign", d.name, "0:00-0:02", "Carlos", "--from", "S1"])
    assert out.exit_code == 0, out.output
    t = sessions.load(d)
    assert [t.speaker_label(s.speaker_id) for s in t.segments] == ["Carlos", "Speaker 2",
                                                                    "Speaker 1"]
    out = run(app, ["sessions", "edit", "test", "1", "Ich hab's|gehört."])
    assert out.exit_code == 0, out.output
    assert [s.text for s in sessions.load(d).segments][1:3] == ["Ich hab's", "gehört."]
    out = run(app, ["sessions", "merge", "latest", "Carlos", "S1", "S2"])
    assert out.exit_code == 0, out.output
    out = run(app, ["sessions", "rename", "latest", "Carlos", "Justus"])
    t = sessions.load(d)
    assert set(t.speakers) == {"S3"} and t.speakers["S3"].label == "Justus"
    out = run(app, ["sessions", "assign", "latest", "9:00", "S3"])
    assert out.exit_code == 1 and "No line at 9:00" in out.output
    out = run(app, ["sessions", "list"])
    assert "2026-01-01_120000_test" in out.output
