"""File transcriptions keep their audio, so `tx refine` can re-run them later."""

import numpy as np
import pytest
import soundfile as sf
import tomli_w
from typer.testing import CliRunner

from transcribe import sessions
from transcribe.audio.base import SAMPLE_RATE
from transcribe.cli.main import app
from transcribe.core.config import Config
from transcribe.storage.recorder import MERGED, load_recording, recorded_channels
from transcribe.storage.transcript_model import Segment, Speaker, Transcript, Word


def tone(seconds, f=440.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    c = Config()
    c.sessions_dir = str(tmp_path / "sessions")
    c.speakers.identify_known = c.speakers.store_voiceprints = False  # no embedder
    c.save()
    return c


@pytest.fixture
def engines(monkeypatch):
    """Fake models: the transcript says how much audio the pipeline was given."""
    calls: list[np.ndarray] = []

    def transcribe_audio(audio, asr, diarizer, languages, num_speakers=None, progress=None,
                         vocabulary=None):
        calls.append(audio)
        text = f" {len(audio) / SAMPLE_RATE:.1f}s"
        seg = Segment(0.0, 1.0, [Word(text, 0.0, 1.0, 0.9, "S1")], "S1", "en", "default")
        return Transcript(segments=[seg], speakers={"S1": Speaker("S1", "Speaker 1")})

    monkeypatch.setattr("transcribe.core.offline.load_engines",
                        lambda config, diarize=True: (None, None))
    monkeypatch.setattr("transcribe.core.offline.transcribe_audio", transcribe_audio)
    return calls


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "meeting.wav"
    sf.write(path, tone(2.0), SAMPLE_RATE, subtype="PCM_16")
    return path


def only_session(config):
    (path,) = config.resolved_sessions_dir().iterdir()
    return path


def test_file_session_keeps_audio_and_refines(config, engines, wav):
    out = CliRunner().invoke(app, ["transcribe", str(wav)])
    assert out.exit_code == 0, out.output
    d = only_session(config)
    assert recorded_channels(d) == [""]  # same layout as a live recording
    assert sf.info(d / MERGED).samplerate == SAMPLE_RATE and sf.info(d / MERGED).channels == 1
    assert np.allclose(load_recording(d), tone(2.0), atol=1e-3)
    assert sessions.playback_audio(d) == d / MERGED

    wav.unlink()  # the original is gone: refine uses the stored audio
    out = CliRunner().invoke(app, ["refine", str(d)])
    assert out.exit_code == 0, out.output
    assert len(engines[-1]) == len(tone(2.0))
    t = sessions.load(d)
    assert t.meta["refined"] and t.meta["title"] == "meeting"


@pytest.mark.parametrize("flag,setting,stored", [
    ("--no-keep-audio", True, False),
    (None, False, False),
    ("--keep-audio", False, True),
])
def test_keep_audio_flag_overrides_config(config, engines, wav, flag, setting, stored):
    config.keep_file_audio = setting
    config.save()
    out = CliRunner().invoke(app, ["transcribe", str(wav), *([flag] if flag else [])])
    assert out.exit_code == 0, out.output
    d = only_session(config)
    assert (d / MERGED).exists() == stored
    assert recorded_channels(d) == ([""] if stored else [])


def old_file_session(config, source):
    """A session written before file audio was kept: transcript + source path only."""
    d = config.resolved_sessions_dir() / "2026-01-01_120000_meeting"
    d.mkdir(parents=True)
    Transcript(meta={"title": "meeting"}).save(d / "transcript.json")
    (d / "session.toml").write_text(tomli_w.dumps({"source": str(source)}), encoding="utf-8")
    return d


def test_refine_old_session_falls_back_to_source_and_stores_it(config, engines, wav):
    d = old_file_session(config, wav)
    out = CliRunner().invoke(app, ["refine", str(d)])
    assert out.exit_code == 0, out.output
    assert len(engines[-1]) == len(tone(2.0))
    assert np.allclose(load_recording(d), tone(2.0), atol=1e-3)  # kept from now on
    assert sessions.load(d).meta["refined"]


def test_refine_old_session_without_source_fails_clearly(config, engines, tmp_path):
    d = old_file_session(config, tmp_path / "moved.wav")
    out = CliRunner().invoke(app, ["refine", str(d)])
    assert out.exit_code == 1
    assert "no longer exists" in " ".join(out.output.split())
    assert "diarize" not in out.output  # no progress bars before the error
    assert not engines


def test_session_audio_respects_keep(config, wav):
    d = old_file_session(config, wav)
    assert len(sessions.session_audio(d, keep=False)[""]) == len(tone(2.0))
    assert not (d / MERGED).exists()


def test_refine_keeps_previous_transcript(config, engines, wav):
    assert CliRunner().invoke(app, ["transcribe", str(wav)]).exit_code == 0
    d = only_session(config)
    first = (d / "transcript.json").read_text()
    for _ in range(2):
        out = CliRunner().invoke(app, ["refine", str(d)])
        assert out.exit_code == 0, out.output
    kept = sorted((d / sessions.VERSIONS).iterdir())
    assert len(kept) == 2 and kept[0].read_text() == first  # same-second names sort in order
    assert all(k.name.startswith("transcript.") for k in kept)
    assert [s.path for s in sessions.list_sessions(config.resolved_sessions_dir())] == [d]
