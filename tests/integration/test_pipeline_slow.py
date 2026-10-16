"""End-to-end on a synthetic German/English dialogue. Downloads ~650 MB of models."""

import pytest

from transcribe.audio.file import load_audio
from transcribe.bench.dialogue import SCRIPT, synthesize
from transcribe.bench.metrics import evaluate
from transcribe.core.config import Config
from transcribe.core.offline import load_engines, transcribe_audio

pytestmark = pytest.mark.slow


def test_dialogue_de_en(tmp_path):
    wav = tmp_path / "dialogue.wav"
    ref = synthesize(wav)
    config = Config()
    config.asr.model = "small"
    asr, diarizer = load_engines(config, diarize=True)
    transcript = transcribe_audio(load_audio(wav), asr, diarizer, ["de", "en"])
    report = evaluate(transcript, ref)
    assert report.num_hyp_speakers == 2
    assert report.speaker_accuracy > 0.95
    assert report.wer_by_language["de"] < 0.15
    assert report.wer_by_language["en"] < 0.15
    assert transcript.meta["speaker_languages"] == {"S1": "de", "S2": "en"}


FOUR_SPEAKERS_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     "speaker-segmentation-models/0-four-speakers-zh.wav")


def test_diarize_real_four_speakers_with_known_count(tmp_path):
    import urllib.request

    from transcribe.diarize.sherpa_offline import SherpaDiarizer

    wav = tmp_path / "four.wav"
    urllib.request.urlretrieve(FOUR_SPEAKERS_URL, wav)
    turns = SherpaDiarizer().diarize(load_audio(wav), num_speakers=4)
    assert len({t.speaker_id for t in turns}) == 4


def test_live_replay_dialogue(tmp_path):
    """Deterministic live replay; checks live quality (refinement off) and refinement."""
    from transcribe.audio.file import FileSource
    from transcribe.live.factory import build_live_session
    from transcribe.live.session import LIVE_TRANSCRIPT
    from transcribe.storage.transcript_model import Transcript

    wav = tmp_path / "dialogue.wav"
    ref = synthesize(wav)
    session = build_live_session(FileSource(wav), Config(), tmp_path / "session")
    final = session.run_sync(refine=True)
    live = Transcript.load(tmp_path / "session" / LIVE_TRANSCRIPT)
    for transcript in (live, final):
        report = evaluate(transcript, ref)
        assert report.num_hyp_speakers == 2
        assert report.speaker_accuracy > 0.95
        assert max(report.wer_by_language.values()) < 0.15


def test_cross_session_identification(tmp_path):
    """Name speakers in session 1; session 2 recognises them, a new voice stays unknown."""
    from transcribe.speakers.embedder import SherpaEmbedder
    from transcribe.speakers.identity import name_speaker, process_session
    from transcribe.speakers.voiceprint_db import VoiceprintDB

    config = Config()
    db = VoiceprintDB(tmp_path / "vp.sqlite")
    asr, diarizer = load_engines(config, diarize=True)
    embedder = SherpaEmbedder.from_config(config)

    def session(name, script):
        wav = tmp_path / f"{name}.wav"
        ref = synthesize(wav, script=script)
        audio = load_audio(wav)
        transcript = transcribe_audio(audio, asr, diarizer, ["de", "en"])
        sdir = tmp_path / name
        sdir.mkdir()
        process_session(sdir, audio, transcript, embedder, config, db)
        transcript.save(sdir / "transcript.json")
        return sdir, transcript, ref

    s1, t1, _ = session("one", SCRIPT)
    first: dict[str, str] = {}
    for seg in t1.segments:
        first.setdefault(seg.speaker_id, seg.text)
    assert first["S1"].startswith("Guten Morgen") and first["S2"].startswith("Good morning")
    assert name_speaker(s1, s1 / "transcript.json", "S1", "Thorsten", config, True, db=db)
    assert name_speaker(s1, s1 / "transcript.json", "S2", "Amy", config, True, db=db)

    _, t2, ref2 = session("two", [
        ("kerstin", "Guten Tag zusammen, ich bin neu im Team und stelle mich kurz vor."),
        ("amy", "Welcome to the team. We are happy to have you here with us this year."),
        ("thorsten", "Willkommen! Ich zeige dir nachher gerne unser Büro und die Küche."),
        ("kerstin", "Vielen Dank, das ist sehr nett. Ich freue mich auf die Zusammenarbeit."),
        ("amy", "If you have any questions about the project, just send me a message."),
    ])
    # 1. Nothing lost: every reference turn has transcribed words (both languages).
    for r in ref2:
        assert any(r.t0 - 0.5 <= seg.t0 <= r.t1 for seg in t2.segments), r
    # 2. Never a confident wrong name: a labelled segment's voice must match its label.
    names = {"thorsten": "Thorsten", "amy": "Amy"}
    for seg in t2.segments:
        spk = t2.speakers[seg.speaker_id]
        voice = next(r.speaker for r in ref2 if r.t0 - 0.5 <= seg.t0 <= r.t1)
        if spk.label in names.values():
            assert names.get(voice) == spk.label, (seg.text, voice, spk)
    # 3. Thorsten (diarized cleanly here) is recognised: labelled or at least suggested.
    thorsten = [t2.speakers[seg.speaker_id] for seg in t2.segments
                if next(r.speaker for r in ref2 if r.t0 - 0.5 <= seg.t0 <= r.t1) == "thorsten"]
    assert any("Thorsten" in (spk.label, spk.suggestion) for spk in thorsten)
    # (Amy is not asserted: diarization merges the two female voices in this clip, and a
    # merged speaker must not be labelled - covered by the consistency checks.)
