"""End-to-end on a synthetic German/English dialogue. Downloads ~650 MB of models."""

import pytest

from transcribe.audio.file import load_audio
from transcribe.bench.dialogue import synthesize
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
