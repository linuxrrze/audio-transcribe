import numpy as np
import soundfile as sf

from transcribe.audio.base import SAMPLE_RATE
from transcribe.storage.recorder import MERGED, PARTS_DIR, AudioRecorder, load_recording


def tone(seconds, f=440.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_parts_rotate_and_merge(tmp_path):
    rec = AudioRecorder(tmp_path, part_seconds=1.0)
    audio = tone(2.5)
    for i in range(0, len(audio), 1600):
        rec.write(audio[i : i + 1600])
    assert len(list((tmp_path / PARTS_DIR).glob("part-*.flac"))) == 3
    rec.finalize()
    assert not (tmp_path / PARTS_DIR).exists()
    merged = load_recording(tmp_path)
    assert len(merged) == len(audio) and np.allclose(merged, audio, atol=1e-3)


def test_interrupted_session_is_recoverable(tmp_path):
    rec = AudioRecorder(tmp_path, part_seconds=1.0)
    rec.write(tone(2.2))  # crash: no finalize(); last part still open
    rec._close_part()  # simulate the OS flushing what was written
    (tmp_path / PARTS_DIR / "part-00003.flac").write_bytes(b"truncated garbage")
    got = load_recording(tmp_path)
    assert abs(len(got) / SAMPLE_RATE - 2.2) < 0.01
    assert not (tmp_path / MERGED).exists()
    sf.info(tmp_path / PARTS_DIR / "part-00000.flac")  # closed parts are valid files
