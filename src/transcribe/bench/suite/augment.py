"""Recording conditions applied to clean benchmark audio (deterministic per seed).

    clean           unchanged
    noise:<snr>     pink-ish noise at <snr> dB (e.g. noise:10)
    reverb:<rt60>   synthetic room, reverberation time in seconds (e.g. reverb:0.6)
    phone           8 kHz telephone band + light noise
    mp3:<kbps>      MP3 round trip (e.g. mp3:32)
    opus:<kbps>     Opus round trip, as in video calls (e.g. opus:16)

Conditions combine with "+": ``reverb:0.5+noise:15``.
"""

from __future__ import annotations

import io

import numpy as np

from transcribe.audio.base import SAMPLE_RATE

CONDITIONS = ["clean", "noise:20", "noise:10", "noise:5", "reverb:0.3", "reverb:0.8",
              "phone", "mp3:32", "opus:16", "reverb:0.5+noise:15"]


def apply(audio: np.ndarray, condition: str, seed: int = 0) -> np.ndarray:
    out = np.asarray(audio, np.float32)
    for part in condition.split("+"):
        kind, _, arg = part.strip().partition(":")
        rng = np.random.default_rng(seed)
        if kind == "clean":
            continue
        if kind == "noise":
            out = add_noise(out, float(arg), rng)
        elif kind == "reverb":
            out = reverb(out, float(arg), rng)
        elif kind == "phone":
            out = phone(out, rng)
        elif kind in ("mp3", "opus"):
            out = codec(out, kind, int(arg))
        else:
            raise ValueError(f"Unknown condition {part!r}; see transcribe.bench.suite.augment")
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    spectrum /= np.sqrt(np.maximum(f, 20.0))  # 1/f power, flat below 20 Hz
    x = np.fft.irfft(spectrum, n)
    return x / (np.std(x) or 1.0)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    active = np.abs(audio) > 1e-3  # speech level, not counting silence
    power = float(np.mean(audio[active] ** 2)) if active.any() else 1e-6
    noise = _pink(len(audio), rng) * np.sqrt(power / 10 ** (snr_db / 10))
    return audio + noise.astype(np.float32)


def reverb(audio: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Convolve with an exponentially decaying noise impulse response (60 dB in rt60)."""
    n = int(rt60 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0  # direct sound
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.fft.irfft(np.fft.rfft(audio, len(audio) + n) * np.fft.rfft(ir, len(audio) + n))
    wet = wet[:len(audio)]
    gain = np.sqrt(np.mean(audio ** 2) / max(float(np.mean(wet ** 2)), 1e-12))
    return (wet * gain).astype(np.float32)  # same loudness as the dry signal


def phone(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """300-3400 Hz band at 8 kHz, back to 16 kHz, light line noise."""
    import soxr

    low = soxr.resample(audio, SAMPLE_RATE, 8000)
    spectrum = np.fft.rfft(low)
    f = np.fft.rfftfreq(len(low), 1 / 8000)
    spectrum[(f < 300) | (f > 3400)] = 0
    band = np.fft.irfft(spectrum, len(low)).astype(np.float32)
    return add_noise(soxr.resample(band, 8000, SAMPLE_RATE)[:len(audio)], 30.0, rng)


def codec(audio: np.ndarray, kind: str, kbps: int) -> np.ndarray:
    """Encode and decode with PyAV (MP3 at 16 kHz, Opus at 48 kHz like WebRTC)."""
    import av
    import soxr

    rate = 48000 if kind == "opus" else SAMPLE_RATE
    samples = soxr.resample(audio, SAMPLE_RATE, rate) if rate != SAMPLE_RATE else audio
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg" if kind == "opus" else "mp3") as out:
        stream = out.add_stream("libopus" if kind == "opus" else "libmp3lame", rate=rate)
        stream.bit_rate = kbps * 1000
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples[None, :].astype(np.float32),
                                           format="flt", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    buf.seek(0)
    with av.open(buf, "r") as inp:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = [f.to_ndarray().ravel() for frame in inp.decode(audio=0)
                  for f in resampler.resample(frame)]
    decoded = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    out_audio = np.zeros(len(audio), np.float32)  # codec delay/padding: same length
    out_audio[:min(len(audio), len(decoded))] = decoded[:len(audio)]
    return out_audio
