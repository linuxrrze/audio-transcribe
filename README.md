# transcribe

Local, privacy-preserving live audio transcription with speaker identification
for Windows, Linux and macOS. CLI (`tx`) and Qt GUI share one headless core.

See [docs/PLAN.md](docs/PLAN.md) for the architecture and roadmap.

## Status
- **M0** core: transcript model, event bus, plugin registry, config, exporters
  (txt/md/srt/vtt/json), LocalAgreement streaming policy, online speaker clustering,
  voiceprint DB + open-set matching.
- **M1** file transcription: `tx transcribe FILE` — faster-whisper + sherpa-onnx
  diarization (pyannote segmentation 3.0 + WeSpeaker ResNet34), fully local, CPU or CUDA.
- **M2** live transcription: `tx live` — microphone capture, streaming ASR
  (LocalAgreement-2), provisional live speaker labels, crash-safe recording, and a
  refinement pass (full offline pipeline) when the session stops.

## Usage
```sh
uv sync --extra asr --extra speakers
uv run tx transcribe meeting.m4a                 # de/en auto, speakers auto
uv run tx transcribe talk.mp3 -n 2 -l de -f srt -o talk.srt
uv run tx export <session-dir> -f md
uv run tx models list

uv sync --extra asr --extra speakers --extra audio
uv run tx devices                                # list microphones
uv run tx live -i "USB" -n 3                     # Ctrl+C stops, then refines
uv run tx live --file meeting.wav                # replay a file in real time (testing)
uv run tx refine <session-dir> -m medium         # re-run refinement, e.g. bigger model
```
Any format ffmpeg reads works (decoded via PyAV). Models download on first use to the
user cache dir (override with `TRANSCRIBE_MODELS_DIR`); sessions are stored under the user
data dir (`sessions_dir` in `config.toml`).

### Mixed German/English
Whisper detects one language per call and can't be limited to a subset, so the pipeline
detects a language **per speaker** (restricted to `--lang`, default `de,en`), then
transcribes runs of same-language turns with the language pinned. Without diarization,
VAD regions serve as the language units. Limitation: one speaker switching language
mid-recording is transcribed in their dominant language.

### Diarization accuracy
Pass `--speakers N` when you know the count — it is the most reliable setting. Otherwise
the clustering threshold (`--diarize-threshold`, default 0.5; lower → more speakers)
decides; it has not been tuned on a real benchmark yet, and on a real 4-speaker sample
the default finds only 2.

### Live mode
- Partial text appears as you speak (dim = may still change); an utterance is finalized
  at the first pause of ~0.6 s. Median latency from end of speech to final text is ~3 s
  on CPU with `small` (16-core test machine). Every decode costs a full Whisper encoder
  pass (~1.3 s for `small` on CPU, regardless of thread count), so on slower machines use
  `-m base` live and set `[live] refine_model = "small"` (or larger) in `config.toml`.
- With several languages allowed, an utterance is only decoded once its language is
  known (≥1.5 s of speech): Whisper pinned to the wrong language translates instead of
  transcribing.
- Live speaker labels are **provisional** (all utterance embeddings are re-clustered
  after each new one; earlier lines may be relabelled). A speaker change without a pause
  stays in one utterance. The refinement pass on stop is authoritative; `--no-refine`
  keeps the live result. Both are saved (`transcript.live.json`, `transcript.json`).
- Audio is recorded as 30 s FLAC parts while running, so a crash loses at most 30 s;
  `tx refine` works on an interrupted session.
- Linux needs the PortAudio library for microphone input (`apt install libportaudio2`);
  the Windows/macOS wheels bundle it.

## Development
```sh
uv sync --extra asr --extra speakers --extra audio
uv run pytest                 # fast unit tests
TX_SLOW=1 uv run pytest       # + end-to-end tests (downloads ~650 MB of models)
uv run python -m transcribe.bench.dialogue out.wav   # synthetic de/en test dialogue
uv run python -m transcribe.bench.latency out.wav small  # real-time live latency
```
Optional extras: `asr`, `speakers`, `audio`, `gui`, `pyannote`.
