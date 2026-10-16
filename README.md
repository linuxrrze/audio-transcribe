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

## Usage
```sh
uv sync --extra asr --extra speakers
uv run tx transcribe meeting.m4a                 # de/en auto, speakers auto
uv run tx transcribe talk.mp3 -n 2 -l de -f srt -o talk.srt
uv run tx export <session-dir> -f md
uv run tx models list
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

## Development
```sh
uv sync --extra asr --extra speakers
uv run pytest                 # fast unit tests
TX_SLOW=1 uv run pytest       # + end-to-end tests (downloads ~650 MB of models)
uv run python -m transcribe.bench.dialogue out.wav   # synthetic de/en test dialogue
```
Optional extras: `asr`, `speakers`, `audio`, `gui`, `pyannote`.
