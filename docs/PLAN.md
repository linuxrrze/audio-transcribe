# Plan: Local Live Transcription Tool with Speaker Identification

## Context
We need a local, privacy-preserving tool for live audio transcription that runs on Windows, Linux and macOS. It should identify speakers, and it should remember voices across sessions using a voice-fingerprint store. It needs a CLI and a GUI. Audio input must be modular so that "record my Zoom/Teams/Jitsi session" can be added later without redesign. The repo (`/home/sandbox/git/transcription`) is empty, so this is a new project.

**Decisions made:** Python core + PySide6 (Qt) GUI · must run on CPU only, uses a GPU when present · German + English (auto-detected, mixed) · live provisional speaker labels plus a refinement pass at the end of a session.

---

## 1. Architecture overview

```
 ┌────────────┐   AudioFrames    ┌──────────┐  speech chunks  ┌────────────┐  words+timestamps
 │ AudioSource│ ───────────────▶ │  VAD +   │ ──────────────▶ │ ASR Engine │ ─────────┐
 │  (plugin)  │  16kHz f32 +ts   │ RingBuf  │                 │  (plugin)  │          ▼
 └────────────┘  + channel tag   └──────────┘                 └────────────┘   ┌────────────┐
                                     │ speech chunks                           │  Aligner   │──▶ TranscriptEvents
                                     ▼                                         │ word→spkr  │    (partial/final)
                              ┌──────────────┐  embeddings  ┌──────────────┐   └────────────┘        │
                              │ Speaker Emb. │ ───────────▶ │ Online       │ ──spk labels──▶ ▲       ▼
                              │  (plugin)    │              │ Clustering + │                    ┌─────────┐
                              └──────────────┘              │ VoiceprintDB │                    │EventBus │─▶ CLI / GUI / Exporters / (later) WebSocket
                                                            └──────────────┘                    └─────────┘
 Session end ─▶ Refinement pass: full-file diarization + optional re-ASR with a bigger model ─▶ final transcript
```

The core is a **headless library** (`transcribe.core`) built around a `Session` object that emits typed events. The CLI and the GUI are thin consumers of the same event stream, so neither frontend contains any pipeline logic.

### Threading model
*(As implemented in M2: a capture thread feeds a queue; one processing thread records, streams ASR and assigns speakers; events go through the EventBus dispatch thread. Completion is signalled with an Event, not `Thread.join`, because a KeyboardInterrupt inside `join()` can leave a still-running thread marked finished.)*
Each stage runs in a worker thread and passes data through bounded `queue.Queue`s. ML inference releases the GIL in CTranslate2 and ONNX Runtime, so threads are enough here; we don't need multiprocessing. The audio callback only copies data into a lock-free ring buffer and must never block. Backpressure: if ASR falls behind, it grows the chunks it processes rather than dropping audio.

---

## 2. Components & recommended technology

### 2.1 Audio input (modular, plugin-based)
`AudioSource` protocol: `open()`, `frames() -> Iterator[AudioFrame]`, `close()`, plus `describe()`, which returns its capabilities: channels, whether it carries speaker hints, and the sample rate.
- `AudioFrame`: `samples: np.ndarray[float32]`, `t_start` (monotonic, in session time), `channel_id` (e.g. `"mic"`, `"system"`, `"participant:alice"`), and an optional `speaker_hint`.
- Resampling to 16 kHz mono happens per channel inside the source adapter (`soxr`).

| Source | Windows | Linux | macOS |
|---|---|---|---|
| Microphone | `sounddevice` (PortAudio) | `sounddevice` | `sounddevice` (mic permission entitlement) |
| System audio (loopback) | WASAPI loopback (`PyAudioWPatch` or `soundcard`) | PipeWire/PulseAudio monitor source (`soundcard` / `pw-record`) | Core Audio Process Taps (macOS 14.2+) through a small bundled Swift helper that streams PCM over stdout; fallback: BlackHole virtual device |
| File | `ffmpeg`/`soundfile` decode, real-time or as-fast-as-possible | same | same |
| Mixer | combines N sources, keeps a separate channel per source | | |

**Designing for meetings later:** a meeting source is just a composite source. Its simplest form is mic + per-app loopback, i.e. "me" vs. "them" on separate channels, which already makes diarization much easier. Richer forms are bot or SDK integrations that deliver **per-participant streams with names**: Jitsi via `lib-jitsi-meet` or a Jibri-style bot; Zoom via the Meeting SDK or RTMS; Teams via the Graph/ACS bot. These would set `channel_id`/`speaker_hint`, and the diarizer must honour them, for example by skipping clustering when a channel is known to carry a single speaker. Sources are discovered via Python entry points (`transcribe.sources`) so that third parties can add them.

### 2.2 VAD
**Silero VAD (ONNX)**. It is tiny, works on CPU and is language-agnostic. It cuts speech into chunks (target 1–8 s, with a forced cut at about 15 s) and drives both ASR and the embedding extraction.

### 2.3 ASR engine (plugin: `transcribe.asr`)
- **Default: `faster-whisper` (CTranslate2)**. int8 on CPU, float16 on CUDA, and it returns word timestamps.
  - CPU default: `small` or `medium` (int8). GPU: `large-v3-turbo`. German quality: offer `primeline/whisper-large-v3-turbo-german` (CT2-converted) as a selectable model.
- **macOS acceleration:** a second backend using `whisper.cpp` (`pywhispercpp`, Metal/CoreML), because CTranslate2 has no Metal backend.
- **Live strategy:** LocalAgreement-2 streaming (the approach used by `whisper_streaming` / SimulStreaming). It re-decodes a growing window and commits a word once two consecutive hypotheses agree. This produces `PartialText` events (grey, may change) and `FinalText` events (committed). Expected latency is about 1.5–3 s on GPU and about 3–6 s with CPU-small.
- Language (implemented in M1): Whisper can't restrict detection to a subset, so the pipeline detects a language **per speaker** from that speaker's longest turns (restricted to `{de, en}`), then transcribes runs of same-language turns with the language pinned. Without diarization, VAD regions are the language units. The user can pin the language.
- Optional future backend: `sherpa-onnx` streaming transducers, for true low-latency and low-power operation.

### 2.4 Speaker diarization — two tiers
**Live tier (provisional):** *(M2 implementation: utterances end at the first VAD gap ≥ 0.4 s between padded regions — including gaps inside the buffer, since the next speaker often starts before the next decode. Each finalized segment ≥ 1 s is embedded with WeSpeaker; all embeddings are re-clustered with sherpa-onnx FastClustering after each new one and mapped onto previous ids, so labels stay stable and earlier lines can be relabelled. This replaced the incremental centroid clusterer: measured on a real 4-speaker clip, raw cosine similarities of different speakers overlapped heavily with same-speaker ones, so one-pass threshold decisions were unreliable.)*
1. For each VAD speech chunk (sub-windowed at about 1.5 s with 50 % overlap), compute a speaker embedding with **WeSpeaker ResNet34-LM (ONNX via `sherpa-onnx`)**. (3D-Speaker CAM++ was evaluated in M1 and dropped: it collapses to one speaker when the speaker count is fixed.) These are CPU-friendly, need no HF token and can be bundled.
2. **Online clustering:** compare each new embedding by cosine similarity against the active speakers' centroids and the voiceprint DB. If it is above threshold τ, assign it and update the centroid with an EMA. Otherwise create `Speaker N`. Run a periodic merge pass for clusters that drift together.
3. Use speaker-change detection to split ASR segments when an embedding switch falls mid-segment.
(Alternative to evaluate: `diart`, which is pyannote-based streaming diarization.)

**Refinement tier (session end, or on demand via `transcribe refine`):**
- Full-recording diarization with **pyannote `speaker-diarization-community-1`** (best quality; optional plugin, since it needs a one-time HF model download / token acceptance). The bundled fallback is the **sherpa-onnx offline diarization** (pyannote segmentation ONNX + embedding + clustering).
- Optionally re-run ASR with a larger model on the recorded audio.
- Map the refined clusters back to the live speakers and voiceprints (Hungarian matching on centroid similarity), so that names the user assigned during the session survive.

**Word→speaker alignment:** each word (with ASR word timestamps) gets the speaker with the largest overlap. Words in overlapping speech get the dominant speaker plus an `overlap: true` flag.

### 2.5 Voice fingerprint store (cross-session identification)
This is the investigation area the request asked for. Findings and design:

- **What a "fingerprint" is:** a set of fixed-length speaker embeddings (192–256-dim float vectors) for each person. We store **several exemplars plus a centroid**, not just one vector. That is more robust when the microphone, room or codec changes (headset vs. laptop mic vs. Zoom codec).
- **Embeddings depend on the model.** Vectors from different embedding models (or model versions) can't be compared with each other. Every embedding is therefore stored with `model_id` + `model_hash`. When the model changes, the stored **enrollment audio snippets** are re-embedded, so short clips are kept as well (opt-in).
- **Schema (SQLite in the `platformdirs` user data dir):**
  - `person(id, display_name, created_at, notes, consent_at)`
  - `voiceprint(id, person_id, model_id, vector BLOB, quality, duration_s, source_session, channel_type, created_at)`
  - `enroll_clip(id, person_id, path, duration_s)` (optional)
- **Matching:** open-set identification. Score = cosine against the person's centroid plus the best-k exemplars. Decision thresholds: "known" ≥ τ_hi, "suggest?" between τ_lo and τ_hi (the GUI asks "Is this Alice?"), otherwise unknown. Thresholds are calibrated per model on a validation set, and a target false-accept rate is shown in the settings.
- **Enrollment paths:**
  1. *Implicit:* after a session, the user renames "Speaker 2" → "Alice", and its clean, single-speaker segments are saved as voiceprints.
  2. *Explicit:* `transcribe speakers enroll "Alice" --record 20s` or from a file.
  - Quality gates: at least about 3 s of net speech per exemplar, no overlap, VAD confidence and SNR above a threshold, and a cap of N exemplars per person (diverse ones are kept, near-duplicates pruned).
- **Updating:** add exemplars only from high-confidence matches or user-confirmed segments. This prevents drift where one person's profile slowly absorbs another person's voice.
- **Management:** list, rename, merge two persons, split, delete a person (removes all vectors and clips), and export/import (JSON + npy) to move profiles between machines.
- **Privacy / legal:** voiceprints are **biometric data** under GDPR Art. 9, which matters for an FAU / EU context. Requirements: purely local, consent recorded per person (`consent_at`), optional encryption at rest (SQLCipher or Fernet with an OS-keyring key), one-click delete, a clear UI indicator when auto-identification is active, and a global "don't store voiceprints" mode. This is identification for convenience only. It is **not** authentication and has no anti-spoofing.

### 2.6 Session storage & export
Session folder: `audio.flac` (or Opus, chunked, so a crash loses at most seconds), `transcript.json` (the canonical model: segments → words with `t0`, `t1`, `speaker_id`, `confidence`, `lang`, `overlap`), and `session.toml` (config + model versions).
Exporters (plugin group `transcribe.exporters`): TXT, Markdown, SRT/WebVTT, JSON; later DOCX.

### 2.7 Frontends
**CLI (`typer` + `rich`)**, command name e.g. `tx`:
- `tx devices`: list mic and loopback sources
- `tx live [--source mic --source system] [--model small] [--lang auto|de|en] [--out DIR]`: rich live view; partial text dimmed, coloured speaker labels
- `tx transcribe FILE`: offline, with full diarization
- `tx refine SESSION`, `tx export SESSION --format srt|md|txt|json`
- `tx speakers list|enroll|rename|merge|delete|export|import`
- `tx models list|download`: models are fetched to a cache dir on first use, with offline mode supported

**GUI (PySide6):**
- Main window: source picker + level meters, start/pause/stop, and a live transcript view (speaker-coloured blocks, grey partials, autoscroll with "jump to live").
- Speaker side panel: active speakers, click to rename or assign an existing person, accept or decline "Is this Alice?" suggestions, merge speakers.
- Session browser: open past sessions, play audio synced to the text (click a word to seek), edit text, refine, export.
- Settings: models, device (CPU/GPU), language, voiceprint privacy options.
- It talks to the core in-process through a Qt adapter that marshals core events into Qt signals. A WebSocket/JSON event API (`tx serve`) can come later, for remote or other UIs.

### 2.8 Configuration & plugins
- TOML config in `platformdirs.user_config_dir`; CLI flags override it.
- Entry-point groups: `transcribe.sources`, `transcribe.asr`, `transcribe.embedders`, `transcribe.diarizers`, `transcribe.exporters`. A registry resolves plugins by name.

---

## 3. Project layout
```
transcription/
  pyproject.toml            # uv-managed; extras: [gpu], [pyannote], [gui], [mac-metal]
  src/transcribe/
    core/      session.py events.py pipeline.py ringbuffer.py config.py registry.py
    audio/     base.py mic.py loopback_win.py loopback_linux.py loopback_mac.py file.py mixer.py
    vad/       silero.py
    asr/       base.py faster_whisper.py whispercpp.py streaming.py   # LocalAgreement
    speakers/  embedder.py online_cluster.py voiceprint_db.py matching.py
    diarize/   base.py sherpa_offline.py pyannote_offline.py align.py refine.py
    storage/   session_store.py transcript_model.py
    export/    txt.py md.py srt.py json.py
    cli/       main.py
    gui/       app.py main_window.py transcript_view.py speaker_panel.py qt_bridge.py
  helpers/macos-audiotap/   # Swift CLI for Core Audio process taps
  tests/    unit/ integration/ fixtures/
  bench/    wer_der.py latency.py
```

---

## 4. Milestones
1. **M0 Skeleton:** pyproject (uv), event model, registry, config, CI matrix (win/linux/mac) with lint (ruff) + mypy + pytest.
2. **M1 Offline file pipeline:** FileSource → VAD → faster-whisper → sherpa offline diarization → alignment → JSON/TXT/SRT export. `tx transcribe` works. This is the baseline for quality metrics.
3. **M2 Live mic:** ring buffer, LocalAgreement streaming, live embeddings + online clustering, rich CLI live view, crash-safe session recording, end-of-session refinement.
4. **M3 Voiceprints:** SQLite store, enrollment (implicit and explicit), open-set matching with calibrated thresholds, `tx speakers` commands, privacy options.
5. **M4 GUI:** PySide6 app covering live view, speaker panel, session browser with synced playback.
6. **M5 System audio:** WASAPI loopback, PipeWire monitor, macOS audio-tap helper; mixer for mic + system as separate channels.
7. **M6 Packaging:** PyInstaller (or Briefcase) builds via GitHub Actions; macOS signing/notarization with mic and audio-capture entitlements; first-run model download wizard.
8. **Later:** meeting sources (per-app capture → Jitsi bot → Zoom/Teams SDKs), optional pyannote plugin, `tx serve` API, summarisation hooks.

## 5. Key risks
- **CPU latency:** mitigate with small int8 models, VAD gating, and a configurable latency/quality trade-off; the refinement pass recovers quality afterwards.
- **macOS system audio** needs native APIs and permissions: isolate it in a helper and provide BlackHole as a fallback.
- **Overlapping speech / similar voices:** live labels are provisional by design; the refinement pass and user correction fix them.
- **Cross-channel voiceprint mismatch** (in person vs. Zoom codec): store exemplars per channel type and prefer the same channel type when matching.
- **Licensing:** Whisper (MIT), faster-whisper (MIT), sherpa-onnx (Apache-2.0), and pyannote (MIT code; models are gated on HF, so it is an optional plugin). Check each model card before bundling. PySide6 is LGPL, which is fine for dynamic linking.

## 6. Verification
- **Unit tests:** ring buffer, LocalAgreement commit logic, alignment, online clustering on synthetic embeddings, voiceprint DB CRUD and matching thresholds, exporters (golden files).
- **Integration:** `tx transcribe tests/fixtures/two_speakers_de_en.wav` produces 2 speakers, and WER/DER fall within bounds. Replay a file through FileSource in real-time mode to test the live path deterministically.
- **Benchmarks (`bench/`):** WER via `jiwer`, DER via `pyannote.metrics` on an AMI subset, VoxConverse, and our own German meeting recordings; measure latency (speech-end → final text) per model/device; record a speaker-ID false-accept/false-reject curve to calibrate τ.
- **Cross-session test:** enroll speakers from session A; in session B with different audio, they are auto-labelled correctly.
- **Manual:** run the CLI live and the GUI on each OS with mic, and later system audio (e.g. a YouTube video plus your own mic).
