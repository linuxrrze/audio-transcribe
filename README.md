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
- **M3** voiceprints: people are recognised across sessions. Name a speaker once
  (`tx speakers name`) or enroll from audio (`tx speakers enroll`); later sessions label
  them automatically or ask "might be X?". Consent, no-storage mode, encryption at rest.
- **M4** desktop app (`tx gui` / `tx-gui`): live view, speaker panel (name, merge,
  "Is this X?" suggestions), session browser with playback synced to the text, export,
  refine, settings and a "Known people" privacy dialog.
- **M6** packaging: PyInstaller bundles (CLI + desktop app) for Linux, Windows and macOS
  built by GitHub Actions; first-run model download; CI on all three platforms.
- **M5** system audio: transcribe what the computer plays (`--source system`), or
  microphone + system audio as separate channels (`--source both`) with echo suppression
  - the basis for recording Zoom/Teams/Jitsi calls.

## Install
```sh
uv sync                      # in the repo: project .venv, then `uv run tx ...`
pip install .                # or into any venv (also `uv pip install .`), then `tx ...`
pip install ".[gui]"         # + desktop app; extras: gui, secure (encryption), system (Windows)
```
Linux also needs the PortAudio library for microphones (`apt install libportaudio2`).
Note: `uv run` always uses the project's `.venv`, not an activated venv (`uv run --active`
uses the active one).

## Usage
```sh
uv run tx transcribe meeting.m4a                 # de/en auto, speakers auto
uv run tx transcribe talk.mp3 -n 2 -l de -f srt -o talk.srt
uv run tx export <session-dir> -f md
uv run tx models list

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

### Custom vocabulary (names, acronyms, jargon)
Whisper guesses unknown names phonetically ("Fanfress" for *Fentriss*). Give it hints:
```sh
tx transcribe folge1.m4a --vocab "Justus, Peter, Bob, Fentriss, Hitchcock"
tx transcribe meeting.m4a --vocab-file vocab.txt
```
`vocab.txt` - one entry per line, `#` comments, `wrong -> right` corrections:
```
# cast
Justus, Peter, Bob
Fentriss
DSGVO          # acronyms and jargon work the same way
Fanfress -> Fentriss          # fix what the model still mishears
Mr Fan Fress -> Mr. Fentriss  # multi-word, keeps the timestamps
```
- Hint words are passed to Whisper as *hotwords* for every 30 s window (not just the
  start); keep the list short (~100 words) - it shares Whisper's prompt space.
- Corrections run on the recognised words (case-insensitive, whole words) in file
  transcription, live mode and refinement.
- Permanent entries: `[asr] vocabulary = [...]` and `[asr.corrections]` in
  `config.toml`, or **File → Settings → Vocabulary** in the app. `--vocab` /
  `--vocab-file` add to them for one run.

### Speaker accuracy
Speakers are found in two passes: the diarizer (pyannote segmentation + WeSpeaker), then
a second pass that cuts the transcript into phrases (sentence ends, pauses ≥ 0.5 s),
embeds each phrase and clusters them again. This fixes the diarizer's typical errors:
merged turns in fast dialogue, mixed clusters, and far too many speakers.

| Word speaker accuracy | diarizer only | + second pass (default) |
|---|---|---|
| AMI meeting ES2004a (4 people) | 45.5 % (55 speakers) | 82.1 % (15) |
| AMI meeting IS1009a (4 people) | 62.5 % (15) | 83.2 % (10) |
| synthetic radio play (6 voices) | 71.2 % (6) | 98.8 % (6) |

On a 50-minute radio play (no reference): 21 → 13 speakers, segments starting
mid-sentence 22 % → 2 %.

- **Pass `--speakers N` when you know the count** - the most reliable setting.
- Without it the default may **over-split** (one person → two labels); merge them in the
  app or with `tx speakers` - merged voices could not be separated again.
- `[speakers] speaker_count = "silhouette"` picks the count from the voices themselves:
  best for meetings with distinct voices (ES2004a: 97.3 %, exactly 4 speakers), but it
  merged a radio play's similar young voices into 3 speakers.
- Similar voices (e.g. several teenage boys in one studio) stay the hardest case.
- Numbers come from three recordings. Reproduce the AMI rows with
  `python -m transcribe.bench.diarization` (downloads ~60 MB; ~10 min on CPU).

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

### Desktop app
```sh
uv sync --extra gui
uv run tx gui
```
![Sessions tab](docs/screenshots/sessions.png)

- **Live:** pick microphone, system audio or both; partial text appears dimmed, finished
  lines get coloured speaker labels. Naming a speaker while recording applies at once;
  the voice is learned (with the consent checkbox ticked) when the session stops.
- **Sessions:** click a timestamp to play from there; the playing line is highlighted.
  Fix diarization by merging speakers, accept or decline suggestions, refine (optionally
  with a known speaker count), export.
- **File → Known people:** see and delete stored voiceprints.
- The GUI is a thin layer over the same core as the CLI (`transcribe.sessions`,
  `transcribe.live.factory`); core events reach the Qt thread through a signal bridge,
  model loading and refinement run in background threads.

### System audio / calls (`--source system|both`)
```sh
uv run tx devices                         # inputs + system-audio sources
uv run tx live -s both -n 3               # you (mic) + call participants (system audio)
uv run tx live -s system --output <id>    # only what a given output device plays
uv run tx live -s both --mix              # one summed stream: cheaper on CPU
```
**Check the devices first:** `tx levels` shows a live meter for every microphone (🎤,
`--input`) and every system-audio source (🔊, `--output`). Speak and play something in
the meeting app: the rows that move are the devices to use. `tx levels -i <mic> --output
<id>` shows only those two. During `tx live`, the status bar has one meter per channel.
In the app, each device picker has a meter that previews the selected device while idle
(and shows the session level while recording); **Check devices…** lists all devices with
meters and a "Use" button.

| Platform | System audio capture | Status |
|---|---|---|
| Linux | PulseAudio/PipeWire monitor via `parec` (pulseaudio-utils) | tested (PulseAudio 17) |
| Windows | WASAPI loopback via `soundcard` (extra `system`) | **untested** |
| macOS 14.2+ | Core Audio process tap, Swift helper `helpers/macos-audiotap` | **not yet compiled/tested** |
| macOS < 14.2 | BlackHole virtual device selected as `--input` | - |

- `both` keeps two channels (`mic`, `system`): remote participants and you are separated
  by channel, each channel is recorded (`audio-mic.flac`, `audio-system.flac`) and
  refined separately. Without a headset the mic hears the speakers: a mic segment that
  repeats overlapping system audio is dropped as echo (`echo_segments_dropped` in the
  transcript meta). Two ASR streams double the CPU load - `--mix` sums both into one
  stream (no echo suppression, and live speaker labels are weaker; refinement helps).

### Voiceprints (speaker identification across sessions)
```sh
uv run tx transcribe meeting.m4a                      # speakers S1, S2, ...
uv run tx speakers name <session> S2 "Alice" --consent  # label + learn Alice's voice
uv run tx speakers enroll "Bob" --file bob.wav --consent  # or --record 30
uv run tx transcribe next_meeting.m4a                  # "S1 is Alice (score 0.91)"
uv run tx speakers identify <session>                   # re-check after new enrollments
uv run tx speakers list | consent NAME --revoke | delete NAME | export F | import F
```
- Each session stores a few **exemplar embeddings** per speaker (`exemplars.npz`, from
  ≥3 s pieces of clean speech that never span a pause). Naming a speaker enrolls them.
- **Decisions:** a speaker is labelled only when the score is ≥ 0.85 *and* leads the best
  competitor by ≥ 0.10 (one person per session speaker); 0.60–0.85 gives a suggestion.
  Calibrated with `python -m transcribe.bench.calibrate --sweep` on synthetic voices:
  0 wrong labels with the same, a different, or a phone-like channel - a channel
  mismatch yields suggestions instead of labels. Real recordings will differ; re-run the
  bench with your own data before relying on it.
- **Safety against diarization errors:** a speaker whose exemplars disagree (two voices
  merged) is never enrolled and never labelled confidently. A speaker with mixed
  languages is transcribed with per-turn language detection, so no speech is dropped.
- **Privacy (GDPR Art. 9 - biometric data):** everything stays local; enrolling requires
  `--consent` (recorded per person); `consent --revoke` deletes the person's voiceprints;
  `[speakers] store_voiceprints = false` disables all storage;
  `encrypt_voiceprints = true` (extra `secure`) encrypts DB vectors and session exemplars
  with a key kept in the OS keyring (or `TRANSCRIBE_VOICEPRINT_KEY`). Exports are
  unencrypted JSON - handle them as sensitive. Identification is for convenience only:
  not authentication, no anti-spoofing.
- Known limitation: diarization sometimes merges similar voices (e.g. two male voices);
  those speakers then stay unlabelled until refined (`tx refine -n N`).

## Installing a release
Download the archive for your platform from the GitHub releases page and unpack it. It
contains `tx` (command line) and `Transcribe` (desktop app); on macOS, `Transcribe.app`.
Models are not bundled: the app offers to download them on first start (or run
`tx models download`, several hundred MB, stored in the user cache directory).

- **Linux:** needs PortAudio and Qt's usual libraries (`apt install libportaudio2 libegl1
  libxkbcommon0`) and, for system audio, `pulseaudio-utils` (also for PipeWire).
- **macOS:** grant microphone and "System Audio Recording" access when asked. Unsigned
  builds must be opened once via right-click → Open.
- **Windows:** system audio uses WASAPI loopback (no extra drivers).

Build a bundle yourself: `uv sync --group packaging --extra gui --extra
secure --extra system && uv run python packaging/build.py`
(on macOS run `helpers/macos-audiotap/build.sh` first). The Linux bundle is ~335 MB
(228 MB archived) and was verified here by transcribing with the frozen `tx`.

## Development
```sh
uv sync --extra secure
uv run pytest                 # fast unit tests (+ system-audio test if PulseAudio runs)
TX_SLOW=1 uv run pytest       # + end-to-end tests (downloads ~650 MB of models)
uv run python -m transcribe.bench.dialogue out.wav   # synthetic de/en test dialogue
uv run python -m transcribe.bench.latency out.wav small  # real-time live latency
uv run python -m transcribe.bench.calibrate --sweep      # voiceprint thresholds
```

