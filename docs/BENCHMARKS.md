# Benchmarks

All numbers measured on CPU unless noted: **AMD Ryzen 7 7735HS** (8 cores / 16 threads,
AVX2, no AVX-512), 125 GB RAM, in a Docker container without GPU access. Whisper via
faster-whisper (CTranslate2, int8 on CPU). Commands to reproduce are given per section.

## Data

| Set | What | Reference |
|---|---|---|
| AMI ES2004a, IS1009a | Real meetings, 4 speakers each, English, 17 + 14 min, headset mix | manual transcript (AMI 1.6.2) and speaker RTTM (BUT setup) |
| Dialogue | Synthetic German/English two-speaker dialogue (Piper TTS), 42 s | script |
| Radio play | "001.m4b", German audio drama, ~10 roles, 51 min | none - indirect statistics only |

Caveats: two meetings are a small sample; synthetic speech is far easier than real
speech; the radio play has no reference, so only plausibility checks are possible.

## Whisper models (AMI meetings)

`python -m transcribe.bench.models --models small medium large-v3-turbo`

WER without filler words (um, uh, ...; AMI annotates them, Whisper drops them).
RTF = ASR time / audio duration (lower is faster). Speaker metrics: see below.

| Model | WER ES2004a | WER IS1009a | RTF | sentence ends / 100 words |
|---|---|---|---|---|
| small | 25.7 % (24.9) | 23.3 % (24.4) | 0.15 | 9-12 |
| medium | 26.1 % (26.4) | 31.1 % (23.1) | 0.49 | 11-13 |
| **large-v3-turbo** | **22.8 %** (21.1) | **21.9 %** (20.1) | 0.42 | 14-15 (**0** without prompt) |

Default settings (with the style prompt, see below); in brackets: without it.

- **large-v3-turbo** is the most accurate (≈2-3 points better than small) at 2.8× the
  CPU time of small - ~0.4× real time, fine for files, too slow for live mode on CPU.
- **medium** is slower than turbo and not more accurate - no reason to use it.
- **Style prompt:** without a prompt, turbo transcribed IS1009a with **no punctuation
  and no capitals at all**, which also broke the sentence-based speaker refinement. A
  short punctuated example sentence as prompt (`asr.style_prompt`, default on) fixes
  that (0 → 14 sentence ends per 100 words) and never leaked into the output (0 of 6
  transcripts). Cost: the transcript becomes more polished and less verbatim - turbo
  loses ≈2 WER points against AMI's strictly verbatim references, and medium on IS1009a
  dropped 10-25 % of the words in some minutes (repetitions, false starts, cross-talk;
  no loops or invented text) = +8 points. For verbatim transcripts set
  `[asr] style_prompt = false` (and prefer `small`).

## Live latency

`python -m transcribe.bench.models --meetings --latency`: real-time replay of the 42 s
dialogue, delay from the end of each utterance to its final text.

| Model | median | max |
|---|---|---|
| small | 4.0 s | 7.6 s |
| medium | 27.8 s | 43.2 s |
| large-v3-turbo | 50.0 s | 90.7 s |

Every live decode re-runs Whisper's encoder on a padded 30 s window (~1.3 s for
`small` on this CPU), so only `small` (or `base`) is usable live without a GPU. The
encoder is exactly what a GPU accelerates most (see whisper.cpp / Vulkan below).

## German: dialogue and radio play

Dialogue (full pipeline, de/en): all three models 0.0 % WER in both languages and
99.1 % word speaker accuracy; pipeline time 22 s (small), 56 s (medium), 85 s (turbo) for
42 s of audio - the synthetic voices are too easy to separate the models.

Radio play (51 min, no reference). How often names are spelled correctly (the cast
appears many times; the counts compare models, they are not recall):

| Model | RTF | speakers | mid-sentence starts | Justus | Peter | Bob | Fentriss | Hitchcock | Claudius | Carlos | Morton |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 0.26 | 12 | 2.4 % | 24 | 24 | 13 | **0** | 3 | 30 | 10 | 2 |
| medium | 0.78 | 12 | 2.6 % | 21 | 24 | 12 | **0** | 3 | 27 | 10 | 8 |
| large-v3-turbo | 0.42 | 12 | 2.1 % | 26 | 26 | 13 | **0** | 5 | 30 | 11 | 7 |

No model ever writes "Fentriss" (always "Fanfress", "Ventress", ...): rare names need the
custom vocabulary (`--vocab`, corrections `Fanfress -> Fentriss`), not a bigger model.

**Custom vocabulary** (`small`, same file, only the eight cast names as hint words - no
correction rules):

| | Fentriss | Morton | Carlos | Claudius | Hitchcock | wrong "F…ss" variants | time |
|---|---|---|---|---|---|---|---|
| without hints | 0 | 1 | 8 | 28 | 4 | 6 (Fanfress, Fendruss, Fenpris, Fentris, Fentus, Ventress) | 863 s |
| with hints | **16** | **9** | 10 | 31 | 5 | **0** | 877 s (+1.6 %) |

Hint words alone fix rare names almost completely at negligible cost - far more
effective than a larger model. Correction rules (`Fanfress -> Fentriss`) remain for
mistakes the hints do not catch.

## Speaker attribution

Word-level, against the AMI speaker annotations (words with exactly one active
speaker). Two views:

- **1:1 accuracy** - each detected speaker maps to one real person; splitting a person
  into two labels counts the smaller part as wrong.
- **Purity** - each detected speaker counts as its majority person (= accuracy after
  the user merges split speakers in the app). Only *mixing* voices is penalised, which
  cannot be repaired afterwards.

(Measured with the previous voice model, WeSpeaker ResNet34; see "Voice model" below for
TitaNet-small.)

| | speakers found | 1:1 accuracy | purity |
|---|---|---|---|
| ES2004a, diarizer only | 55 | 45 % | 99 % |
| ES2004a, + second pass (small / medium / turbo) | 15 | 79 % / 84 % / 79 % | 98 % / 98 % / 97 % |
| IS1009a, diarizer only | 15 | 62-64 % | 95-96 % |
| IS1009a, + second pass (small / medium / turbo) | 9 | 83 % / 61 % / 55 % | 90 % / 95 % / 88 % |

(4 real speakers per meeting; with the style prompt.)

- The diarizer's clusters are pure but massively over-split (55 labels for 4 people) -
  unusable without the second pass, which brings ES2004a to 15 labels at 98 % purity.
- The second pass costs a few points of purity on IS1009a (95 → 88-90 %): some voices
  get mixed. The 1:1 differences between models there (55-83 %) mostly reflect how the
  9 clusters split the 4 people, i.e. what merging fixes.
- Without punctuation (turbo without style prompt) the second pass failed: phrases of
  10 s spanning several speakers, purity 71 %. Hence the style prompt.
- Word timestamps are equally good for all three models (median error 0.06-0.08 s vs.
  the AMI word times), so they are not the cause of differences.
- `[speakers] speaker_count = "silhouette"` reached 97 % 1:1 accuracy with exactly 4
  speakers on ES2004a, but merged the radio play's similar voices into 3 speakers.

Reproduce: `python -m transcribe.bench.diarization` and `python -m transcribe.bench.models`.

## Live speaker labels

`python -m transcribe.bench.live_speakers` replays a recording through live labelling:
utterances cut as in live mode (an utterance ends at a 0.4 s pause), Whisper `small`
words, then the live code labels piece by piece. Share of speech (10 ms frames with one
reference speaker) labelled correctly under the best one-to-one mapping of labels to
people; in brackets the number of labels. "Ceiling" = every piece labelled with its true
majority speaker.

| Live labelling | Radio play (6 voices) | AMI IS1009a | AMI ES2004a |
|---|---|---|---|
| Before: whole utterances, ResNet34, all re-clustered after each utterance | 29 % (1) | 73 % (5) | 51 % (33) |
| Whole utterances, TitaNet, centroids | 29 % (1) | 75 % (5) | 76 % (8) |
| **Sentences, TitaNet, centroids (default)** | **91 %** (5) | **84 %** (9) | **79 %** (12) |
| Sentences, ResNet34, centroids | 29 % (1) | 62 % (1) | 75 % (5) |
| Ceiling: whole utterances / sentences | 41 % / 98 % | 83 % / 91 % | 77 % / 81 % |

(Centroid threshold 0.4. "Before" was measured on the same recordings with the previous
tracker in an experiment script; it is no longer part of the code.)

What made the difference, in order of impact:

1. **Labels per sentence.** In a radio play nobody waits 0.4 s before answering: 40
   lines became 9 utterances, so per-utterance labels could at best be 41 % right. Cutting
   finished utterances at sentence ends (Whisper's punctuation, not at "Mr."/"Dr."/"z.B.")
   raises the ceiling to 98 %. pyannote's segmentation model as change detector inside
   utterances reached 72 % on the play and added nothing on top of sentences.
2. **Voice model.** Ten models compared on single-speaker pieces (nearest neighbour from
   the same speaker / equal error rate, IS1009a): WeSpeaker ResNet34 88 % / 28.5 %,
   **NeMo TitaNet-small 98 % / 13.3 %** at half the time (21 vs 41 ms per piece);
   TitaNet-large and 3D-Speaker CAM++ were similar, ERes2Net and larger ResNets slower
   and not better. TitaNet-small is now the model everywhere (diarization, second pass,
   voiceprints).
3. **Stable assignment.** Each piece joins the most similar speaker centroid if the
   cosine similarity is ≥ 0.4, else starts a new speaker; pieces under 1 s never start
   one. Re-clustering everything after each utterance relabelled earlier lines and
   over-split ES2004a into 33 labels. Threshold 0.3 merged voices in the play, 0.5
   over-split the meetings (18-24 labels).

On the user's radio play (first 6 minutes, no reference, checked by reading): before,
three labels with whole exchanges of Justus, Peter and Mr. Fentriss under one; after,
Justus, Peter, the fake Mr. Fentriss and the narrator each consistently under their own
label, errors left at short interjections.

## Voice model: TitaNet-small replaces ResNet34

Speaker embeddings are used by the diarizer, the second pass, live labels and
voiceprints. NeMo TitaNet-small (CC-BY-4.0, 192-dim) is the default since it halved the
speaker confusions of WeSpeaker ResNet34 on single-speaker pieces of IS1009a (see
"Live speaker labels") at twice the speed. Full offline pipeline, word speaker accuracy /
purity, `python -m transcribe.bench.diarization [--embedding-model ...]` (radio play: same
pipeline, experiment script):

| Second pass (default) | ResNet34 (thr 0.5) | **TitaNet-small (thr 0.5)** |
|---|---|---|
| ES2004a | 69.0 % / 98.5 % (15 labels) | **79.5 % / 99.2 %** (15) |
| IS1009a | 83.1 % / 90.1 % (9) | 81.1 % / **97.2 %** (15) |
| Radio play | 88.5 % / 92.6 % (6) | **96.7 % / 99.2 %** (7) |
| ES2004a, `speaker_count = "silhouette"` | 98.5 % (4) | **99.2 %** (4) |
| IS1009a, `speaker_count = "silhouette"` | 86.9 % / 89.6 % (5) | 92.7 % / 92.7 % (3) |

TitaNet mixes far fewer voices (purity); on IS1009a it splits people into more labels,
which merging fixes. The diarizer's clustering threshold was swept from 0.3 to 0.8: 0.6
scored highest on these three recordings (IS1009a 94.9 %, play 99.2 %), but merged a
female English and a female German voice in the short cross-session test dialogue
(`tests/integration`), so the default stays at 0.5.

**Voiceprint thresholds** recalibrated for TitaNet (`python -m transcribe.bench.calibrate
--sweep`, 20 enrolled synthetic voices + 5 strangers): genuine scores ≥ 0.92 on the same
channel, ≥ 0.80 through another microphone, ≥ 0.66 through a phone-like channel;
strangers ≤ 0.66. Auto-label at ≥ 0.75 (was 0.85 for ResNet34): 0 wrong names on all
channels; 20/20 recognised on the same and on another microphone, 8 recognised + 12
suggested on the phone channel. The consistency check before enrolling (exemplars
within 0.70 of their medoid, ≥ 75 % agreeing) holds for TitaNet: genuine AMI speakers
87-95 % consistent, different people's pieces ≤ 0.60 similar.

## whisper.cpp: CPU vs AMD iGPU (Vulkan)

Measured by the user on a second machine with an **AMD Radeon 780M** iGPU (RDNA3,
"RADV PHOENIX", flash attention and cooperative-matrix cores in use), whisper.cpp v1.9.5,
CPU build vs Vulkan build on the same machine.

**Micro-benchmark** (`whisper-bench`, model `small`):

| Stage | CPU | iGPU (Vulkan) | Speedup |
|---|---|---|---|
| Encoder (one 30 s window) | 1336 ms | 190 ms | **7.0×** |
| Prompt processing / token | 1.26 ms | 0.18 ms | 7.0× |
| Decoder, single tokens | 7.29 ms | 6.80 ms | 1.07× |
| Decoder, batched (beam search) | 2.17 ms | 1.66 ms | 1.3× |
| Total | 9.1 s | 3.2 s | **2.8×** |

The GPU accelerates the encoder; token-by-token decoding barely benefits.

**Real meetings**, `large-v3-turbo` (`python -m transcribe.bench.whispercpp`):

| Meeting | Build | VAD | WER | repeated lines | time | RTF |
|---|---|---|---|---|---|---|
| ES2004a (17 min) | CPU | - | 29.2 % | 62 | 597 s | 0.56 |
| | Vulkan | - | **58.2 %** | **138** | 129 s | 0.12 |
| | CPU | ✓ | 22.4 % | 17 | 344 s | 0.33 |
| | **Vulkan** | ✓ | **20.9 %** | 7 | **85 s** | **0.079** |
| IS1009a (14 min) | CPU | - | 22.4 % | 9 | 345 s | 0.40 |
| | Vulkan | - | 23.7 % | 6 | 74 s | 0.086 |
| | CPU | ✓ | 22.6 % | 4 | 260 s | 0.31 |
| | **Vulkan** | ✓ | **21.5 %** | 5 | **68 s** | **0.080** |

- **The iGPU is 3.8-4.7× faster** than the CPU of the same machine with
  `large-v3-turbo` - more than for `small` (2.8×), because turbo's large encoder
  dominates. At ~0.08× real time, turbo would even be usable for live mode.
- **VAD is mandatory.** Without it, Whisper hallucinated in non-speech stretches of
  ES2004a: the Vulkan run wrote "We're going to be able to do that." 88 times in a row
  (WER 58 %), the CPU run "I'll go first." 27 times. The GPU's different arithmetic
  (fp16) made the loop worse, but it is a Whisper behaviour, not a Vulkan bug. With
  whisper.cpp's Silero VAD: no loops, and both builds at 21-23 % WER. The main
  pipeline always uses a VAD filter, the whisper.cpp engine included.
- VAD also saves time (silence is skipped): CPU 0.56 → 0.33, iGPU 0.12 → 0.08 RTF.
- Accuracy with VAD matches faster-whisper with the same model (21-23 %, see above).
- faster-whisper (CTranslate2) supports only NVIDIA GPUs. For AMD/Intel iGPUs (and
  Apple Metal) there is now the whisper.cpp engine (below) - with Vulkan, the most
  accurate model runs ~5× faster than `large-v3-turbo` on the CPU and ~2× faster than
  `small` on the CPU.

Reproduce: `scripts/build_whispercpp.sh`, then
`python -m transcribe.bench.whispercpp --binary cpu=... --binary vulkan=... --models large-v3-turbo [--vad] --out-dir runs/`.

## whisper.cpp engine

`[asr] engine = "whisper.cpp"` (`--engine whisper.cpp`) runs the same Whisper models
through whisper.cpp's `whisper-server`, for GPUs that faster-whisper cannot use. Measured
here on the CPU build (the test machine has no usable GPU), against faster-whisper on the
same CPU, full pipeline:

`python -m transcribe.bench.models --engine whisper.cpp --device cpu --models small`

| Engine, model | WER ES2004a / IS1009a | word start error | purity (second pass) | RTF (CPU) |
|---|---|---|---|---|
| faster-whisper `small` | 25.7 % / 23.3 % | 0.05 / 0.05 s | 98.5 % / 90.1 % | 0.15-0.17 |
| whisper.cpp `small` | 23.7 % / 23.8 % | 0.05 / 0.05 s | 98.1 % / 90.3 % | 0.16 |
| faster-whisper `large-v3-turbo` | 22.8 % / 21.9 % | | 97 % / 88 % | 0.42 |
| whisper.cpp `large-v3-turbo` | 22.9 % / 23.7 % | 0.04 / 0.05 s | 97.7 % / 94.2 % | 0.45-0.48 |

German/English dialogue (`--dialogue`): 0.0 % WER in both languages, 100 % speaker
accuracy. On the CPU both engines are equally fast; the point of whisper.cpp is the GPU
(Radeon 780M: ~4-5× faster than its CPU, see above).

Making whisper.cpp match took four fixes, each found on these meetings:

1. **VAD in the engine.** whisper-server v1.9.5 with `--vad` returns word times on the
   shortened (silence-free) timeline. The engine cuts the speech out itself (Silero,
   faster-whisper's settings) and maps times back.
2. **DTW word timestamps.** whisper.cpp's token timestamps are rough: median start
   error 0.26-0.39 s (faster-whisper: 0.05 s), which cost 4 points of speaker purity on
   ES2004a. DTW (`-dtw <model>`) is precise once read correctly: `t_dtw` marks where a
   token *ends*; a word starts where the previous one ended. whisper.cpp silently drops
   DTW when flash attention is on, so the engine turns flash attention off (`-nfa`).
   `[asr] whispercpp_dtw = false` restores flash attention for speed if word timing
   does not matter.
3. **Controlled context.** whisper.cpp feeds the text decoded so far back into the next
   window. On IS1009a one window came out in lower case without punctuation after
   minute 3, and every later window copied it (sentence ends 11.6 → 2.7 per 100 words,
   purity 86 %). With no context at all, it dropped more speech (ES2004a 31.7 % WER).
   Now the engine decodes ≤28 s windows itself, passes the previous window's text as
   prompt only if it is well-formed (capitals and sentence punctuation), and limits
   whisper.cpp's context to exactly that prompt - counted with whisper.cpp's own
   tokenizer, read from the model file, so the hint words are never cut off.
4. **One-pass language detection.** The server's language probabilities cost two
   encoder passes; a request with a fixed language and 50 ms duration returns the same
   probabilities after one (4.0 → 2.0 s per call on the CPU).

**Live mode on the CPU** (`--latency`, `small`): median delay 8.2 s (5.4 s with
`whispercpp_dtw = false`), faster-whisper 4.0 s - every live decode runs the encoder,
and whisper.cpp's CPU encoder without flash attention is slower. On a CPU, use
faster-whisper live; whisper.cpp is meant for the GPU, where the encoder is 7× faster.

**On the GPU** (AMD Radeon 780M iGPU, Vulkan build, measured by the user; same
benchmark):

| Model | Device | WER ES2004a / IS1009a | word start error | purity | RTF | live latency median (max) |
|---|---|---|---|---|---|---|
| `small` | 780M | 22.8 % / 23.9 % | 0.05 / 0.06 s | 98.1 % / 90.6 % | 0.073-0.077 | **1.8 s** (2.0 s) |
| `large-v3-turbo` | 780M | 23.1 % / 23.2 % | 0.05 / 0.05 s | 97.3 % / 88.8 % | **0.092-0.096** | **4.0 s** (5.3 s) |
| `large-v3-turbo` | same machine's CPU | 23.0 % / 23.7 % | 0.04 / 0.05 s | 97.7 % / 94.2 % | 0.37-0.39 | - |

- **Live mode with the most accurate model.** `large-v3-turbo` reaches 4.0 s median
  latency - as fast as faster-whisper `small` on the CPU (4.0 s), where turbo took 50 s.
  `small` on the iGPU gives the snappiest live view (1.8 s).
- **Files: 4× faster than the CPU, same accuracy.** Turbo transcribes the 17 min meeting
  in 1.7 min instead of 6.7 min; WER and word timing are unchanged.
- **On the GPU, `small` is barely faster than turbo** (0.075 vs 0.094): the encoder is no
  longer the bottleneck, token-by-token decoding is (the GPU speeds it up only 1.07×,
  see `whisper-bench` above), and turbo's decoder has 4 layers to `small`'s 12. With a
  GPU, turbo is the right model for files and the refinement pass.
- **DTW costs ~15 % speed:** `whisper-cli` with flash attention reached RTF 0.08, the
  engine with DTW (flash attention off) 0.09-0.10. Worth it - without DTW speaker purity
  was ~4 points lower (CPU runs above).
- Purity on IS1009a with turbo differs between GPU (88.8 %) and CPU (94.2 %) at equal WER
  and timing: fp16 on the GPU produces slightly different text and phrase boundaries,
  which the speaker pass groups differently. faster-whisper turbo was at 88 % as well -
  the usual spread on this meeting, not a GPU defect.

Recommended setup with such a GPU: `engine = "whisper.cpp"`, `model = "small"` for the
live view and `[live] refine_model = "large-v3-turbo"` for the final transcript (or turbo
for both, if 4 s live latency is fine; files: `tx transcribe -m large-v3-turbo`).

To measure on a machine with a GPU (Vulkan / Metal build from
`scripts/build_whispercpp.sh`):

```sh
uv run python -m transcribe.bench.models --engine whisper.cpp --models small large-v3-turbo --latency
uv run python -m transcribe.bench.models --engine whisper.cpp --device cpu --models large-v3-turbo
```

## Benchmark suite

`python -m transcribe.bench.suite` measures "who said what" on many recordings at once:
datasets in one format, recording conditions, setting sweeps, standard metrics and a
fixed split between tuning (`dev`) and reporting (`test`) data.

```sh
uv run python -m transcribe.bench.suite build ami --split dev           # 18 meetings
uv run python -m transcribe.bench.suite build ami --split test          # 16 meetings
uv run python -m transcribe.bench.suite build voxconverse --split dev --limit 50
uv run python -m transcribe.bench.suite build synth --lang de --split dev --count 20
uv run python -m transcribe.bench.suite build synth --lang de --commonvoice ~/cv/de
uv run python -m transcribe.bench.suite add-session <corrected session> --dataset local
uv run python -m transcribe.bench.suite list

uv run python -m transcribe.bench.suite run tune --data ami synth-de-piper --split dev \
    --conditions clean noise:10 phone --set speakers.diarize_threshold=0.4,0.5,0.6 --live
uv run python -m transcribe.bench.suite report tune
uv run python -m transcribe.bench.suite report tune --compare speakers.diarize_threshold=0.5
```

- **Datasets:** AMI meetings (official dev/test split, transcripts and speaker turns),
  VoxConverse (debates and news, many speakers, speaker turns only), synthetic German or
  English conversations (8 German Piper voices, or real voices from a Common Voice
  download; fast turns, interruptions, pauses), and your own sessions after correcting
  them in the app (`add-session`; they stay on your computer).
- **Conditions:** `clean`, `noise:<SNR dB>`, `reverb:<RT60 s>`, `phone`, `mp3:<kbps>`,
  `opus:<kbps>`, combinable (`reverb:0.5+noise:15`).
- **Settings:** `--set section.key=v1,v2` for any config value; all combinations run.
  Whisper output and speaker turns are cached per recording and condition, so sweeping
  speaker settings only re-runs the cheap steps; runs are resumable.
- **Metrics** (`bench/suite/metrics.py`, standard definitions, no extra dependencies):
  WER, **cpWER** (speaker-attributed WER: a word counts as wrong if misrecognised *or*
  attributed to the wrong person), DER with and without 0.25 s collar, impurity (share
  of speech under someone else's label), speaker-count error, speed; with `--live` the
  same for live labels. Rates are summed over recordings with 95 % bootstrap confidence
  intervals.
- **Paired comparison:** `report NAME --compare section.key=value` compares every
  setting with that one on the same recordings: difference per rate, its 95 % bootstrap
  interval and how many recordings got better / worse (`paired.md`). Differences between
  recordings cancel out, so it shows much smaller effects than the per-setting intervals
  (which overlap even where one setting is worse on 15 of 18 meetings). Other swept keys
  stay fixed: each setting is compared with the one that differs only in the given keys.
- **Regression check:** `report NAME --baseline summary.json --tolerance 0.02` exits 1 if
  a rate got worse by more than 2 points.
- Data and caches live in `~/.cache/transcribe/bench` (AMI ~60 MB per meeting,
  VoxConverse test ~4.3 GB).
- **Nightly CI** runs a small subset (6 German conversations + 1 AMI meeting, clean and
  10 dB noise, live labels) against `tests/bench/baseline-ci.json` and fails if a rate
  gets worse by more than 5 points (quantised Whisper differs slightly between CPUs).

### First results (dev split, `small`, CPU)

`run first --data synth-de-piper ami --split dev --conditions clean noise:10 phone --live`
- 4 AMI dev meetings (1.6 h) and 10 synthetic German conversations (2-6 voices, 8.5 min).

| Data | Condition | WER | cpWER | DER | DER live | impurity / live | speaker count error |
|---|---|---|---|---|---|---|---|
| AMI | clean | 18.4 % | 33.3 % | 18.2 % | **11.6 %** | 2.3 % / 1.8 % | 7.5 |
| AMI | noise 10 dB | 24.3 % | 45.7 % | 23.1 % | 15.6 % | 2.3 % / 2.5 % | 9.8 |
| AMI | phone | 23.7 % | 43.9 % | 25.0 % | **47.7 %** | 3.2 % / **32 %** | 8.0 |
| German conversations | clean | 5.6 % | 17.2 % | 7.2 % | 4.3 % | 4.3 % / 6.0 % | 1.0 |
| German conversations | noise 10 dB | 19.8 % | 39.3 % | 27.2 % | 18.5 % | 8.1 % / 14 % | 1.4 |
| German conversations | phone | 10.2 % | 31.2 % | 14.9 % | **39.3 %** | 9.5 % / **38 %** | 0.9 |

(95 % intervals in `summary.md`; with 4 meetings they are wide, e.g. AMI cpWER 25-43 %.)

What this shows, to work on next:

1. **Live labels collapsed on telephone audio** (impurity 32-38 % vs 2-6 % clean), while
   the offline labels stayed pure: band-limited voices sound more alike (TitaNet
   similarity of *different* speakers rises from ~0.1 to ~0.35, the same speaker stays
   ~0.7), and the live threshold 0.4 merged them. **Fixed:** live labels now detect
   telephone-band channels (energy above 4 kHz vs. the speech band, median over recent
   sentences: ~1e-7 on a phone line, ~1e-2 otherwise - also with noise, reverb, MP3 or
   Opus) and use `live_threshold_narrowband = 0.6` there (best of 0.4-0.7 on both data
   sets; a single 0.6 for all audio would cost AMI clean 4.6 points):

   | live DER / impurity | before | after |
   |---|---|---|
   | AMI phone | 47.7 % / 32.2 % | **27.7 % / 2.2 %** |
   | German conversations phone | 39.3 % / 37.7 % | **5.3 % / 2.4 %** |
   | clean, noise 10 dB, Opus 16 kbit/s (both) | | unchanged |

   Centring the embeddings on the session mean (removing the common channel component)
   aligned the thresholds too, but halved the separation with only four speakers (AMI
   clean EER 6.3 -> 12.3 %) - not used.
2. **Attribution, not recognition, dominates on meetings:** cpWER 33 % vs WER 18 % on
   AMI. The offline second pass splits 4 people into ~11.5 labels (speaker count error
   7.5) - pure, but each split costs cpWER; the live labels get closer to 4 and beat the
   offline DER (11.6 vs 18.2 %).
3. **Noise costs more in German than on AMI** (WER +14 vs +6 points at 10 dB).
4. **VAD on telephone-band audio:** Silero finds only ~73 % of the speech in band-limited
   AMI audio (69.5 of 94.8 s in a 2-minute excerpt); Whisper alone transcribes 92 % of
   the words. (An earlier version of the `phone` condition added full-band noise after
   band-limiting, which made the VAD collapse to 12 % - not realistic, fixed.)
   **Fixed:** speech detection detects telephone-band audio (`audio/bandwidth.py`, as for
   the live labels) and uses `asr.vad_threshold_narrowband = 0.25` there instead of 0.5
   (`asr.vad_threshold`, unchanged for wideband audio). On the first data (4 meetings,
   10 conversations) any value of 0.15-0.35 lowered the phone WER by 1-2 points vs 0.5
   (AMI 23.7 -> 22.2 %, German 9.6 -> 9.0 %); the speaker metrics jumped by up to 14
   points between neighbouring values without a trend. A second run on all 18 AMI dev
   meetings (9.7 h) and 40 German conversations, `phone` only, compared 0.2, 0.25 and
   0.35 on the same recordings (`report vad-nb-sweep2 --compare
   asr.vad_threshold_narrowband=0.25`; difference to 0.25 in points, 95 % interval):

   | vs 0.25 | AMI WER | AMI cpWER | AMI DER | German WER | German cpWER | German DER |
   |---|---|---|---|---|---|---|
   | 0.2 | -0.3 (-0.9 - +0.4) | -0.8 (-7.9 - +7.1) | -0.2 (-5.0 - +4.8) | +0.1 (-0.5 - +0.7) | -0.5 (-1.8 - +0.5) | -0.2 (-0.9 - +0.5) |
   | 0.35 | **+1.2 (+0.6 - +1.8)** | -6.3 (-14.2 - +2.0) | -3.8 (-8.7 - +1.6) | +0.8 (-0.0 - +1.9) | -0.7 (-2.5 - +0.8) | -0.5 (-1.5 - +0.4) |

   0.2 and 0.25 are indistinguishable; 0.35 costs words (AMI: worse in 15 of 18
   meetings). No speaker metric differs significantly - the jumps on the small set were
   noise, probably because the second pass re-clusters the transcript's phrases, so a
   few changed words can move whole stretches to another label. 0.25 stays.
   At 0.25 the 18 meetings are much harder than the first 4 (phone WER 33.8 vs 22.2 %,
   cpWER 76 %, DER 47 %, live DER 35.5 %, speaker count error 10.6 with 4 people);
   German: WER 9.0 %, cpWER 54 %, DER 27 %, live DER 7.0 %.
5. **Many speakers in a short recording:** in one 5-voice conversation the second pass
   merged the voices into 2 (cpWER 88 %) - its speaker-count estimate ignores speakers
   with < 3 s of speech; the live labels got this recording right.
