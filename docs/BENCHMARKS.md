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
  pipeline always uses faster-whisper's VAD filter; a whisper.cpp engine must too.
- VAD also saves time (silence is skipped): CPU 0.56 → 0.33, iGPU 0.12 → 0.08 RTF.
- Accuracy with VAD matches faster-whisper with the same model (21-23 %, see above).
- faster-whisper (CTranslate2) supports only NVIDIA GPUs. Using AMD/Intel iGPUs (and
  Apple Metal) requires a whisper.cpp engine plugin - the measurements say it is worth
  it: with Vulkan, the most accurate model runs ~5× faster than `large-v3-turbo` on the
  CPU and ~2× faster than `small` does today.

Reproduce: `scripts/build_whispercpp.sh`, then
`python -m transcribe.bench.whispercpp --binary cpu=... --binary vulkan=... --models large-v3-turbo [--vad] --out-dir runs/`.
