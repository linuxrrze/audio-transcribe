"""Dataset builders: AMI, VoxConverse, synthetic conversations, your own sessions.

Each builder writes recordings into ``Dataset(name)`` (see data.py). Downloads are cached;
building again only adds what is missing.
"""

from __future__ import annotations

import csv
import json
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from transcribe.audio.base import SAMPLE_RATE
from transcribe.bench.suite.data import Dataset, Recording, RefTurn, bench_dir

ProgressFn = Callable[[str], None]

# --- AMI (meetings, English; official BUT diarization-setup split) --------------------
AMI_LISTS = ("https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
             "lists/{split}.meetings.txt")


def ami_meetings(split: str) -> list[str]:
    with urllib.request.urlopen(AMI_LISTS.format(split=split), timeout=60) as resp:
        return resp.read().decode().split()


def build_ami(split: str = "test", limit: int | None = None,
              progress: ProgressFn = print) -> Dataset:
    """AMI headset-mix meetings with reference turns (RTTM) and manual transcripts."""
    from transcribe.bench.ami import cache_dir, fetch, load_rttm

    ds = Dataset("ami")
    have = {r.id for r in ds.recordings()}
    meetings = ami_meetings(split)[:limit]
    archive = cache_dir() / "ami_public_manual_1.6.2.zip"
    for m in meetings:
        if m in have:
            continue
        progress(f"ami {m}")
        wav, rttm = fetch(m)
        turns = [RefTurn(a, b, s) for a, b, s in load_rttm(rttm)]
        _attach_ami_words(archive, m, turns)
        import soundfile as sf

        ds.write([Recording(m, "ami", split, str(wav), "en", turns,
                            meta={"duration_s": sf.info(wav).duration})])
    return ds


def _attach_ami_words(archive: Path, meeting: str, turns: list[RefTurn]) -> None:
    """Put each manually transcribed word into the reference turn of its speaker."""
    from transcribe.bench.ami import AMI_MANUAL

    if not archive.exists():
        urllib.request.urlretrieve(AMI_MANUAL, archive)
    with zipfile.ZipFile(archive) as z:
        meetings = ET.fromstring(z.read("corpusResources/meetings.xml"))
        agents = {}
        for mt in meetings.iter():
            if mt.get("observation") == meeting:
                agents = {sp.get("nxt_agent"): sp.get("global_name") for sp in mt}
        by_speaker: dict[str, list[RefTurn]] = defaultdict(list)
        for t in turns:
            by_speaker[t.speaker].append(t)
        texts: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for name in z.namelist():
            m = re.fullmatch(rf"words/{meeting}\.([A-Z])\.words\.xml", name)
            speaker = agents.get(m.group(1)) if m else None
            if speaker is None or speaker not in by_speaker:
                continue
            own = by_speaker[speaker]
            for el in ET.fromstring(z.read(name)).iter("w"):
                if el.get("punc") == "true" or not el.text or el.get("starttime") is None:
                    continue
                t0 = float(el.get("starttime", 0))
                turn = min(own, key=lambda t: 0 if t.t0 <= t0 <= t.t1
                           else min(abs(t0 - t.t0), abs(t0 - t.t1)))
                texts[id(turn)].append((t0, el.text))
    for t in turns:
        t.text = " ".join(w for _, w in sorted(texts.get(id(t), [])))


# --- VoxConverse (debates, news, talk shows; many speakers; diarization only) ----------
VOX_AUDIO = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_{split}_wav.zip"
VOX_RTTM = "https://raw.githubusercontent.com/joonson/voxconverse/master/{split}/{id}.rttm"
VOX_LIST = "https://api.github.com/repos/joonson/voxconverse/contents/{split}"


def build_voxconverse(split: str = "test", limit: int | None = None,
                      progress: ProgressFn = print) -> Dataset:
    """VoxConverse audio (one zip per split: dev ~2 GB, test ~4.3 GB) + reference RTTMs.
    No transcripts: scored for speakers only."""
    from transcribe.bench.ami import load_rttm

    ds = Dataset("voxconverse")
    downloads = bench_dir() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"voxconverse_{split}_wav.zip"
    if not archive.exists():
        progress(f"downloading {archive.name} (several GB, once)")
        _download(VOX_AUDIO.format(split=split), archive)
    with urllib.request.urlopen(VOX_LIST.format(split=split), timeout=60) as resp:
        ids = sorted(x["name"][:-5] for x in json.load(resp) if x["name"].endswith(".rttm"))
    have = {r.id for r in ds.recordings()}
    (ds.dir / "audio").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".wav")}
        for rid in ids[:limit]:
            if rid in have or rid not in members:
                continue
            progress(f"voxconverse {rid}")
            out = ds.dir / "audio" / f"{rid}.wav"
            out.write_bytes(z.read(members[rid]))
            rttm = ds.dir / "audio" / f"{rid}.rttm"
            urllib.request.urlretrieve(VOX_RTTM.format(split=split, id=rid), rttm)
            turns = [RefTurn(a, b, s) for a, b, s in load_rttm(rttm)]
            import soundfile as sf

            ds.write([Recording(rid, "voxconverse", split, f"audio/{rid}.wav", "", turns,
                                has_text=False, meta={"duration_s": sf.info(out).duration})])
    return ds


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


# --- synthetic conversations (German / English, known ground truth) -------------------
LINES = {
    "de": [
        "Hast du das gehört?", "Ja, ganz deutlich.", "Was?", "Genau.", "Nein.",
        "Da hat jemand um Hilfe gerufen.", "Wir sollten sofort nachsehen.",
        "Moment, ich hole noch schnell meine Taschenlampe.", "Bist du sicher?",
        "Ich glaube, das kam aus dem alten Haus am Ende der Straße.",
        "Halt! Wer seid ihr und was wollt ihr hier?", "Wir suchen einen Papagei.",
        "Einen Papagei? Hier gibt es keinen Papagei.", "Aber wir haben ihn doch gehört.",
        "Das ist eine lange Geschichte.", "Erzähl schon, wir haben Zeit.",
        "Der Termin am Donnerstag passt mir leider nicht.", "Wie wäre es mit Freitag?",
        "Freitag um zehn Uhr wäre gut.", "Dann schicke ich allen eine Einladung.",
        "Hat jemand die Zahlen vom letzten Quartal?", "Die liegen im gemeinsamen Ordner.",
        "Der Umsatz ist um zwölf Prozent gestiegen.", "Das klingt doch vielversprechend.",
        "Ich bin mir da nicht so sicher.", "Warum nicht?", "Die Kosten steigen schneller.",
        "Können wir das bis nächste Woche klären?", "Ich kümmere mich darum.",
        "Danke, das ist nett.", "Gern geschehen.", "Und was machen wir jetzt?",
        "Wir warten, bis es dunkel wird.", "Das gefällt mir überhaupt nicht.",
        "Keine Sorge, es wird schon nichts passieren.", "Hm.", "Na gut.",
        "Sag mal, hast du heute schon etwas gegessen?", "Nur ein Brötchen am Morgen.",
        "Dann gehen wir gleich in die Kantine.", "Gute Idee, ich habe großen Hunger.",
    ],
    "en": [
        "Did you hear that?", "Yes, quite clearly.", "What?", "Exactly.", "No.",
        "Somebody called for help over there.", "We should go and check right away.",
        "Wait, let me grab my torch first.", "Are you sure about that?",
        "Thursday does not work for me, unfortunately.", "How about Friday instead?",
        "Friday at ten would be fine.", "I will send everyone an invitation.",
        "Does anyone have the numbers from last quarter?", "They are in the shared folder.",
        "Revenue went up by twelve percent.", "That sounds promising.",
        "I am not so sure about that.", "Why not?", "Costs are rising even faster.",
        "Can we sort this out by next week?", "I will take care of it.",
        "Thanks, that is kind of you.", "You are welcome.", "So what do we do now?",
        "We wait until it gets dark.", "Okay.", "Right.",
        "Have you had anything to eat today?", "Just a sandwich this morning.",
    ],
}
PIPER_VOICES = {
    # German text needs German voices (English voices read it unintelligibly)
    "de": ["thorsten", "kerstin", "karlsson", "eva", "ramona", "pavoque", "miro", "dii"],
    "en": ["amy", *[f"vctk-{i}" for i in (5, 18, 30, 47, 63, 81, 99, 104)]],
}


def _gap(rng: random.Random) -> float:
    """Pause before the next turn: mostly short, sometimes long, sometimes overlapping."""
    x = rng.random()
    if x < 0.08:
        return -rng.uniform(0.05, 0.4)  # interruption: overlapping speech
    if x < 0.40:
        return rng.uniform(0.05, 0.3)  # faster than live mode's 0.4 s utterance end
    if x < 0.85:
        return rng.uniform(0.3, 0.9)
    return rng.uniform(0.9, 2.0)


Voice = Callable[[str], tuple[np.ndarray, str]]  # text to say -> (audio, text spoken)


def conversation(rng: random.Random, voices: dict[str, Voice], lines: list[str],
                 turns: int) -> tuple[np.ndarray, list[RefTurn]]:
    """Mix ``turns`` turns of random speakers (no immediate repeats) into one signal."""
    names = list(voices)
    parts: list[tuple[int, np.ndarray]] = []
    ref: list[RefTurn] = []
    t, last = 0.5, None
    for _ in range(turns):
        speaker = rng.choice([n for n in names if n != last] or names)
        audio, text = voices[speaker](rng.choice(lines))
        start = max(0.0, t + (_gap(rng) if ref else 0.0))
        parts.append((int(start * SAMPLE_RATE), audio))
        ref.append(RefTurn(start, start + len(audio) / SAMPLE_RATE, speaker, text))
        t, last = start + len(audio) / SAMPLE_RATE, speaker
    out = np.zeros(int((t + 0.5) * SAMPLE_RATE), np.float32)
    for i, a in parts:
        out[i:i + len(a)] += a
    return np.clip(out, -1, 1), ref


def build_synth(language: str = "de", split: str = "dev", count: int = 10,
                speakers: tuple[int, int] = (2, 6), turns: tuple[int, int] = (20, 40),
                commonvoice: Path | None = None, name: str | None = None,
                progress: ProgressFn = print) -> Dataset:
    """Synthetic conversations: Piper voices (default) or real voices from a Common Voice
    download (``commonvoice``: folder with validated.tsv and clips/). Seeds differ per
    split, so dev and test never share a conversation."""
    import soundfile as sf

    source = "cv" if commonvoice else "piper"
    ds = Dataset(name or f"synth-{language}-{source}")
    (ds.dir / "audio").mkdir(parents=True, exist_ok=True)
    have = {r.id for r in ds.recordings()}
    pool = _commonvoice_pool(commonvoice) if commonvoice else None
    base = 0 if split == "dev" else 100_000
    for i in range(count):
        rid = f"{split}-{i:03d}"
        if rid in have:
            continue
        rng = random.Random(base + i)
        n = rng.randint(*speakers)
        if pool is not None:
            chosen = rng.sample(sorted(pool), min(n, len(pool)))
            voices = {f"cv{k}": _cv_voice(pool[c], rng) for k, c in enumerate(chosen)}
            lines = [""]  # Common Voice: the text comes with the clip
        else:
            from transcribe.bench.dialogue import speak

            chosen = rng.sample(PIPER_VOICES[language], min(n, len(PIPER_VOICES[language])))
            def piper(name: str) -> Voice:
                return lambda text: (speak(name, text), text)

            voices = {c: piper(c) for c in chosen}
            lines = LINES[language]
        progress(f"{ds.name} {rid}: {len(voices)} speakers")
        audio, ref = conversation(rng, voices, lines, rng.randint(*turns))
        sf.write(ds.dir / "audio" / f"{rid}.flac", audio, SAMPLE_RATE)
        ds.write([Recording(rid, ds.name, split, f"audio/{rid}.flac", language, ref,
                            meta={"duration_s": len(audio) / SAMPLE_RATE, "seed": base + i,
                                  "source": source})])
    return ds


def _commonvoice_pool(folder: Path, min_clips: int = 15) -> dict[str, list[tuple[Path, str]]]:
    """Common Voice speakers (client_id) with at least ``min_clips`` validated clips."""
    clips: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    with open(folder / "validated.tsv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            clips[row["client_id"]].append((folder / "clips" / row["path"], row["sentence"]))
    return {k: v for k, v in clips.items() if len(v) >= min_clips}


def _cv_voice(clips: list[tuple[Path, str]], rng: random.Random) -> Voice:
    """A Common Voice speaker: says one of their clips (the sentence comes with it)."""
    from transcribe.audio.file import load_audio
    from transcribe.vad.silero import speech_regions

    def speak(_text: str) -> tuple[np.ndarray, str]:
        path, sentence = rng.choice(clips)
        audio = load_audio(path)
        regions = speech_regions(audio, min_silence_ms=500, pad_ms=50)
        if regions:  # recordings start and end with silence: keep the speech
            audio = audio[int(regions[0][0] * SAMPLE_RATE):int(regions[-1][1] * SAMPLE_RATE)]
        return audio, sentence

    return speak


# --- your own sessions (corrected in the app or with `tx sessions`) -------------------
def add_session(session_dir: Path, dataset: str = "local", split: str = "test") -> Recording:
    """Use a corrected session as reference: its speakers and text become the truth."""
    from transcribe import sessions

    t = sessions.load_numbered(session_dir)
    audio = sessions.playback_audio(session_dir)
    if audio is None:
        raise FileNotFoundError(f"No audio for {session_dir}")
    langs = t.meta.get("languages") or []
    turns = [RefTurn(s.t0, s.t1, t.speaker_label(s.speaker_id) if s.speaker_id else "?",
                     s.text) for s in t.segments if s.words]
    rec = Recording(session_dir.name, dataset, split, str(audio.resolve()),
                    langs[0] if isinstance(langs, list) and len(langs) == 1 else "", turns,
                    meta={"duration_s": max((x.t1 for x in turns), default=0.0)})
    Dataset(dataset).write([rec])
    return rec
