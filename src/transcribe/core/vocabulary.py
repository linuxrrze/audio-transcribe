"""Custom vocabulary: hint words for the recognizer and correction rules.

- **Terms** (names, acronyms, jargon: "Justus", "DSGVO", "FAU") are passed to Whisper
  as hotwords, which bias every 30 s decoding window towards these spellings.
- **Corrections** (``wrong -> right``) fix what the model still gets wrong, e.g.
  ``Fanfress -> Fentriss``. They apply to whole words (case-insensitive), may span
  several words (``Mr Fan Fress -> Mr. Fentriss``), and keep the word timestamps.

Text format (``--vocab-file``, the GUI settings): one entry per line, ``#`` starts a
comment, ``->`` makes a correction; a correction's target is also used as a hint word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from transcribe.storage.transcript_model import Word

# Whisper's prompt holds at most 223 tokens; hint words compete with the previous text
# for that space, so a very long list is cut (most important terms first).
MAX_HOTWORD_CHARS = 600

_TOKEN = re.compile(r"\w[\w'-]*", re.UNICODE)


def _norm(text: str) -> str:
    return " ".join(m.group(0).lower() for m in _TOKEN.finditer(text))


@dataclass
class Vocabulary:
    terms: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)  # wrong -> right

    @classmethod
    def parse(cls, text: str) -> Vocabulary:
        """Lines or comma-separated terms; ``a -> b`` lines are corrections."""
        vocab = cls()
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "->" in line:
                wrong, right = (part.strip() for part in line.split("->", 1))
                if wrong and right:
                    vocab.corrections[wrong] = right
                continue
            vocab.terms += [t.strip() for t in line.split(",") if t.strip()]
        return vocab

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        return cls.parse(path.read_text(encoding="utf-8"))

    @classmethod
    def from_config(cls, terms: list[str], corrections: dict[str, str]) -> Vocabulary:
        return cls(list(terms), dict(corrections))

    def merged(self, other: Vocabulary) -> Vocabulary:
        return Vocabulary(list(dict.fromkeys(self.terms + other.terms)),
                          {**self.corrections, **other.corrections})

    def __bool__(self) -> bool:
        return bool(self.terms or self.corrections)

    def to_text(self) -> str:
        lines = list(self.terms)
        lines += [f"{wrong} -> {right}" for wrong, right in self.corrections.items()]
        return "\n".join(lines)

    def hotwords(self, max_chars: int = MAX_HOTWORD_CHARS) -> str | None:
        """Hint words for Whisper: terms plus correction targets, deduplicated."""
        words = list(dict.fromkeys(self.terms + list(self.corrections.values())))
        out: list[str] = []
        size = 0
        for w in words:
            if size + len(w) + 2 > max_chars:
                break
            out.append(w)
            size += len(w) + 2
        return ", ".join(out) or None

    def correct(self, words: list[Word]) -> list[Word]:
        """Apply corrections; matching words are merged into one word spanning them."""
        rules = sorted(((_norm(w).split(), r) for w, r in self.corrections.items()
                        if _norm(w)), key=lambda rule: -len(rule[0]))
        if not rules:
            return words
        tokens = [_norm(w.text) for w in words]
        out: list[Word] = []
        i = 0
        while i < len(words):
            for pattern, right in rules:
                n = len(pattern)
                if tokens[i:i + n] == pattern:
                    first, last = words[i], words[i + n - 1]
                    lead = first.text[: len(first.text) - len(first.text.lstrip())]
                    m = re.search(r"[^\w']*$", last.text.rstrip())
                    trail = m.group(0) if m else ""  # keep trailing punctuation
                    out.append(Word(lead + right + trail, first.t0, last.t1,
                                    min(w.confidence for w in words[i:i + n]),
                                    first.speaker_id, first.overlap))
                    i += n
                    break
            else:
                out.append(words[i])
                i += 1
        return out
