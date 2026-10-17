"""User-interface translations (gettext style): ``_("Text")``, ``ngettext(one, many, n)``.

Source strings are English. Catalogs are gettext .po files in ``transcribe/locale/``
(``de.po``), read directly - no compiled .mo files, no extra dependency; edit them with
any PO editor (Poedit, Weblate, ...). ``python scripts/i18n.py`` collects new strings
from the code into the catalogs; ``tests/unit/test_i18n.py`` checks that every string is
translated and keeps its placeholders.

Language: ``TRANSCRIBE_LANG``, else ``[ui] language`` in config.toml, else the system
language; anything without a catalog falls back to English. Strings are translated when
first used, so the language is fixed for the lifetime of the process (the app applies a
new language after a restart).

Placeholders use ``str.format``: write ``_("Saved to {path}").format(path=p)``, never an
f-string inside ``_()`` (it could not be looked up).
"""

from __future__ import annotations

import ast
import locale
import os
import re
from pathlib import Path

LANGUAGES = {"en": "English", "de": "Deutsch"}
LOCALE_DIR = Path(__file__).parent / "locale"
_DEFAULT_SPEAKER = re.compile(r"Speaker (\d+)")

_lang: str | None = None
_catalog: dict[str, str | list[str]] = {}


def detect() -> str:
    """The language to use: TRANSCRIBE_LANG, config, system locale - "en" if unsupported."""
    wanted = os.environ.get("TRANSCRIBE_LANG", "").strip()
    if not wanted:
        try:
            from transcribe.core.config import Config

            wanted = Config.load().ui.language
        except Exception:  # noqa: BLE001 - a broken config must not break the UI text
            wanted = "auto"
    if wanted in ("", "auto"):
        wanted = system_language()
    code = wanted.replace("-", "_").split("_")[0].lower()
    return code if code in LANGUAGES else "en"


def system_language() -> str:
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        if value := os.environ.get(var, ""):
            return value.split(":")[0]
    try:
        name = locale.getlocale()[0] or ""
    except ValueError:
        name = ""
    # Windows reports names like "German_Germany"
    return {"german": "de", "english": "en"}.get(name.split("_")[0].lower(), name)


def set_language(lang: str | None = None) -> str:
    """Load the catalog for ``lang`` (default: detect()). Returns the language used."""
    global _lang, _catalog
    _lang = lang if lang in LANGUAGES else detect()
    path = LOCALE_DIR / f"{_lang}.po"
    _catalog = read_po(path) if _lang != "en" and path.exists() else {}
    return _lang


def language() -> str:
    return _lang or set_language()


def _(message: str) -> str:
    if _lang is None:
        set_language()
    text = _catalog.get(message)
    return text if isinstance(text, str) and text else message


def ngettext(singular: str, plural: str, n: int) -> str:
    """``singular`` for n == 1, else ``plural`` (English and German share this rule)."""
    if _lang is None:
        set_language()
    forms = _catalog.get(singular)
    if isinstance(forms, list) and len(forms) == 2 and all(forms):
        return forms[0 if n == 1 else 1]
    return singular if n == 1 else plural


def N_(message: str) -> str:
    """Mark a string for translation without translating it yet (module-level tables)."""
    return message


def display_label(label: str) -> str:
    """A speaker label for display: default labels ("Speaker 3") in the UI language."""
    if m := _DEFAULT_SPEAKER.fullmatch(label):
        return _("Speaker {n}").format(n=m.group(1))
    return label


# --- .po files ------------------------------------------------------------------------
def read_po(path: Path) -> dict[str, str | list[str]]:
    """msgid -> msgstr (or [singular, plural] for plural entries); fuzzy entries skipped."""
    catalog: dict[str, str | list[str]] = {}
    for entry in parse_po(path.read_text(encoding="utf-8")):
        if "fuzzy" in entry["flags"] or not entry["msgid"]:
            continue
        catalog[entry["msgid"]] = (entry["msgstr_plural"] if entry["msgid_plural"]
                                   else entry["msgstr"])
    return catalog


def parse_po(text: str) -> list[dict]:
    """Entries of a .po file: msgid, msgid_plural, msgstr, msgstr_plural, flags,
    comments (lines starting with '#'), in file order."""
    entries: list[dict] = []
    cur: dict | None = None
    key: str | None = None

    def new() -> dict:
        return {"msgid": "", "msgid_plural": "", "msgstr": "", "msgstr_plural": [],
                "flags": set(), "comments": []}

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if cur is not None:
                entries.append(cur)
            cur, key = None, None
            continue
        if cur is None:
            cur = new()
        if line.startswith("#,"):
            cur["flags"] |= {f.strip() for f in line[2:].split(",")}
        elif line.startswith("#"):
            cur["comments"].append(line)
        elif line.startswith('"'):
            value = _unquote(line)
            if key and key.startswith("msgstr["):
                cur["msgstr_plural"][-1] += value
            elif key:
                cur[key] += value
        else:
            word, _sep, rest = line.partition(" ")
            if word.startswith("msgstr["):
                cur["msgstr_plural"].append(_unquote(rest))
                key = word
            elif word in ("msgid", "msgid_plural", "msgstr"):
                cur[word] = _unquote(rest)
                key = word
    if cur is not None:
        entries.append(cur)
    return entries


def _unquote(s: str) -> str:
    return ast.literal_eval(s.strip())  # PO strings use the same escapes as Python


def quote(s: str) -> str:
    """A string as PO literal (inverse of _unquote)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") \
        .replace("\t", "\\t") + '"'
