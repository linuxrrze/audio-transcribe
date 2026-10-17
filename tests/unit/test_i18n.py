"""Translations: catalogs complete and consistent, language detection, PO reading."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from transcribe import i18n

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
# HTML tags and rich style markup ([red]...[/]); "[required]" etc. are plain text
TAG = re.compile(r"</?(?:a|b|i|p|br|span)\b[^>]*>"
                 r"|\[/?(?:red|green|yellow|cyan|blue|magenta|bold|dim|italic|b|i)?\]")


@pytest.fixture
def lang():
    yield i18n.set_language
    i18n.set_language("en")


def test_catalogs_up_to_date_and_complete():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "i18n.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("po", sorted(i18n.LOCALE_DIR.glob("*.po")), ids=lambda p: p.stem)
def test_translations_keep_placeholders_and_markup(po):
    for e in i18n.parse_po(po.read_text(encoding="utf-8")):
        if not e["msgid"]:
            continue
        sources = [e["msgid"], e["msgid_plural"] or e["msgid"]]
        targets = e["msgstr_plural"] if e["msgid_plural"] else [e["msgstr"]]
        for src, dst in zip(sources, targets, strict=False):
            assert sorted(PLACEHOLDER.findall(src)) == sorted(PLACEHOLDER.findall(dst)), \
                (e["msgid"], dst)
            assert sorted(TAG.findall(src)) == sorted(TAG.findall(dst)), (e["msgid"], dst)
            assert src.endswith("…") == dst.endswith("…"), (e["msgid"], dst)
            assert ("&" in src) == ("&" in dst), (e["msgid"], dst)  # menu mnemonics


def test_german(lang):
    lang("de")
    assert i18n.language() == "de"
    assert i18n.display_label("Speaker 3") == "Sprecher 3"
    assert i18n.display_label("Anna") == "Anna"  # names are not translated
    lang("en")
    assert i18n._("Speaker {n}") == "Speaker {n}"
    assert i18n.display_label("Speaker 3") == "Speaker 3"


def test_unknown_message_and_plural_fallback(lang):
    lang("de")
    assert i18n._("A string nobody translated") == "A string nobody translated"
    assert i18n.ngettext("{n} file", "{n} files", 2) == "{n} files"


def test_detect(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRANSCRIBE_LANG", "de")
    assert i18n.detect() == "de"
    monkeypatch.setenv("TRANSCRIBE_LANG", "fr_FR")  # no catalog: English
    assert i18n.detect() == "en"
    monkeypatch.delenv("TRANSCRIBE_LANG")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")  # config "auto": system language
    assert i18n.detect() == "de"
    (tmp_path / "transcribe").mkdir()
    (tmp_path / "transcribe" / "config.toml").write_text('[ui]\nlanguage = "en"\n')
    assert i18n.detect() == "en"  # the config wins over the system


def test_po_roundtrip(tmp_path):
    original, german = 'Say "hi"\n\tnow', "Sag „hallo“\n\tjetzt"  # real newline + tab
    po = tmp_path / "x.po"
    po.write_text("\n".join([
        'msgid ""', 'msgstr ""', '"Language: de\\n"', "",
        "#: a.py:1", f"msgid {i18n.quote(original)}", f"msgstr {i18n.quote(german)}", "",
        "#, fuzzy", 'msgid "Old"', 'msgstr "Alt"', "",
        'msgid "{n} line"', 'msgid_plural "{n} lines"', 'msgstr[0] "{n} Zeile"',
        'msgstr[1] ""', '"{n} Zeilen"', "",
    ]), encoding="utf-8")
    cat = i18n.read_po(po)
    assert cat[original] == german
    assert "Old" not in cat  # fuzzy: not used
    assert cat["{n} line"] == ["{n} Zeile", "{n} Zeilen"]
