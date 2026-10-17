#!/usr/bin/env python3
"""Collect translatable strings from the code into the .po catalogs.

    python scripts/i18n.py            # update src/transcribe/locale/*.po
    python scripts/i18n.py --check    # exit 1 if a catalog is out of date or incomplete

Finds _("..."), N_("...") and ngettext("...", "...", n) calls with literal strings.
Existing translations are kept; new strings get an empty msgstr (English is shown until
they are translated); strings no longer in the code are dropped.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "transcribe"
sys.path.insert(0, str(ROOT / "src"))
from transcribe.i18n import LOCALE_DIR, parse_po, quote  # noqa: E402


def extract() -> dict[tuple[str, str], list[str]]:
    """(msgid, msgid_plural) -> source locations, in order of first appearance."""
    found: dict[tuple[str, str], list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            args = node.args
            strs = [a.value for a in args if isinstance(a, ast.Constant)
                    and isinstance(a.value, str)]
            if node.func.id in ("_", "N_") and len(args) == 1 and len(strs) == 1:
                key = (strs[0], "")
            elif node.func.id == "ngettext" and len(args) == 3 and len(strs) >= 2 \
                    and all(isinstance(a, ast.Constant) for a in args[:2]):
                key = (strs[0], strs[1])
            else:
                continue
            # "/" on every platform: the catalog must not depend on where it was updated
            where = f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
            found.setdefault(key, []).append(where)
    return found


def render(found: dict, old: dict[str, dict], lang: str) -> str:
    out = [f'# Transcribe - {lang} translation. Update with: python scripts/i18n.py',
           'msgid ""', 'msgstr ""',
           quote("Content-Type: text/plain; charset=UTF-8\n"),
           quote(f"Language: {lang}\n"),
           quote("Plural-Forms: nplurals=2; plural=(n != 1);\n"), ""]
    for (msgid, plural), where in found.items():
        prev = old.get(msgid, {})
        out += [f"#: {w}" for w in where[:3]]
        if "fuzzy" in prev.get("flags", ()):
            out.append("#, fuzzy")
        out.append(f"msgid {quote(msgid)}")
        if plural:
            forms = prev.get("msgstr_plural") or ["", ""]
            out.append(f"msgid_plural {quote(plural)}")
            out += [f"msgstr[{i}] {quote(f)}" for i, f in enumerate((forms + ["", ""])[:2])]
        else:
            out.append(f"msgstr {quote(prev.get('msgstr', ''))}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    check = "--check" in sys.argv
    found = extract()
    status = 0
    for po in sorted(LOCALE_DIR.glob("*.po")):
        old = {e["msgid"]: e for e in parse_po(po.read_text(encoding="utf-8")) if e["msgid"]}
        text = render(found, old, po.stem)
        missing = [m for (m, p) in found if not (old.get(m, {}).get("msgstr_plural") if p
                                                   else old.get(m, {}).get("msgstr"))]
        if check:
            if text != po.read_text(encoding="utf-8"):
                print(f"{po.name}: out of date - run python scripts/i18n.py")
                status = 1
            if missing:
                print(f"{po.name}: {len(missing)} untranslated, e.g. {missing[:3]}")
                status = 1
        else:
            po.write_text(text, encoding="utf-8")
            print(f"{po.name}: {len(found)} strings, {len(missing)} untranslated")
    return status


if __name__ == "__main__":
    sys.exit(main())
