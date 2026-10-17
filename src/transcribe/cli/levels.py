"""Terminal rendering of device levels (`tx levels`, live view)."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from transcribe.audio.levels import FLOOR_DB, Level, meter_fraction
from transcribe.i18n import N_, _

WIDTH = 30
STATUS_STYLE = {"signal": "green", "silent": "yellow", "no data": "dim", "error": "red"}
STATUS_TEXT = {"signal": N_("signal"), "silent": N_("silent"), "no data": N_("no data"),
               "error": N_("error")}


def meter(level: Level, width: int = WIDTH) -> Text:
    """▮-bar for the RMS level with a │ peak marker; green / yellow (> -12) / red (> -3)."""
    filled = round(meter_fraction(level.rms_db) * width)
    peak = min(width - 1, round(meter_fraction(level.peak_db) * width))
    text = Text()
    for i in range(width):
        pos_db = FLOOR_DB + (i + 1) / width * -FLOOR_DB
        style = "red" if pos_db > -3 else "yellow" if pos_db > -12 else "green"
        if i < filled:
            text.append("▮", style=style)
        elif i == peak and level.peak_db > FLOOR_DB:
            text.append("│", style=style)
        else:
            text.append("▯", style="dim")
    return text


def levels_table(rows: list[tuple[str, str, bool]], snapshot: dict[str, Level],
                 title: str | None = None) -> Table:
    """``rows``: (key, label, is_default) in display order."""
    table = Table(title=title or _("Audio levels"), title_justify="left")
    table.add_column(_("Device"), overflow="fold", max_width=48)
    table.add_column(_("Level"))
    table.add_column("dBFS", justify="right")
    table.add_column(_("max"), justify="right")
    table.add_column(_("Status"))
    for key, label, is_default in rows:
        lv = snapshot.get(key, Level())
        db = f"{lv.rms_db:5.0f}" if lv.receiving else "  -"
        status = Text(lv.error or _(STATUS_TEXT[lv.status]), style=STATUS_STYLE[lv.status])
        table.add_row(label + (" *" if is_default else ""), meter(lv), db,
                      f"{lv.max_db:4.0f}" if lv.max_db > FLOOR_DB else "-", status)
    return table
