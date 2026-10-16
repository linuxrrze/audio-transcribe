"""Speaker colours that work on light and dark backgrounds."""

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2",
           "#8c564b", "#bcbd22", "#7f7f7f"]


def speaker_color(speaker_id: str | None) -> str:
    if not speaker_id or not speaker_id[1:].isdigit():
        return "#7f7f7f"
    return PALETTE[(int(speaker_id[1:]) - 1) % len(PALETTE)]
