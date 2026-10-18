"""Small widget helpers that keep long texts (translations, device names, paths) from
widening windows."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLineEdit

MIN_COMBO_CHARS = 16  # compact combo boxes: at least this many characters wide


def compact_combo(combo: QComboBox) -> QComboBox:
    """The field may be narrower than its longest entry; the drop-down shows it in full
    (call ``fit_popup`` again after changing the entries)."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(MIN_COMBO_CHARS)
    fit_popup(combo)
    return combo


def fit_popup(combo: QComboBox) -> None:
    """Make the drop-down list wide enough for the longest entry."""
    metrics = combo.fontMetrics()
    longest = max((metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())),
                  default=0)
    view = combo.view()
    # text plus item padding, scroll bar and frame
    view.setMinimumWidth(longest + 3 * metrics.averageCharWidth()
                         + view.verticalScrollBar().sizeHint().width() + 2 * view.frameWidth())


def path_text(text: str) -> QLineEdit:
    """Read-only, selectable text for a path: unlike a label it never widens the window
    (a long path scrolls, the tooltip shows it in full)."""
    field = QLineEdit(text)
    field.setReadOnly(True)
    field.setFrame(False)
    field.setToolTip(text)
    field.setCursorPosition(0)
    field.setStyleSheet("QLineEdit { background: transparent; }")
    return field
