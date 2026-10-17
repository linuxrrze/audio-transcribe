"""Level meter widget (-60..0 dBFS, colour zones, peak hold) and a device check dialog."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from transcribe.audio.levels import (
    FLOOR_DB,
    PEAK_HOLD_S,
    DeviceEntry,
    Level,
    LevelMonitor,
    device_source,
    list_devices,
    meter_fraction,
)
from transcribe.i18n import N_, _

GREEN, YELLOW, RED = QColor("#2ca02c"), QColor("#e6b800"), QColor("#d62728")
STATUS_TEXT = {"signal": "", "silent": N_("silent"), "no data": N_("no data"),
               "error": N_("error")}


class LevelMeter(QWidget):
    """Horizontal meter. Feed it with ``set_level`` (a Level) or ``set_db`` (RMS dBFS)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.rms_db = FLOOR_DB
        self.peak_db = FLOOR_DB
        self._peak_at = 0.0
        self.status = "no data"
        self.setToolTip(_("Signal level (dBFS)"))

    LABEL_W = 64  # space for "-34 dB" / "no data" (also translated) right of the bar

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(200, 16)

    def set_db(self, rms_db: float) -> None:
        now = time.monotonic()
        rms_db = max(FLOOR_DB, rms_db)
        self.rms_db = rms_db
        if rms_db >= self.peak_db or now - self._peak_at > PEAK_HOLD_S:
            self.peak_db, self._peak_at = rms_db, now
        self.status = "signal" if rms_db > -50 else "silent"
        self.update()

    def set_level(self, level: Level) -> None:
        self.rms_db, self.peak_db = level.rms_db, level.peak_db
        self.status = level.status
        self.setToolTip(level.error or _("{rms} dBFS (max {peak})").format(
            rms=f"{level.rms_db:.0f}", peak=f"{level.max_db:.0f}"))
        self.update()

    def reset(self) -> None:
        self.rms_db = self.peak_db = FLOOR_DB
        self.status = "no data"
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        full = QRectF(self.rect())
        r = full.adjusted(0.5, 0.5, -0.5 - self.LABEL_W, -0.5)
        p.setPen(self.palette().mid().color())
        p.setBrush(self.palette().base())
        p.drawRect(r)
        w = r.width() - 2
        fill = meter_fraction(self.rms_db) * w
        # zones: green up to -12 dBFS, yellow to -3, red above
        for start_db, end_db, color in ((FLOOR_DB, -12, GREEN), (-12, -3, YELLOW), (-3, 0, RED)):
            x0, x1 = meter_fraction(start_db) * w, meter_fraction(end_db) * w
            if fill > x0:
                p.fillRect(QRectF(r.left() + 1 + x0, r.top() + 1, min(fill, x1) - x0,
                                  r.height() - 2), color)
        if self.peak_db > FLOOR_DB:
            x = r.left() + 1 + meter_fraction(self.peak_db) * w
            p.fillRect(QRectF(x - 1, r.top() + 1, 2, r.height() - 2),
                       RED if self.peak_db > -3 else self.palette().text().color())
        text = STATUS_TEXT.get(self.status, "")
        label = _(text) if text else f"{self.rms_db:.0f} dB"
        p.setPen(self.palette().text().color())
        font = p.font()
        font.setPointSizeF(max(6.0, font.pointSizeF() * 0.8))
        p.setFont(font)
        p.drawText(QRectF(r.right() + 4, full.top(), self.LABEL_W - 4, full.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)


class DeviceCheckDialog(QDialog):
    """Meters for every microphone and system-audio device; "Use" picks one."""

    def __init__(self, on_use: Callable[[DeviceEntry], None], parent=None,
                 devices: tuple[list[DeviceEntry], list[str]] | None = None,
                 source_factory: Callable = device_source) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Check audio devices"))
        self.resize(640, 420)
        entries, errors = devices or list_devices()
        intro = QLabel(_("Speak into your microphone and play something in the meeting app. "
                         "The devices whose meters move are the ones to use."))
        intro.setWordWrap(True)
        grid = QGridLayout()
        self.meters: dict[str, LevelMeter] = {}
        row = 0
        for kind, title in (("input", _("Microphones / inputs")),
                            ("output", _("System audio (what the computer plays)"))):
            grid.addWidget(QLabel(f"<b>{title}</b>"), row, 0, 1, 3)
            row += 1
            for e in (e for e in entries if e.kind == kind):
                name = _("{device} (default)").format(device=e.name) if e.is_default else e.name
                grid.addWidget(QLabel(name), row, 0)
                meter = LevelMeter()
                self.meters[e.key] = meter
                grid.addWidget(meter, row, 1)
                use = QPushButton(_("Use"))
                use.clicked.connect(self._use_handler(on_use, e))
                grid.addWidget(use, row, 2)
                row += 1
        for err in errors:
            grid.addWidget(QLabel(f"<span style='color:#d62728'>{err}</span>"), row, 0, 1, 3)
            row += 1
        grid.setColumnStretch(1, 1)
        inner = QWidget()
        inner.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        close = QPushButton(_("Close"))
        close.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)
        self.monitor = LevelMonitor({e.key: source_factory(e) for e in entries}).start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    def _use_handler(self, on_use: Callable[[DeviceEntry], None],
              entry: DeviceEntry) -> Callable[[], None]:
        def use() -> None:
            on_use(entry)
            self.accept()

        return use

    def _refresh(self) -> None:
        for key, level in self.monitor.snapshot().items():
            self.meters[key].set_level(level)

    def done(self, result: int) -> None:
        self.timer.stop()
        self.monitor.stop()
        super().done(result)
