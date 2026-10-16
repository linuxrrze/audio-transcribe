"""GUI entry point (`tx-gui` or `tx gui`)."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The GUI needs the 'gui' extra: pip install 'transcribe[gui]'", file=sys.stderr)
        return 1
    from transcribe.core.config import Config
    from transcribe.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Transcribe")
    window = MainWindow(Config.load())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
