"""GUI entry point (`tx-gui` or `tx gui`)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The GUI needs the 'gui' extra: pip install 'transcribe[gui]'", file=sys.stderr)
        return 1
    from transcribe.core.config import Config, required_models
    from transcribe.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0], *argv])
    app.setApplicationName("Transcribe")
    config = Config.load()
    window = MainWindow(config)
    window.show()
    if "--smoke-test" in argv:  # packaging check: the window builds, then quit
        QTimer.singleShot(500, app.quit)
        code = app.exec()
        print("smoke test ok")
        return code

    def first_run() -> None:
        from transcribe.core.models import missing_models
        from transcribe.gui.dialogs import ModelSetupDialog

        missing = missing_models(*required_models(config))
        if missing:
            ModelSetupDialog(missing, window).exec()

    QTimer.singleShot(0, first_run)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
