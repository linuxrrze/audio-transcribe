"""GUI entry point (`tx-gui` or `tx gui`)."""

from __future__ import annotations

import os
import sys

# Qt 6 on Wayland logs every text-field focus change of the input-method protocol
# ("qt.qpa.wayland.textinput: ... Trying to disable ... but ... is focused") - harmless
# noise. Silenced unless the user asks for that category explicitly.
QUIET_QT_LOGGING = "qt.qpa.wayland.textinput=false"


def quiet_qt_logging() -> None:
    """Add QUIET_QT_LOGGING to QT_LOGGING_RULES (before the QApplication exists)."""
    rules = os.environ.get("QT_LOGGING_RULES", "")
    if "qt.qpa.wayland.textinput" not in rules:
        os.environ["QT_LOGGING_RULES"] = ";".join(r for r in (rules, QUIET_QT_LOGGING) if r)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quiet_qt_logging()
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        from transcribe.i18n import _

        print(_("The GUI needs the 'gui' extra: pip install 'transcribe[gui]'"),
              file=sys.stderr)
        return 1
    from transcribe.core.config import Config
    from transcribe.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([sys.argv[0], *argv])
    app.setApplicationName("Transcribe")
    install_qt_translations(app)
    config = Config.load()
    window = MainWindow(config)
    window.show()
    if "--smoke-test" in argv:  # packaging check: the window builds, then quit
        QTimer.singleShot(500, app.quit)
        code = _run(app, window)
        print("smoke test ok")
        return code

    def first_run() -> None:  # no config.toml yet, or models missing: the setup wizard
        from transcribe.core.setup import needs_setup

        if needs_setup(config):
            window.setup()

    QTimer.singleShot(0, first_run)
    return _run(app, window)


def install_qt_translations(app) -> bool:
    """Qt's own texts (standard buttons like Save/Cancel, file dialogs) in the UI language,
    from the translations that ship with Qt. Returns whether any were loaded."""
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    from transcribe.i18n import language

    lang = language()
    if lang == "en":
        return False
    QLocale.setDefault(QLocale(lang))  # number and date formats in dialogs
    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if not translator.load(QLocale(lang), "qtbase", "_", path):
        return False
    app.installTranslator(translator)
    return True


def _run(app, window) -> int:
    """Run the event loop, then tear Qt down deterministically.

    Leaving widgets (and the QtMultimedia player) to interpreter shutdown destroys them
    in arbitrary order, which can segfault on exit.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    code = app.exec()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
