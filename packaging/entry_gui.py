"""PyInstaller entry point for the Transcribe desktop app."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # see entry_cli.py
    from transcribe.gui.app import main

    sys.exit(main())
