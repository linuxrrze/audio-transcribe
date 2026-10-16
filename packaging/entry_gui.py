"""PyInstaller entry point for the Transcribe desktop app."""

import sys

from transcribe.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
