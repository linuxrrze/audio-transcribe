"""PyInstaller entry point for the `tx` command line tool."""

import multiprocessing

if __name__ == "__main__":
    # Frozen apps start worker processes (e.g. diarization) by re-running this
    # executable; freeze_support() makes such a child run its task instead of the CLI.
    multiprocessing.freeze_support()
    from transcribe.cli.main import app

    app()
