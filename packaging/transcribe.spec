# PyInstaller spec: one folder with the `tx` CLI and the `Transcribe` GUI (shared libs).
# Build: uv run python packaging/build.py   (models are downloaded on first run)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 - provided by PyInstaller
MAC = sys.platform == "darwin"

datas, binaries = [], []
for pkg in ("faster_whisper",):  # Silero VAD model shipped as package data
    datas += collect_data_files(pkg)
for pkg in ("ctranslate2", "sherpa_onnx", "onnxruntime", "soxr"):
    binaries += collect_dynamic_libs(pkg)
datas += copy_metadata("transcribe")  # entry points (registry also has built-ins)
hidden = [
    "transcribe.asr.faster_whisper", "transcribe.diarize.sherpa_offline",
    "transcribe.speakers.embedder", "transcribe.audio.mic", "transcribe.audio.system",
    "transcribe.export.txt", "transcribe.export.md", "transcribe.export.srt",
    "transcribe.export.json", "keyring.backends",
]
helper = ROOT / "helpers/macos-audiotap/.build/apple/Products/Release/transcribe-audiotap"
if MAC and helper.exists():
    binaries.append((str(helper), "."))

excludes = ["tkinter", "matplotlib", "torch", "IPython", "pytest"]


def analysis(script):
    return Analysis(  # noqa: F821
        [str(ROOT / "packaging" / script)], pathex=[str(ROOT / "src")], binaries=binaries,
        datas=datas, hiddenimports=hidden, excludes=excludes, noarchive=False)


cli, gui = analysis("entry_cli.py"), analysis("entry_gui.py")
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="tx",  # noqa: F821
              console=True, codesign_identity=None,
              entitlements_file=str(ROOT / "packaging/macos/entitlements.plist"))
gui_exe = EXE(PYZ(gui.pure), gui.scripts, [], exclude_binaries=True,  # noqa: F821
              name="Transcribe", console=False, codesign_identity=None,
              entitlements_file=str(ROOT / "packaging/macos/entitlements.plist"))
coll = COLLECT(cli_exe, gui_exe, cli.binaries, cli.datas, gui.binaries, gui.datas,  # noqa: F821
               name="transcribe")

if MAC:
    app = BUNDLE(  # noqa: F821
        coll, name="Transcribe.app", bundle_identifier="org.transcribe.desktop",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSMicrophoneUsageDescription":
                "Transcribe records your microphone to create a local transcript.",
            "NSAudioCaptureUsageDescription":
                "Transcribe records system audio (e.g. calls) to create a local transcript.",
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
        })
