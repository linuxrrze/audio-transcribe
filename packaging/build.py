"""Build the desktop bundle with PyInstaller, smoke-test it and archive it.

Usage: uv run python packaging/build.py [--skip-smoke]
Output: dist/transcribe-<version>-<platform>.(zip|tar.gz)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def version() -> str:
    import tomllib

    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def run(cmd: list[str], **kw) -> str:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def smoke_test(bundle: Path) -> None:
    exe = ".exe" if os.name == "nt" else ""
    tx = bundle / f"tx{exe}"
    print(run([str(tx), "version"]).strip())
    plugins = run([str(tx), "plugins"])
    assert "faster-whisper" in plugins and "sherpa-onnx" in plugins, plugins
    env = {**os.environ, "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen")}
    gui = bundle / f"Transcribe{exe}"
    if sys.platform == "darwin":
        gui = DIST / "Transcribe.app" / "Contents" / "MacOS" / "Transcribe"
    out = run([str(gui), "--smoke-test"], env=env)
    assert "smoke test ok" in out, out
    print("smoke tests passed")


def archive(bundle: Path) -> Path:
    system = {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux")
    name = DIST / f"transcribe-{version()}-{system}-{platform.machine().lower()}"
    if system == "macos":
        src_dir, base = DIST, "Transcribe.app"
        return Path(shutil.make_archive(str(name), "zip", src_dir, base))
    fmt = "zip" if system == "windows" else "gztar"
    return Path(shutil.make_archive(str(name), fmt, DIST, bundle.name))


def main() -> None:
    shutil.rmtree(DIST, ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(ROOT / "build"),
         str(ROOT / "packaging" / "transcribe.spec")])
    bundle = DIST / "transcribe"
    if "--skip-smoke" not in sys.argv:
        smoke_test(bundle)
    print(f"archive: {archive(bundle)}")


if __name__ == "__main__":
    main()
