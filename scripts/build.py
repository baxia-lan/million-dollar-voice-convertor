#!/usr/bin/env python3
"""Build script: package VoiceConv as a standalone desktop app using PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
ENTRY = SRC / "voiceconv" / "app.py"
DIST = ROOT / "dist"


def build():
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=VoiceConv",
        "--windowed",
        f"--distpath={DIST}",
        f"--workpath={ROOT / 'build'}",
        f"--specpath={ROOT / 'build'}",
        # Include the entire source package
        f"--paths={SRC}",
        # Hidden imports that PyInstaller may miss
        "--hidden-import=voiceconv",
        "--hidden-import=voiceconv.gui",
        "--hidden-import=voiceconv.gui.main_window",
        "--hidden-import=voiceconv.core",
        "--hidden-import=voiceconv.separation",
        "--hidden-import=voiceconv.svc",
        "--hidden-import=voiceconv.judge",
        "--hidden-import=voiceconv.compliance",
        "--hidden-import=voiceconv.export",
        "--hidden-import=soundfile",
        "--hidden-import=pyworld",
        "--hidden-import=scipy.signal",
        "--hidden-import=scipy.fft",
        # Collect all data files for libraries that need them
        "--collect-data=resemblyzer",
        "--noconfirm",
        "--clean",
        str(ENTRY),
    ]

    print(f"Building VoiceConv...")
    print(f"  Entry point: {ENTRY}")
    print(f"  Output: {DIST}")
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        print(f"\nBuild successful! Output at: {DIST / 'VoiceConv'}")
    else:
        print(f"\nBuild failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    build()
