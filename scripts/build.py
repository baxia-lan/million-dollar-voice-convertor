#!/usr/bin/env python3
"""Build script: package VoiceConv as a standalone desktop app using PyInstaller."""

from __future__ import annotations

import platform
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
        "--hidden-import=torch",
        "--hidden-import=torchaudio",
        "--hidden-import=demucs.api",
        # Collect all data files for libraries that need them
        "--collect-data=resemblyzer",
        "--noconfirm",
        "--clean",
    ]

    # Platform-specific options
    if sys.platform == "darwin":
        cmd.append("--osx-bundle-identifier=com.voiceconv.app")
        icon_path = ROOT / "src" / "voiceconv" / "gui" / "resources" / "icon.icns"
        if icon_path.exists():
            cmd.append(f"--icon={icon_path}")
        # Build universal binary on Apple Silicon when possible
        if platform.machine() == "arm64":
            cmd.append("--target-arch=arm64")
    elif sys.platform == "win32":
        icon_path = ROOT / "src" / "voiceconv" / "gui" / "resources" / "icon.ico"
        if icon_path.exists():
            cmd.append(f"--icon={icon_path}")

    cmd.append(str(ENTRY))

    print("Building VoiceConv...")
    print(f"  Platform: {sys.platform} ({platform.machine()})")
    print(f"  Entry point: {ENTRY}")
    print(f"  Output: {DIST}")
    print()

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        out = DIST / "VoiceConv"
        if sys.platform == "darwin":
            out = DIST / "VoiceConv.app"
        print(f"\nBuild successful! Output at: {out}")
    else:
        print(f"\nBuild failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    build()
