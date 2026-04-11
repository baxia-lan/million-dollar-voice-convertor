#!/usr/bin/env python3
"""Launch VoiceConv application."""

import sys
from pathlib import Path

# Ensure src/ is on the path when running directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from voiceconv.app import main

if __name__ == "__main__":
    main()
