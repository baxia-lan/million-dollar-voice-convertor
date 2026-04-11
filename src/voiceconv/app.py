"""Application entry point."""

from __future__ import annotations

import logging
import sys

from voiceconv.logging_config import setup_logging


def main():
    """Launch VoiceConv GUI application."""
    setup_logging(level=logging.INFO)

    # Import PySide6 here to fail fast with a clear message
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "ERROR: PySide6 is required. Install with: pip install PySide6",
            file=sys.stderr,
        )
        sys.exit(1)

    from voiceconv.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VoiceConv")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
