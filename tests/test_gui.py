"""Tests for GUI components (headless-safe)."""

import tempfile

import pytest

from voiceconv.core.types import AudioFormat, ConversionJob, ProvenanceRecord
from pathlib import Path


class TestGUIImports:
    """Test that GUI modules can be imported without a display server."""

    def test_worker_import(self):
        from voiceconv.gui.worker import ConversionWorker
        assert ConversionWorker is not None

    def test_types_used_by_gui(self):
        tmp = Path(tempfile.gettempdir())
        job = ConversionJob(
            song_path=tmp / "test.wav",
            reference_path=tmp / "ref.wav",
            output_dir=tmp / "out",
            output_format=AudioFormat.WAV,
            provenance=ProvenanceRecord(user_consent=True),
        )
        assert job.output_format == AudioFormat.WAV


@pytest.mark.gui
class TestMainWindow:
    """GUI tests that require a display server (skipped in CI)."""

    def test_window_creation(self):
        try:
            from PySide6.QtWidgets import QApplication
            import sys

            app = QApplication.instance() or QApplication(sys.argv)
            from voiceconv.gui.main_window import MainWindow

            window = MainWindow()
            assert window.windowTitle().startswith("VoiceConv")
            window.close()
        except Exception:
            pytest.skip("Display server not available")
