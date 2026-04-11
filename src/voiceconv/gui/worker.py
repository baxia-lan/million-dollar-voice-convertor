"""Background worker thread for running the conversion pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from voiceconv.core.types import ConversionJob, JobStatus


class ConversionWorker(QThread):
    """Runs ConversionPipeline in a background thread.

    Signals:
        progress_updated(status_str, percent, message)
        finished_ok(output_path)
        finished_error(error_message)
    """

    progress_updated = Signal(str, float, str)
    finished_ok = Signal(str)
    finished_error = Signal(str)

    def __init__(self, pipeline, job: ConversionJob, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._job = job

    def _on_progress(self, status: JobStatus, pct: float, msg: str):
        self.progress_updated.emit(status.value, pct, msg)

    def run(self):
        try:
            output_path = self._pipeline.run(
                self._job, on_progress=self._on_progress
            )
            self.finished_ok.emit(str(output_path))
        except Exception as e:
            self.finished_error.emit(str(e))
