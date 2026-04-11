"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from voiceconv.compliance import ComplianceManager
from voiceconv.core.pipeline import ConversionPipeline
from voiceconv.core.types import AudioFormat, ConversionJob, ProvenanceRecord
from voiceconv.export import Exporter
from voiceconv.judge import QualityJudge
from voiceconv.separation.base import SeparatorBackend
from voiceconv.separation.demucs_adapter import DemucsSeparator
from voiceconv.svc.base import SVCBackend
from voiceconv.svc.world_svc import WorldSVC

from .worker import ConversionWorker

logger = logging.getLogger(__name__)

AUDIO_FILTER = "Audio Files (*.wav *.flac *.mp3 *.ogg *.m4a);;All Files (*)"


class MainWindow(QMainWindow):
    """VoiceConv main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceConv — Speech-to-Singing Voice Conversion")
        self.setMinimumSize(700, 600)
        self._worker: ConversionWorker | None = None
        self._setup_ui()
        self._setup_backends()

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # Title
        title = QLabel("VoiceConv")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        subtitle = QLabel(
            "Zero-shot speech-to-singing voice conversion. "
            "Preserves pitch, timing, lyrics, and expression — only changes timbre."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Input section ──
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)

        # Song input
        song_row = QHBoxLayout()
        song_row.addWidget(QLabel("Song:"))
        self._song_path = QLineEdit()
        self._song_path.setPlaceholderText("Path to song file (wav, mp3, flac...)")
        self._song_path.setReadOnly(True)
        song_row.addWidget(self._song_path)
        self._btn_song = QPushButton("Browse...")
        self._btn_song.clicked.connect(self._browse_song)
        song_row.addWidget(self._btn_song)
        input_layout.addLayout(song_row)

        # Reference speech input
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Speech Reference:"))
        self._ref_path = QLineEdit()
        self._ref_path.setPlaceholderText(
            "Path to your speech sample (any content, not singing)"
        )
        self._ref_path.setReadOnly(True)
        ref_row.addWidget(self._ref_path)
        self._btn_ref = QPushButton("Browse...")
        self._btn_ref.clicked.connect(self._browse_reference)
        ref_row.addWidget(self._btn_ref)
        input_layout.addLayout(ref_row)

        layout.addWidget(input_group)

        # ── Settings section ──
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Backend selection
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("SVC Backend:"))
        self._backend_combo = QComboBox()
        backend_row.addWidget(self._backend_combo)
        settings_layout.addLayout(backend_row)

        # Separator selection
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Separator:"))
        self._sep_combo = QComboBox()
        sep_row.addWidget(self._sep_combo)
        settings_layout.addLayout(sep_row)

        # Conversion strength
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Conversion Strength:"))
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setMinimum(10)
        self._strength_slider.setMaximum(100)
        self._strength_slider.setValue(80)
        self._strength_label = QLabel("0.80")
        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        strength_row.addWidget(self._strength_slider)
        strength_row.addWidget(self._strength_label)
        settings_layout.addLayout(strength_row)

        # Output format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Output Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["wav", "flac", "mp3", "ogg"])
        fmt_row.addWidget(self._format_combo)
        settings_layout.addLayout(fmt_row)

        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Dir:"))
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Output directory (default: next to song)")
        out_row.addWidget(self._output_dir)
        self._btn_outdir = QPushButton("Browse...")
        self._btn_outdir.clicked.connect(self._browse_output)
        out_row.addWidget(self._btn_outdir)
        settings_layout.addLayout(out_row)

        layout.addWidget(settings_group)

        # ── Consent ──
        self._consent_check = QCheckBox(
            "I confirm I have the right to use these audio materials "
            "and consent to voice conversion processing."
        )
        layout.addWidget(self._consent_check)

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self._btn_convert = QPushButton("Convert")
        self._btn_convert.setStyleSheet(
            "font-size: 16px; padding: 8px 24px; font-weight: bold;"
        )
        self._btn_convert.clicked.connect(self._start_conversion)
        btn_row.addWidget(self._btn_convert)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_conversion)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        # ── Progress ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

        # ── Log output ──
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

        # Status bar
        self.setStatusBar(QStatusBar())

    def _setup_backends(self):
        """Detect and register available backends."""
        # Separators
        separators = [DemucsSeparator()]
        for sep in separators:
            available = sep.is_available()
            label = f"{sep.name} {'✓' if available else '(not installed)'}"
            self._sep_combo.addItem(label, userData=sep)

        # SVC backends
        backends = [WorldSVC()]
        for be in backends:
            available = be.is_available()
            label = f"{be.name} {'✓' if available else '(not installed)'}"
            self._backend_combo.addItem(label, userData=be)

    # ── File browsing ────────────────────────────────────────────────

    def _browse_song(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Song", "", AUDIO_FILTER
        )
        if path:
            self._song_path.setText(path)

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Speech Reference", "", AUDIO_FILTER
        )
        if path:
            self._ref_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)

    # ── Conversion logic ─────────────────────────────────────────────

    def _validate_inputs(self) -> bool:
        if not self._song_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a song file.")
            return False
        if not self._ref_path.text():
            QMessageBox.warning(
                self, "Missing Input", "Please select a speech reference file."
            )
            return False
        if not Path(self._song_path.text()).exists():
            QMessageBox.warning(self, "File Not Found", "Song file does not exist.")
            return False
        if not Path(self._ref_path.text()).exists():
            QMessageBox.warning(
                self, "File Not Found", "Reference file does not exist."
            )
            return False
        if not self._consent_check.isChecked():
            QMessageBox.warning(
                self,
                "Consent Required",
                "Please check the consent box to proceed.",
            )
            return False
        return True

    def _build_pipeline(self) -> ConversionPipeline:
        separator: SeparatorBackend = self._sep_combo.currentData()
        svc_backend: SVCBackend = self._backend_combo.currentData()

        # Apply conversion strength if backend supports it
        if isinstance(svc_backend, WorldSVC):
            strength = self._strength_slider.value() / 100.0
            svc_backend = WorldSVC(conversion_strength=strength)

        fmt_str = self._format_combo.currentText()
        exporter = Exporter(default_format=AudioFormat(fmt_str))

        return ConversionPipeline(
            separator=separator,
            svc_backend=svc_backend,
            judge=QualityJudge(),
            exporter=exporter,
            compliance=ComplianceManager(),
        )

    def _start_conversion(self):
        if not self._validate_inputs():
            return

        song_path = Path(self._song_path.text())
        ref_path = Path(self._ref_path.text())

        output_dir = self._output_dir.text()
        if not output_dir:
            output_dir = str(song_path.parent / "voiceconv_output")

        fmt_str = self._format_combo.currentText()
        job = ConversionJob(
            song_path=song_path,
            reference_path=ref_path,
            output_dir=Path(output_dir),
            output_format=AudioFormat(fmt_str),
            provenance=ProvenanceRecord(user_consent=True),
        )

        pipeline = self._build_pipeline()

        self._set_running(True)
        self._log_text.clear()
        self._log("Starting conversion...")

        self._worker = ConversionWorker(pipeline, job, parent=self)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_success)
        self._worker.finished_error.connect(self._on_error)
        self._worker.start()

    def _cancel_conversion(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self._set_running(False)
            self._log("Conversion cancelled.")
            self._status_label.setText("Cancelled")

    def _set_running(self, running: bool):
        self._btn_convert.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_song.setEnabled(not running)
        self._btn_ref.setEnabled(not running)
        if not running:
            self._progress_bar.setValue(0)

    def _log(self, msg: str):
        self._log_text.append(msg)

    # ── Worker callbacks ─────────────────────────────────────────────

    def _on_progress(self, status: str, pct: float, msg: str):
        self._progress_bar.setValue(int(pct))
        self._status_label.setText(f"[{status}] {msg}")
        if msg:
            self._log(f"[{pct:.0f}%] {msg}")

    def _on_success(self, output_path: str):
        self._set_running(False)
        self._status_label.setText("Done!")
        self._log(f"Output saved to: {output_path}")
        QMessageBox.information(
            self,
            "Conversion Complete",
            f"Output saved to:\n{output_path}\n\n"
            "Check the output directory for all stems and provenance record.",
        )

    def _on_error(self, error_msg: str):
        self._set_running(False)
        self._status_label.setText("Failed")
        self._log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Conversion Failed", error_msg)
