"""Main application window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
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
    QSpinBox,
    QStatusBar,
    QTabWidget,
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
from voiceconv.svc.openvoice_svc import OpenVoiceSVC
from voiceconv.svc.seedvc_svc import SeedVCSVC
from voiceconv.svc.world_svc import WorldSVC

from .worker import ConversionWorker

logger = logging.getLogger(__name__)

AUDIO_FILTER = "Audio Files (*.wav *.flac *.mp3 *.ogg *.m4a);;All Files (*)"


# ── Fine-tune worker thread ────────────────────────────────────────

class FineTuneWorker(QThread):
    progress_updated = Signal(int, int, str)
    finished_ok = Signal(str)  # checkpoint path
    finished_error = Signal(str)

    def __init__(self, recordings_dir, output_dir, steps, parent=None):
        super().__init__(parent)
        self._recordings_dir = recordings_dir
        self._output_dir = output_dir
        self._steps = steps

    def run(self):
        try:
            from voiceconv.finetune import VoiceFineTuner

            finetuner = VoiceFineTuner()
            ckpt = finetuner.run(
                recordings_dir=self._recordings_dir,
                output_dir=self._output_dir,
                steps=self._steps,
                on_progress=lambda cur, total, msg: self.progress_updated.emit(cur, total, msg),
            )
            self.finished_ok.emit(str(ckpt))
        except Exception as e:
            self.finished_error.emit(str(e))


# ── Main window ─────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """VoiceConv main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceConv — Singing Voice Conversion")
        self.setMinimumSize(750, 700)
        self._worker: ConversionWorker | None = None
        self._ft_worker: FineTuneWorker | None = None
        self._custom_checkpoint: str | None = None
        self._setup_ui()
        self._setup_backends()

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # Title
        title = QLabel("VoiceConv")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        subtitle = QLabel(
            "Few-shot singing voice conversion. "
            "Preserves pitch, timing, lyrics, and expression — only changes timbre."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Tabs ──
        tabs = QTabWidget()
        tabs.addTab(self._build_convert_tab(), "Convert")
        tabs.addTab(self._build_finetune_tab(), "Fine-Tune")
        layout.addWidget(tabs)

        # ── Log output ──
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

        self.setStatusBar(QStatusBar())

    def _build_convert_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Input section ──
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)

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

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Speech Reference:"))
        self._ref_path = QLineEdit()
        self._ref_path.setPlaceholderText("Path to your speech sample (any content)")
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

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("SVC Backend:"))
        self._backend_combo = QComboBox()
        backend_row.addWidget(self._backend_combo)
        settings_layout.addLayout(backend_row)

        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("Separator:"))
        self._sep_combo = QComboBox()
        sep_row.addWidget(self._sep_combo)
        settings_layout.addLayout(sep_row)

        # Custom checkpoint
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Custom Checkpoint:"))
        self._ckpt_path = QLineEdit()
        self._ckpt_path.setPlaceholderText("(optional) Fine-tuned .pth file")
        self._ckpt_path.setReadOnly(True)
        ckpt_row.addWidget(self._ckpt_path)
        self._btn_ckpt = QPushButton("Browse...")
        self._btn_ckpt.clicked.connect(self._browse_checkpoint)
        ckpt_row.addWidget(self._btn_ckpt)
        btn_clear_ckpt = QPushButton("Clear")
        btn_clear_ckpt.clicked.connect(lambda: self._ckpt_path.clear())
        ckpt_row.addWidget(btn_clear_ckpt)
        settings_layout.addLayout(ckpt_row)

        # Diffusion steps
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Diffusion Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(5, 100)
        self._steps_spin.setValue(50)
        steps_row.addWidget(self._steps_spin)
        settings_layout.addLayout(steps_row)

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

        return tab

    def _build_finetune_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc = QLabel(
            "Fine-tune the singing model on your voice recordings (15-30 min of clean audio). "
            "This creates a personalized checkpoint for higher-quality voice conversion."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Recordings directory
        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("Recordings Dir:"))
        self._ft_recordings_dir = QLineEdit()
        self._ft_recordings_dir.setPlaceholderText("Directory with your clean voice recordings")
        rec_row.addWidget(self._ft_recordings_dir)
        btn_rec = QPushButton("Browse...")
        btn_rec.clicked.connect(
            lambda: self._ft_recordings_dir.setText(
                QFileDialog.getExistingDirectory(self, "Select Recordings Directory") or self._ft_recordings_dir.text()
            )
        )
        rec_row.addWidget(btn_rec)
        layout.addLayout(rec_row)

        # Output directory
        ft_out_row = QHBoxLayout()
        ft_out_row.addWidget(QLabel("Output Dir:"))
        self._ft_output_dir = QLineEdit()
        self._ft_output_dir.setPlaceholderText("Where to save the fine-tuned model")
        ft_out_row.addWidget(self._ft_output_dir)
        btn_ft_out = QPushButton("Browse...")
        btn_ft_out.clicked.connect(
            lambda: self._ft_output_dir.setText(
                QFileDialog.getExistingDirectory(self, "Select Output Directory") or self._ft_output_dir.text()
            )
        )
        ft_out_row.addWidget(btn_ft_out)
        layout.addLayout(ft_out_row)

        # Training steps
        ft_steps_row = QHBoxLayout()
        ft_steps_row.addWidget(QLabel("Training Steps:"))
        self._ft_steps_spin = QSpinBox()
        self._ft_steps_spin.setRange(100, 5000)
        self._ft_steps_spin.setValue(1000)
        self._ft_steps_spin.setSingleStep(100)
        ft_steps_row.addWidget(self._ft_steps_spin)
        layout.addLayout(ft_steps_row)

        # Start button
        self._btn_finetune = QPushButton("Start Fine-Tuning")
        self._btn_finetune.setStyleSheet(
            "font-size: 14px; padding: 6px 20px; font-weight: bold;"
        )
        self._btn_finetune.clicked.connect(self._start_finetune)
        layout.addWidget(self._btn_finetune)

        # Progress
        self._ft_progress = QProgressBar()
        self._ft_progress.setRange(0, 100)
        self._ft_progress.setValue(0)
        layout.addWidget(self._ft_progress)

        self._ft_status = QLabel("Ready")
        layout.addWidget(self._ft_status)

        layout.addStretch()
        return tab

    def _setup_backends(self):
        """Detect and register available backends."""
        separators = [DemucsSeparator()]
        for sep in separators:
            available = sep.is_available()
            label = f"{sep.name} {'✓' if available else '(not installed)'}"
            self._sep_combo.addItem(label, userData=sep)

        backends = [SeedVCSVC(), OpenVoiceSVC(), WorldSVC()]
        for be in backends:
            available = be.is_available()
            label = f"{be.name} {'✓' if available else '(not installed)'}"
            self._backend_combo.addItem(label, userData=be)

    # ── File browsing ────────────────────────────────────────────────

    def _browse_song(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Song", "", AUDIO_FILTER)
        if path:
            self._song_path.setText(path)

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Speech Reference", "", AUDIO_FILTER)
        if path:
            self._ref_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Fine-Tuned Checkpoint", "", "PyTorch Checkpoint (*.pth);;All Files (*)"
        )
        if path:
            self._ckpt_path.setText(path)

    # ── Conversion logic ─────────────────────────────────────────────

    def _validate_inputs(self) -> bool:
        if not self._song_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a song file.")
            return False
        if not self._ref_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a speech reference file.")
            return False
        if not Path(self._song_path.text()).exists():
            QMessageBox.warning(self, "File Not Found", "Song file does not exist.")
            return False
        if not Path(self._ref_path.text()).exists():
            QMessageBox.warning(self, "File Not Found", "Reference file does not exist.")
            return False
        if not self._consent_check.isChecked():
            QMessageBox.warning(self, "Consent Required", "Please check the consent box to proceed.")
            return False
        return True

    def _build_pipeline(self) -> ConversionPipeline:
        separator: SeparatorBackend = self._sep_combo.currentData()
        svc_backend: SVCBackend = self._backend_combo.currentData()

        steps = self._steps_spin.value()
        custom_ckpt = self._ckpt_path.text() or None

        if isinstance(svc_backend, SeedVCSVC):
            svc_backend = SeedVCSVC(
                diffusion_steps=steps,
                auto_f0_adjust=False,
                custom_checkpoint=Path(custom_ckpt) if custom_ckpt else None,
            )
        elif isinstance(svc_backend, OpenVoiceSVC):
            svc_backend = OpenVoiceSVC(tau=0.1)
        elif isinstance(svc_backend, WorldSVC):
            svc_backend = WorldSVC(conversion_strength=0.8)

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

    # ── Fine-tuning ──────────────────────────────────────────────────

    def _start_finetune(self):
        rec_dir = self._ft_recordings_dir.text()
        out_dir = self._ft_output_dir.text()
        if not rec_dir or not Path(rec_dir).is_dir():
            QMessageBox.warning(self, "Missing Input", "Please select a recordings directory.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing Input", "Please select an output directory.")
            return

        steps = self._ft_steps_spin.value()
        self._btn_finetune.setEnabled(False)
        self._ft_status.setText("Fine-tuning...")
        self._ft_progress.setValue(0)
        self._log(f"Starting fine-tuning: {rec_dir} → {out_dir} ({steps} steps)")

        self._ft_worker = FineTuneWorker(rec_dir, out_dir, steps, parent=self)
        self._ft_worker.progress_updated.connect(self._on_ft_progress)
        self._ft_worker.finished_ok.connect(self._on_ft_success)
        self._ft_worker.finished_error.connect(self._on_ft_error)
        self._ft_worker.start()

    def _on_ft_progress(self, current: int, total: int, msg: str):
        if total > 0:
            pct = int(current / total * 100)
            self._ft_progress.setValue(pct)
        self._ft_status.setText(msg)
        self._log(f"[FT] {msg}")

    def _on_ft_success(self, ckpt_path: str):
        self._btn_finetune.setEnabled(True)
        self._ft_status.setText(f"Done! Checkpoint: {ckpt_path}")
        self._ft_progress.setValue(100)
        self._log(f"Fine-tuning complete: {ckpt_path}")

        # Auto-set the checkpoint in the Convert tab
        self._ckpt_path.setText(ckpt_path)

        QMessageBox.information(
            self,
            "Fine-Tuning Complete",
            f"Checkpoint saved to:\n{ckpt_path}\n\n"
            "It has been automatically selected in the Convert tab.",
        )

    def _on_ft_error(self, error_msg: str):
        self._btn_finetune.setEnabled(True)
        self._ft_status.setText("Failed")
        self._log(f"[FT] ERROR: {error_msg}")
        QMessageBox.critical(self, "Fine-Tuning Failed", error_msg)
