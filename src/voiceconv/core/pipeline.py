"""Main conversion pipeline orchestrating all modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from voiceconv.compliance import ComplianceManager
from voiceconv.core.audio_io import load_audio
from voiceconv.core.preprocessing import (
    normalize_loudness,
    trim_silence,
    validate_reference,
    validate_song,
)
from voiceconv.core.types import (
    AudioClip,
    AudioFormat,
    ConversionJob,
    ConversionResult,
    JobStatus,
    JudgementResult,
    SeparationResult,
)
from voiceconv.export import Exporter
from voiceconv.judge import QualityJudge
from voiceconv.separation.base import SeparatorBackend
from voiceconv.svc.base import SVCBackend

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[JobStatus, float, str], None]


class ConversionPipeline:
    """Orchestrates the full voice conversion pipeline.

    Pipeline stages:
    1. Load & validate inputs
    2. Separate vocals from song
    3. Convert vocal timbre using speech reference
    4. Judge output quality
    5. Mix converted vocal with instrumental
    6. Export final output with provenance
    """

    def __init__(
        self,
        separator: SeparatorBackend,
        svc_backend: SVCBackend,
        judge: QualityJudge | None = None,
        exporter: Exporter | None = None,
        compliance: ComplianceManager | None = None,
    ):
        self.separator = separator
        self.svc_backend = svc_backend
        self.judge = judge or QualityJudge()
        self.exporter = exporter or Exporter()
        self.compliance = compliance or ComplianceManager()

    def run(
        self,
        job: ConversionJob,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Run the full pipeline. Returns path to final output.

        Raises on failure with descriptive error.
        """

        def _progress(status: JobStatus, pct: float, msg: str = ""):
            job.status = status
            job.progress = pct
            if on_progress:
                on_progress(status, pct, msg)
            if msg:
                logger.info("[%.0f%%] %s", pct, msg)

        try:
            # --- Stage 0: Compliance check ---
            self.compliance.require_consent(job)
            provenance = self.compliance.initialize_provenance(job)
            self.compliance.record_backends(
                provenance, self.separator.name, self.svc_backend.name
            )

            # --- Stage 1: Load & preprocess ---
            _progress(JobStatus.SEPARATING, 5, "Loading song...")
            song = load_audio(job.song_path)
            validate_song(song)

            _progress(JobStatus.SEPARATING, 10, "Loading reference speech...")
            reference = load_audio(job.reference_path)
            validate_reference(reference)
            reference = trim_silence(reference)
            reference = normalize_loudness(reference)

            # --- Stage 2: Vocal separation ---
            _progress(JobStatus.SEPARATING, 15, "Separating vocals...")
            sep_result: SeparationResult = self.separator.separate(song)
            _progress(JobStatus.SEPARATING, 45, "Separation complete.")

            # --- Stage 3: Voice conversion ---
            _progress(JobStatus.CONVERTING, 50, "Converting voice timbre...")
            conv_result: ConversionResult = self.svc_backend.convert(
                sep_result.vocals, reference
            )
            _progress(JobStatus.CONVERTING, 75, "Conversion complete.")

            # --- Stage 4: Quality judgement ---
            _progress(JobStatus.JUDGING, 78, "Assessing quality...")
            judgement: JudgementResult = self.judge.judge(
                sep_result.vocals, reference, conv_result.converted_vocal
            )
            self.compliance.record_judgement(provenance, judgement)
            _progress(JobStatus.JUDGING, 85, "Quality assessment done.")

            # --- Stage 5: Mix & export ---
            _progress(JobStatus.EXPORTING, 87, "Mixing tracks...")
            mixed = self.exporter.mix_tracks(
                conv_result.converted_vocal, sep_result.instrumental
            )

            _progress(JobStatus.EXPORTING, 90, "Exporting files...")
            paths = self.exporter.export_stems(
                vocal=sep_result.vocals,
                instrumental=sep_result.instrumental,
                converted_vocal=conv_result.converted_vocal,
                mixed=mixed,
                output_dir=job.output_dir,
                fmt=job.output_format,
            )

            # Save provenance
            self.compliance.save_provenance(provenance, job.output_dir)
            job.provenance = provenance

            _progress(JobStatus.DONE, 100, "Done!")
            return paths["final_mix"]

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error("Pipeline failed: %s", e, exc_info=True)
            raise
