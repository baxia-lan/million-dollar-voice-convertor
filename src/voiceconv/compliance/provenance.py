"""Compliance and provenance tracking.

Every conversion job produces a provenance record documenting:
- Input file hashes (non-reversible)
- Backends used
- Preservation guarantees
- User consent acknowledgement
- Quality judgement results
- Timestamps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from voiceconv.core.audio_io import audio_hash
from voiceconv.core.types import ConversionJob, JudgementResult, ProvenanceRecord

logger = logging.getLogger(__name__)


class ComplianceManager:
    """Manage consent, provenance, and compliance records."""

    def __init__(self, provenance_dir: Path | None = None):
        self._provenance_dir = provenance_dir

    def require_consent(self, job: ConversionJob) -> None:
        """Verify user has acknowledged consent for voice conversion.

        Raises ValueError if consent not given.
        """
        if not job.provenance.user_consent:
            raise ValueError(
                "User consent required before processing. "
                "Please acknowledge that you have the right to use "
                "the provided audio materials and consent to voice conversion."
            )

    def initialize_provenance(self, job: ConversionJob) -> ProvenanceRecord:
        """Create provenance record with input hashes."""
        record = job.provenance
        record.source_song_hash = audio_hash(job.song_path)
        record.reference_speech_hash = audio_hash(job.reference_path)
        record.preserve_f0 = True
        record.preserve_duration = True
        record.preserve_lyrics = True
        record.preserve_expression = True
        logger.info("Provenance initialized: job_id=%s", record.job_id)
        return record

    def record_backends(
        self,
        record: ProvenanceRecord,
        separation_backend: str,
        svc_backend: str,
    ) -> None:
        """Record which backends were used."""
        record.separation_backend = separation_backend
        record.svc_backend = svc_backend

    def record_judgement(
        self,
        record: ProvenanceRecord,
        judgement: JudgementResult,
    ) -> None:
        """Attach quality judgement to provenance."""
        record.judgement = judgement

    def save_provenance(
        self, record: ProvenanceRecord, output_dir: Path
    ) -> Path:
        """Save provenance record as JSON alongside output."""
        save_dir = self._provenance_dir or output_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        path = save_dir / f"provenance_{record.job_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Provenance saved: %s", path)
        return path

    def validate_record(self, record: ProvenanceRecord) -> list[str]:
        """Validate a provenance record, return list of issues."""
        issues = []
        if not record.source_song_hash:
            issues.append("Missing source song hash")
        if not record.reference_speech_hash:
            issues.append("Missing reference speech hash")
        if not record.separation_backend:
            issues.append("Missing separation backend info")
        if not record.svc_backend:
            issues.append("Missing SVC backend info")
        if not record.user_consent:
            issues.append("User consent not recorded")
        if not record.preserve_f0:
            issues.append("F0 preservation not guaranteed")
        return issues
