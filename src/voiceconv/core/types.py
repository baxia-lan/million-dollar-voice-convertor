"""Shared data types used across the application."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class AudioFormat(Enum):
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    OGG = "ogg"


class JobStatus(Enum):
    PENDING = "pending"
    SEPARATING = "separating"
    CONVERTING = "converting"
    JUDGING = "judging"
    EXPORTING = "exporting"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AudioClip:
    """In-memory audio with metadata."""

    samples: np.ndarray  # shape (n_samples,) mono float32 [-1, 1]
    sample_rate: int
    source_path: Path | None = None
    name: str = ""

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def n_samples(self) -> int:
        return len(self.samples)


@dataclass
class SeparationResult:
    """Output of vocal/instrumental separation."""

    vocals: AudioClip
    instrumental: AudioClip


@dataclass
class ConversionResult:
    """Output of SVC backend."""

    converted_vocal: AudioClip
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgementResult:
    """Quality assessment scores."""

    speaker_similarity: float  # 0-1, cosine similarity of embeddings
    f0_correlation: float  # 0-1, pitch preservation
    signal_to_noise: float  # dB
    overall_pass: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceRecord:
    """Tracks processing lineage for compliance."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_song_hash: str = ""
    reference_speech_hash: str = ""
    separation_backend: str = ""
    svc_backend: str = ""
    preserve_f0: bool = True
    preserve_duration: bool = True
    preserve_lyrics: bool = True
    preserve_expression: bool = True
    judgement: JudgementResult | None = None
    user_consent: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "source_song_hash": self.source_song_hash,
            "reference_speech_hash": self.reference_speech_hash,
            "separation_backend": self.separation_backend,
            "svc_backend": self.svc_backend,
            "preserve_f0": self.preserve_f0,
            "preserve_duration": self.preserve_duration,
            "preserve_lyrics": self.preserve_lyrics,
            "preserve_expression": self.preserve_expression,
            "user_consent": self.user_consent,
            "notes": self.notes,
        }
        if self.judgement:
            d["judgement"] = {
                "speaker_similarity": self.judgement.speaker_similarity,
                "f0_correlation": self.judgement.f0_correlation,
                "signal_to_noise": self.judgement.signal_to_noise,
                "overall_pass": self.judgement.overall_pass,
            }
        return d


@dataclass
class ConversionJob:
    """Full pipeline job."""

    song_path: Path
    reference_path: Path
    output_dir: Path
    output_format: AudioFormat = AudioFormat.WAV
    status: JobStatus = JobStatus.PENDING
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    progress: float = 0.0  # 0-100
    error: str = ""
