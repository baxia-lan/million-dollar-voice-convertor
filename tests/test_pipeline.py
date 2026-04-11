"""Tests for the conversion pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from voiceconv.compliance import ComplianceManager
from voiceconv.core.pipeline import ConversionPipeline
from voiceconv.core.types import (
    AudioClip,
    AudioFormat,
    ConversionJob,
    ConversionResult,
    JobStatus,
    JudgementResult,
    ProvenanceRecord,
    SeparationResult,
)
from voiceconv.export import Exporter
from voiceconv.judge import QualityJudge
from voiceconv.separation.base import SeparatorBackend
from voiceconv.svc.base import SVCBackend


class MockSeparator(SeparatorBackend):
    @property
    def name(self):
        return "mock-separator"

    def is_available(self):
        return True

    def separate(self, mix):
        half = len(mix.samples) // 2
        return SeparationResult(
            vocals=AudioClip(
                samples=mix.samples[:half].copy(),
                sample_rate=mix.sample_rate,
                name="vocals",
            ),
            instrumental=AudioClip(
                samples=mix.samples[half:].copy(),
                sample_rate=mix.sample_rate,
                name="instrumental",
            ),
        )


class MockSVC(SVCBackend):
    @property
    def name(self):
        return "mock-svc"

    def is_available(self):
        return True

    def convert(self, source_vocal, reference_speech):
        # Return source with slight modification (simulating timbre change)
        converted = source_vocal.samples * 0.9
        return ConversionResult(
            converted_vocal=AudioClip(
                samples=converted.astype(np.float32),
                sample_rate=source_vocal.sample_rate,
                name="converted",
            ),
            metadata={"backend": "mock"},
        )


class TestConversionPipeline:
    @pytest.fixture
    def pipeline(self):
        return ConversionPipeline(
            separator=MockSeparator(),
            svc_backend=MockSVC(),
            judge=QualityJudge(),
            exporter=Exporter(),
            compliance=ComplianceManager(),
        )

    def test_full_pipeline(self, pipeline, song_wav, speech_wav, tmp_dir):
        job = ConversionJob(
            song_path=song_wav,
            reference_path=speech_wav,
            output_dir=tmp_dir / "output",
            provenance=ProvenanceRecord(user_consent=True),
        )
        output = pipeline.run(job)
        assert Path(output).exists()
        assert job.status == JobStatus.DONE
        assert job.progress == 100.0

    def test_pipeline_without_consent_fails(self, pipeline, song_wav, speech_wav, tmp_dir):
        job = ConversionJob(
            song_path=song_wav,
            reference_path=speech_wav,
            output_dir=tmp_dir / "output",
            provenance=ProvenanceRecord(user_consent=False),
        )
        with pytest.raises(ValueError, match="consent"):
            pipeline.run(job)

    def test_pipeline_provenance_created(self, pipeline, song_wav, speech_wav, tmp_dir):
        job = ConversionJob(
            song_path=song_wav,
            reference_path=speech_wav,
            output_dir=tmp_dir / "output",
            provenance=ProvenanceRecord(user_consent=True),
        )
        pipeline.run(job)
        provenance_files = list((tmp_dir / "output").glob("provenance_*.json"))
        assert len(provenance_files) == 1

    def test_pipeline_progress_callback(self, pipeline, song_wav, speech_wav, tmp_dir):
        job = ConversionJob(
            song_path=song_wav,
            reference_path=speech_wav,
            output_dir=tmp_dir / "output",
            provenance=ProvenanceRecord(user_consent=True),
        )
        progress_log = []
        pipeline.run(job, on_progress=lambda s, p, m: progress_log.append((s, p, m)))
        assert len(progress_log) > 0
        # Last progress should be 100%
        assert progress_log[-1][1] == 100.0
