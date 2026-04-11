"""Tests for compliance and provenance module."""

import json
from pathlib import Path

import pytest

from voiceconv.compliance.provenance import ComplianceManager
from voiceconv.core.types import (
    AudioFormat,
    ConversionJob,
    JudgementResult,
    ProvenanceRecord,
)


class TestComplianceManager:
    @pytest.fixture
    def manager(self):
        return ComplianceManager()

    def test_require_consent_raises_without(self, manager, tmp_dir, wav_file, speech_wav):
        job = ConversionJob(
            song_path=wav_file,
            reference_path=speech_wav,
            output_dir=tmp_dir,
            provenance=ProvenanceRecord(user_consent=False),
        )
        with pytest.raises(ValueError, match="consent"):
            manager.require_consent(job)

    def test_require_consent_passes_with(self, manager, tmp_dir, wav_file, speech_wav):
        job = ConversionJob(
            song_path=wav_file,
            reference_path=speech_wav,
            output_dir=tmp_dir,
            provenance=ProvenanceRecord(user_consent=True),
        )
        manager.require_consent(job)  # Should not raise

    def test_initialize_provenance(self, manager, tmp_dir, wav_file, speech_wav):
        job = ConversionJob(
            song_path=wav_file,
            reference_path=speech_wav,
            output_dir=tmp_dir,
            provenance=ProvenanceRecord(user_consent=True),
        )
        record = manager.initialize_provenance(job)
        assert record.source_song_hash != ""
        assert record.reference_speech_hash != ""
        assert record.preserve_f0 is True

    def test_save_provenance(self, manager, tmp_dir):
        record = ProvenanceRecord(
            source_song_hash="abc123",
            reference_speech_hash="def456",
            separation_backend="test-sep",
            svc_backend="test-svc",
            user_consent=True,
        )
        path = manager.save_provenance(record, tmp_dir)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["source_song_hash"] == "abc123"
        assert data["svc_backend"] == "test-svc"

    def test_validate_record_complete(self, manager):
        record = ProvenanceRecord(
            source_song_hash="abc",
            reference_speech_hash="def",
            separation_backend="sep",
            svc_backend="svc",
            user_consent=True,
        )
        issues = manager.validate_record(record)
        assert len(issues) == 0

    def test_validate_record_missing_fields(self, manager):
        record = ProvenanceRecord()
        issues = manager.validate_record(record)
        assert len(issues) > 0

    def test_record_judgement(self, manager):
        record = ProvenanceRecord()
        judgement = JudgementResult(
            speaker_similarity=0.8,
            f0_correlation=0.95,
            signal_to_noise=15.0,
            overall_pass=True,
        )
        manager.record_judgement(record, judgement)
        assert record.judgement is judgement
        d = record.to_dict()
        assert d["judgement"]["speaker_similarity"] == 0.8
