"""Tests for quality judge module."""

import numpy as np
import pytest

from voiceconv.core.types import AudioClip
from voiceconv.judge.quality import QualityJudge


class TestQualityJudge:
    @pytest.fixture
    def judge(self):
        return QualityJudge()

    def test_judge_returns_result(self, judge, mono_sine, speech_like):
        result = judge.judge(mono_sine, speech_like, mono_sine)
        assert 0 <= result.speaker_similarity <= 1.0
        assert 0 <= result.f0_correlation <= 1.0
        assert isinstance(result.signal_to_noise, float)
        assert isinstance(result.overall_pass, bool)

    def test_identical_signals_high_f0_correlation(self, judge, speech_like):
        """Same signal as source and converted should have high F0 correlation."""
        result = judge.judge(speech_like, speech_like, speech_like)
        assert result.f0_correlation > 0.8

    def test_spectral_embedding_fallback(self, judge, mono_sine):
        """Spectral fallback should return valid embedding."""
        emb = judge._spectral_embedding(mono_sine)
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 0.01

    def test_snr_positive_for_clean_signal(self, judge, mono_sine):
        snr = judge._compute_snr(mono_sine)
        assert snr > 0
