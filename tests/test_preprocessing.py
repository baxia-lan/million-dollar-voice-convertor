"""Tests for audio preprocessing."""

import numpy as np
import pytest

from voiceconv.core.preprocessing import (
    normalize_loudness,
    trim_silence,
    validate_reference,
    validate_song,
)
from voiceconv.core.types import AudioClip


class TestValidation:
    def test_valid_song(self, song_like):
        validate_song(song_like)  # Should not raise

    def test_short_song_raises(self, sample_rate):
        short = AudioClip(
            samples=np.zeros(100, dtype=np.float32),
            sample_rate=sample_rate,
            name="short",
        )
        with pytest.raises(ValueError, match="too short"):
            validate_song(short)

    def test_valid_reference(self, speech_like):
        validate_reference(speech_like)  # Should not raise

    def test_short_reference_raises(self, sample_rate):
        short = AudioClip(
            samples=np.zeros(100, dtype=np.float32),
            sample_rate=sample_rate,
            name="short",
        )
        with pytest.raises(ValueError, match="too short"):
            validate_reference(short)


class TestTrimSilence:
    def test_trims_leading_silence(self, sample_rate):
        silence = np.zeros(sample_rate, dtype=np.float32)
        tone = 0.5 * np.sin(
            2 * np.pi * 440 * np.arange(sample_rate) / sample_rate
        ).astype(np.float32)
        clip = AudioClip(
            samples=np.concatenate([silence, tone]),
            sample_rate=sample_rate,
            name="padded",
        )
        trimmed = trim_silence(clip)
        assert trimmed.n_samples < clip.n_samples

    def test_no_trim_on_active_signal(self, mono_sine):
        trimmed = trim_silence(mono_sine)
        # Should be same or very close in length
        assert trimmed.n_samples >= mono_sine.n_samples * 0.9


class TestNormalizeLoudness:
    def test_normalizes_peak(self, sample_rate):
        quiet = AudioClip(
            samples=np.full(1000, 0.01, dtype=np.float32),
            sample_rate=sample_rate,
            name="quiet",
        )
        normalized = normalize_loudness(quiet, target_db=-3.0)
        expected_peak = 10 ** (-3.0 / 20.0)
        assert abs(np.abs(normalized.samples).max() - expected_peak) < 0.01

    def test_silent_audio_unchanged(self, sample_rate):
        silent = AudioClip(
            samples=np.zeros(1000, dtype=np.float32),
            sample_rate=sample_rate,
            name="silent",
        )
        result = normalize_loudness(silent)
        assert np.allclose(result.samples, 0.0)
