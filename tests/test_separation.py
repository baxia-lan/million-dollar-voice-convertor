"""Tests for separation module."""

import pytest

from voiceconv.separation.base import SeparatorBackend
from voiceconv.separation.demucs_adapter import DemucsSeparator


class TestSeparatorInterface:
    def test_demucs_is_separator(self):
        sep = DemucsSeparator()
        assert isinstance(sep, SeparatorBackend)

    def test_demucs_name(self):
        sep = DemucsSeparator()
        assert "demucs" in sep.name

    def test_demucs_availability_check(self):
        sep = DemucsSeparator()
        result = sep.is_available()
        assert isinstance(result, bool)

    @pytest.mark.slow
    def test_demucs_separation(self, song_like):
        sep = DemucsSeparator()
        if not sep.is_available():
            pytest.skip("demucs not installed")
        result = sep.separate(song_like)
        assert result.vocals.n_samples > 0
        assert result.instrumental.n_samples > 0
        assert result.vocals.sample_rate == song_like.sample_rate
