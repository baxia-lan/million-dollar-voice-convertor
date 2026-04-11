"""Tests for export module."""

from pathlib import Path

import numpy as np
import pytest

from voiceconv.core.types import AudioClip, AudioFormat
from voiceconv.export.exporter import Exporter


class TestExporter:
    @pytest.fixture
    def exporter(self):
        return Exporter()

    def test_mix_tracks_same_length(self, exporter, mono_sine, speech_like):
        mixed = exporter.mix_tracks(mono_sine, speech_like)
        assert mixed.n_samples == max(mono_sine.n_samples, speech_like.n_samples)
        assert mixed.samples.dtype == np.float32

    def test_mix_tracks_no_clipping(self, exporter, sample_rate):
        loud = AudioClip(
            samples=np.ones(1000, dtype=np.float32),
            sample_rate=sample_rate,
            name="loud",
        )
        mixed = exporter.mix_tracks(loud, loud)
        assert np.abs(mixed.samples).max() <= 1.0

    def test_mix_with_gain(self, exporter, mono_sine, speech_like):
        mixed = exporter.mix_tracks(
            mono_sine, speech_like, vocal_gain_db=-6.0, instrumental_gain_db=-3.0
        )
        assert mixed.n_samples > 0

    def test_export_creates_file(self, exporter, mono_sine, tmp_dir):
        path = exporter.export(mono_sine, tmp_dir, "test_output")
        assert path.exists()
        assert path.suffix == ".wav"

    def test_export_stems(self, exporter, mono_sine, speech_like, tmp_dir):
        mixed = exporter.mix_tracks(mono_sine, speech_like)
        paths = exporter.export_stems(
            vocal=mono_sine,
            instrumental=speech_like,
            converted_vocal=mono_sine,
            mixed=mixed,
            output_dir=tmp_dir,
        )
        assert len(paths) == 4
        for name, path in paths.items():
            assert Path(path).exists(), f"{name} not found"
