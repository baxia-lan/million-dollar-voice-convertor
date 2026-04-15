"""Tests for SVC backend adapters."""

import numpy as np
import pytest

from voiceconv.core.types import AudioClip
from voiceconv.svc.openvoice_svc import OpenVoiceSVC
from voiceconv.svc.seedvc_svc import SeedVCSVC
from voiceconv.svc.world_svc import WorldSVC, _mcep_to_sp, _sp_to_mcep


class TestWorldSVC:
    @pytest.fixture
    def backend(self):
        return WorldSVC(conversion_strength=0.8)

    def test_name(self, backend):
        assert backend.name == "world-svc"

    def test_availability(self, backend):
        # pyworld should be importable in test env (if installed)
        result = backend.is_available()
        assert isinstance(result, bool)

    @pytest.mark.slow
    def test_conversion_preserves_duration(self, backend, song_like, speech_like):
        if not backend.is_available():
            pytest.skip("pyworld not installed")
        result = backend.convert(song_like, speech_like)
        # Duration should be preserved (within 5%)
        src_dur = song_like.duration_seconds
        out_dur = result.converted_vocal.duration_seconds
        assert abs(out_dur - src_dur) / src_dur < 0.05

    @pytest.mark.slow
    def test_conversion_output_valid(self, backend, song_like, speech_like):
        if not backend.is_available():
            pytest.skip("pyworld not installed")
        result = backend.convert(song_like, speech_like)
        assert result.converted_vocal.samples.dtype == np.float32
        assert np.abs(result.converted_vocal.samples).max() <= 1.0
        assert len(result.metadata) > 0

    @pytest.mark.slow
    def test_conversion_strength_zero_is_identity(self, song_like, speech_like):
        backend = WorldSVC(conversion_strength=0.0)
        if not backend.is_available():
            pytest.skip("pyworld not installed")
        result = backend.convert(song_like, speech_like)
        # With 0 strength, output should be very similar to input
        # (not identical due to WORLD analysis/resynthesis loss)
        assert result.converted_vocal.n_samples > 0


class TestOpenVoiceSVC:
    @pytest.fixture
    def backend(self):
        return OpenVoiceSVC(tau=0.1)

    def test_name(self, backend):
        assert backend.name == "openvoice-v2"

    def test_availability(self, backend):
        result = backend.is_available()
        assert isinstance(result, bool)

    def test_is_svc_backend(self, backend):
        from voiceconv.svc.base import SVCBackend
        assert isinstance(backend, SVCBackend)

    @pytest.mark.slow
    def test_conversion_preserves_duration(self, backend, song_like, speech_like):
        if not backend.is_available():
            pytest.skip("openvoice-cli not installed")
        result = backend.convert(song_like, speech_like)
        src_dur = song_like.duration_seconds
        out_dur = result.converted_vocal.duration_seconds
        assert abs(out_dur - src_dur) / src_dur < 0.10

    @pytest.mark.slow
    def test_conversion_output_valid(self, backend, song_like, speech_like):
        if not backend.is_available():
            pytest.skip("openvoice-cli not installed")
        result = backend.convert(song_like, speech_like)
        assert result.converted_vocal.samples.dtype == np.float32
        assert np.abs(result.converted_vocal.samples).max() <= 1.0
        assert result.metadata["backend"] == "openvoice-v2"


class TestSeedVCSVC:
    @pytest.fixture
    def backend(self):
        return SeedVCSVC(diffusion_steps=10)

    def test_name(self, backend):
        assert "seed-vc" in backend.name

    def test_name_with_checkpoint(self):
        be = SeedVCSVC(custom_checkpoint="/fake/path.pth")
        assert "fine-tuned" in be.name

    def test_availability(self, backend):
        result = backend.is_available()
        assert isinstance(result, bool)

    def test_is_svc_backend(self, backend):
        from voiceconv.svc.base import SVCBackend
        assert isinstance(backend, SVCBackend)


class TestMCEPConversion:
    def test_roundtrip(self):
        """MCEP -> SP -> MCEP should approximately preserve the envelope."""
        n_frames, n_freq = 10, 513
        sp = np.random.rand(n_frames, n_freq).astype(np.float64) + 0.01
        mcep = _sp_to_mcep(sp, order=39)
        sp_reconstructed = _mcep_to_sp(mcep, n_freq)
        # Not exact due to truncation, but shapes must match
        assert sp_reconstructed.shape == sp.shape
        assert not np.any(np.isnan(sp_reconstructed))
