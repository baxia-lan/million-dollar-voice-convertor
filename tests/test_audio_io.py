"""Tests for audio I/O module."""

from pathlib import Path

import numpy as np
import pytest

from voiceconv.core.audio_io import audio_hash, load_audio, save_audio
from voiceconv.core.types import AudioClip, AudioFormat


class TestLoadAudio:
    def test_load_wav(self, wav_file):
        clip = load_audio(wav_file)
        assert isinstance(clip, AudioClip)
        assert clip.sample_rate > 0
        assert clip.n_samples > 0
        assert clip.samples.dtype == np.float32

    def test_load_nonexistent_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_dir / "no_such_file.wav")

    def test_load_unsupported_format_raises(self, tmp_dir):
        bad = tmp_dir / "test.xyz"
        bad.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_audio(bad)

    def test_load_resamples(self, wav_file):
        clip = load_audio(wav_file, target_sr=16000)
        assert clip.sample_rate == 16000

    def test_load_normalizes_to_unit_range(self, wav_file):
        clip = load_audio(wav_file)
        assert np.abs(clip.samples).max() <= 1.0 + 1e-6


class TestSaveAudio:
    def test_save_wav(self, mono_sine, tmp_dir):
        path = save_audio(mono_sine, tmp_dir / "out.wav", AudioFormat.WAV)
        assert path.exists()
        assert path.suffix == ".wav"

    def test_save_flac(self, mono_sine, tmp_dir):
        path = save_audio(mono_sine, tmp_dir / "out.flac", AudioFormat.FLAC)
        assert path.exists()

    def test_roundtrip(self, mono_sine, tmp_dir):
        path = save_audio(mono_sine, tmp_dir / "rt.wav", AudioFormat.WAV)
        loaded = load_audio(path, target_sr=mono_sine.sample_rate)
        # Allow some loss from PCM16 quantization
        assert abs(loaded.duration_seconds - mono_sine.duration_seconds) < 0.1


class TestAudioHash:
    def test_hash_deterministic(self, wav_file):
        h1 = audio_hash(wav_file)
        h2 = audio_hash(wav_file)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_files_different_hash(self, wav_file, speech_wav):
        assert audio_hash(wav_file) != audio_hash(speech_wav)
