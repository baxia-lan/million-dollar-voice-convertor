"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from voiceconv.core.types import AudioClip


@pytest.fixture
def sample_rate():
    return 22050


@pytest.fixture
def mono_sine(sample_rate) -> AudioClip:
    """A 2-second 440Hz sine wave."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioClip(samples=samples, sample_rate=sample_rate, name="sine_440")


@pytest.fixture
def speech_like(sample_rate) -> AudioClip:
    """A 3-second signal mimicking speech (multi-frequency + envelope)."""
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fundamental + harmonics with amplitude envelope
    envelope = np.clip(np.sin(2 * np.pi * 3 * t) * 0.5 + 0.5, 0.1, 1.0)
    signal = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
        + 0.05 * np.sin(2 * np.pi * 600 * t)
    )
    samples = (signal * envelope).astype(np.float32)
    return AudioClip(samples=samples, sample_rate=sample_rate, name="speech_like")


@pytest.fixture
def song_like(sample_rate) -> AudioClip:
    """A 4-second signal mimicking a song (melody + accompaniment)."""
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Melody: pitch-varying vocal-like signal
    f0 = 220 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Vibrato
    phase = np.cumsum(2 * np.pi * f0 / sample_rate)
    vocal = 0.3 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    # Accompaniment
    accomp = 0.1 * np.sin(2 * np.pi * 110 * t) + 0.05 * np.sin(2 * np.pi * 165 * t)
    samples = (vocal + accomp).astype(np.float32)
    return AudioClip(samples=samples, sample_rate=sample_rate, name="song_like")


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def wav_file(mono_sine, tmp_dir) -> Path:
    """Write a sine wave to a .wav file and return its path."""
    import soundfile as sf

    path = tmp_dir / "test_sine.wav"
    sf.write(str(path), mono_sine.samples, mono_sine.sample_rate)
    return path


@pytest.fixture
def speech_wav(speech_like, tmp_dir) -> Path:
    """Write speech-like audio to a .wav file."""
    import soundfile as sf

    path = tmp_dir / "test_speech.wav"
    sf.write(str(path), speech_like.samples, speech_like.sample_rate)
    return path


@pytest.fixture
def song_wav(song_like, tmp_dir) -> Path:
    """Write song-like audio to a .wav file."""
    import soundfile as sf

    path = tmp_dir / "test_song.wav"
    sf.write(str(path), song_like.samples, song_like.sample_rate)
    return path
