"""Audio preprocessing: trim silence, normalize, validate."""

from __future__ import annotations

import logging

import numpy as np

from .types import AudioClip

logger = logging.getLogger(__name__)

MIN_DURATION = 1.0  # seconds
MAX_DURATION_SONG = 600.0  # 10 minutes
MAX_DURATION_REF = 120.0  # 2 minutes


def validate_song(clip: AudioClip) -> None:
    """Validate a song input."""
    if clip.duration_seconds < MIN_DURATION:
        raise ValueError(
            f"Song too short ({clip.duration_seconds:.1f}s). "
            f"Minimum {MIN_DURATION}s required."
        )
    if clip.duration_seconds > MAX_DURATION_SONG:
        raise ValueError(
            f"Song too long ({clip.duration_seconds:.1f}s). "
            f"Maximum {MAX_DURATION_SONG:.0f}s supported."
        )


def validate_reference(clip: AudioClip) -> None:
    """Validate a speech reference input."""
    if clip.duration_seconds < MIN_DURATION:
        raise ValueError(
            f"Reference too short ({clip.duration_seconds:.1f}s). "
            f"Minimum {MIN_DURATION}s required."
        )
    if clip.duration_seconds > MAX_DURATION_REF:
        raise ValueError(
            f"Reference too long ({clip.duration_seconds:.1f}s). "
            f"Maximum {MAX_DURATION_REF:.0f}s supported."
        )


def trim_silence(
    clip: AudioClip,
    threshold_db: float = -40.0,
    min_silence_ms: int = 200,
) -> AudioClip:
    """Trim leading and trailing silence from audio."""
    threshold = 10 ** (threshold_db / 20.0)
    frame_len = int(clip.sample_rate * min_silence_ms / 1000)

    abs_samples = np.abs(clip.samples)

    # Find first frame above threshold
    start = 0
    for i in range(0, len(abs_samples) - frame_len, frame_len):
        if np.max(abs_samples[i : i + frame_len]) > threshold:
            start = max(0, i - frame_len)
            break

    # Find last frame above threshold
    end = len(abs_samples)
    for i in range(len(abs_samples) - frame_len, 0, -frame_len):
        if np.max(abs_samples[i : i + frame_len]) > threshold:
            end = min(len(abs_samples), i + 2 * frame_len)
            break

    trimmed = clip.samples[start:end]
    if len(trimmed) < int(clip.sample_rate * 0.5):
        logger.warning("Trim resulted in very short audio, keeping original")
        return clip

    logger.info(
        "Trimmed silence: %.1fs -> %.1fs",
        clip.duration_seconds,
        len(trimmed) / clip.sample_rate,
    )
    return AudioClip(
        samples=trimmed,
        sample_rate=clip.sample_rate,
        source_path=clip.source_path,
        name=clip.name,
    )


def normalize_loudness(clip: AudioClip, target_db: float = -3.0) -> AudioClip:
    """Normalize audio loudness to target peak dB."""
    peak = np.abs(clip.samples).max()
    if peak < 1e-8:
        return clip

    target_peak = 10 ** (target_db / 20.0)
    gain = target_peak / peak
    normalized = (clip.samples * gain).astype(np.float32)

    return AudioClip(
        samples=normalized,
        sample_rate=clip.sample_rate,
        source_path=clip.source_path,
        name=clip.name,
    )
