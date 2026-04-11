"""Audio import/export with format detection and resampling."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from .types import AudioClip, AudioFormat

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}
DEFAULT_SR = 44100


def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> AudioClip:
    """Load an audio file, convert to mono float32, resample if needed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # soundfile handles wav/flac/ogg natively
    # For mp3/m4a, try soundfile first then fall back to pydub
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        data, sr = _load_with_pydub(path)

    # Convert to mono by averaging channels
    if data.ndim == 2 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    elif data.ndim == 2:
        data = data[:, 0]

    # Resample if needed
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr

    # Normalize to [-1, 1]
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak

    data = data.astype(np.float32)
    logger.info("Loaded %s: %.1fs @ %dHz", path.name, len(data) / sr, sr)
    return AudioClip(samples=data, sample_rate=sr, source_path=path, name=path.stem)


def _load_with_pydub(path: Path) -> tuple[np.ndarray, int]:
    """Fallback loader using pydub for formats soundfile can't handle."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(str(path))
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels)
    else:
        samples = samples.reshape(-1, 1)
    # pydub returns int samples, normalize
    max_val = float(2 ** (seg.sample_width * 8 - 1))
    samples = samples / max_val
    return samples, sr


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy."""
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(data, up, down).astype(np.float32)


def save_audio(
    clip: AudioClip,
    path: Path,
    fmt: AudioFormat = AudioFormat.WAV,
) -> Path:
    """Save an AudioClip to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct extension
    if path.suffix.lower() != f".{fmt.value}":
        path = path.with_suffix(f".{fmt.value}")

    data = np.clip(clip.samples, -1.0, 1.0)

    if fmt in (AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OGG):
        subtype = "PCM_16" if fmt == AudioFormat.WAV else None
        sf.write(str(path), data, clip.sample_rate, subtype=subtype)
    elif fmt == AudioFormat.MP3:
        _save_with_pydub(data, clip.sample_rate, path, "mp3")
    else:
        sf.write(str(path), data, clip.sample_rate)

    logger.info("Saved %s (%.1fs)", path.name, clip.duration_seconds)
    return path


def _save_with_pydub(
    data: np.ndarray, sr: int, path: Path, fmt: str
) -> None:
    """Save using pydub for mp3 etc."""
    from pydub import AudioSegment

    int_data = (data * 32767).astype(np.int16)
    seg = AudioSegment(
        int_data.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    seg.export(str(path), format=fmt)


def audio_hash(path: Path) -> str:
    """SHA-256 hash of audio file for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
