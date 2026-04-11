"""WORLD vocoder-based voice conversion backend.

This backend uses the WORLD vocoder to decompose audio into F0, spectral
envelope, and aperiodicity. Voice conversion is performed by transforming
the spectral envelope (mel-cepstral coefficients) to match the target
speaker's characteristics while preserving F0, duration, and expression.

Approach:
1. Analyze source vocal with WORLD -> F0, SP, AP
2. Analyze reference speech with WORLD -> F0_ref, SP_ref, AP_ref
3. Convert SP to mel-cepstral coefficients (MCEPs)
4. Global variance normalization: shift source MCEPs to target distribution
5. Apply frequency warping for vocal tract length normalization
6. Resynthesize with original F0 + transformed SP + original AP
"""

from __future__ import annotations

import logging

import numpy as np

from voiceconv.core.types import AudioClip, ConversionResult

from .base import SVCBackend

logger = logging.getLogger(__name__)

# WORLD analysis parameters
FRAME_PERIOD_MS = 5.0
MCEP_ORDER = 39  # 40 coefficients including c0


def _sp_to_mcep(sp: np.ndarray, order: int = MCEP_ORDER) -> np.ndarray:
    """Convert spectral envelope to mel-cepstral coefficients via DCT."""
    # Log spectral envelope
    log_sp = np.log(sp + 1e-16)
    # Apply DCT-II along frequency axis to get cepstral coefficients
    from scipy.fft import dct

    mcep = dct(log_sp, type=2, axis=1, norm="ortho")
    return mcep[:, : order + 1]


def _mcep_to_sp(mcep: np.ndarray, n_freq: int) -> np.ndarray:
    """Convert mel-cepstral coefficients back to spectral envelope."""
    from scipy.fft import idct

    # Pad to original frequency dimension
    padded = np.zeros((mcep.shape[0], n_freq), dtype=mcep.dtype)
    padded[:, : mcep.shape[1]] = mcep
    log_sp = idct(padded, type=2, axis=1, norm="ortho")
    return np.exp(log_sp)


def _frequency_warp(mcep: np.ndarray, alpha: float) -> np.ndarray:
    """Apply frequency warping to MCEPs for vocal tract length normalization.

    alpha > 0: shift formants up (shorter vocal tract)
    alpha < 0: shift formants down (longer vocal tract)
    """
    if abs(alpha) < 1e-6:
        return mcep.copy()

    warped = np.zeros_like(mcep)
    warped[:, 0] = mcep[:, 0]
    order = mcep.shape[1] - 1

    for i in range(order):
        warped[:, i + 1] = mcep[:, i + 1] + alpha * (
            mcep[:, min(i + 2, order)] if i + 2 <= order else 0
        )
    return warped


class WorldSVC(SVCBackend):
    """WORLD vocoder-based voice conversion.

    Performs timbre conversion via spectral envelope transformation
    while strictly preserving F0 (pitch), duration, and expression.
    Works on CPU, no GPU required. No model downloads needed.
    """

    def __init__(
        self,
        mcep_order: int = MCEP_ORDER,
        warp_alpha: float = 0.0,
        conversion_strength: float = 0.8,
    ):
        self._mcep_order = mcep_order
        self._warp_alpha = warp_alpha
        self._strength = np.clip(conversion_strength, 0.0, 1.0)

    @property
    def name(self) -> str:
        return "world-svc"

    def is_available(self) -> bool:
        try:
            import pyworld  # noqa: F401
            return True
        except ImportError:
            return False

    def convert(
        self,
        source_vocal: AudioClip,
        reference_speech: AudioClip,
    ) -> ConversionResult:
        import pyworld as pw

        sr = source_vocal.sample_rate
        source = source_vocal.samples.astype(np.float64)
        reference = reference_speech.samples.astype(np.float64)

        # --- Analyze source vocal ---
        logger.info("Analyzing source vocal with WORLD...")
        f0_src, t_src = pw.harvest(source, sr, frame_period=FRAME_PERIOD_MS)
        sp_src = pw.cheaptrick(source, f0_src, t_src, sr)
        ap_src = pw.d4c(source, f0_src, t_src, sr)

        # --- Analyze reference speech ---
        logger.info("Analyzing reference speech with WORLD...")
        f0_ref, t_ref = pw.harvest(reference, sr, frame_period=FRAME_PERIOD_MS)
        sp_ref = pw.cheaptrick(reference, f0_ref, t_ref, sr)

        # --- Convert spectral envelope via MCEP transformation ---
        logger.info("Converting spectral envelope...")
        n_freq = sp_src.shape[1]

        mcep_src = _sp_to_mcep(sp_src, self._mcep_order)
        mcep_ref = _sp_to_mcep(sp_ref, self._mcep_order)

        # Compute statistics from voiced frames only
        voiced_src = f0_src > 0
        voiced_ref = f0_ref > 0

        if voiced_src.sum() < 2 or voiced_ref.sum() < 2:
            logger.warning("Insufficient voiced frames, using all frames")
            voiced_src = np.ones(len(f0_src), dtype=bool)
            voiced_ref = np.ones(len(f0_ref), dtype=bool)

        mean_src = mcep_src[voiced_src].mean(axis=0)
        std_src = mcep_src[voiced_src].std(axis=0) + 1e-8
        mean_ref = mcep_ref[voiced_ref].mean(axis=0)
        std_ref = mcep_ref[voiced_ref].std(axis=0) + 1e-8

        # Global variance normalization: shift source distribution toward target
        mcep_converted = (mcep_src - mean_src) / std_src * std_ref + mean_ref

        # Blend with original based on conversion strength
        mcep_converted = (
            self._strength * mcep_converted + (1 - self._strength) * mcep_src
        )

        # Optional frequency warping for vocal tract adjustment
        if abs(self._warp_alpha) > 1e-6:
            mcep_converted = _frequency_warp(mcep_converted, self._warp_alpha)

        # Convert back to spectral envelope
        sp_converted = _mcep_to_sp(mcep_converted, n_freq)

        # Ensure non-negative and reasonable range
        sp_converted = np.maximum(sp_converted, 1e-16)

        # --- Resynthesize with ORIGINAL F0 (preserving pitch & timing) ---
        logger.info("Resynthesizing with original F0...")
        converted = pw.synthesize(
            f0_src,  # PRESERVE F0
            sp_converted,  # CONVERTED spectral envelope
            ap_src,  # PRESERVE aperiodicity/expression
            sr,
            frame_period=FRAME_PERIOD_MS,
        )

        converted = converted.astype(np.float32)

        # Normalize
        peak = np.abs(converted).max()
        if peak > 0:
            converted = converted / peak * 0.95

        logger.info(
            "Conversion complete: %.1fs output", len(converted) / sr
        )
        return ConversionResult(
            converted_vocal=AudioClip(
                samples=converted,
                sample_rate=sr,
                name=f"{source_vocal.name}_converted",
            ),
            metadata={
                "backend": self.name,
                "mcep_order": self._mcep_order,
                "conversion_strength": float(self._strength),
                "warp_alpha": self._warp_alpha,
                "source_voiced_frames": int(voiced_src.sum()),
                "source_total_frames": len(f0_src),
            },
        )
