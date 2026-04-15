"""OpenVoice V2 neural voice conversion backend.

Uses the OpenVoice ToneColorConverter (VITS-based) for high-quality
zero-shot timbre transfer. Preserves the vocal performance (melody contour,
rhythm, lyrics, expression) while replacing speaker identity.

Note: OpenVoice may introduce a global F0 offset as a side effect of
speaker embedding transfer. The F0 contour *shape* (melody) is preserved
with ~0.80 correlation. Duration and lyrics are exactly preserved.
"""

from __future__ import annotations

import logging
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceconv.core.types import AudioClip, ConversionResult

from .base import SVCBackend

logger = logging.getLogger(__name__)

_OPENVOICE_CACHE = Path.home() / ".cache" / "openvoice_v2"


class OpenVoiceSVC(SVCBackend):
    """OpenVoice V2 neural voice conversion.

    High-quality zero-shot timbre transfer using a pre-trained VITS model.
    Automatically downloads the checkpoint (~131 MB) on first use.
    """

    def __init__(self, tau: float = 0.1, device: str | None = None):
        self._tau = tau
        self._device = device or self._detect_device()
        self._converter = None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    def name(self) -> str:
        return "openvoice-v2"

    def is_available(self) -> bool:
        try:
            from openvoice_cli.api import ToneColorConverter  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self):
        """Lazy-load the ToneColorConverter and download checkpoint if needed."""
        if self._converter is not None:
            return

        from openvoice_cli.api import ToneColorConverter
        from openvoice_cli.downloader import download_checkpoint

        ckpt_dir = _OPENVOICE_CACHE
        config_path = ckpt_dir / "config.json"
        ckpt_path = ckpt_dir / "checkpoint.pth"

        if not config_path.exists() or not ckpt_path.exists():
            logger.info("Downloading OpenVoice V2 checkpoint...")
            download_checkpoint(str(ckpt_dir))

        logger.info("Loading OpenVoice model on %s...", self._device)
        self._converter = ToneColorConverter(
            str(config_path), device=self._device
        )
        self._converter.load_ckpt(str(ckpt_path))
        logger.info("OpenVoice model loaded.")

    def convert(
        self,
        source_vocal: AudioClip,
        reference_speech: AudioClip,
    ) -> ConversionResult:
        self._ensure_model()

        sr_orig = source_vocal.sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            src_path = tmp / "source.wav"
            ref_path = tmp / "reference.wav"
            out_path = tmp / "converted.wav"

            sf.write(str(src_path), source_vocal.samples, sr_orig, subtype="FLOAT")
            sf.write(
                str(ref_path),
                reference_speech.samples,
                reference_speech.sample_rate,
                subtype="FLOAT",
            )

            # Extract speaker embeddings
            logger.info("Extracting speaker embeddings...")
            src_se = self._converter.extract_se([str(src_path)])
            tgt_se = self._converter.extract_se([str(ref_path)])

            # Neural tone-color conversion
            logger.info("Running OpenVoice conversion (tau=%.2f)...", self._tau)
            self._converter.convert(
                audio_src_path=str(src_path),
                src_se=src_se,
                tgt_se=tgt_se,
                output_path=str(out_path),
                tau=self._tau,
            )

            converted, sr_conv = sf.read(str(out_path))

        if converted.ndim == 2:
            converted = converted.mean(axis=1)

        # Resample to original sample rate if needed
        if sr_conv != sr_orig:
            from scipy.signal import resample_poly

            g = gcd(sr_conv, sr_orig)
            converted = resample_poly(converted, sr_orig // g, sr_conv // g)

        converted = converted.astype(np.float32)

        # Normalize
        peak = np.abs(converted).max()
        if peak > 0:
            converted = converted / peak * 0.95

        logger.info(
            "OpenVoice conversion complete: %.1fs output",
            len(converted) / sr_orig,
        )

        return ConversionResult(
            converted_vocal=AudioClip(
                samples=converted,
                sample_rate=sr_orig,
                name=f"{source_vocal.name}_converted",
            ),
            metadata={
                "backend": self.name,
                "tau": self._tau,
                "device": self._device,
            },
        )
