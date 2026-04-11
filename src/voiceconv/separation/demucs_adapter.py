"""Demucs-based vocal/instrumental separation adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voiceconv.core.types import AudioClip, SeparationResult

from .base import SeparatorBackend

logger = logging.getLogger(__name__)


class DemucsSeparator(SeparatorBackend):
    """Separation using Meta's Demucs (htdemucs model).

    Demucs separates audio into 4 stems: drums, bass, other, vocals.
    We recombine drums+bass+other as the instrumental track.
    """

    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        self._model_name = model_name
        self._device = device or self._detect_device()
        self._model = None

    @staticmethod
    def _detect_device() -> str:
        """Pick the best available device (CUDA > MPS > CPU)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def name(self) -> str:
        return f"demucs/{self._model_name}"

    def is_available(self) -> bool:
        try:
            import demucs.api  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            import demucs.api
            logger.info("Loading Demucs model '%s'...", self._model_name)
            self._model = demucs.api.Separator(
                model=self._model_name,
                device=self._device,
            )
            logger.info("Demucs model loaded.")

    def separate(self, mix: AudioClip) -> SeparationResult:
        self._load_model()
        import torch

        # Demucs expects (channels, samples) tensor
        # Convert mono to stereo for demucs
        waveform = np.stack([mix.samples, mix.samples])  # (2, n_samples)
        tensor = torch.from_numpy(waveform).float()

        logger.info("Running Demucs separation on %.1fs audio...", mix.duration_seconds)
        _, separated = self._model.separate_tensor(tensor, sr=mix.sample_rate)

        # separated is dict: stem_name -> (channels, samples)
        vocals_tensor = separated["vocals"]
        # Recombine non-vocal stems as instrumental
        inst_stems = [v for k, v in separated.items() if k != "vocals"]
        instrumental_tensor = sum(inst_stems)

        # Convert to mono numpy
        vocals_np = vocals_tensor.mean(dim=0).numpy().astype(np.float32)
        inst_np = instrumental_tensor.mean(dim=0).numpy().astype(np.float32)

        logger.info("Separation complete.")
        return SeparationResult(
            vocals=AudioClip(
                samples=vocals_np,
                sample_rate=mix.sample_rate,
                name=f"{mix.name}_vocals",
            ),
            instrumental=AudioClip(
                samples=inst_np,
                sample_rate=mix.sample_rate,
                name=f"{mix.name}_instrumental",
            ),
        )
