"""Demucs-based vocal/instrumental separation adapter.

Compatible with demucs 4.x (uses demucs.pretrained + demucs.apply).
"""

from __future__ import annotations

import logging

import numpy as np

from voiceconv.core.types import AudioClip, SeparationResult

from .base import SeparatorBackend

logger = logging.getLogger(__name__)


class DemucsSeparator(SeparatorBackend):
    """Separation using Meta's Demucs (htdemucs model).

    Demucs separates audio into 4 stems: drums, bass, other, vocals.
    We recombine drums+bass+other as the instrumental track.
    """

    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        device: str | None = None,
        shifts: int = 3,
        overlap: float = 0.5,
    ):
        self._model_name = model_name
        self._device = device or self._detect_device()
        self._shifts = shifts
        self._overlap = overlap
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
            from demucs.pretrained import get_model  # noqa: F401
            from demucs.apply import apply_model  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from demucs.pretrained import get_model
            logger.info("Loading Demucs model '%s' on %s...", self._model_name, self._device)
            self._model = get_model(self._model_name)
            if self._device != "cpu":
                import torch
                self._model.to(torch.device(self._device))
            logger.info("Demucs model loaded.")

    def separate(self, mix: AudioClip) -> SeparationResult:
        self._load_model()
        import torch
        from demucs.apply import apply_model

        # Demucs expects (batch, channels, samples) tensor at model.samplerate
        # Convert mono to stereo for demucs
        waveform = np.stack([mix.samples, mix.samples])  # (2, n_samples)
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, 2, n_samples)

        device = torch.device(self._device)

        logger.info(
            "Running Demucs separation on %.1fs audio (device=%s)...",
            mix.duration_seconds,
            self._device,
        )
        with torch.no_grad():
            sources = apply_model(
                self._model,
                tensor.to(device),
                shifts=self._shifts,
                overlap=self._overlap,
            )
        # sources shape: (1, n_sources, channels, samples)
        sources = sources.cpu()

        # Map source indices to names
        source_names = self._model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        vocals_idx = source_names.index("vocals")

        vocals_tensor = sources[0, vocals_idx]  # (channels, samples)
        # Recombine non-vocal stems as instrumental
        inst_indices = [i for i in range(len(source_names)) if i != vocals_idx]
        instrumental_tensor = sources[0, inst_indices].sum(dim=0)  # (channels, samples)

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
