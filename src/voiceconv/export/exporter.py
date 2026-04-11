"""Export module: mix converted vocal with instrumental, save to file."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voiceconv.core.audio_io import save_audio
from voiceconv.core.types import AudioClip, AudioFormat

logger = logging.getLogger(__name__)


class Exporter:
    """Mix and export final audio."""

    def __init__(self, default_format: AudioFormat = AudioFormat.WAV):
        self.default_format = default_format

    def mix_tracks(
        self,
        vocal: AudioClip,
        instrumental: AudioClip,
        vocal_gain_db: float = 0.0,
        instrumental_gain_db: float = 0.0,
    ) -> AudioClip:
        """Mix converted vocal with instrumental track.

        Args:
            vocal: Converted vocal track.
            instrumental: Instrumental track from separation.
            vocal_gain_db: Volume adjustment for vocal in dB.
            instrumental_gain_db: Volume adjustment for instrumental in dB.

        Returns:
            Mixed AudioClip.
        """
        sr = vocal.sample_rate

        # Apply gain
        v_gain = 10 ** (vocal_gain_db / 20.0)
        i_gain = 10 ** (instrumental_gain_db / 20.0)

        v = vocal.samples * v_gain
        inst = instrumental.samples * i_gain

        # Pad shorter track
        max_len = max(len(v), len(inst))
        if len(v) < max_len:
            v = np.pad(v, (0, max_len - len(v)))
        if len(inst) < max_len:
            inst = np.pad(inst, (0, max_len - len(inst)))

        mixed = v + inst

        # Prevent clipping
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed = mixed / peak * 0.95

        mixed = mixed.astype(np.float32)
        logger.info("Mixed vocal + instrumental: %.1fs", len(mixed) / sr)

        return AudioClip(
            samples=mixed,
            sample_rate=sr,
            name=f"{vocal.name}_final",
        )

    def export(
        self,
        clip: AudioClip,
        output_dir: Path,
        filename: str = "output",
        fmt: AudioFormat | None = None,
    ) -> Path:
        """Export audio clip to file.

        Returns path to saved file.
        """
        fmt = fmt or self.default_format
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{filename}.{fmt.value}"
        return save_audio(clip, path, fmt)

    def export_stems(
        self,
        vocal: AudioClip,
        instrumental: AudioClip,
        converted_vocal: AudioClip,
        mixed: AudioClip,
        output_dir: Path,
        fmt: AudioFormat | None = None,
    ) -> dict[str, Path]:
        """Export all stems and final mix."""
        fmt = fmt or self.default_format
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        paths["original_vocal"] = save_audio(
            vocal, output_dir / f"original_vocal.{fmt.value}", fmt
        )
        paths["instrumental"] = save_audio(
            instrumental, output_dir / f"instrumental.{fmt.value}", fmt
        )
        paths["converted_vocal"] = save_audio(
            converted_vocal, output_dir / f"converted_vocal.{fmt.value}", fmt
        )
        paths["final_mix"] = save_audio(
            mixed, output_dir / f"final_mix.{fmt.value}", fmt
        )
        logger.info("Exported %d stems to %s", len(paths), output_dir)
        return paths
