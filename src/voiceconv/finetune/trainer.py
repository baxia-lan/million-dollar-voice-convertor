"""Voice fine-tuning wrapper for personalized Seed-VC adaptation.

Takes 15-30 minutes of clean user recordings, splits them into
5-25 second segments, and fine-tunes the Seed-VC 44kHz singing model
to the user's voice. Produces a checkpoint that can be loaded by
SeedVCSVC for higher-quality personalized conversion.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Segment duration bounds for Seed-VC fine-tuning dataset
_MIN_SEGMENT_SEC = 3.0
_MAX_SEGMENT_SEC = 25.0
_DEFAULT_STEPS = 1000
_DEFAULT_SAVE_INTERVAL = 500


class VoiceFineTuner:
    """Fine-tune Seed-VC on user's voice recordings.

    Usage:
        finetuner = VoiceFineTuner()
        ckpt_path = finetuner.run(
            recordings_dir="/path/to/user/recordings",
            output_dir="/path/to/output",
            steps=1000,
        )
        # Then use ckpt_path as custom_checkpoint in SeedVCSVC
    """

    def __init__(self, device: str | None = None):
        self._device = device or self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"

    def run(
        self,
        recordings_dir: str | Path,
        output_dir: str | Path,
        steps: int = _DEFAULT_STEPS,
        save_interval: int = _DEFAULT_SAVE_INTERVAL,
        batch_size: int = 1,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """Run fine-tuning pipeline.

        Args:
            recordings_dir: Directory with user's clean audio recordings.
            output_dir: Where to save the fine-tuned checkpoint.
            steps: Training steps (default 1000).
            save_interval: Save checkpoint every N steps.
            batch_size: Training batch size.
            on_progress: Callback(current_step, total_steps, message).

        Returns:
            Path to the best fine-tuned checkpoint.
        """
        recordings_dir = Path(recordings_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Prepare training data
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        if on_progress:
            on_progress(0, steps, "Preparing training segments...")

        n_segments = self._prepare_segments(recordings_dir, segments_dir)
        if n_segments == 0:
            raise ValueError(
                f"No valid audio segments found in {recordings_dir}. "
                "Need audio files (wav/mp3/flac) with at least 3 seconds of content."
            )
        logger.info("Prepared %d training segments in %s", n_segments, segments_dir)

        # Step 2: Get config and pretrained checkpoint paths
        config_path, pretrained_ckpt_path = self._resolve_model_paths()

        # Step 3: Patch the config to use our output dir
        patched_config_path = self._patch_config(config_path, output_dir)

        # Step 4: Run training
        if on_progress:
            on_progress(0, steps, "Starting fine-tuning...")

        ckpt_path = self._train(
            config_path=patched_config_path,
            pretrained_ckpt_path=pretrained_ckpt_path,
            data_dir=str(segments_dir),
            output_dir=output_dir,
            steps=steps,
            save_interval=save_interval,
            batch_size=batch_size,
            on_progress=on_progress,
        )

        logger.info("Fine-tuning complete. Checkpoint: %s", ckpt_path)
        return ckpt_path

    def _prepare_segments(self, recordings_dir: Path, segments_dir: Path) -> int:
        """Split long recordings into segments suitable for fine-tuning."""
        import librosa

        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
        n_segments = 0

        for audio_file in sorted(recordings_dir.rglob("*")):
            if audio_file.suffix.lower() not in audio_extensions:
                continue

            try:
                audio, sr = librosa.load(str(audio_file), sr=44100, mono=True)
            except Exception as e:
                logger.warning("Failed to load %s: %s", audio_file, e)
                continue

            duration = len(audio) / sr

            if duration < _MIN_SEGMENT_SEC:
                logger.warning("Skipping %s (%.1fs < %.1fs min)", audio_file.name, duration, _MIN_SEGMENT_SEC)
                continue

            if duration <= _MAX_SEGMENT_SEC:
                # Short enough — use as-is
                out_path = segments_dir / f"seg_{n_segments:04d}.wav"
                sf.write(str(out_path), audio, sr)
                n_segments += 1
            else:
                # Split into segments with small overlap
                segment_len = int(_MAX_SEGMENT_SEC * sr)
                hop = int((_MAX_SEGMENT_SEC - 1.0) * sr)  # 1s overlap
                pos = 0
                while pos + int(_MIN_SEGMENT_SEC * sr) < len(audio):
                    end = min(pos + segment_len, len(audio))
                    segment = audio[pos:end]
                    if len(segment) / sr >= _MIN_SEGMENT_SEC:
                        out_path = segments_dir / f"seg_{n_segments:04d}.wav"
                        sf.write(str(out_path), segment, sr)
                        n_segments += 1
                    pos += hop

        return n_segments

    def _resolve_model_paths(self) -> tuple[str, str]:
        """Get the 44kHz singing model config and pretrained checkpoint paths."""
        from seed_vc.hf_utils import load_custom_model_from_hf

        ckpt_path, config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )
        return config_path, ckpt_path

    def _patch_config(self, config_path: str, output_dir: Path) -> str:
        """Patch the config to set log_dir to our output directory."""
        import yaml

        config = yaml.safe_load(open(config_path))
        config["log_dir"] = str(output_dir / "runs")
        config["batch_size"] = 1  # Safe default for CPU/low-memory

        patched_path = str(output_dir / "config_finetune.yml")
        with open(patched_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return patched_path

    def _train(
        self,
        config_path: str,
        pretrained_ckpt_path: str,
        data_dir: str,
        output_dir: Path,
        steps: int,
        save_interval: int,
        batch_size: int,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> Path:
        """Run the actual Seed-VC fine-tuning.

        We import and instantiate the Trainer directly, fixing the
        broken relative imports in the pip package.
        """
        # Fix Seed-VC's broken non-relative imports by adding its package dir to sys.path
        seedvc_pkg_dir = str(Path(__file__).resolve().parents[0])
        seedvc_install_dir = None
        try:
            import seed_vc
            seedvc_install_dir = str(Path(seed_vc.__file__).parent)
        except ImportError:
            pass

        if seedvc_install_dir and seedvc_install_dir not in sys.path:
            sys.path.insert(0, seedvc_install_dir)

        from seed_vc.train import Trainer

        logger.info(
            "Starting fine-tuning: steps=%d, batch=%d, device=%s",
            steps, batch_size, self._device,
        )

        trainer = Trainer(
            config_path=config_path,
            pretrained_ckpt_path=pretrained_ckpt_path,
            data_dir=data_dir,
            run_name="voiceconv_finetune",
            batch_size=batch_size,
            num_workers=0,
            steps=steps,
            save_interval=save_interval,
            max_epochs=10000,
            device=self._device,
        )

        trainer.train()

        # Find the latest checkpoint
        runs_dir = output_dir / "runs" / "voiceconv_finetune"
        checkpoints = sorted(runs_dir.glob("DiT_epoch_*_step_*.pth"))
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found in {runs_dir}")

        latest = max(checkpoints, key=lambda p: int(p.stem.split("_step_")[-1]))
        # Copy to a stable location
        final_ckpt = output_dir / "finetuned_checkpoint.pth"
        shutil.copy2(str(latest), str(final_ckpt))

        return final_ckpt
