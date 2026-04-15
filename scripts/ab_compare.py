#!/usr/bin/env python3
"""A/B comparison: zero-shot vs few-shot Seed-VC.

Usage:
    python scripts/ab_compare.py --song ~/Downloads/dtr.wav --ref ~/Downloads/recording.wav

Requires: both test files and a fine-tuned checkpoint (from GUI or VoiceFineTuner).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(song_path, ref_path, output_dir, custom_checkpoint=None, label=""):
    """Run the full pipeline and return output path + timing."""
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from voiceconv.core.types import ConversionJob, AudioFormat, ProvenanceRecord
    from voiceconv.core.pipeline import ConversionPipeline
    from voiceconv.separation.demucs_adapter import DemucsSeparator
    from voiceconv.svc.seedvc_svc import SeedVCSVC
    from voiceconv.judge.quality import QualityJudge
    from voiceconv.export.exporter import Exporter
    from voiceconv.compliance.provenance import ComplianceManager

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    svc = SeedVCSVC(
        diffusion_steps=50,
        auto_f0_adjust=False,
        custom_checkpoint=Path(custom_checkpoint) if custom_checkpoint else None,
    )

    pipeline = ConversionPipeline(
        separator=DemucsSeparator(),
        svc_backend=svc,
        judge=QualityJudge(),
        exporter=Exporter(default_format=AudioFormat.WAV),
        compliance=ComplianceManager(),
    )

    job = ConversionJob(
        song_path=Path(song_path),
        reference_path=Path(ref_path),
        output_dir=out,
        output_format=AudioFormat.WAV,
        provenance=ProvenanceRecord(user_consent=True),
    )

    t0 = time.time()
    result = pipeline.run(job)
    elapsed = time.time() - t0
    logger.info("[%s] Done in %.0fs: %s", label, elapsed, result)
    return result, elapsed


def compute_metrics(ref_path, orig_vocal_path, converted_path):
    """Compute speaker similarity and F0 correlation."""
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()

    ref, ref_sr = sf.read(ref_path)
    if ref.ndim == 2: ref = ref.mean(axis=1)
    ref_emb = encoder.embed_utterance(preprocess_wav(ref, source_sr=ref_sr))

    orig, orig_sr = sf.read(orig_vocal_path)
    if orig.ndim == 2: orig = orig.mean(axis=1)
    orig_emb = encoder.embed_utterance(preprocess_wav(orig, source_sr=orig_sr))

    conv, conv_sr = sf.read(converted_path)
    if conv.ndim == 2: conv = conv.mean(axis=1)
    conv_emb = encoder.embed_utterance(preprocess_wav(conv, source_sr=conv_sr))

    sim_you = float(np.dot(conv_emb, ref_emb))
    sim_orig = float(np.dot(conv_emb, orig_emb))

    return {"speaker_sim_you": sim_you, "speaker_sim_orig": sim_orig}


def main():
    parser = argparse.ArgumentParser(description="A/B compare zero-shot vs few-shot")
    parser.add_argument("--song", required=True, help="Path to song file")
    parser.add_argument("--ref", required=True, help="Path to reference speech")
    parser.add_argument("--checkpoint", help="Path to fine-tuned checkpoint (.pth)")
    parser.add_argument("--output", default="ab_compare_output", help="Output directory")
    args = parser.parse_args()

    out_base = Path(args.output)

    # Run zero-shot
    print("\n=== Running ZERO-SHOT Seed-VC ===")
    zs_result, zs_time = run_pipeline(
        args.song, args.ref, out_base / "zero_shot", label="zero-shot"
    )

    if args.checkpoint:
        # Run few-shot
        print("\n=== Running FEW-SHOT Seed-VC ===")
        fs_result, fs_time = run_pipeline(
            args.song, args.ref, out_base / "few_shot",
            custom_checkpoint=args.checkpoint, label="few-shot"
        )

    # Compute metrics
    print("\n=== Computing Metrics ===")
    zs_metrics = compute_metrics(
        args.ref,
        out_base / "zero_shot" / "original_vocal.wav",
        out_base / "zero_shot" / "converted_vocal.wav",
    )

    print(f"\n{'Metric':<25s} {'Zero-Shot':>10s}", end="")
    if args.checkpoint:
        fs_metrics = compute_metrics(
            args.ref,
            out_base / "few_shot" / "original_vocal.wav",
            out_base / "few_shot" / "converted_vocal.wav",
        )
        print(f" {'Few-Shot':>10s}", end="")
    print()
    print("-" * 50)

    print(f"{'Speaker sim → you':<25s} {zs_metrics['speaker_sim_you']:>10.4f}", end="")
    if args.checkpoint:
        print(f" {fs_metrics['speaker_sim_you']:>10.4f}", end="")
    print()

    print(f"{'Speaker sim → orig':<25s} {zs_metrics['speaker_sim_orig']:>10.4f}", end="")
    if args.checkpoint:
        print(f" {fs_metrics['speaker_sim_orig']:>10.4f}", end="")
    print()

    print(f"{'Time':<25s} {zs_time:>9.0f}s", end="")
    if args.checkpoint:
        print(f" {fs_time:>9.0f}s", end="")
    print()

    # Output paths
    print(f"\nZero-shot output: {out_base / 'zero_shot' / 'final_mix.wav'}")
    if args.checkpoint:
        print(f"Few-shot output:  {out_base / 'few_shot' / 'final_mix.wav'}")


if __name__ == "__main__":
    main()
