"""Seed-VC singing voice conversion backend (44kHz F0-conditioned).

Directly loads the singing-specific model from app_svc pipeline:
- DiT diffusion transformer at 44100 Hz
- RMVPE for accurate F0 extraction (singing-optimized)
- BigVGAN v2 44kHz vocoder for high-fidelity synthesis
- Whisper encoder for content features
- CAM++ for speaker embedding

Supports both zero-shot (pretrained checkpoint) and few-shot
(user-fine-tuned checkpoint) modes.
"""

from __future__ import annotations

import logging
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voiceconv.core.types import AudioClip, ConversionResult

from .base import SVCBackend

logger = logging.getLogger(__name__)

_MAX_REF_SECONDS = 25
_models_cache = None


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_singing_models(device: torch.device):
    """Load the 44kHz F0-conditioned singing model directly (app_svc path)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    import yaml
    from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch
    from seed_vc.hf_utils import load_custom_model_from_hf

    logger.info("Loading Seed-VC 44kHz singing model on %s...", device)

    # Load DiT model (F0-conditioned, 44kHz, BigVGAN)
    dit_ckpt_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    )
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    model, _, _, _ = load_checkpoint(
        model, None, dit_ckpt_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Speaker encoder: CAM++
    from seed_vc.modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None,
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)

    # Vocoder: BigVGAN 44kHz
    # Patch BigVGAN._from_pretrained to accept newer huggingface_hub kwargs
    from seed_vc.modules.bigvgan import bigvgan
    _orig_from_pretrained = bigvgan.BigVGAN._from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched_from_pretrained(cls, **kwargs):
        kwargs.setdefault("proxies", None)
        kwargs.setdefault("resume_download", False)
        return _orig_from_pretrained.__func__(cls, **kwargs)

    bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained
    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    # Content encoder: Whisper
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None, output_attentions=False,
                    output_hidden_states=False, return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    else:
        raise ValueError(f"Unsupported speech tokenizer: {speech_tokenizer_type}")

    # F0 extractor: RMVPE
    from seed_vc.modules.rmvpe import RMVPE
    rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)
    f0_fn = rmvpe.infer_from_audio

    # Mel spectrogram function
    from seed_vc.modules.audio import mel_spectrogram
    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": hop_length,
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
        "center": False,
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    _models_cache = {
        "model": model,
        "semantic_fn": semantic_fn,
        "vocoder_fn": bigvgan_model,
        "campplus_model": campplus_model,
        "to_mel": to_mel,
        "f0_fn": f0_fn,
        "sr": sr,
        "hop_length": hop_length,
        "device": device,
        "config": config,
        "dit_ckpt_path": dit_ckpt_path,
    }

    logger.info("Seed-VC 44kHz singing model loaded (sr=%d).", sr)
    return _models_cache


def _adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def _crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


class SeedVCSVC(SVCBackend):
    """Seed-VC 44kHz F0-conditioned singing voice conversion."""

    def __init__(
        self,
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
        auto_f0_adjust: bool = False,
        pitch_shift: int = 0,
        custom_checkpoint: Path | None = None,
        device: str | None = None,
    ):
        self._diffusion_steps = diffusion_steps
        self._cfg_rate = inference_cfg_rate
        self._auto_f0_adjust = auto_f0_adjust
        self._pitch_shift = pitch_shift
        self._custom_checkpoint = custom_checkpoint
        self._device_str = device

    @property
    def name(self) -> str:
        suffix = " (fine-tuned)" if self._custom_checkpoint else ""
        return f"seed-vc-singing{suffix}"

    def is_available(self) -> bool:
        try:
            from seed_vc.modules.commons import build_model  # noqa: F401
            return True
        except ImportError:
            return False

    def convert(
        self,
        source_vocal: AudioClip,
        reference_speech: AudioClip,
    ) -> ConversionResult:
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        device = torch.device(self._device_str) if self._device_str else _detect_device()
        models = _load_singing_models(device)

        # Optionally load custom checkpoint
        if self._custom_checkpoint and Path(self._custom_checkpoint).exists():
            from seed_vc.modules.commons import load_checkpoint
            logger.info("Loading custom checkpoint: %s", self._custom_checkpoint)
            models["model"], _, _, _ = load_checkpoint(
                models["model"], None, str(self._custom_checkpoint),
                load_only_params=True, ignore_modules=[], is_distributed=False,
            )
            for key in models["model"]:
                models["model"][key].eval().to(device)

        inference_module = models["model"]
        semantic_fn = models["semantic_fn"]
        vocoder_fn = models["vocoder_fn"]
        campplus_model = models["campplus_model"]
        mel_fn = models["to_mel"]
        f0_fn = models["f0_fn"]
        sr = models["sr"]
        hop_length = models["hop_length"]

        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length
        max_context_window = sr // hop_length * 30

        sr_orig = source_vocal.sample_rate

        # Write to temp files and load at model SR
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src_path = tmp / "source.wav"
            ref_path = tmp / "reference.wav"
            sf.write(str(src_path), source_vocal.samples, sr_orig, subtype="FLOAT")
            ref_samples = reference_speech.samples
            ref_sr = reference_speech.sample_rate
            max_ref = int(_MAX_REF_SECONDS * ref_sr)
            if len(ref_samples) > max_ref:
                ref_samples = ref_samples[:max_ref]
            sf.write(str(ref_path), ref_samples, ref_sr, subtype="FLOAT")

            import librosa
            source_audio = librosa.load(str(src_path), sr=sr)[0]
            ref_audio = librosa.load(str(ref_path), sr=sr)[0]

        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[:sr * _MAX_REF_SECONDS]).unsqueeze(0).float().to(device)

        logger.info(
            "Running Seed-VC singing conversion (44kHz, F0-conditioned, "
            "steps=%d, cfg=%.2f) on %.1fs audio...",
            self._diffusion_steps, self._cfg_rate, source_vocal.duration_seconds,
        )

        # Resample to 16kHz for content features
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        # Content encoding (chunked for long audio)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                S_alt = semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        S_ori = semantic_fn(ref_waves_16k)

        # Mel spectrograms
        mel = mel_fn(source_audio.to(device).float())
        mel2 = mel_fn(ref_audio.to(device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * 1.0)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # Speaker embedding
        feat2 = kaldi.fbank(ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        # F0 extraction
        F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]

        # F0 adjustment
        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log_f0_alt = log_f0_alt.clone()

        if self._auto_f0_adjust:
            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]
            median_log_f0_ori = torch.median(torch.log(voiced_F0_ori + 1e-5))
            median_log_f0_alt = torch.median(torch.log(voiced_F0_alt + 1e-5))
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori

        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if self._pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = _adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], self._pitch_shift)

        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )
        interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
            shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode="nearest"
        ).squeeze(1)

        # Chunked diffusion + vocoding
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    mel2, style2, None, self._diffusion_steps,
                    inference_cfg_rate=self._cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1):]

            # Clone + detach from autocast/inference graph before vocoder
            vc_wave = vocoder_fn(vc_target.float().clone()).squeeze().detach().cpu()
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].numpy())
                    break
                generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].numpy())
                previous_chunk = vc_wave[0, -overlap_wave_len:]
            elif is_last_chunk:
                output_wave = _crossfade(previous_chunk.numpy(), vc_wave[0].numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                break
            else:
                output_wave = _crossfade(previous_chunk.numpy(), vc_wave[0, :-overlap_wave_len].numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]

            processed_frames += vc_target.size(2) - overlap_frame_len

        # Concatenate all chunks
        full_audio = np.concatenate(generated_wave_chunks).astype(np.float32)

        # Resample to original SR if needed
        if sr != sr_orig:
            from scipy.signal import resample_poly
            g = gcd(sr, sr_orig)
            full_audio = resample_poly(full_audio, sr_orig // g, sr // g).astype(np.float32)

        # Normalize
        peak = np.abs(full_audio).max()
        if peak > 0:
            full_audio = full_audio / peak * 0.95

        logger.info("Seed-VC singing conversion complete: %.1fs output", len(full_audio) / sr_orig)

        return ConversionResult(
            converted_vocal=AudioClip(
                samples=full_audio,
                sample_rate=sr_orig,
                name=f"{source_vocal.name}_converted",
            ),
            metadata={
                "backend": self.name,
                "diffusion_steps": self._diffusion_steps,
                "inference_cfg_rate": self._cfg_rate,
                "f0_condition": True,
                "auto_f0_adjust": self._auto_f0_adjust,
                "model_sr": sr,
                "custom_checkpoint": str(self._custom_checkpoint) if self._custom_checkpoint else None,
            },
        )
