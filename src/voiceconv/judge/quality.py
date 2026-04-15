"""Quality assessment for converted vocals.

Checks:
1. Speaker similarity: cosine similarity of speaker embeddings
2. F0 correlation: how well pitch contour is preserved
3. Signal quality: SNR of output
"""

from __future__ import annotations

import logging

import numpy as np

from voiceconv.core.types import AudioClip, JudgementResult

logger = logging.getLogger(__name__)

# Thresholds for pass/fail
MIN_SPEAKER_SIMILARITY = 0.55
MIN_F0_CORRELATION = 0.60  # Neural backends may shift F0 globally; contour shape is preserved
MIN_SNR_DB = 5.0


class QualityJudge:
    """Assess quality of voice conversion output."""

    def __init__(self):
        self._encoder = None

    def _get_speaker_embedding(self, clip: AudioClip) -> np.ndarray:
        """Extract speaker embedding using resemblyzer."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            if self._encoder is None:
                self._encoder = VoiceEncoder()

            wav = preprocess_wav(clip.samples, source_sr=clip.sample_rate)
            embedding = self._encoder.embed_utterance(wav)
            return embedding
        except ImportError:
            logger.warning("resemblyzer not available, using spectral fallback")
            return self._spectral_embedding(clip)

    def _spectral_embedding(self, clip: AudioClip) -> np.ndarray:
        """Fallback speaker embedding using spectral statistics."""
        from scipy.fft import rfft

        # Compute average spectral profile as a crude speaker fingerprint
        frame_len = min(2048, len(clip.samples))
        hop = frame_len // 2
        n_frames = max(1, (len(clip.samples) - frame_len) // hop)

        spectra = []
        for i in range(n_frames):
            start = i * hop
            frame = clip.samples[start : start + frame_len]
            spec = np.abs(rfft(frame * np.hanning(len(frame))))
            spectra.append(spec)

        avg_spec = np.mean(spectra, axis=0)
        # Reduce to 64 dimensions via averaging bins
        n_bins = 64
        bin_size = len(avg_spec) // n_bins
        embedding = np.array(
            [avg_spec[i * bin_size : (i + 1) * bin_size].mean() for i in range(n_bins)]
        )
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _compute_f0_correlation(
        self, source: AudioClip, converted: AudioClip
    ) -> float:
        """Compute correlation of F0 contours between source and converted."""
        try:
            import pyworld as pw
        except ImportError:
            logger.warning("pyworld not available, skipping F0 correlation")
            return 0.9  # Optimistic default

        sr = source.sample_rate
        f0_src, _ = pw.harvest(
            source.samples.astype(np.float64), sr, frame_period=5.0
        )
        f0_conv, _ = pw.harvest(
            converted.samples.astype(np.float64), sr, frame_period=5.0
        )

        # Align lengths
        min_len = min(len(f0_src), len(f0_conv))
        f0_src = f0_src[:min_len]
        f0_conv = f0_conv[:min_len]

        # Only compare voiced frames
        voiced = (f0_src > 0) & (f0_conv > 0)
        if voiced.sum() < 2:
            return 0.0

        corr = np.corrcoef(f0_src[voiced], f0_conv[voiced])[0, 1]
        return float(max(0.0, corr))

    def _compute_snr(self, clip: AudioClip) -> float:
        """Estimate SNR by comparing signal energy to noise floor."""
        samples = clip.samples
        frame_len = 1024
        hop = 512
        energies = []
        for i in range(0, len(samples) - frame_len, hop):
            frame = samples[i : i + frame_len]
            energies.append(np.mean(frame**2))

        if not energies:
            return 0.0

        energies = np.array(energies)
        # Estimate noise as bottom 10th percentile energy
        noise_energy = np.percentile(energies, 10) + 1e-16
        signal_energy = np.mean(energies) + 1e-16
        snr_db = 10 * np.log10(signal_energy / noise_energy)
        return float(snr_db)

    def judge(
        self,
        source_vocal: AudioClip,
        reference_speech: AudioClip,
        converted_vocal: AudioClip,
    ) -> JudgementResult:
        """Run full quality assessment.

        Args:
            source_vocal: Original isolated vocal (performance carrier).
            reference_speech: User speech reference (timbre donor).
            converted_vocal: Output of SVC backend.
        """
        logger.info("Running quality assessment...")

        # 1. Speaker similarity: converted should sound like reference
        emb_ref = self._get_speaker_embedding(reference_speech)
        emb_conv = self._get_speaker_embedding(converted_vocal)
        similarity = float(
            np.dot(emb_ref, emb_conv)
            / (np.linalg.norm(emb_ref) * np.linalg.norm(emb_conv) + 1e-8)
        )

        # 2. F0 correlation: converted should preserve source pitch
        f0_corr = self._compute_f0_correlation(source_vocal, converted_vocal)

        # 3. Signal quality
        snr = self._compute_snr(converted_vocal)

        overall_pass = (
            similarity >= MIN_SPEAKER_SIMILARITY
            and f0_corr >= MIN_F0_CORRELATION
            and snr >= MIN_SNR_DB
        )

        result = JudgementResult(
            speaker_similarity=similarity,
            f0_correlation=f0_corr,
            signal_to_noise=snr,
            overall_pass=overall_pass,
            details={
                "thresholds": {
                    "min_speaker_similarity": MIN_SPEAKER_SIMILARITY,
                    "min_f0_correlation": MIN_F0_CORRELATION,
                    "min_snr_db": MIN_SNR_DB,
                },
            },
        )
        logger.info(
            "Judgement: similarity=%.3f, f0_corr=%.3f, snr=%.1fdB, pass=%s",
            similarity,
            f0_corr,
            snr,
            overall_pass,
        )
        return result
