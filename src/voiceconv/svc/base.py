"""Abstract base class for singing voice conversion backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voiceconv.core.types import AudioClip, ConversionResult


class SVCBackend(ABC):
    """Interface that all SVC backends must implement.

    Contract:
    - preserve_f0=True: output F0 contour matches source vocal
    - preserve_duration=True: output length matches source vocal
    - preserve_lyrics=True: phonetic content from source is retained
    - preserve_expression=True: vibrato, dynamics, articulation from source
    - Only timbre (speaker identity) comes from the reference speech
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""

    @abstractmethod
    def convert(
        self,
        source_vocal: AudioClip,
        reference_speech: AudioClip,
    ) -> ConversionResult:
        """Convert source vocal timbre to match reference speaker.

        Args:
            source_vocal: Isolated vocal track (performance carrier).
            reference_speech: User speech sample (timbre donor).

        Returns:
            ConversionResult with the converted vocal.
        """
