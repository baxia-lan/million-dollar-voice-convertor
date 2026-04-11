"""Abstract base class for vocal/instrumental separation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voiceconv.core.types import AudioClip, SeparationResult


class SeparatorBackend(ABC):
    """Interface that all separation backends must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""

    @abstractmethod
    def separate(self, mix: AudioClip) -> SeparationResult:
        """Separate a mix into vocals and instrumental.

        Args:
            mix: Full song audio clip.

        Returns:
            SeparationResult with vocals and instrumental.
        """
