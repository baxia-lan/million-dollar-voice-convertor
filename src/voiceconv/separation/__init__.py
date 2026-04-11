"""Vocal/instrumental separation adapters."""

from .base import SeparatorBackend
from .demucs_adapter import DemucsSeparator

__all__ = ["SeparatorBackend", "DemucsSeparator"]
