"""WAY-EEG-GAL loader and benchmark for hand pose regression."""
from .loader import WAYGALLoader, extract_me_windows, extract_targets

__all__ = ["WAYGALLoader", "extract_me_windows", "extract_targets"]