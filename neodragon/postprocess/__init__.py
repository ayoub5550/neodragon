# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Neodragon-X post-processing (optional, runs after generation)."""
from .rife import RIFEInterpolator, interpolate_frames

__all__ = ["RIFEInterpolator", "interpolate_frames"]
