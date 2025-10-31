"""
Upstream lm-watermarking sources bundled for reference.

This package keeps the original modules in one place while making them
importable via ``upstream.lm_watermarking``.  We insert the package
directory onto ``sys.path`` so that legacy absolute imports such as
``import homoglyphs`` inside the upstream files continue to work
without modification.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(__file__)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

# Convenience re-export for users who still expect the canonical names.
try:
    from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor
except Exception:  # pragma: no cover - optional convenience import
    WatermarkDetector = None  # type: ignore
    WatermarkLogitsProcessor = None  # type: ignore

__all__ = [
    "WatermarkDetector",
    "WatermarkLogitsProcessor",
]
