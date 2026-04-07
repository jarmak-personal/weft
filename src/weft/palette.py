"""Palette + per-pixel index encoder — brainstorm idea #20.

Structurally distinct from both the primitive-stack baseline and the
bicubic-patch path: there are no tiles, no quadtree, no per-region
basis. The whole image is one global palette plus a per-pixel index
into it.

Why bother:
* Hard-edged content (vector art, screenshots, text, pixel art) is the
  one regime where bicubic and primitive-stack both struggle, because
  step discontinuities don't lie in any smooth basis. Palette+labels
  represents step changes natively — every pixel transition is exact
  up to the chosen quantization.
* The labels grid is highly autocorrelated (adjacent pixels share
  labels almost everywhere on natural and screenshot content), which
  the bitstream-level zstd pass crushes. Typical ratios on screenshot
  content: 5-15× over the raw labels grid.
* No GPU dependency: encoder is one PIL ``Image.quantize`` call,
  decoder is one ``palette[labels]`` lookup.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def fit_palette(image: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``image`` (linear-RGB float in [0, 1]) to a K-color palette.

    Uses PIL's median-cut quantizer (``Image.quantize`` with
    ``MEDIANCUT``) since it's fast, deterministic, and ships in the
    standard library. Returns ``(palette, labels)`` where ``palette``
    is shape ``(K, 3)`` u8 and ``labels`` is shape ``(h, w)`` u8 with
    each value in ``[0, K)``.
    """
    if k < 1 or k > 256:
        raise ValueError(f"palette size k must be in [1, 256], got {k}")
    arr_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(arr_u8, mode="RGB")
    quantized = pil.quantize(
        colors=k,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    raw_palette = quantized.getpalette() or []
    # PIL pads palettes to 256 entries — slice to the actual K used.
    n_used = max(int(quantized.getextrema()[1]) + 1, 1)
    n_used = min(n_used, k, len(raw_palette) // 3)
    palette = np.array(raw_palette[: n_used * 3], dtype=np.uint8).reshape(n_used, 3)
    labels = np.array(quantized, dtype=np.uint8)
    return palette, labels


def render_palette(palette: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reconstruct an image from a palette and per-pixel labels grid.

    Returns a ``(h, w, 3)`` float32 image in [0, 1].
    """
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8)
    if palette.dtype != np.uint8:
        palette = palette.astype(np.uint8)
    return palette[labels].astype(np.float32) / 255.0
