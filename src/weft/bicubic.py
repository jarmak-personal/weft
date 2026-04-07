"""Bicubic Bézier patch tiles — brainstorm idea #11.

This is a structurally-different encoder path: instead of greedy primitive
search per tile, each tile is approximated by a single 4×4 Bézier control
grid, fit by closed-form linear least squares. The decoder evaluates the
bicubic in NumPy on the CPU; no OptiX/BVH involvement.

Why bother:
* Encoder is one np.linalg.pinv per tile axis (precomputed once per tile
  size) → orders of magnitude faster than greedy primitive search.
* 16 RGB control points × 8 bits = 48 bytes per tile, fixed and uniform.
* Smooth-content tiles compress beautifully via the closed form; sharp
  edges are where the bicubic struggles — and that's exactly the regime
  where the existing primitive-stack encoder shines, so the two are
  complementary.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Bernstein basis (precomputed per tile size)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _bernstein4_basis(n: int) -> np.ndarray:
    """Cubic Bernstein basis evaluated at n uniform points in [0, 1].

    Returns shape ``(n, 4)`` where row i is ``[B0(t_i), B1(t_i), B2(t_i), B3(t_i)]``.
    """
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.stack([
        (1.0 - t) ** 3,
        3.0 * (1.0 - t) ** 2 * t,
        3.0 * (1.0 - t) * t ** 2,
        t ** 3,
    ], axis=-1)


@lru_cache(maxsize=8)
def _bernstein4_pinvs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Precomputed pseudoinverses for the closed-form bicubic fit.

    Returns ``(B_pinv, BT_pinv)`` where B is the Bernstein basis ``(n, 4)``.
    For an n×n tile patch P with channels c, the LS-optimal control grid
    Q ``(4, 4, c)`` satisfies ``P[:, :, k] = B @ Q[:, :, k] @ B.T``, so
    ``Q[:, :, k] = B_pinv @ P[:, :, k] @ BT_pinv``.
    """
    B = _bernstein4_basis(n)
    return np.linalg.pinv(B), np.linalg.pinv(B.T)


# ---------------------------------------------------------------------------
# Per-tile fit (encoder side)
# ---------------------------------------------------------------------------

def fit_tile(tile: np.ndarray) -> np.ndarray:
    """Closed-form least-squares bicubic fit of a single tile patch.

    Input ``tile`` shape ``(h, w, 3)`` linear-RGB float32 in [0, 1].
    Output: ``(4, 4, 3)`` float32 control grid (clipped to [0, 1]).
    """
    h, w, c = tile.shape
    if h != w:
        # Ragged edge tiles: pad to square so the basis matches.
        pad_h = max(0, w - h)
        pad_w = max(0, h - w)
        if pad_h or pad_w:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        h = w = max(h, w)
    Bv_pinv, Bu_pinv_T = _bernstein4_pinvs(h)
    P = np.empty((4, 4, c), dtype=np.float32)
    for ch in range(c):
        P[:, :, ch] = (Bv_pinv @ tile[:, :, ch].astype(np.float64) @ Bu_pinv_T).astype(np.float32)
    return np.clip(P, 0.0, 1.0)


def fit_tiles(image: np.ndarray, quad_tiles: list[Any]) -> np.ndarray:
    """Fit one bicubic control grid per QuadTile against ``image``.

    Returns shape ``(n_tiles, 4, 4, 3)`` float32. Delegates patch
    extraction to ``weft.quadtree.extract_adaptive_tiles`` so that the
    image is edge-padded to a multiple of ``TILE_SIZE_MAX`` exactly the
    way the existing adaptive encoder does — otherwise tiles that
    straddle the original right/bottom edges produce empty slices and
    crash on cartoon-family-style non-multiple-of-32 images.
    """
    from .quadtree import extract_adaptive_tiles
    patches = extract_adaptive_tiles(image, quad_tiles)
    n = len(quad_tiles)
    out = np.empty((n, 4, 4, 3), dtype=np.float32)
    for i, patch in enumerate(patches):
        out[i] = fit_tile(patch)
    return out


# ---------------------------------------------------------------------------
# Per-tile evaluation (decoder side)
# ---------------------------------------------------------------------------

def eval_tile(control_grid: np.ndarray, size: int) -> np.ndarray:
    """Evaluate a (4, 4, 3) bicubic control grid at ``size×size`` pixels.

    Returns ``(size, size, 3)`` float32 in [0, 1].
    """
    B = _bernstein4_basis(size)               # (size, 4)
    out = np.empty((size, size, 3), dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = (B @ control_grid[:, :, ch].astype(np.float64) @ B.T).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def render_image(
    control_grids: np.ndarray,
    quad_tiles: list[Any],
    *,
    width: int,
    height: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Reconstruct a full image from per-tile bicubic control grids.

    ``control_grids`` shape ``(n_tiles, 4, 4, 3)`` aligned with ``quad_tiles``
    in the same order they were emitted by ``fit_tiles``. Returns the
    reconstructed ``(height, width, 3)`` float32 image in [0, 1].

    The tiles may extend past ``(height, width)`` because the encoder
    quadtree pads the source image up to a multiple of ``TILE_SIZE_MAX``
    before splitting. We render into an oversized canvas that covers the
    tile extents and then crop back to the requested output dimensions.

    ``target_width`` / ``target_height``: when provided and different
    from ``width`` / ``height``, each tile is evaluated at a scaled
    pixel size directly from the Bernstein basis. Bicubic patches are
    analytic in normalized [0, 1] control-point space so any output
    size reads off the same underlying surface — no bilinear upscale
    artifacts. Set to None (default) for bit-identical legacy behavior.
    """
    if control_grids.shape[0] != len(quad_tiles):
        raise ValueError(
            f"control grid count {control_grids.shape[0]} does not match "
            f"quad tile count {len(quad_tiles)}"
        )
    target_w = int(target_width) if target_width is not None else width
    target_h = int(target_height) if target_height is not None else height
    if not quad_tiles:
        return np.zeros((target_h, target_w, 3), dtype=np.float32)
    scale_x = target_w / float(width)
    scale_y = target_h / float(height)
    scaled = not (target_w == width and target_h == height)
    canvas_h = max(target_h, int(round(max((qt.y + qt.size) for qt in quad_tiles) * scale_y)))
    canvas_w = max(target_w, int(round(max((qt.x + qt.size) for qt in quad_tiles) * scale_x)))
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    for cg, qt in zip(control_grids, quad_tiles):
        if scaled:
            tx0 = int(round(qt.x * scale_x))
            ty0 = int(round(qt.y * scale_y))
            tx1 = int(round((qt.x + qt.size) * scale_x))
            ty1 = int(round((qt.y + qt.size) * scale_y))
            sz_x = max(1, tx1 - tx0)
            sz_y = max(1, ty1 - ty0)
            tile_px = max(sz_x, sz_y)
            tile = eval_tile(cg, tile_px)
            out[ty0:ty0 + sz_y, tx0:tx0 + sz_x] = tile[:sz_y, :sz_x]
        else:
            out[qt.y:qt.y + qt.size, qt.x:qt.x + qt.size] = eval_tile(cg, qt.size)
    return out[:target_h, :target_w]
