"""Deterministic software renderer for WEFT primitives.

This module is the reference path for encode scoring and decode fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from .constants import TILE_SIZE
from .primitives import Primitive, TileRecord


@dataclass(slots=True)
class RenderConfig:
    tile_size: int = TILE_SIZE
    deblock: bool = True


_grid_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_scaled_grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _grid(tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) pixel sample positions for a ``tile_size``-side tile.

    Sampling at pixel CENTERS (half-integer positions: 0.5, 1.5, …, tile_size-0.5)
    is the standard graphics convention. Critically, it ensures that samples
    are never exactly on primitive edges when primitives are stored at
    integer-ish coordinates — which would otherwise cause float32 sign-flip
    instability in the triangle barycentric-coverage test and produce
    alternating 0/1 pixel patterns along triangle edges.

    The GPU primitive-decode kernel (``gpu_render.decode_tile_pixels``)
    samples at the same half-integer positions, so both renderers produce
    bit-equivalent output on the same bitstream.
    """
    cached = _grid_cache.get(tile_size)
    if cached is not None:
        return cached
    base = np.arange(tile_size, dtype=np.float32) + 0.5
    ys, xs = np.meshgrid(base, base, indexing="ij")
    _grid_cache[tile_size] = (xs, ys)
    return xs, ys


def _scaled_grid(source_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) that sample the [0, ``source_size``] primitive
    coordinate space at ``output_size``-resolution.

    Used by the scale-independent rendering path: primitives in a tile
    are stored in ``[0, source_size]`` pixel-space coordinates (the
    encoder's convention), but at decode time we want to resample that
    same geometry onto a larger or smaller output grid. This grid
    places ``output_size`` pixel centers uniformly across the
    ``[0, source_size]`` primitive coordinate range so that a 2×
    target gives twice the samples per primitive unit.

    When ``output_size == source_size`` this reduces to ``_grid`` and
    produces bit-identical output.
    """
    key = (source_size, output_size)
    cached = _scaled_grid_cache.get(key)
    if cached is not None:
        return cached
    if output_size == source_size:
        xs, ys = _grid(source_size)
    else:
        step = float(source_size) / float(output_size)
        base = (np.arange(output_size, dtype=np.float32) + 0.5) * step
        ys, xs = np.meshgrid(base, base, indexing="ij")
    _scaled_grid_cache[key] = (xs, ys)
    return xs, ys


def _blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return src * alpha[..., None] + dst * (1.0 - alpha[..., None])


def _point_line_distance(px: np.ndarray, py: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    vx = x1 - x0
    vy = y1 - y0
    den = vx * vx + vy * vy + 1e-8
    t = ((px - x0) * vx + (py - y0) * vy) / den
    t = np.clip(t, 0.0, 1.0)
    cx = x0 + t * vx
    cy = y0 + t * vy
    dx = px - cx
    dy = py - cy
    return np.sqrt(dx * dx + dy * dy)


def _eval_curve_distance(px: np.ndarray, py: np.ndarray, p0: tuple[float, float], c: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    pts = []
    for t in ts:
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * c[0] + t * t * p1[0]
        y = u * u * p0[1] + 2.0 * u * t * c[1] + t * t * p1[1]
        pts.append((x, y))

    best = np.full_like(px, 1e9, dtype=np.float32)
    for i in range(len(pts) - 1):
        d = _point_line_distance(px, py, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        best = np.minimum(best, d)
    return best


def _inside_triangle(px: np.ndarray, py: np.ndarray, tri: tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1, x2, y2 = tri
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(den) < 1e-6:
        return np.zeros_like(px, dtype=np.float32)

    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
    c = 1.0 - a - b
    inside = (a >= 0.0) & (b >= 0.0) & (c >= 0.0)
    return inside.astype(np.float32)


def render_tile(
    primitives: Iterable[Primitive],
    tile_size: int = TILE_SIZE,
    residual_rgb: tuple[int, int, int] | None = None,
    source_size: int | None = None,
) -> np.ndarray:
    """Render a tile's primitives at ``tile_size``-pixel resolution.

    When ``source_size`` is given and differs from ``tile_size``, the
    renderer samples the tile at ``tile_size`` pixels but maps those
    samples onto the ``[0, source_size]`` coordinate space of the
    primitives. This is the scale-independent rendering path: set
    ``source_size`` to the tile's encoded size and ``tile_size`` to
    the target (larger or smaller) size, and primitives — which are
    analytic — render crisply at any resolution.

    When ``source_size is None`` the function reduces to the legacy
    "render at physical pixel size" behavior and is bit-identical to
    the pre-scaling implementation.
    """
    if source_size is None:
        source_size = tile_size
    xs, ys = _scaled_grid(source_size, tile_size)
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    prims = list(primitives)
    if not prims:
        if residual_rgb is not None:
            tile += np.array(residual_rgb, dtype=np.float32)[None, None, :] / 255.0
        return np.clip(tile, 0.0, 1.0)

    # Per-tile bicubic fast path: if the tile is exactly one PRIM_BICUBIC
    # primitive (the per-tile-hybrid encoder picked bicubic over a
    # primitive stack for this tile), evaluate the Bernstein basis at
    # the pixel grid and return. The 48 geom values are the (4, 4, 3)
    # row-major control points in [0, 1] linear RGB; we delegate to
    # the bicubic module so the basis matrices are lru-cached across
    # tile sizes. Bicubic uses normalized [0, 1] control points so
    # the ``source_size`` parameter doesn't apply — any tile_size
    # renders the same analytic surface.
    if len(prims) == 1 and prims[0].kind == 5:  # PRIM_BICUBIC
        from .bicubic import eval_tile as _bicubic_eval
        cp = np.asarray(prims[0].geom, dtype=np.float32).reshape(4, 4, 3)
        tile = _bicubic_eval(cp, tile_size)
        if residual_rgb is not None:
            tile += np.array(residual_rgb, dtype=np.float32)[None, None, :] / 255.0
        return np.clip(tile, 0.0, 1.0)

    # Fast path: if all primitives are const patches, vectorize the entire composite.
    if all(p.kind == 0 for p in prims):
        for prim in prims:
            a = max(0.0, min(1.0, prim.alpha))
            c = np.array(prim.color0, dtype=np.float32)
            tile = c[None, None, :] * a + tile * (1.0 - a)
        if residual_rgb is not None:
            tile += np.array(residual_rgb, dtype=np.float32)[None, None, :] / 255.0
        return np.clip(tile, 0.0, 1.0)

    for prim in prims:
        a = np.float32(max(0.0, min(1.0, prim.alpha)))
        c0r, c0g, c0b = prim.color0

        if prim.kind == 0:  # const patch — inline for speed
            tile[:, :, 0] = c0r * a + tile[:, :, 0] * (1.0 - a)
            tile[:, :, 1] = c0g * a + tile[:, :, 1] * (1.0 - a)
            tile[:, :, 2] = c0b * a + tile[:, :, 2] * (1.0 - a)
            continue

        if prim.kind == 1:  # linear patch
            x0, y0, x1, y1 = prim.geom
            vx, vy = x1 - x0, y1 - y0
            den = vx * vx + vy * vy + 1e-8
            t = np.clip(((xs - x0) * vx + (ys - y0) * vy) / den, 0.0, 1.0)
            c1 = prim.color1 if prim.color1 is not None else prim.color0
            t3 = t[..., None]
            color0 = np.array(prim.color0, dtype=np.float32)
            color1 = np.array(c1, dtype=np.float32)
            src = color0 * (1.0 - t3) + color1 * t3
            alpha = np.full((tile_size, tile_size), a, dtype=np.float32)
        elif prim.kind == 2:  # line
            x0, y0, x1, y1, thickness = prim.geom
            dist = _point_line_distance(xs, ys, x0, y0, x1, y1)
            sigma = max(0.5, float(thickness))
            inv_2s2 = 1.0 / (2.0 * sigma * sigma)
            alpha = np.exp(-(dist * dist) * inv_2s2, dtype=np.float32) * a
            src = np.empty_like(tile)
            src[:, :, 0] = c0r; src[:, :, 1] = c0g; src[:, :, 2] = c0b
        elif prim.kind == 3:  # quadratic curve
            x0, y0, cx, cy, x1, y1, thickness = prim.geom
            dist = _eval_curve_distance(xs, ys, (x0, y0), (cx, cy), (x1, y1))
            sigma = max(0.5, float(thickness))
            inv_2s2 = 1.0 / (2.0 * sigma * sigma)
            alpha = np.exp(-(dist * dist) * inv_2s2, dtype=np.float32) * a
            src = np.empty_like(tile)
            src[:, :, 0] = c0r; src[:, :, 1] = c0g; src[:, :, 2] = c0b
        elif prim.kind == 4:  # triangle polygon
            alpha = _inside_triangle(xs, ys, prim.geom).astype(np.float32) * a
            src = np.empty_like(tile)
            src[:, :, 0] = c0r; src[:, :, 1] = c0g; src[:, :, 2] = c0b
        else:
            continue

        a3 = alpha[..., None]
        tile = src * a3 + tile * (1.0 - a3)

    if residual_rgb is not None:
        tile += np.array(residual_rgb, dtype=np.float32)[None, None, :] / 255.0

    return np.clip(tile, 0.0, 1.0)


def render_scene_tiled(
    tiles: list[TileRecord],
    width: int,
    height: int,
    tile_size: int = TILE_SIZE,
    deblock: bool = True,
    residual_maps: list[np.ndarray] | None = None,
    res1_grid_size: int = 4,
) -> np.ndarray:
    cols = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    if len(tiles) != cols * rows:
        raise ValueError("tile count mismatch")

    out = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.float32)
    idx = 0
    for ty in range(rows):
        for tx in range(cols):
            tile = tiles[idx]
            tile_img = render_tile(tile.primitives, tile_size=tile_size, residual_rgb=tile.residual_rgb)
            if residual_maps is not None:
                tile_img = np.clip(tile_img + upsample_residual_map(residual_maps[idx], tile_size=tile_size, grid_size=res1_grid_size), 0.0, 1.0)
            y0 = ty * tile_size
            x0 = tx * tile_size
            out[y0 : y0 + tile_size, x0 : x0 + tile_size, :] = tile_img
            idx += 1

    out = out[:height, :width, :]
    if deblock:
        out = deblock_image(out, tile_size)
    return np.clip(out, 0.0, 1.0)


def render_scene_adaptive(
    records: list[TileRecord],
    quad_tiles: list,
    width: int,
    height: int,
    residual_maps: list[np.ndarray] | None = None,
    res1_grid_size: int = 4,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Render a scene with variable-size adaptive tiles (quadtree layout).

    *quad_tiles* is a list of QuadTile objects (from weft.quadtree) that
    specify each tile's (x, y, size).

    ``target_width`` / ``target_height``: when provided and different
    from ``width`` / ``height``, enables scale-independent rendering.
    Each quad tile is rendered at a scaled pixel size
    (``qt.size * target_width / width``) with the primitive coordinate
    space preserved, so analytic primitives (linear ramps, lines,
    curves, triangles, bicubic patches) stay crisp at any target
    resolution. When target==source, this is bit-identical to the
    legacy path.

    Tiles whose primitive lists include the raster-based RES1 residual
    grid still pay for that grid at the target resolution via
    ``upsample_residual_map``, which does a simple bilinear upsample
    from the 4×4 grid — adequate because RES1 is low-frequency by
    construction.
    """
    from .constants import TILE_SIZE_MAX

    target_w = int(target_width) if target_width is not None else width
    target_h = int(target_height) if target_height is not None else height
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target dimensions must be > 0")

    scale_x = target_w / float(width)
    scale_y = target_h / float(height)
    # For now we require isotropic scaling since tile sizes are square.
    # Non-uniform targets would need rectangular tiles downstream.
    scaled = not (target_w == width and target_h == height)

    pad_target_h = (TILE_SIZE_MAX - target_h % TILE_SIZE_MAX) % TILE_SIZE_MAX
    pad_target_w = (TILE_SIZE_MAX - target_w % TILE_SIZE_MAX) % TILE_SIZE_MAX
    out = np.zeros(
        (target_h + pad_target_h, target_w + pad_target_w, 3),
        dtype=np.float32,
    )

    for i, (rec, qt) in enumerate(zip(records, quad_tiles)):
        if not scaled:
            tile_img = render_tile(rec.primitives, tile_size=qt.size, residual_rgb=rec.residual_rgb)
            tx0, ty0 = qt.x, qt.y
            ts_x, ts_y = qt.size, qt.size
        else:
            # Compute integer target rect for this tile. Using
            # rounding rather than floor/ceil so adjacent tiles tile
            # without gaps or overlaps when scales aren't integers.
            tx0 = int(round(qt.x * scale_x))
            ty0 = int(round(qt.y * scale_y))
            tx1 = int(round((qt.x + qt.size) * scale_x))
            ty1 = int(round((qt.y + qt.size) * scale_y))
            ts_x = max(1, tx1 - tx0)
            ts_y = max(1, ty1 - ty0)
            # Use the larger dimension as tile_size (square render),
            # then crop. For isotropic scaling ts_x == ts_y.
            tile_px = max(ts_x, ts_y)
            tile_img = render_tile(
                rec.primitives,
                tile_size=tile_px,
                residual_rgb=rec.residual_rgb,
                source_size=qt.size,
            )
            # Crop to the exact integer rect if anisotropic.
            if ts_x != tile_px or ts_y != tile_px:
                tile_img = tile_img[:ts_y, :ts_x, :]

        if residual_maps is not None and i < len(residual_maps):
            tile_img = np.clip(
                tile_img + upsample_residual_map(
                    residual_maps[i], tile_size=tile_img.shape[0], grid_size=res1_grid_size,
                ),
                0.0, 1.0,
            )

        out[ty0:ty0 + tile_img.shape[0], tx0:tx0 + tile_img.shape[1], :] = tile_img

    return np.clip(out[:target_h, :target_w, :], 0.0, 1.0)


def deblock_image(img: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    out = img.copy()
    h, w, _ = out.shape

    # Light seam smoothing at tile boundaries.
    for x in range(tile_size, w, tile_size):
        xl = x - 1
        xr = x
        avg = 0.5 * (out[:, xl, :] + out[:, xr, :])
        out[:, xl, :] = avg
        out[:, xr, :] = avg

    for y in range(tile_size, h, tile_size):
        yt = y - 1
        yb = y
        avg = 0.5 * (out[yt, :, :] + out[yb, :, :])
        out[yt, :, :] = avg
        out[yb, :, :] = avg

    return out


def render_scene_upscaled(
    tiles: list[TileRecord],
    source_width: int,
    source_height: int,
    out_width: int,
    out_height: int,
    tile_size: int = TILE_SIZE,
    deterministic_seed: int = 0,
    residual_maps: list[np.ndarray] | None = None,
    res1_grid_size: int = 4,
) -> np.ndarray:
    scale_x = out_width / float(source_width)
    scale_y = out_height / float(source_height)
    max_scale = max(scale_x, scale_y)
    spp = 4 if max_scale > 1.5 else 1

    base = render_scene_tiled(
        tiles,
        source_width,
        source_height,
        tile_size=tile_size,
        deblock=True,
        residual_maps=residual_maps,
        res1_grid_size=res1_grid_size,
    )
    if out_width == source_width and out_height == source_height:
        return base

    if spp == 1:
        # Bilinear upscale for <=1.5x while retaining deterministic path.
        ys = np.linspace(0.0, source_height - 1, out_height, dtype=np.float32)
        xs = np.linspace(0.0, source_width - 1, out_width, dtype=np.float32)
        yi = np.floor(ys).astype(np.int32)
        xi = np.floor(xs).astype(np.int32)
        yf = ys - yi
        xf = xs - xi
        yi1 = np.clip(yi + 1, 0, source_height - 1)
        xi1 = np.clip(xi + 1, 0, source_width - 1)

        out = np.empty((out_height, out_width, 3), dtype=np.float32)
        for oy, (y0, y1, wy) in enumerate(zip(yi, yi1, yf)):
            row0 = base[y0]
            row1 = base[y1]
            c00 = row0[xi]
            c01 = row0[xi1]
            c10 = row1[xi]
            c11 = row1[xi1]
            c0 = c00 * (1.0 - xf[:, None]) + c01 * xf[:, None]
            c1 = c10 * (1.0 - xf[:, None]) + c11 * xf[:, None]
            out[oy] = c0 * (1.0 - wy) + c1 * wy
        return np.clip(out, 0.0, 1.0)

    # 4-spp deterministic jittered supersample.
    rng = np.random.default_rng(deterministic_seed)
    jitter = np.array([[0.125, 0.375], [0.625, 0.125], [0.375, 0.875], [0.875, 0.625]], dtype=np.float32)
    rng.shuffle(jitter)

    out = np.zeros((out_height, out_width, 3), dtype=np.float32)
    for jx, jy in jitter:
        ys = (np.arange(out_height, dtype=np.float32) + jy) / out_height * source_height
        xs = (np.arange(out_width, dtype=np.float32) + jx) / out_width * source_width
        ys = np.clip(ys, 0.0, source_height - 1.0)
        xs = np.clip(xs, 0.0, source_width - 1.0)

        yi = np.floor(ys).astype(np.int32)
        xi = np.floor(xs).astype(np.int32)
        yi1 = np.clip(yi + 1, 0, source_height - 1)
        xi1 = np.clip(xi + 1, 0, source_width - 1)
        yf = ys - yi
        xf = xs - xi

        for oy, (y0, y1, wy) in enumerate(zip(yi, yi1, yf)):
            row0 = base[y0]
            row1 = base[y1]
            c00 = row0[xi]
            c01 = row0[xi1]
            c10 = row1[xi]
            c11 = row1[xi1]
            c0 = c00 * (1.0 - xf[:, None]) + c01 * xf[:, None]
            c1 = c10 * (1.0 - xf[:, None]) + c11 * xf[:, None]
            out[oy] += c0 * (1.0 - wy) + c1 * wy

    out *= 1.0 / float(spp)
    return np.clip(out, 0.0, 1.0)


def decode_hash(image_linear_rgb: np.ndarray) -> str:
    digest = hashlib.sha256(np.clip(image_linear_rgb, 0.0, 1.0).astype(np.float32).tobytes()).hexdigest()
    return digest


def upsample_residual_map(residual_map: np.ndarray, tile_size: int, grid_size: int = 4) -> np.ndarray:
    """Upsample a low-res residual map to tile size via bilinear interpolation.

    The encoder (:func:`weft.encoder._quantize_residual_map`) builds each grid
    cell as the mean of a ``(tile_size/grid_size) × (tile_size/grid_size)``
    block of pixel-center error values. For half-pixel sampling, grid cell
    ``(gy, gx)`` therefore represents the error at tile-local pixel center::

        center = (gy * tile_size/grid_size + tile_size/(2*grid_size),
                  gx * tile_size/grid_size + tile_size/(2*grid_size))

    To sample the residual at pixel center ``(py+0.5, px+0.5)``, we solve for
    the grid-space coordinate ``fx = px_center * grid_size/tile_size - 0.5``
    and bilinearly interpolate. Pixels outside the cell-center range
    (corners of the tile) are clamped to the edge of the grid.

    """
    if residual_map.shape != (grid_size, grid_size, 3):
        raise ValueError("invalid residual map shape")

    pixel_centers = np.arange(tile_size, dtype=np.float32) + 0.5
    raw = pixel_centers * (float(grid_size) / float(tile_size)) - 0.5
    clamped = np.clip(raw, 0.0, grid_size - 1)
    xi = np.floor(clamped).astype(np.int32)
    xf = clamped - xi
    xi1 = np.clip(xi + 1, 0, grid_size - 1)
    yi, yi1, yf = xi, xi1, xf  # square tile

    # Fancy-index all four corners in one shot, then bilinear-blend.
    # Shapes:
    #   c?? : (tile_size, tile_size, 3)
    #   wx  : (1, tile_size, 1)
    #   wy  : (tile_size, 1, 1)
    c00 = residual_map[yi[:, None], xi[None, :]]
    c01 = residual_map[yi[:, None], xi1[None, :]]
    c10 = residual_map[yi1[:, None], xi[None, :]]
    c11 = residual_map[yi1[:, None], xi1[None, :]]
    wx = xf[None, :, None]
    wy = yf[:, None, None]
    c0 = c00 * (1.0 - wx) + c01 * wx
    c1 = c10 * (1.0 - wx) + c11 * wx
    return c0 * (1.0 - wy) + c1 * wy
