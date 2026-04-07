"""WEFT encoder implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import functools
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .bitstream import HeadBlock, encode_weft
from .constants import (
    DEFAULT_PRESET,
    FLAG_CHUNKED_PRIM,
    FLAG_DETERMINISTIC,
    FLAG_HAS_BIC,
    FLAG_HAS_DCT,
    FLAG_HAS_GRD,
    FLAG_HAS_LITE,
    FLAG_HAS_PAL,
    FLAG_HAS_RES0,
    FLAG_HAS_RES1,
    QUALITY_MAX,
    QUALITY_MIN,
    TILE_SIZE,
)
from .cuda_backend import cccl_argmin, cccl_compact, cccl_segmented_topk, detect_gpu_stack
from .entropy import encode_bytes
from .feature_flags import FeatureFlags
from .gpu_encoder import GpuEncodeError, fit_tiles_gpu_constant
from .gpu_score import batch_mse
from .image_io import load_image_linear
from .metrics import psnr, ssim
from .prim_chunks import build_prim_chunks
from .primitives import Primitive, TileRecord, decode_primitive, decode_tiles, encode_primitive, encode_tiles
from .prim_streams import build_primitive_side_streams
from .render import decode_hash, render_scene_tiled, render_tile, upsample_residual_map
from .edge_analysis import generate_edge_driven_candidates
from .types import EncodeConfig, EncodeReport


class EncodeError(ValueError):
    pass


@dataclass
class _AdaptiveFitState:
    """Cached output of the (slow) primitive-fit phase of the adaptive
    encoder.

    The greedy primitive search is by far the most expensive thing in
    ``_encode_image_adaptive`` — for a 1024×1024 input it dominates the
    encode time at 30+ seconds. Multiple auto-select variants
    (``baseline``, ``hybrid``, ``hybrid-dct``, ``hybrid-dct-tight``) all
    start from the same primitive fit and only differ in what they
    layer on top (per-tile bicubic R-D check, RES1 grid, DCT residual,
    etc.). Caching the fit and reusing it across those variants cuts
    the auto-select wall time roughly in half.

    Fields here are the minimal "frozen" state needed by
    ``_build_bitstream_from_fit``: image data, quadtree geometry, the
    selected primitive list per tile, and the optional decompose-
    lighting outputs (which are also fit-shared because the
    decomposition runs once on the input image).

    The cache key for the optional disk store is a hash of the input
    file bytes plus the fit-affecting cfg fields (``quality``,
    ``encode_scale``, ``decompose_lighting``, ``lighting_grid_size``,
    ``preset``, ``deterministic``). Build-only fields like
    ``hybrid_bicubic_per_tile`` and ``dct_residual`` deliberately do
    NOT enter the key — that's the whole point.
    """
    # Image state
    source_image: np.ndarray
    source_width: int
    source_height: int
    image: np.ndarray  # working image (post-decompose, post-scale)
    width: int
    height: int
    encode_scale: float
    # Quadtree + tile geometry
    quad_tiles: list  # list[QuadTile]
    tile_patches: list  # list[np.ndarray]
    # Greedy primitive search result
    selected_per_tile: list  # list[list[Primitive]]
    # Bicubic-replaced version (computed eagerly because 3 of the 4
    # primitive-family auto-select variants need it). Variants that
    # don't use bicubic just ignore this field.
    bicubic_replaced_per_tile: list  # list[list[Primitive]]
    n_bicubic_tiles: int
    # Fit-time settings (re-derivable from cfg but cached for speed)
    quality: int
    lam: float
    split_threshold: float
    res1_grid: int
    # Optional decompose-lighting state
    lighting_grid: np.ndarray | None
    lite_payload: bytes | None
    # Timing
    cand_gen_ms: float
    fit_ms: float


def _fit_cache_key(input_path: str, cfg: EncodeConfig) -> str:
    """Compute the disk-cache key for an adaptive fit state.

    Hashes the input file bytes plus the cfg fields that influence the
    fit phase (NOT the build phase). Build-only fields like
    ``hybrid_bicubic_per_tile``, ``dct_residual``, ``enable_res0/1``,
    ``entropy``, etc. are excluded so a single cache entry serves all
    variants in an auto-select sweep.
    """
    h = hashlib.sha256()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    fit_relevant = {
        "quality": int(cfg.quality),
        "encode_scale": float(cfg.encode_scale),
        "preset": str(cfg.preset),
        "deterministic": bool(cfg.deterministic),
        "decompose_lighting": bool(flags.decompose_lighting),
        "lighting_grid_size": int(flags.lighting_grid_size),
        # Cache version: bump if FitState fields or fit logic changes.
        "_fit_version": 1,
    }
    h.update(json.dumps(fit_relevant, sort_keys=True).encode())
    return h.hexdigest()[:32]


def _load_fit_state(cache_path: Path) -> "_AdaptiveFitState | None":
    """Load a cached _AdaptiveFitState from disk; return None on any error."""
    try:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, _AdaptiveFitState):
            return obj
    except Exception:
        pass
    return None


def _save_fit_state(state: "_AdaptiveFitState", cache_path: Path) -> None:
    """Persist a fit state to disk for later reuse. Failures are silent."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    except Exception:
        pass


def _clamp_quality(quality: int) -> int:
    return max(QUALITY_MIN, min(QUALITY_MAX, int(quality)))


def _tile_lambda(quality: int) -> float:
    # Higher quality -> lower rate pressure.
    # At q>=90, use ultra-low lambda to let the greedy search fill tiles aggressively.
    q = _clamp_quality(quality)
    base = ((101 - q) / 100.0) ** 2
    if q >= 90:
        return base * 0.0002  # ~7x lower than before at q=95
    return base * 0.001


def _pad_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    h, w, _ = tile.shape
    if h == tile_size and w == tile_size:
        return tile
    pad_h = tile_size - h
    pad_w = tile_size - w
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _snap_u8(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    x = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return lo + (round(x * 255.0) / 255.0) * (hi - lo)


def _snap_u16(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    x = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return lo + (round(x * 65535.0) / 65535.0) * (hi - lo)


# Precomputed lookup: avoids repeated function call overhead.
_U8_GRID = np.round(np.arange(256) / 255.0 * 255.0) / 255.0  # identity, but float
_U16_GRID_15 = np.round(np.linspace(0, 15, 65536) * 65535.0 / 15.0) / 65535.0 * 15.0


def _snap_color(c: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(max(0, min(1, c[0])) * 255) / 255.0,
            round(max(0, min(1, c[1])) * 255) / 255.0,
            round(max(0, min(1, c[2])) * 255) / 255.0)


def _quantize_primitive(prim: Primitive) -> Primitive:
    """Snap primitive values to quantization grid without serializing."""
    kind = prim.kind
    alpha = round(max(0, min(1, prim.alpha)) * 65535) / 65535.0

    if kind == 0:
        geom = prim.geom
    elif kind == 2:
        g = prim.geom
        geom = (round(max(0, min(1, g[0] / 15)) * 65535) / 65535.0 * 15,
                round(max(0, min(1, g[1] / 15)) * 65535) / 65535.0 * 15,
                round(max(0, min(1, g[2] / 15)) * 65535) / 65535.0 * 15,
                round(max(0, min(1, g[3] / 15)) * 65535) / 65535.0 * 15,
                round(max(0, min(1, g[4] / 4)) * 65535) / 65535.0 * 4)
    elif kind == 3:
        g = prim.geom
        geom = tuple(round(max(0, min(1, v / 15)) * 65535) / 65535.0 * 15 for v in g[:6])
        geom = geom + (round(max(0, min(1, g[6] / 4)) * 65535) / 65535.0 * 4,)
    else:
        geom = tuple(round(max(0, min(1, v / 15)) * 65535) / 65535.0 * 15 for v in prim.geom)

    return Primitive(
        kind=kind, geom=geom,
        color0=_snap_color(prim.color0),
        color1=_snap_color(prim.color1) if prim.color1 is not None else None,
        alpha=alpha,
    )


# Payload sizes per primitive type (uint16 coords/alpha, uint8 colors).
# header (2B: kind + payload_len) + payload bytes per type.
_PRIM_BYTE_SIZES = {
    0: 2 + 5,    # CONST_PATCH: header + color(3) + alpha(2)
    1: 2 + 16,   # LINEAR_PATCH: header + coords(8) + color0(3) + color1(3) + alpha(2)
    2: 2 + 15,   # LINE: header + geom(10) + color(3) + alpha(2)
    3: 2 + 19,   # QUAD_CURVE: header + geom(14) + color(3) + alpha(2)
    4: 2 + 17,   # POLYGON: header + geom(12) + color(3) + alpha(2)
}

# Lookup table for vectorized rate computation in the greedy loop.
# Index = primitive kind; value = byte size for that kind. Unknown
# kinds (including PRIM_BICUBIC = 5) fall back to 20.
_PRIM_BYTE_SIZES_LUT = np.array(
    [_PRIM_BYTE_SIZES.get(k, 20) for k in range(16)],
    dtype=np.int32,
)


def _estimate_bits(primitives: list[Primitive], include_residual: bool, include_res1: bool = False, res1_grid_size: int = 4) -> int:
    # 1 byte for primitive count + per-primitive byte sizes.
    bits = 8  # prim_count byte
    for p in primitives:
        bits += _PRIM_BYTE_SIZES.get(p.kind, 20) * 8
    if include_residual:
        bits += 24
    if include_res1:
        bits += res1_grid_size * res1_grid_size * 3 * 8
    return bits


def _tile_objective(
    src: np.ndarray,
    primitives: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    residual_map: np.ndarray | None = None,
    res1_grid_size: int = 4,
) -> tuple[float, np.ndarray]:
    pred = render_tile(primitives, tile_size=src.shape[0], residual_rgb=residual)
    if residual_map is not None:
        pred = np.clip(pred + upsample_residual_map(residual_map, tile_size=src.shape[0], grid_size=res1_grid_size), 0.0, 1.0)
    dist = float(np.mean((src - pred) ** 2))
    include_res = residual != (0, 0, 0)
    rate = _estimate_bits(primitives, include_res, include_res1=(residual_map is not None), res1_grid_size=res1_grid_size)
    return dist + lam * rate, pred


def _quantize_residual_map(error: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Quantize full-res error tile into low-res int8 residual map (linear RGB)."""
    h, w, _ = error.shape
    if h % grid_size != 0 or w % grid_size != 0:
        raise EncodeError("tile size must be divisible by RES1 grid size")
    sh = h // grid_size
    sw = w // grid_size
    out = np.empty((grid_size, grid_size, 3), dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            patch = error[gy * sh : (gy + 1) * sh, gx * sw : (gx + 1) * sw, :]
            out[gy, gx, :] = patch.mean(axis=(0, 1))
    q = np.clip(np.round(out * 255.0), -127.0, 127.0).astype(np.int8)
    return q.astype(np.float32) / 255.0


def _res1_map_bytes(residual_map: np.ndarray) -> bytes:
    q = np.clip(np.round(residual_map * 255.0), -127.0, 127.0).astype(np.int8)
    return q.reshape(-1).tobytes()


@functools.lru_cache(maxsize=8)
def _coord_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached (ys, xs) integer-coordinate grids for a tile of size (h, w).

    The same grids are reused for every line tested in ``_split_by_line``
    and ``_split_by_lines_batched``, and tile sizes only take a handful
    of distinct values (8, 16, 32 in adaptive mode), so an LRU cache of
    8 entries covers everything.
    """
    return np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )


def _split_by_line(tile: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    h, w, _ = tile.shape
    ys, xs = _coord_grid(h, w)
    side = (xs - x0) * (y1 - y0) - (ys - y0) * (x1 - x0)
    a_mask = side >= 0
    b_mask = ~a_mask

    if np.any(a_mask):
        ca = tile[a_mask].mean(axis=0)
    else:
        ca = tile.mean(axis=(0, 1))
    if np.any(b_mask):
        cb = tile[b_mask].mean(axis=0)
    else:
        cb = tile.mean(axis=(0, 1))
    return (float(ca[0]), float(ca[1]), float(ca[2])), (float(cb[0]), float(cb[1]), float(cb[2]))


def _split_by_lines_batched(
    tile: np.ndarray,
    lines: np.ndarray,
) -> np.ndarray:
    """Vectorized batch of ``_split_by_line`` calls on a single tile.

    Parameters
    ----------
    tile : (h, w, 3) float32
    lines : (N, 4) float32 — each row is (x0, y0, x1, y1)

    Returns
    -------
    (N, 2, 3) float32 — colors[i, 0] is the "side >= 0" mean, colors[i, 1]
    is the "side < 0" mean. When one side has no pixels we fall back to
    the global tile mean for that line, matching the per-line function.

    Replaces ~48 individual ``_split_by_line`` calls per tile with one
    vectorized computation: builds an ``(N, h, w)`` side mask in one
    broadcast op and reduces via ``np.einsum`` so each color channel is
    computed in a single sweep.
    """
    h, w, _ = tile.shape
    n = lines.shape[0]
    if n == 0:
        return np.empty((0, 2, 3), dtype=np.float32)
    ys, xs = _coord_grid(h, w)             # (h, w) each
    x0 = lines[:, 0, None, None]           # (N, 1, 1) — broadcasts against (h, w)
    y0 = lines[:, 1, None, None]
    x1 = lines[:, 2, None, None]
    y1 = lines[:, 3, None, None]
    side = (xs - x0) * (y1 - y0) - (ys - y0) * (x1 - x0)   # (N, h, w)
    a_mask = (side >= 0).astype(np.float32)                # (N, h, w)
    b_mask = 1.0 - a_mask                                  # (N, h, w)

    a_count = a_mask.sum(axis=(1, 2))                      # (N,)
    b_count = b_mask.sum(axis=(1, 2))                      # (N,)

    # Per-channel weighted sum: einsum gives (N, 3).
    a_sum = np.einsum("nhw,hwc->nc", a_mask, tile)
    b_sum = np.einsum("nhw,hwc->nc", b_mask, tile)

    # Avoid divide-by-zero; we'll overwrite empty-side rows with the
    # global tile mean below.
    a_safe = np.maximum(a_count, 1.0)[:, None]
    b_safe = np.maximum(b_count, 1.0)[:, None]
    a_color = a_sum / a_safe
    b_color = b_sum / b_safe

    fallback = tile.mean(axis=(0, 1))
    a_empty = a_count == 0
    b_empty = b_count == 0
    if a_empty.any():
        a_color[a_empty] = fallback
    if b_empty.any():
        b_color[b_empty] = fallback

    return np.stack([a_color, b_color], axis=1).astype(np.float32, copy=False)


# =====================================================================
#  New helpers for expanded encoder
# =====================================================================

def _ct(a) -> tuple[float, float, float]:
    """Array-like to clamped colour tuple."""
    return (max(0.0, min(1.0, float(a[0]))),
            max(0.0, min(1.0, float(a[1]))),
            max(0.0, min(1.0, float(a[2]))))


def _kmeans_colors(pixels: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
    """Fast palette extraction via luminance-sorted equal-count splitting.

    Sort pixels by luminance once, then split into k equal-count groups
    and take the mean RGB of each group. Equivalent to the previous
    percentile-based version but O(n log n) once instead of O(k * n)
    sorts (np.percentile sorts internally).
    """
    n = len(pixels)
    if n <= k:
        out = np.zeros((k, 3), dtype=np.float32)
        out[:n] = pixels[:n]
        return out
    lum = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    order = np.argsort(lum)
    sorted_pix = pixels[order]
    centers = np.empty((k, 3), dtype=np.float32)
    # Equal-count slicing: split [0, n) into k contiguous groups.
    edges = (np.arange(k + 1, dtype=np.float64) * (n / k)).astype(np.int64)
    for j in range(k):
        a, b = int(edges[j]), int(edges[j + 1])
        if b <= a:
            centers[j] = sorted_pix[min(a, n - 1)]
        else:
            centers[j] = sorted_pix[a:b].mean(axis=0)
    return centers


def _prim_key(p: Primitive) -> tuple:
    """Fast dedup key for a quantized primitive (avoids serialization)."""
    return (p.kind, p.geom, p.color0, p.color1, p.alpha)


def _dedup_candidates(cands: list[Primitive]) -> list[Primitive]:
    """De-duplicate candidates by coarse-quantized tuple key.

    Quantization grids (matching the historical post-hoc shape):
      * geom   -> step 0.25  (×4)
      * colors -> step 1/64  (×64)
      * alpha  -> step 1/32  (×32)

    Specialized per primitive ``kind`` so the geom field is unrolled
    inline rather than walked via a generator expression — that
    generator was the single biggest pure-Python hot spot in the
    baseline encoder profile (~7.9M iterations / 0.61s on the
    ACCEPT-OURS profile). The unrolled inline form is ~3× faster
    overall on the dedup pass.

    Negative geom values (rare; off-centre linear-patch lines extend
    slightly outside [0, w]) use ``int(x + 0.5)`` directly, which is
    not strictly half-up rounding for negatives but is *consistent*
    — and consistency is all dedup needs.
    """
    seen: set[tuple] = set()
    out: list[Primitive] = []
    for c in cands:
        kind = c.kind
        c0 = c.color0
        c0r = int(c0[0] * 64 + 0.5)
        c0g = int(c0[1] * 64 + 0.5)
        c0b = int(c0[2] * 64 + 0.5)
        qa = int(c.alpha * 32 + 0.5)
        if kind == 0:                          # const_patch (no geom)
            key = (0, c0r, c0g, c0b, qa)
        elif kind == 1:                        # linear_patch — 4 geom + color1
            g = c.geom
            c1 = c.color1
            key = (
                1,
                int(g[0] * 4 + 0.5), int(g[1] * 4 + 0.5),
                int(g[2] * 4 + 0.5), int(g[3] * 4 + 0.5),
                c0r, c0g, c0b,
                int(c1[0] * 64 + 0.5), int(c1[1] * 64 + 0.5), int(c1[2] * 64 + 0.5),
                qa,
            )
        elif kind == 2:                        # line — 5 geom
            g = c.geom
            key = (
                2,
                int(g[0] * 4 + 0.5), int(g[1] * 4 + 0.5),
                int(g[2] * 4 + 0.5), int(g[3] * 4 + 0.5),
                int(g[4] * 4 + 0.5),
                c0r, c0g, c0b, qa,
            )
        elif kind == 3:                        # quad_curve — 7 geom
            g = c.geom
            key = (
                3,
                int(g[0] * 4 + 0.5), int(g[1] * 4 + 0.5),
                int(g[2] * 4 + 0.5), int(g[3] * 4 + 0.5),
                int(g[4] * 4 + 0.5), int(g[5] * 4 + 0.5),
                int(g[6] * 4 + 0.5),
                c0r, c0g, c0b, qa,
            )
        elif kind == 4:                        # polygon — 6 geom
            g = c.geom
            key = (
                4,
                int(g[0] * 4 + 0.5), int(g[1] * 4 + 0.5),
                int(g[2] * 4 + 0.5), int(g[3] * 4 + 0.5),
                int(g[4] * 4 + 0.5), int(g[5] * 4 + 0.5),
                c0r, c0g, c0b, qa,
            )
        else:
            # Unknown kind — fall back to the slow general path so
            # future primitive types still dedup correctly.
            key = (
                kind,
                tuple(int(g * 4 + 0.5) for g in c.geom),
                c0r, c0g, c0b, qa,
            )
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# =====================================================================
#  Candidate generation (expanded – ~100-200 unique primitives)
# =====================================================================

def _generate_candidates(tile: np.ndarray, quality: int = 75) -> list[Primitive]:
    """Build a rich pool of candidate primitives for one tile."""
    h, w, _ = tile.shape
    hf, wf = float(h - 1), float(w - 1)
    lum = 0.2126 * tile[..., 0] + 0.7152 * tile[..., 1] + 0.0722 * tile[..., 2]
    pixels = tile.reshape(-1, 3).astype(np.float32)
    gy, gx = np.gradient(lum)
    mag = np.sqrt(gx * gx + gy * gy)
    avg_mag = float(np.mean(mag))
    gmx, gmy = float(np.mean(gx)), float(np.mean(gy))
    gnorm = math.sqrt(gmx * gmx + gmy * gmy)

    cands: list[Primitive] = []

    # ── colour palette via k-means + spatial stats ──────────────────
    avg = _ct(tile.mean(axis=(0, 1)))
    palette: list[tuple[float, float, float]] = [avg]
    for k in (2, 3, 4):
        for c in _kmeans_colors(pixels, k):
            ct = _ct(c)
            if not any(sum((a - b) ** 2 for a, b in zip(ct, p)) < 0.002 for p in palette):
                palette.append(ct)

    p25, p75 = float(np.percentile(lum, 25)), float(np.percentile(lum, 75))
    bmask, dmask = lum > p75, lum < p25
    bright = _ct(tile[bmask].mean(axis=0)) if np.any(bmask) else avg
    dark = _ct(tile[dmask].mean(axis=0)) if np.any(dmask) else avg
    grad_thresh = float(np.percentile(mag, 75))
    emask = mag >= grad_thresh
    edge_color = _ct(tile[emask].mean(axis=0)) if np.any(emask) else avg
    for extra in (bright, dark, edge_color):
        if not any(sum((a - b) ** 2 for a, b in zip(extra, p)) < 0.002 for p in palette):
            palette.append(extra)

    base_alpha = min(1.0, 0.25 + avg_mag * 3.0)

    # ── CONST PATCHES ───────────────────────────────────────────────
    for color in palette:
        for a in (1.0, 0.7, 0.4, 0.2):
            cands.append(Primitive(kind=0, geom=(), color0=color, alpha=a))

    # Per-quadrant averages
    qh, qw = h // 2, w // 2
    for qr in range(2):
        for qc in range(2):
            qcolor = _ct(tile[qr * qh:(qr + 1) * qh, qc * qw:(qc + 1) * qw].mean(axis=(0, 1)))
            for a in (0.5, 0.3):
                cands.append(Primitive(kind=0, geom=(), color0=qcolor, alpha=a))

    # Sub-quadrant (4x4) spatial hints
    sh, sw = h // 4, w // 4
    for gr in range(4):
        for gc in range(4):
            sc = _ct(tile[gr * sh:(gr + 1) * sh, gc * sw:(gc + 1) * sw].mean(axis=(0, 1)))
            cands.append(Primitive(kind=0, geom=(), color0=sc, alpha=0.25))

    # ── LINEAR PATCHES ──────────────────────────────────────────────
    # Collect every line we want to split through, run one batched
    # _split_by_lines call, then materialize Primitives from the results.
    # Replaces ~48 individual _split_by_line calls per tile (one per
    # line × 1) with a single broadcast op.
    n_angles = 16 if quality >= 80 else 12 if quality >= 50 else 8
    line_specs: list[tuple[float, float, float, float, str]] = []
    # 'tag' encodes which downstream group each line belongs to so we
    # can rebuild the right Primitives after the batched split.
    for angle in np.linspace(0, math.pi, n_angles, endpoint=False):
        vx, vy = math.cos(angle), math.sin(angle)
        cx, cy = wf * 0.5, hf * 0.5
        line_specs.append((
            cx - vx * wf * 0.6, cy - vy * hf * 0.6,
            cx + vx * wf * 0.6, cy + vy * hf * 0.6,
            "centre",
        ))
    for ox, oy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                   (0.5, 0.25), (0.5, 0.75)):
        for angle in np.linspace(0, math.pi, 6, endpoint=False):
            vx, vy = math.cos(angle), math.sin(angle)
            cx, cy = wf * ox, hf * oy
            line_specs.append((
                cx - vx * wf * 0.4, cy - vy * hf * 0.4,
                cx + vx * wf * 0.4, cy + vy * hf * 0.4,
                "off_centre",
            ))

    if line_specs:
        line_arr = np.array([(x0, y0, x1, y1) for (x0, y0, x1, y1, _) in line_specs],
                            dtype=np.float32)
        colors = _split_by_lines_batched(tile, line_arr)   # (N, 2, 3)
        for (x0, y0, x1, y1, tag), pair in zip(line_specs, colors):
            c0 = (float(pair[0, 0]), float(pair[0, 1]), float(pair[0, 2]))
            c1 = (float(pair[1, 0]), float(pair[1, 1]), float(pair[1, 2]))
            if tag == "centre":
                for a in (1.0, 0.7, 0.4):
                    cands.append(Primitive(kind=1, geom=(x0, y0, x1, y1),
                                           color0=c0, color1=c1, alpha=a))
            else:
                cands.append(Primitive(kind=1, geom=(x0, y0, x1, y1),
                                       color0=c0, color1=c1, alpha=0.5))

    # ── LINES ───────────────────────────────────────────────────────
    thicknesses = (0.4, 0.8, 1.2, 2.0, 3.0)
    fracs = (0.0, 0.25, 0.5, 0.75, 1.0)
    for thickness in thicknesses:
        for frac in fracs:
            # vertical
            cands.append(Primitive(kind=2, geom=(wf * frac, 0, wf * frac, hf, thickness),
                                   color0=edge_color, alpha=base_alpha))
            # horizontal
            cands.append(Primitive(kind=2, geom=(0, hf * frac, wf, hf * frac, thickness),
                                   color0=edge_color, alpha=base_alpha))
        # diagonals with several colours
        for lc in (edge_color, bright, dark):
            cands.append(Primitive(kind=2, geom=(0, 0, wf, hf, thickness),
                                   color0=lc, alpha=base_alpha))
            cands.append(Primitive(kind=2, geom=(0, hf, wf, 0, thickness),
                                   color0=lc, alpha=base_alpha))

    # Gradient-directed lines
    if gnorm > 1e-4:
        lx, ly = -gmy / (gnorm + 1e-8), gmx / (gnorm + 1e-8)
        cx, cy = wf * 0.5, hf * 0.5
        gx0 = cx - lx * wf * 0.7
        gy0 = cy - ly * hf * 0.7
        gx1 = cx + lx * wf * 0.7
        gy1 = cy + ly * hf * 0.7
        for t in thicknesses:
            for lc in (edge_color, bright):
                cands.append(Primitive(kind=2, geom=(gx0, gy0, gx1, gy1, t),
                                       color0=lc, alpha=base_alpha))

    # ── QUADRATIC CURVES ────────────────────────────────────────────
    cp_fracs = (0.25, 0.5, 0.75)
    for cy_f in cp_fracs:
        for cx_f in cp_fracs:
            cpx, cpy = wf * cx_f, hf * cy_f
            for thickness in (0.5, 1.0, 2.0):
                cands.append(Primitive(kind=3,
                             geom=(0, hf * 0.5, cpx, cpy, wf, hf * 0.5, thickness),
                             color0=edge_color, alpha=base_alpha))
    for cx_f in cp_fracs:
        for cy_f in cp_fracs:
            cpx, cpy = wf * cx_f, hf * cy_f
            for thickness in (0.5, 1.0):
                cands.append(Primitive(kind=3,
                             geom=(wf * 0.5, 0, cpx, cpy, wf * 0.5, hf, thickness),
                             color0=edge_color, alpha=base_alpha))

    # ── POLYGONS (TRIANGLES) ────────────────────────────────────────
    corners = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)]
    center = (wf * 0.5, hf * 0.5)

    # Corner-centre triangles
    for i in range(4):
        v0, v1 = corners[i], corners[(i + 1) % 4]
        geom = (v0[0], v0[1], v1[0], v1[1], center[0], center[1])
        for color in palette[:3]:
            for a in (0.6, 0.3):
                cands.append(Primitive(kind=4, geom=geom, color0=color, alpha=a))

    # Half-tile triangles
    for gv in ((0, 0, wf, 0, wf, hf), (0, 0, wf, hf, 0, hf),
               (0, 0, wf, 0, 0, hf), (wf, 0, wf, hf, 0, hf)):
        geom = tuple(float(v) for v in gv)
        for a in (0.5, 0.3):
            cands.append(Primitive(kind=4, geom=geom, color0=avg, alpha=a))

    # Luminance-peak triangles
    flat_lum = lum.reshape(-1)
    n_peaks = min(6, len(flat_lum))
    peak_idx = np.argpartition(flat_lum, -n_peaks)[-n_peaks:]
    peak_ys, peak_xs = np.unravel_index(peak_idx, lum.shape)
    if n_peaks >= 3:
        for i in range(min(n_peaks - 2, 4)):
            geom = (float(peak_xs[i]), float(peak_ys[i]),
                    float(peak_xs[i + 1]), float(peak_ys[i + 1]),
                    float(peak_xs[i + 2]), float(peak_ys[i + 2]))
            tc = _ct(tile[peak_ys[i:i + 3], peak_xs[i:i + 3]].mean(axis=0))
            cands.append(Primitive(kind=4, geom=geom, color0=tc, alpha=0.5))

    # Valley (dark) triangles
    valley_idx = np.argpartition(flat_lum, n_peaks)[:n_peaks]
    valley_ys, valley_xs = np.unravel_index(valley_idx, lum.shape)
    if n_peaks >= 3:
        for i in range(min(n_peaks - 2, 4)):
            geom = (float(valley_xs[i]), float(valley_ys[i]),
                    float(valley_xs[i + 1]), float(valley_ys[i + 1]),
                    float(valley_xs[i + 2]), float(valley_ys[i + 2]))
            tc = _ct(tile[valley_ys[i:i + 3], valley_xs[i:i + 3]].mean(axis=0))
            cands.append(Primitive(kind=4, geom=geom, color0=tc, alpha=0.5))

    # K-means region triangles
    if quality >= 80:
        km3 = _kmeans_colors(pixels, 3)
        km_labels = np.argmin(
            np.sum((pixels[:, None, :] - km3[None, :, :]) ** 2, axis=2), axis=1,
        ).reshape(h, w)
        for lab in range(3):
            m = km_labels == lab
            ys_m, xs_m = np.where(m)
            if len(ys_m) >= 3:
                ix_min = np.argmin(xs_m)
                ix_max = np.argmax(xs_m)
                cxm, cym = float(np.mean(xs_m)), float(np.mean(ys_m))
                geom = (float(xs_m[ix_min]), float(ys_m[ix_min]),
                        float(xs_m[ix_max]), float(ys_m[ix_max]),
                        cxm, cym)
                tc = _ct(km3[lab])
                for a in (0.5, 0.3):
                    cands.append(Primitive(kind=4, geom=geom, color0=tc, alpha=a))

    return _dedup_candidates(cands)


# =====================================================================
#  Greedy building blocks
# =====================================================================

def _greedy_add(
    tile: np.ndarray,
    selected: list[Primitive],
    pool: list[Primitive],
    max_primitives: int,
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Greedily append primitives from *pool* that reduce the objective.

    Attempts GPU-parallel candidate scoring first; falls back to CPU
    if GPU is unavailable.
    """
    best_obj, best_pred = _tile_objective(
        tile, selected, residual, lam, res1_grid_size=res1_grid_size)
    pool = list(pool)

    # Try to use GPU batch objective evaluation (renders all candidates in one launch).
    _gpu_batch = None
    try:
        from .gpu_render import gpu_batch_objectives
        _gpu_batch = gpu_batch_objectives
    except Exception:
        pass

    while pool and len(selected) < max_primitives:
        # Build all trial primitive sets.
        trial_sets = [selected + [cand] for cand in pool]

        # Score all candidates at once.
        gpu_mse = _gpu_batch(tile, trial_sets) if _gpu_batch is not None else None

        if gpu_mse is not None:
            dist_values = gpu_mse
        else:
            pred_values_cpu: list[np.ndarray] = []
            for cand in pool:
                _, pred = _tile_objective(
                    tile, selected + [cand], residual, lam,
                    res1_grid_size=res1_grid_size)
                pred_values_cpu.append(pred)
            dist_values = batch_mse(tile, pred_values_cpu)

        obj_values: list[float] = []
        for i, cand in enumerate(pool):
            rate = _estimate_bits(
                selected + [cand],
                include_residual=(residual != (0, 0, 0)),
                include_res1=False,
                res1_grid_size=res1_grid_size,
            )
            obj_values.append(float(dist_values[i]) + lam * rate)

        min_idx = cccl_argmin(obj_values)
        if obj_values[min_idx] >= best_obj:
            break
        selected = selected + [pool[min_idx]]
        best_obj = obj_values[min_idx]

        # Render the winner to get the prediction image for next iteration.
        _, best_pred = _tile_objective(
            tile, selected, residual, lam, res1_grid_size=res1_grid_size)
        pool.pop(min_idx)

    return selected, best_obj, best_pred


def _one_pass_remove(
    tile: np.ndarray,
    selected: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Repeatedly drop the first primitive whose removal lowers the objective."""
    best_obj, best_pred = _tile_objective(
        tile, selected, residual, lam, res1_grid_size=res1_grid_size)
    changed = True
    while changed and len(selected) > 1:
        changed = False
        for i in range(1, len(selected)):
            trial = selected[:i] + selected[i + 1:]
            obj, pred = _tile_objective(
                tile, trial, residual, lam, res1_grid_size=res1_grid_size)
            if obj + 1e-9 < best_obj:
                selected = trial
                best_obj = obj
                best_pred = pred
                changed = True
                break
    return selected, best_obj, best_pred


# =====================================================================
#  Refinement passes (colour, alpha, coordinates)
# =====================================================================

def _compute_alpha_masks(
    primitives: list[Primitive], tile_size: int,
) -> list[np.ndarray]:
    """Compute per-pixel alpha contribution mask for each primitive.

    Returns list of (tile_size, tile_size) float32 arrays representing
    the effective alpha each primitive contributes at each pixel, accounting
    for the back-to-front compositing order.
    """
    from .render import _grid, _point_line_distance, _eval_curve_distance, _inside_triangle
    xs, ys = _grid(tile_size)
    n = len(primitives)
    # Raw alpha per primitive (before compositing order)
    raw_alphas = []
    for prim in primitives:
        a = np.float32(max(0.0, min(1.0, prim.alpha)))
        if prim.kind == 0:
            alpha = np.full((tile_size, tile_size), a, dtype=np.float32)
        elif prim.kind == 1:
            alpha = np.full((tile_size, tile_size), a, dtype=np.float32)
        elif prim.kind == 2:
            x0, y0, x1, y1, thickness = prim.geom
            dist = _point_line_distance(xs, ys, x0, y0, x1, y1)
            sigma = max(0.5, float(thickness))
            alpha = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32) * a
        elif prim.kind == 3:
            x0, y0, cx, cy, x1, y1, thickness = prim.geom
            dist = _eval_curve_distance(xs, ys, (x0, y0), (cx, cy), (x1, y1))
            sigma = max(0.5, float(thickness))
            alpha = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32) * a
        elif prim.kind == 4:
            alpha = _inside_triangle(xs, ys, prim.geom).astype(np.float32) * a
        else:
            alpha = np.zeros((tile_size, tile_size), dtype=np.float32)
        raw_alphas.append(alpha)

    # Effective contribution masks accounting for compositing order.
    # Back-to-front: primitive i's contribution is alpha_i * prod(1 - alpha_j for j > i)
    masks = []
    for i in range(n):
        mask = raw_alphas[i].copy()
        # Visibility from layers composited after this one
        for j in range(i + 1, n):
            mask = mask * (1.0 - raw_alphas[j])
        masks.append(mask)
    return masks


def _lstsq_optimize_colors(
    tile: np.ndarray,
    selected: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Solve for optimal primitive colors via least-squares.

    Given fixed geometry and alpha for each primitive, find the colors
    that minimize ||target - composited_result||^2.
    """
    tile_size = tile.shape[0]
    n = len(selected)
    if n == 0:
        return selected, float("inf"), np.zeros_like(tile)

    masks = _compute_alpha_masks(selected, tile_size)

    # For each channel, set up: A @ colors = target
    # where A[pixel, prim] = mask[prim][pixel] and target = tile - residual_bias
    target = tile.copy()
    if residual != (0, 0, 0):
        bias = np.array(residual, dtype=np.float32) / 255.0
        target = target - bias[None, None, :]

    npix = tile_size * tile_size
    new_selected = list(selected)

    for ch in range(3):
        # Build the linear system
        A = np.zeros((npix, n), dtype=np.float32)
        for i in range(n):
            A[:, i] = masks[i].ravel()

        b = target[:, :, ch].ravel()

        # Solve via least-squares (fast for small N)
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        colors = np.clip(result, 0.0, 1.0)

        # Update primitive colors
        for i in range(n):
            prim = new_selected[i]
            c0 = list(prim.color0)
            c0[ch] = float(colors[i])
            new_prim = Primitive(
                kind=prim.kind, geom=prim.geom,
                color0=(c0[0], c0[1], c0[2]),
                color1=prim.color1, alpha=prim.alpha,
            )
            new_selected[i] = _quantize_primitive(new_prim)

    # Also handle color1 for linear patches via per-half solve
    for i in range(n):
        prim = new_selected[i]
        if prim.kind == 1 and prim.color1 is not None:
            from .render import _grid
            xs, ys = _grid(tile_size)
            x0, y0, x1, y1 = prim.geom
            v = np.array([x1 - x0, y1 - y0], dtype=np.float32)
            den = float(v[0] * v[0] + v[1] * v[1] + 1e-8)
            t = ((xs - x0) * v[0] + (ys - y0) * v[1]) / den
            t = np.clip(t, 0.0, 1.0)
            # Weighted average of target in the t>0.5 region
            mask_hi = (t > 0.5) & (masks[i] > 0.01)
            if mask_hi.any():
                c1_opt = _ct(target[mask_hi].mean(axis=0))
                new_selected[i] = _quantize_primitive(Primitive(
                    kind=1, geom=prim.geom,
                    color0=prim.color0, color1=c1_opt, alpha=prim.alpha,
                ))

    best_obj, best_pred = _tile_objective(
        tile, new_selected, residual, lam, res1_grid_size=res1_grid_size)
    return new_selected, best_obj, best_pred


def _refine_colors_pass(
    tile: np.ndarray,
    selected: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Optimize colors via least-squares, then do a heuristic nudge pass."""
    # First: global least-squares solve
    selected, best_obj, best_pred = _lstsq_optimize_colors(
        tile, selected, residual, lam, res1_grid_size)

    # Then: fine-tune with small per-primitive nudges for quantization recovery
    mean_err = (tile - best_pred).mean(axis=(0, 1))
    for i in range(len(selected)):
        prim = selected[i]
        base = np.array(prim.color0, dtype=np.float32)
        for f in (0.3, 0.6, 1.0, -0.3):
            nc = np.clip(base + mean_err * f, 0, 1)
            tp = _quantize_primitive(Primitive(
                kind=prim.kind, geom=prim.geom,
                color0=_ct(nc), color1=prim.color1, alpha=prim.alpha,
            ))
            trial = selected[:i] + [tp] + selected[i + 1:]
            obj, pred = _tile_objective(
                tile, trial, residual, lam, res1_grid_size=res1_grid_size)
            if obj < best_obj:
                selected = trial
                best_obj = obj
                best_pred = pred
                mean_err = (tile - best_pred).mean(axis=(0, 1))

    return selected, best_obj, best_pred


def _refine_alpha_pass(
    tile: np.ndarray,
    selected: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Try alpha perturbations — batch GPU scoring when available."""
    best_obj, best_pred = _tile_objective(
        tile, selected, residual, lam, res1_grid_size=res1_grid_size)
    _deltas = (-0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.25)

    # Try GPU batched evaluation.
    try:
        from .gpu_render import gpu_batch_objectives
    except ImportError:
        gpu_batch_objectives = None

    for i in range(len(selected)):
        prim = selected[i]
        trials: list[list[Primitive]] = []
        trial_prims: list[Primitive] = []
        for delta in _deltas:
            na = max(0.01, min(1.0, prim.alpha + delta))
            tp = _quantize_primitive(Primitive(
                kind=prim.kind, geom=prim.geom,
                color0=prim.color0, color1=prim.color1, alpha=na,
            ))
            trial = selected[:i] + [tp] + selected[i + 1:]
            trials.append(trial)
            trial_prims.append(tp)

        # Batch MSE evaluation.
        gpu_mses = None
        if gpu_batch_objectives is not None:
            gpu_mses = gpu_batch_objectives(tile, trials)

        if gpu_mses is not None:
            for j, trial in enumerate(trials):
                rate = _estimate_bits(trial, include_residual=(residual != (0, 0, 0)), res1_grid_size=res1_grid_size)
                obj = float(gpu_mses[j]) + lam * rate
                if obj < best_obj:
                    selected = trial
                    best_obj = obj
                    _, best_pred = _tile_objective(tile, selected, residual, lam, res1_grid_size=res1_grid_size)
        else:
            for trial in trials:
                obj, pred = _tile_objective(tile, trial, residual, lam, res1_grid_size=res1_grid_size)
                if obj < best_obj:
                    selected = trial
                    best_obj = obj
                    best_pred = pred

    return selected, best_obj, best_pred


def _refine_coords_pass(
    tile: np.ndarray,
    selected: list[Primitive],
    residual: tuple[int, int, int],
    lam: float,
    res1_grid_size: int,
) -> tuple[list[Primitive], float, np.ndarray]:
    """Try small geometry perturbations for every non-const primitive."""
    best_obj, best_pred = _tile_objective(
        tile, selected, residual, lam, res1_grid_size=res1_grid_size)

    for i in range(len(selected)):
        prim = selected[i]
        if prim.kind == 0 or not prim.geom:
            continue
        geom = list(prim.geom)
        for gi in range(len(geom)):
            is_thickness = (prim.kind == 2 and gi == 4) or (prim.kind == 3 and gi == 6)
            lo, hi = (0.0, 4.0) if is_thickness else (0.0, 15.0)
            deltas = (-0.3, -0.15, 0.15, 0.3) if is_thickness else (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
            for d in deltas:
                trial_geom = list(geom)
                trial_geom[gi] = max(lo, min(hi, geom[gi] + d))
                tp = _quantize_primitive(Primitive(
                    kind=prim.kind, geom=tuple(trial_geom),
                    color0=prim.color0, color1=prim.color1, alpha=prim.alpha,
                ))
                trial = selected[:i] + [tp] + selected[i + 1:]
                obj, pred = _tile_objective(
                    tile, trial, residual, lam, res1_grid_size=res1_grid_size)
                if obj < best_obj:
                    selected = trial
                    geom = list(tp.geom)
                    best_obj = obj
                    best_pred = pred

    return selected, best_obj, best_pred


# =====================================================================
#  Residual-error re-seeding
# =====================================================================

def _boundary_seam_cost(
    img_a: np.ndarray,
    img_b: np.ndarray,
    axis: int,
    strip_width: int = 2,
) -> float:
    """Compute seam discontinuity between two adjacent tile images.

    *axis* 0 = vertical seam (A is left, B is right).
    *axis* 1 = horizontal seam (A is top, B is bottom).
    """
    if axis == 0:
        strip_a = img_a[:, -strip_width:, :]
        strip_b = img_b[:, :strip_width, :]
    else:
        strip_a = img_a[-strip_width:, :, :]
        strip_b = img_b[:strip_width, :, :]
    return float(np.mean((strip_a - strip_b) ** 2))


def refine_tile_boundaries(
    tiles: list[np.ndarray],
    records: list[TileRecord],
    cols: int,
    rows: int,
    tile_size: int,
    quality: int,
    seam_threshold: float = 0.005,
) -> list[TileRecord]:
    """Post-processing pass that reduces seam artifacts between adjacent tiles.

    For each tile boundary with high discontinuity, adjust the border-adjacent
    primitives' colors to minimize the seam. This replaces the naive deblock
    averaging.
    """
    lam = _tile_lambda(quality)

    def _render(rec: TileRecord, ts: int) -> np.ndarray:
        return render_tile(rec.primitives, tile_size=ts, residual_rgb=rec.residual_rgb)

    # Process vertical seams (left-right pairs)
    for ty in range(rows):
        for tx in range(cols - 1):
            idx_a = ty * cols + tx
            idx_b = ty * cols + tx + 1
            img_a = _render(records[idx_a], tile_size)
            img_b = _render(records[idx_b], tile_size)

            cost = _boundary_seam_cost(img_a, img_b, axis=0)
            if cost < seam_threshold:
                continue

            # Compute the average color at the boundary
            boundary_avg = (img_a[:, -1, :] + img_b[:, 0, :]) * 0.5

            # Try adjusting the rightmost const patches in tile A
            # and leftmost const patches in tile B
            for idx, tile_img, side in [(idx_a, tiles[idx_a], "right"), (idx_b, tiles[idx_b], "left")]:
                rec = records[idx]
                for i, prim in enumerate(rec.primitives):
                    if prim.kind != 0:
                        continue
                    # Nudge color toward boundary average
                    base = np.array(prim.color0, dtype=np.float32)
                    target = boundary_avg.mean(axis=0)
                    for blend in (0.1, 0.2, 0.3):
                        nc = np.clip(base * (1 - blend) + target * blend, 0, 1)
                        tp = _quantize_primitive(Primitive(
                            kind=0, geom=(), color0=_ct(nc), alpha=prim.alpha,
                        ))
                        trial = list(rec.primitives)
                        trial[i] = tp
                        trial_rec = TileRecord(primitives=trial, residual_rgb=rec.residual_rgb)
                        trial_img = _render(trial_rec, tile_size)

                        # Check: did we improve the seam without hurting tile quality?
                        obj_before, _ = _tile_objective(tile_img, rec.primitives, rec.residual_rgb, lam)
                        obj_after, _ = _tile_objective(tile_img, trial, rec.residual_rgb, lam)

                        if side == "right":
                            new_cost = _boundary_seam_cost(trial_img, img_b, axis=0)
                        else:
                            new_cost = _boundary_seam_cost(img_a, trial_img, axis=0)

                        # Accept if seam improved and tile quality didn't degrade much
                        if new_cost < cost and obj_after < obj_before * 1.05:
                            records[idx] = trial_rec
                            if side == "right":
                                img_a = trial_img
                            else:
                                img_b = trial_img
                            cost = new_cost
                            break

    # Process horizontal seams (top-bottom pairs)
    for ty in range(rows - 1):
        for tx in range(cols):
            idx_a = ty * cols + tx
            idx_b = (ty + 1) * cols + tx
            img_a = _render(records[idx_a], tile_size)
            img_b = _render(records[idx_b], tile_size)

            cost = _boundary_seam_cost(img_a, img_b, axis=1)
            if cost < seam_threshold:
                continue

            boundary_avg = (img_a[-1, :, :] + img_b[0, :, :]) * 0.5

            for idx, tile_img, side in [(idx_a, tiles[idx_a], "bottom"), (idx_b, tiles[idx_b], "top")]:
                rec = records[idx]
                for i, prim in enumerate(rec.primitives):
                    if prim.kind != 0:
                        continue
                    base = np.array(prim.color0, dtype=np.float32)
                    target = boundary_avg.mean(axis=0)
                    for blend in (0.1, 0.2, 0.3):
                        nc = np.clip(base * (1 - blend) + target * blend, 0, 1)
                        tp = _quantize_primitive(Primitive(
                            kind=0, geom=(), color0=_ct(nc), alpha=prim.alpha,
                        ))
                        trial = list(rec.primitives)
                        trial[i] = tp
                        trial_rec = TileRecord(primitives=trial, residual_rgb=rec.residual_rgb)
                        trial_img = _render(trial_rec, tile_size)

                        obj_before, _ = _tile_objective(tile_img, rec.primitives, rec.residual_rgb, lam)
                        obj_after, _ = _tile_objective(tile_img, trial, rec.residual_rgb, lam)

                        if side == "bottom":
                            new_cost = _boundary_seam_cost(trial_img, img_b, axis=1)
                        else:
                            new_cost = _boundary_seam_cost(img_a, trial_img, axis=1)

                        if new_cost < cost and obj_after < obj_before * 1.05:
                            records[idx] = trial_rec
                            if side == "bottom":
                                img_a = trial_img
                            else:
                                img_b = trial_img
                            cost = new_cost
                            break

    return records


def _generate_residual_candidates(
    tile: np.ndarray,
    pred: np.ndarray,
    quality: int,
) -> list[Primitive]:
    """Create candidates targeting high-error regions of *tile - pred*.

    Generates a rich pool covering all primitive types, with colors and
    geometry derived from the actual error pattern.
    """
    h, w, _ = tile.shape
    hf, wf = float(h - 1), float(w - 1)
    error = tile - pred
    err_lum = np.abs(error).mean(axis=2)
    thresh = float(np.percentile(err_lum, 60))
    hi_mask = err_lum >= thresh

    cands: list[Primitive] = []
    if not np.any(hi_mask):
        return cands

    # Colors from error regions.
    hi_color = _ct(np.clip(tile[hi_mask].mean(axis=0), 0, 1))
    corr_color = _ct(np.clip(pred[hi_mask].mean(axis=0) + error[hi_mask].mean(axis=0), 0, 1))
    target_color = _ct(np.clip(tile[hi_mask].mean(axis=0), 0, 1))

    # ── CONST PATCHES at multiple alphas ──
    for color in (hi_color, corr_color, target_color):
        for a in (0.6, 0.4, 0.25, 0.15, 0.08):
            cands.append(Primitive(kind=0, geom=(), color0=color, alpha=a))

    # Sub-region const patches: split high-error mask into quadrants.
    qh, qw = h // 2, w // 2
    for qr in range(2):
        for qc in range(2):
            q_mask = hi_mask[qr * qh:(qr + 1) * qh, qc * qw:(qc + 1) * qw]
            q_tile = tile[qr * qh:(qr + 1) * qh, qc * qw:(qc + 1) * qw]
            if q_mask.any():
                qc_color = _ct(np.clip(q_tile[q_mask].mean(axis=0), 0, 1))
                for a in (0.35, 0.2, 0.1):
                    cands.append(Primitive(kind=0, geom=(), color0=qc_color, alpha=a))

    # ── LINEAR PATCHES along error gradient ──
    ely, elx = np.gradient(err_lum)
    emx, emy = float(np.mean(elx)), float(np.mean(ely))
    en = math.sqrt(emx * emx + emy * emy) + 1e-8
    vx, vy = emx / en, emy / en
    for offset in (0.5, 0.3, 0.7):
        cx, cy = wf * offset, hf * offset
        x0 = cx - vx * wf * 0.6
        y0 = cy - vy * hf * 0.6
        x1 = cx + vx * wf * 0.6
        y1 = cy + vy * hf * 0.6
        c0, c1 = _split_by_line(tile, x0, y0, x1, y1)
        for a in (0.5, 0.3, 0.15):
            cands.append(Primitive(kind=1, geom=(x0, y0, x1, y1), color0=c0, color1=c1, alpha=a))
    # Perpendicular direction too.
    pvx, pvy = -vy, vx
    cx, cy = wf * 0.5, hf * 0.5
    x0 = cx - pvx * wf * 0.6
    y0 = cy - pvy * hf * 0.6
    x1 = cx + pvx * wf * 0.6
    y1 = cy + pvy * hf * 0.6
    c0, c1 = _split_by_line(tile, x0, y0, x1, y1)
    for a in (0.4, 0.2):
        cands.append(Primitive(kind=1, geom=(x0, y0, x1, y1), color0=c0, color1=c1, alpha=a))

    # ── LINES at error centroid and peaks ──
    ey_arr, ex_arr = np.where(hi_mask)
    ecx, ecy = float(np.mean(ex_arr)), float(np.mean(ey_arr))
    for t in (0.3, 0.6, 1.0, 1.5, 2.5):
        for lc in (hi_color, target_color):
            for a in (0.5, 0.3, 0.15):
                # Vertical and horizontal through error centroid.
                cands.append(Primitive(kind=2, geom=(ecx, 0, ecx, hf, t), color0=lc, alpha=a))
                cands.append(Primitive(kind=2, geom=(0, ecy, wf, ecy, t), color0=lc, alpha=a))
    # Diagonal through error centroid.
    for t in (0.5, 1.0):
        cands.append(Primitive(kind=2, geom=(0, 0, wf, hf, t), color0=hi_color, alpha=0.25))
        cands.append(Primitive(kind=2, geom=(0, hf, wf, 0, t), color0=hi_color, alpha=0.25))

    # ── CURVES through error peaks ──
    if len(ex_arr) >= 6:
        # Sort error points by x, pick start/mid/end.
        order = np.argsort(ex_arr)
        p0 = (float(ex_arr[order[0]]), float(ey_arr[order[0]]))
        pm = (float(ex_arr[order[len(order) // 2]]), float(ey_arr[order[len(order) // 2]]))
        p1 = (float(ex_arr[order[-1]]), float(ey_arr[order[-1]]))
        ctrl = (2 * pm[0] - (p0[0] + p1[0]) / 2, 2 * pm[1] - (p0[1] + p1[1]) / 2)
        ctrl = (max(0.0, min(wf, ctrl[0])), max(0.0, min(hf, ctrl[1])))
        for t in (0.5, 1.0, 2.0):
            cands.append(Primitive(kind=3,
                geom=(p0[0], p0[1], ctrl[0], ctrl[1], p1[0], p1[1], t),
                color0=hi_color, alpha=0.3))

    # ── TRIANGLES covering error bounding box ──
    if len(ex_arr) >= 3:
        ix_min, ix_max = np.argmin(ex_arr), np.argmax(ex_arr)
        iy_min, iy_max = np.argmin(ey_arr), np.argmax(ey_arr)
        # Bounding-box triangle pair.
        geom1 = (float(ex_arr[ix_min]), float(ey_arr[ix_min]),
                 float(ex_arr[ix_max]), float(ey_arr[ix_max]),
                 ecx, ecy)
        geom2 = (float(ex_arr[ix_min]), float(ey_arr[iy_min]),
                 float(ex_arr[ix_max]), float(ey_arr[iy_max]),
                 ecx, ecy)
        for geom in (geom1, geom2):
            for a in (0.4, 0.2, 0.1):
                cands.append(Primitive(kind=4, geom=geom, color0=hi_color, alpha=a))
                cands.append(Primitive(kind=4, geom=geom, color0=target_color, alpha=a))

    return _dedup_candidates(cands)


# =====================================================================
#  Tile fitting entry point
# =====================================================================

def _fit_tile(
    tile: np.ndarray,
    quality: int,
    max_primitives: int,
    enable_res0: bool,
    enable_res1: bool,
    res1_grid_size: int = 4,
) -> tuple[TileRecord, np.ndarray | None]:
    lam = _tile_lambda(quality)
    # Build a large candidate pool from both sources.
    cands_edge = generate_edge_driven_candidates(tile, quality)
    cands_template = _generate_candidates(tile, quality)
    candidates = _dedup_candidates(cands_edge + cands_template)
    if not candidates:
        avg = tile.mean(axis=(0, 1))
        return TileRecord(primitives=[Primitive(kind=0, geom=(), color0=_ct(avg), alpha=1.0)]), None

    selected: list[Primitive] = [candidates[0]]
    residual: tuple[int, int, int] = (0, 0, 0)
    residual_map: np.ndarray | None = None

    best_obj, best_pred = _tile_objective(
        tile, selected, residual, lam, res1_grid_size=res1_grid_size)

    # ── Phase 1: greedy fill from full candidate pool ─────────────
    selected, best_obj, best_pred = _greedy_add(
        tile, selected, candidates[1:], max_primitives,
        residual, lam, res1_grid_size)

    # ── Phase 2: iterative refine + re-seed rounds ────────────────
    # More rounds = more compute = higher quality.
    n_rounds = 8 if quality >= 90 else 5 if quality >= 70 else 3 if quality >= 50 else 1
    mse_floor = 0.001  # Skip refinement if MSE is already very low.
    prev_obj = best_obj
    for _round in range(n_rounds):
        cur_mse = float(np.mean((tile - best_pred) ** 2))
        if cur_mse < mse_floor:
            break  # Tile is already well-represented.

        # Refine what we have.
        selected, best_obj, best_pred = _refine_colors_pass(
            tile, selected, residual, lam, res1_grid_size)
        if len(selected) > 1:
            selected, best_obj, best_pred = _refine_alpha_pass(
                tile, selected, residual, lam, res1_grid_size)
            selected, best_obj, best_pred = _refine_coords_pass(
                tile, selected, residual, lam, res1_grid_size)
            selected, best_obj, best_pred = _one_pass_remove(
                tile, selected, residual, lam, res1_grid_size)

        # Early exit if no improvement this round.
        if best_obj >= prev_obj - 1e-7 and _round > 0:
            break
        prev_obj = best_obj

        # Re-seed: generate fresh candidates targeting residual error.
        if len(selected) < max_primitives:
            resid_cands = _generate_residual_candidates(tile, best_pred, quality)
            existing_keys = {encode_primitive(p) for p in selected}
            fresh = [c for c in candidates if encode_primitive(c) not in existing_keys]
            pool = _dedup_candidates(resid_cands + fresh[:30])
            if pool:
                selected, best_obj, best_pred = _greedy_add(
                    tile, selected, pool, max_primitives,
                    residual, lam, res1_grid_size)

    # ── Phase 3: final prune ───────────────────────────────────────
    selected, best_obj, best_pred = _one_pass_remove(
        tile, selected, residual, lam, res1_grid_size)

    # ── Phase 4: RES0 scalar bias ──────────────────────────────────
    if enable_res0:
        mean_err = tile - best_pred
        bias = mean_err.mean(axis=(0, 1))
        rb = int(np.clip(np.round(float(bias[0]) * 255.0), -127, 127))
        gb = int(np.clip(np.round(float(bias[1]) * 255.0), -127, 127))
        bb = int(np.clip(np.round(float(bias[2]) * 255.0), -127, 127))
        test_res = (rb, gb, bb)
        obj_res, pred_res = _tile_objective(
            tile, selected, test_res, lam, res1_grid_size=res1_grid_size)
        if obj_res < best_obj:
            residual = test_res
            best_obj = obj_res
            best_pred = pred_res

    # ── Phase 5: RES1 correction grid ──────────────────────────────
    if enable_res1:
        err = tile - best_pred
        cand_map = _quantize_residual_map(err, grid_size=res1_grid_size)
        obj_res1, pred_res1 = _tile_objective(
            tile, selected, residual, lam,
            residual_map=cand_map, res1_grid_size=res1_grid_size)
        if obj_res1 < best_obj:
            residual_map = cand_map
            best_obj = obj_res1
            best_pred = pred_res1

    return TileRecord(primitives=selected, residual_rgb=residual), residual_map


def _extract_tiles(image: np.ndarray, tile_size: int) -> tuple[list[np.ndarray], int, int]:
    h, w, _ = image.shape
    cols = (w + tile_size - 1) // tile_size
    rows = (h + tile_size - 1) // tile_size
    tiles: list[np.ndarray] = []
    for ty in range(rows):
        for tx in range(cols):
            y0 = ty * tile_size
            x0 = tx * tile_size
            crop = image[y0 : min(y0 + tile_size, h), x0 : min(x0 + tile_size, w), :]
            tiles.append(_pad_tile(crop, tile_size))
    return tiles, cols, rows


def _apply_single_image_maxcompute_preset(cfg: EncodeConfig) -> None:
    cfg.quality = max(int(cfg.quality), 95)
    cfg.max_primitives = max(int(cfg.max_primitives), 48)
    cfg.enable_res0 = True
    cfg.enable_res1 = True
    ff = dict(cfg.feature_flags)
    ff.setdefault("candidate_bank", "rich18")
    ff.setdefault("multi_rounds", 2)
    ff.setdefault("adaptive_tile_budget", True)
    ff.setdefault("edge_weighted_objective", True)
    ff.setdefault("target_bpp", 6.0)
    ff.setdefault("enable_res2", True)
    ff.setdefault("search_mode", "greedy")
    ff.setdefault("beam_width", 4)
    ff.setdefault("mcmc_steps", 0)
    ff.setdefault("stochastic_restarts", 0)
    ff.setdefault("early_exit_patience", 0)
    ff.setdefault("container_v2_blocks", True)
    ff.setdefault("split_entropy_streams", True)
    ff.setdefault("neighbor_delta_coding", True)
    ff.setdefault("maxcompute_fit_passes", 512)
    cfg.feature_flags = ff


def _primitive_from_model(
    *,
    model: int,
    c0: np.ndarray,
    c1: np.ndarray,
    alpha: float,
    candidate_bank: str,
) -> Primitive:
    if candidate_bank == "rich18":
        if model == 0:
            return Primitive(kind=0, geom=(), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if 1 <= model <= 8:
            geoms = [
                (0.0, 0.0, 15.0, 0.0),
                (0.0, 0.0, 15.0, 6.0),
                (0.0, 0.0, 15.0, 15.0),
                (0.0, 6.0, 15.0, 15.0),
                (0.0, 15.0, 15.0, 15.0),
                (0.0, 15.0, 15.0, 6.0),
                (0.0, 15.0, 15.0, 0.0),
                (0.0, 6.0, 15.0, 0.0),
            ]
            g = geoms[model - 1]
            return Primitive(
                kind=1,
                geom=g,
                color0=(float(c0[0]), float(c0[1]), float(c0[2])),
                color1=(float(c1[0]), float(c1[1]), float(c1[2])),
                alpha=alpha,
            )
        if model == 9:
            return Primitive(kind=2, geom=(7.5, 0.0, 7.5, 15.0, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 10:
            return Primitive(kind=2, geom=(0.0, 7.5, 15.0, 7.5, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 11:
            return Primitive(kind=2, geom=(0.0, 0.0, 15.0, 15.0, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 12:
            return Primitive(kind=4, geom=(0.0, 0.0, 15.0, 0.0, 0.0, 15.0), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 13:
            return Primitive(kind=4, geom=(15.0, 0.0, 15.0, 15.0, 0.0, 0.0), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 14:
            return Primitive(kind=4, geom=(0.0, 15.0, 15.0, 15.0, 0.0, 0.0), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 15:
            return Primitive(kind=3, geom=(0.0, 7.5, 7.5, 0.0, 15.0, 7.5, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        if model == 16:
            return Primitive(kind=3, geom=(0.0, 7.5, 7.5, 15.0, 15.0, 7.5, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)
        return Primitive(kind=3, geom=(7.5, 0.0, 0.0, 7.5, 7.5, 15.0, 1.5), color0=(float(c0[0]), float(c0[1]), float(c0[2])), alpha=alpha)

    c0t = (float(c0[0]), float(c0[1]), float(c0[2]))
    c1t = (float(c1[0]), float(c1[1]), float(c1[2]))
    if int(model) == 1:
        return Primitive(kind=1, geom=(0.0, 0.0, 15.0, 0.0), color0=c0t, color1=c1t, alpha=alpha)
    if int(model) == 2:
        return Primitive(kind=1, geom=(0.0, 0.0, 0.0, 15.0), color0=c0t, color1=c1t, alpha=alpha)
    if int(model) == 3:
        return Primitive(kind=1, geom=(0.0, 0.0, 15.0, 15.0), color0=c0t, color1=c1t, alpha=alpha)
    if int(model) == 4:
        return Primitive(kind=1, geom=(0.0, 15.0, 15.0, 0.0), color0=c0t, color1=c1t, alpha=alpha)
    return Primitive(kind=0, geom=(), color0=c0t, alpha=alpha)


def _tile_edge_strength(tile: np.ndarray) -> float:
    y = 0.2126 * tile[..., 0] + 0.7152 * tile[..., 1] + 0.0722 * tile[..., 2]
    gx = np.diff(y, axis=1, append=y[:, -1:])
    gy = np.diff(y, axis=0, append=y[-1:, :])
    gm = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(gm))


def _hierarchical_subtile_candidates(tile: np.ndarray, level: str) -> list[Primitive]:
    """Generate explicit 8x8-in-16x16 candidates via quadrant triangle pairs."""
    if level not in {"mid", "max"}:
        return []
    h, w, _ = tile.shape
    if h < 16 or w < 16:
        return []
    out: list[Primitive] = []
    alpha_base = 0.72 if level == "mid" else 0.82
    for qy in range(2):
        for qx in range(2):
            x0 = float(qx * 8)
            y0 = float(qy * 8)
            x1 = float(qx * 8 + 7)
            y1 = float(qy * 8 + 7)
            patch = tile[qy * 8 : qy * 8 + 8, qx * 8 : qx * 8 + 8, :]
            c = patch.mean(axis=(0, 1))
            color = (float(c[0]), float(c[1]), float(c[2]))
            # Two triangles approximating a local 8x8 constant patch.
            out.append(Primitive(kind=4, geom=(x0, y0, x1, y0, x0, y1), color0=color, alpha=alpha_base))
            out.append(Primitive(kind=4, geom=(x1, y0, x1, y1, x0, y1), color0=color, alpha=alpha_base))
            if level == "max":
                # Add an oriented linear split candidate for subtile texture.
                out.append(
                    Primitive(
                        kind=1,
                        geom=(x0, y0, x1, y1),
                        color0=color,
                        color1=(
                            float(np.clip(color[0] * 1.05, 0.0, 1.0)),
                            float(np.clip(color[1] * 1.05, 0.0, 1.0)),
                            float(np.clip(color[2] * 1.05, 0.0, 1.0)),
                        ),
                        alpha=0.58,
                    )
                )
    return [_quantize_primitive(p) for p in out]


def _encode_res2_sparse(
    *,
    tiles: list[np.ndarray],
    records: list[TileRecord],
    max_atoms_per_tile: int = 8,
) -> dict[str, Any]:
    """Build sparse high-frequency residual atoms per tile.

    Each atom stores (pixel_index, dr, dg, db) with int8 deltas in linear RGB.
    """
    out_tiles: list[list[list[int]]] = []
    for tile, rec in zip(tiles, records):
        pred = render_tile(rec.primitives, tile_size=tile.shape[0], residual_rgb=rec.residual_rgb)
        err = tile - pred
        lum = 0.2126 * err[..., 0] + 0.7152 * err[..., 1] + 0.0722 * err[..., 2]
        gx = np.diff(lum, axis=1, append=lum[:, -1:])
        gy = np.diff(lum, axis=0, append=lum[-1:, :])
        hf = np.sqrt(gx * gx + gy * gy)
        flat = hf.reshape(-1)
        k = min(max_atoms_per_tile, int(flat.size))
        if k <= 0:
            out_tiles.append([])
            continue
        picks = np.argpartition(flat, -k)[-k:]
        atoms: list[list[int]] = []
        for pix in picks.tolist():
            y = pix // tile.shape[1]
            x = pix % tile.shape[1]
            d = np.clip(np.round(err[y, x, :] * 255.0), -127.0, 127.0).astype(np.int32)
            if int(np.max(np.abs(d))) < 2:
                continue
            atoms.append([int(pix), int(d[0]), int(d[1]), int(d[2])])
        out_tiles.append(atoms)
    return {
        "format": "sparse_atoms_v1",
        "tile_size": 16,
        "max_atoms_per_tile": int(max_atoms_per_tile),
        "tiles": out_tiles,
    }


def _res2_single_atom_objective(
    *,
    tile: np.ndarray,
    pred: np.ndarray,
    primitives: list[Primitive],
    lam: float,
) -> float:
    """Objective of a single best sparse residual atom over current prediction."""
    err = tile - pred
    lum = 0.2126 * err[..., 0] + 0.7152 * err[..., 1] + 0.0722 * err[..., 2]
    idx = int(np.argmax(np.abs(lum.reshape(-1))))
    y = idx // tile.shape[1]
    x = idx % tile.shape[1]
    corr = pred.copy()
    delta = np.clip(np.round(err[y, x, :] * 255.0), -127.0, 127.0).astype(np.float32) / 255.0
    corr[y, x, :] = np.clip(corr[y, x, :] + delta, 0.0, 1.0)
    dist = float(np.mean((tile - corr) ** 2))
    rate = float(_estimate_bits(primitives, include_residual=False, include_res1=False, res1_grid_size=4) + 32)
    return dist + lam * rate


def _allocate_tile_budgets(
    *,
    complexity: np.ndarray,
    max_primitives_per_tile: int,
    tile_count: int,
    width: int,
    height: int,
    target_bpp: float | None,
    adaptive: bool,
) -> np.ndarray:
    cap = max(1, int(max_primitives_per_tile))
    budgets = np.ones(tile_count, dtype=np.int32)
    total_cap = tile_count * cap
    total_target = total_cap

    if target_bpp is not None and target_bpp > 0.0:
        target_bits = float(target_bpp) * float(width * height)
        base_bits = float(tile_count * 32)
        extra_bits = max(0.0, target_bits - base_bits)
        extra_prim_budget = int(extra_bits // 72.0)
        total_target = max(tile_count, min(total_cap, tile_count + extra_prim_budget))

    extras = max(0, total_target - tile_count)
    if extras == 0:
        return budgets

    if adaptive and complexity.size == tile_count and float(np.sum(complexity)) > 0.0:
        w = np.maximum(complexity.astype(np.float64), 1e-9)
        w = w / float(np.sum(w))
        raw = w * float(extras)
        add = np.floor(raw).astype(np.int32)
        rem = int(extras - int(np.sum(add)))
        if rem > 0:
            frac_order = np.argsort(-(raw - add))
            add[frac_order[:rem]] += 1
    else:
        add = np.zeros(tile_count, dtype=np.int32)
        add[:extras] = 1
        if extras > tile_count:
            for i in range(tile_count, extras):
                add[i % tile_count] += 1
    budgets = np.minimum(cap, budgets + add)
    return budgets


def _tile_model_order(
    *,
    tile_idx: int,
    fit,
    active_models: np.ndarray,
    active_model_set: set[int],
    lam_model: float,
    rate_bits_all: np.ndarray,
    rd_model_selection: bool,
    topn: int,
) -> np.ndarray:
    if rd_model_selection:
        model_obj = fit.model_mse[tile_idx, active_models] + (lam_model * rate_bits_all[active_models])
    else:
        model_obj = np.full((len(active_models),), 1e9, dtype=np.float32)
        preferred = int(fit.model_ids[tile_idx])
        if preferred in active_model_set:
            pref_idx = int(np.where(active_models == preferred)[0][0])
        else:
            pref_idx = int(np.argmin(fit.model_mse[tile_idx, active_models]))
        model_obj[pref_idx] = 0.0
    order_local = np.argsort(model_obj)[: min(topn, len(model_obj))]
    return active_models[order_local]


def _beam_search_tile_models(
    *,
    tile: np.ndarray,
    fit,
    tile_idx: int,
    rounds: int,
    budget: int,
    beam_width: int,
    alpha_schedule: list[float],
    candidate_bank: str,
    active_models: np.ndarray,
    active_model_set: set[int],
    lam_model: float,
    rate_bits_all: np.ndarray,
    rd_model_selection: bool,
    lam_eff: float,
    extra_candidates: list[Primitive] | None = None,
) -> list[Primitive]:
    beams: list[list[Primitive]] = [[]]
    for round_idx in range(rounds):
        alpha = alpha_schedule[min(round_idx, len(alpha_schedule) - 1)]
        expanded: list[tuple[float, list[Primitive]]] = []
        for seq in beams:
            if len(seq) >= budget:
                obj, _ = _tile_objective(tile, seq, (0, 0, 0), lam_eff, res1_grid_size=4)
                expanded.append((float(obj), list(seq)))
                continue
            model_order = _tile_model_order(
                tile_idx=tile_idx,
                fit=fit,
                active_models=active_models,
                active_model_set=active_model_set,
                lam_model=lam_model,
                rate_bits_all=rate_bits_all,
                rd_model_selection=rd_model_selection,
                topn=4,
            )
            existing = {encode_primitive(p) for p in seq}
            seeded = False
            for model in model_order:
                c0 = fit.candidate_c0[tile_idx, int(model), :]
                c1 = fit.candidate_c1[tile_idx, int(model), :]
                cand = _quantize_primitive(
                    _primitive_from_model(
                        model=int(model),
                        c0=c0,
                        c1=c1,
                        alpha=alpha,
                        candidate_bank=candidate_bank,
                    )
                )
                key = encode_primitive(cand)
                if key in existing:
                    continue
                trial = seq + [cand]
                obj, _ = _tile_objective(tile, trial, (0, 0, 0), lam_eff, res1_grid_size=4)
                expanded.append((float(obj), trial))
                seeded = True
            if extra_candidates:
                for cand in extra_candidates:
                    key = encode_primitive(cand)
                    if key in existing:
                        continue
                    trial = seq + [cand]
                    obj, _ = _tile_objective(tile, trial, (0, 0, 0), lam_eff, res1_grid_size=4)
                    expanded.append((float(obj), trial))
                    seeded = True
            if not seeded:
                obj, _ = _tile_objective(tile, seq, (0, 0, 0), lam_eff, res1_grid_size=4)
                expanded.append((float(obj), list(seq)))
        expanded.sort(key=lambda x: x[0])
        beams = [seq for _, seq in expanded[:beam_width]]
        if not beams:
            break
    if not beams:
        return []
    return beams[0]


def _mcmc_refine_hard_tiles(
    *,
    records: list[TileRecord],
    tile_images: list[np.ndarray],
    fit,
    budgets: np.ndarray,
    active_models: np.ndarray,
    candidate_bank: str,
    rounds: int,
    lam_eff: float,
    mcmc_steps: int,
    stochastic_restarts: int,
    edge_strength: np.ndarray | None,
    extra_candidates_by_tile: dict[int, list[Primitive]] | None = None,
) -> None:
    if mcmc_steps <= 0:
        return
    hard: list[int] = list(np.argsort(-np.min(fit.model_mse[:, active_models], axis=1))[: max(1, len(records) // 5)])
    if edge_strength is not None:
        thr = float(np.percentile(edge_strength, 80))
        hard = sorted(set(hard + [i for i, e in enumerate(edge_strength.tolist()) if e >= thr]))
    rng = np.random.default_rng(12345)
    alpha_schedule = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3]
    for idx in hard:
        tile = tile_images[idx]
        current = list(records[idx].primitives)
        cur_obj, _ = _tile_objective(tile, current, (0, 0, 0), lam_eff, res1_grid_size=4)
        best = list(current)
        best_obj = float(cur_obj)
        restarts = max(0, int(stochastic_restarts))
        for restart in range(restarts + 1):
            state = list(current if restart == 0 else best[: max(1, min(len(best), int(budgets[idx])) // 2)])
            state_obj, _ = _tile_objective(tile, state, (0, 0, 0), lam_eff, res1_grid_size=4)
            temp = 0.02
            for step in range(mcmc_steps):
                proposal = list(state)
                if proposal and rng.random() < 0.35:
                    drop = int(rng.integers(0, len(proposal)))
                    proposal = proposal[:drop] + proposal[drop + 1 :]
                if len(proposal) < int(budgets[idx]):
                    use_extra = bool(extra_candidates_by_tile and idx in extra_candidates_by_tile and extra_candidates_by_tile[idx] and rng.random() < 0.35)
                    if use_extra:
                        ecands = extra_candidates_by_tile[idx]
                        cand = ecands[int(rng.integers(0, len(ecands)))]
                    else:
                        model = int(active_models[int(rng.integers(0, len(active_models)))])
                        c0 = fit.candidate_c0[idx, model, :]
                        c1 = fit.candidate_c1[idx, model, :]
                        alpha = alpha_schedule[min(step % rounds, len(alpha_schedule) - 1)]
                        cand = _quantize_primitive(
                            _primitive_from_model(
                                model=model,
                                c0=c0,
                                c1=c1,
                                alpha=alpha,
                                candidate_bank=candidate_bank,
                            )
                        )
                    key = encode_primitive(cand)
                    if key not in {encode_primitive(p) for p in proposal}:
                        proposal.append(cand)
                prop_obj, _ = _tile_objective(tile, proposal, (0, 0, 0), lam_eff, res1_grid_size=4)
                accept = False
                if prop_obj <= state_obj:
                    accept = True
                else:
                    delta = float(prop_obj - state_obj)
                    prob = float(np.exp(-delta / max(temp, 1e-5)))
                    if rng.random() < prob:
                        accept = True
                if accept:
                    state = proposal
                    state_obj = float(prop_obj)
                    if state_obj < best_obj:
                        best = list(state)
                        best_obj = state_obj
                temp *= 0.97
        if best_obj + 1e-9 < cur_obj:
            records[idx] = TileRecord(primitives=best, residual_rgb=(0, 0, 0))


def _encode_image_gpu_baseline(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    if cfg.preset not in {DEFAULT_PRESET, "stylized-v1", "rtx-single-maxcompute"}:
        raise EncodeError(f"unsupported preset: {cfg.preset}")
    if cfg.preset == "rtx-single-maxcompute":
        _apply_single_image_maxcompute_preset(cfg)
    quality = _clamp_quality(cfg.quality)
    if cfg.tile_size != TILE_SIZE:
        raise EncodeError(f"v1 requires tile_size={TILE_SIZE}")
    if cfg.entropy != "chunked-rans":
        raise EncodeError("GPU-only decode requires chunked-rans PRIM encoding")
    if cfg.preset == "rtx-heavy-v2":
        cfg.max_primitives = max(cfg.max_primitives, 1)
        preset_id = 2
    else:
        preset_id = 1

    image, width, height = load_image_linear(input_path)
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    # Technique-level controls (campaign toggles).
    if flags.mixed_action_beam_level in {"mid", "max"}:
        flags.search_mode = "beam"
        flags.beam_width = max(flags.beam_width, 6 if flags.mixed_action_beam_level == "mid" else 10)
    if flags.hierarchical_tiling_level in {"mid", "max"}:
        flags.multi_rounds = max(flags.multi_rounds, 6 if flags.hierarchical_tiling_level == "mid" else 10)
        cfg.max_primitives = max(cfg.max_primitives, 56 if flags.hierarchical_tiling_level == "mid" else 72)
    if flags.residual_patch_borrow_level in {"mid", "max"}:
        flags.enable_res2 = True
    try:
        fit = fit_tiles_gpu_constant(
            image=image,
            tile_size=cfg.tile_size,
            iterations=max(1, int(flags.maxcompute_fit_passes)),
        )
    except GpuEncodeError as exc:
        raise EncodeError(f"GPU baseline encode failed: {exc}") from exc
    # Candidate-bank RD model selection.
    if flags.candidate_bank == "rich18":
        # 0 const, 1..8 linear, 9..11 line, 12..14 triangle, 15..17 curve
        rate_bits_all = np.array(
            [32.0] + [72.0] * 8 + [64.0] * 3 + [80.0] * 3 + [88.0] * 3,
            dtype=np.float32,
        )
        active_models = np.arange(min(int(fit.model_mse.shape[1]), 18), dtype=np.int32)
    else:
        rate_bits_all = np.array([32.0, 72.0, 72.0, 72.0, 72.0], dtype=np.float32)
        active_models = np.arange(min(int(fit.model_mse.shape[1]), 5), dtype=np.int32)
    if flags.primitive_dictionary_level in {"mid", "max"}:
        rate_bits_all = rate_bits_all * (0.92 if flags.primitive_dictionary_level == "mid" else 0.84)
    if flags.entropy_context_model_level in {"mid", "max"}:
        rate_bits_all = rate_bits_all * (0.95 if flags.entropy_context_model_level == "mid" else 0.88)
    active_model_set = {int(m) for m in active_models.tolist()}
    lam_model = (((101.0 - float(quality)) / 100.0) ** 2) * 2.0e-4

    tile_images, _, _ = _extract_tiles(image, cfg.tile_size)
    tile_count = int(fit.model_ids.shape[0])
    tile_complexity = np.min(fit.model_mse, axis=1).astype(np.float64)
    edge_strength = np.array([_tile_edge_strength(t) for t in tile_images], dtype=np.float64)
    if flags.edge_weighted_objective:
        tile_complexity = tile_complexity + (0.25 * edge_strength)
    if flags.edge_budget_boost_level in {"mid", "max"}:
        tile_complexity = tile_complexity + ((0.3 if flags.edge_budget_boost_level == "mid" else 0.6) * edge_strength)
    per_tile_cap = min(max(1, int(cfg.max_primitives)), max(1, int(flags.multi_rounds)))
    budgets = _allocate_tile_budgets(
        complexity=tile_complexity,
        max_primitives_per_tile=per_tile_cap,
        tile_count=tile_count,
        width=width,
        height=height,
        target_bpp=flags.target_bpp,
        adaptive=flags.adaptive_tile_budget,
    )
    hierarchical_tile_ids: set[int] = set()
    extra_candidates_by_tile: dict[int, list[Primitive]] = {}
    if flags.hierarchical_tiling_level in {"mid", "max"}:
        q = 75 if flags.hierarchical_tiling_level == "mid" else 60
        thr = float(np.percentile(tile_complexity, q))
        for i in range(tile_count):
            if float(tile_complexity[i]) >= thr:
                ecands = _hierarchical_subtile_candidates(tile_images[i], flags.hierarchical_tiling_level)
                if ecands:
                    hierarchical_tile_ids.add(i)
                    extra_candidates_by_tile[i] = ecands

    records: list[TileRecord] = [TileRecord(primitives=[], residual_rgb=(0, 0, 0)) for _ in range(tile_count)]
    rounds = max(1, int(flags.multi_rounds))
    alpha_schedule = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3]
    if flags.subpixel_primitives_level in {"mid", "max"}:
        alpha_schedule = [1.0, 0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.44, 0.36, 0.28]
    tile_lam = _tile_lambda(quality)
    lam_eff = tile_lam
    if rounds > 1:
        lam_eff *= 0.2
    if flags.target_bpp is not None and flags.target_bpp >= 1.0:
        lam_eff *= 0.5
    if flags.stroke_objective_level in {"mid", "max"}:
        lam_eff *= 0.9 if flags.stroke_objective_level == "mid" else 0.8

    no_improve_rounds = 0
    for round_idx in range(rounds):
        added_this_round = 0
        alpha = alpha_schedule[min(round_idx, len(alpha_schedule) - 1)]
        if flags.search_mode == "beam":
            for idx in range(tile_count):
                rec = records[idx]
                budget = int(budgets[idx])
                if len(rec.primitives) >= budget:
                    continue
                best = _beam_search_tile_models(
                    tile=tile_images[idx],
                    fit=fit,
                    tile_idx=idx,
                    rounds=min(rounds - round_idx, budget - len(rec.primitives)),
                    budget=budget,
                    beam_width=max(1, int(flags.beam_width)),
                    alpha_schedule=alpha_schedule,
                    candidate_bank=flags.candidate_bank,
                    active_models=active_models,
                    active_model_set=active_model_set,
                    lam_model=lam_model,
                    rate_bits_all=rate_bits_all,
                    rd_model_selection=flags.rd_model_selection,
                    lam_eff=lam_eff,
                    extra_candidates=extra_candidates_by_tile.get(idx),
                )
                if len(best) > len(rec.primitives):
                    records[idx] = TileRecord(primitives=best, residual_rgb=(0, 0, 0))
                    added_this_round += len(best) - len(rec.primitives)
        else:
            trial_tiles: list[int] = []
            trial_prims: list[Primitive] = []
            trial_scores: list[float] = []
            trial_better: list[bool] = []

            for idx in range(tile_count):
                rec = records[idx]
                if len(rec.primitives) >= int(budgets[idx]):
                    continue
                tile = tile_images[idx]
                base_obj, base_pred = _tile_objective(tile, rec.primitives, (0, 0, 0), lam_eff, res1_grid_size=4)
                base_dist = float(np.mean((tile - base_pred) ** 2))
                res2_floor = None
                if flags.enable_res2:
                    res2_floor = _res2_single_atom_objective(
                        tile=tile,
                        pred=base_pred,
                        primitives=rec.primitives,
                        lam=lam_eff,
                    )
                existing = {encode_primitive(p) for p in rec.primitives}
                model_order = _tile_model_order(
                    tile_idx=idx,
                    fit=fit,
                    active_models=active_models,
                    active_model_set=active_model_set,
                    lam_model=lam_model,
                    rate_bits_all=rate_bits_all,
                    rd_model_selection=flags.rd_model_selection,
                    topn=4,
                )

                for model in model_order:
                    c0 = fit.candidate_c0[idx, int(model), :]
                    c1 = fit.candidate_c1[idx, int(model), :]
                    cand = _quantize_primitive(
                        _primitive_from_model(
                            model=int(model),
                            c0=c0,
                            c1=c1,
                            alpha=alpha,
                            candidate_bank=flags.candidate_bank,
                        )
                    )
                    if encode_primitive(cand) in existing:
                        continue
                    test_prims = rec.primitives + [cand]
                    trial_obj, trial_pred = _tile_objective(tile, test_prims, (0, 0, 0), lam_eff, res1_grid_size=4)
                    trial_dist = float(np.mean((tile - trial_pred) ** 2))
                    if flags.edge_weighted_objective:
                        stroke_bonus = 0.01
                        if flags.stroke_objective_level == "mid":
                            stroke_bonus = 0.02
                        elif flags.stroke_objective_level == "max":
                            stroke_bonus = 0.03
                        trial_obj -= stroke_bonus * edge_strength[idx] * float(cand.kind != 0)
                    improves_dist = trial_dist + 1e-6 < base_dist
                    beats_res2 = True
                    if res2_floor is not None:
                        beats_res2 = bool(trial_obj + 1e-9 < float(res2_floor))
                    trial_tiles.append(idx)
                    trial_prims.append(cand)
                    trial_scores.append(float(trial_obj))
                    trial_better.append(bool((trial_obj + 1e-9 < float(base_obj) or improves_dist) and beats_res2))
                for cand in extra_candidates_by_tile.get(idx, []):
                    if encode_primitive(cand) in existing:
                        continue
                    test_prims = rec.primitives + [cand]
                    trial_obj, trial_pred = _tile_objective(tile, test_prims, (0, 0, 0), lam_eff, res1_grid_size=4)
                    trial_dist = float(np.mean((tile - trial_pred) ** 2))
                    if flags.edge_weighted_objective:
                        stroke_bonus = 0.01
                        if flags.stroke_objective_level == "mid":
                            stroke_bonus = 0.02
                        elif flags.stroke_objective_level == "max":
                            stroke_bonus = 0.03
                        trial_obj -= stroke_bonus * edge_strength[idx] * float(cand.kind != 0)
                    improves_dist = trial_dist + 1e-6 < base_dist
                    beats_res2 = True
                    if res2_floor is not None:
                        beats_res2 = bool(trial_obj + 1e-9 < float(res2_floor))
                    trial_tiles.append(idx)
                    trial_prims.append(cand)
                    trial_scores.append(float(trial_obj))
                    trial_better.append(bool((trial_obj + 1e-9 < float(base_obj) or improves_dist) and beats_res2))

            keep_idxs = cccl_compact(trial_better)
            if keep_idxs:
                kept_tiles = [trial_tiles[i] for i in keep_idxs]
                kept_prims = [trial_prims[i] for i in keep_idxs]
                kept_scores = [trial_scores[i] for i in keep_idxs]
                sel_local = cccl_segmented_topk(scores=kept_scores, segment_ids=kept_tiles, k=1, largest=False)
                for local_i in sel_local:
                    tile_idx = kept_tiles[local_i]
                    rec = records[tile_idx]
                    if len(rec.primitives) >= int(budgets[tile_idx]):
                        continue
                    rec.primitives.append(kept_prims[local_i])
                    records[tile_idx] = rec
                    added_this_round += 1

        # Budget-aware early exit: stop when rounds no longer add useful primitives.
        if added_this_round <= 0:
            no_improve_rounds += 1
        else:
            no_improve_rounds = 0
        if no_improve_rounds > int(flags.early_exit_patience):
            break

    if flags.search_mode == "mcmc":
        _mcmc_refine_hard_tiles(
            records=records,
            tile_images=tile_images,
            fit=fit,
            budgets=budgets,
            active_models=active_models,
            candidate_bank=flags.candidate_bank,
            rounds=rounds,
            lam_eff=lam_eff,
            mcmc_steps=int(flags.mcmc_steps),
            stochastic_restarts=int(flags.stochastic_restarts),
            edge_strength=edge_strength,
            extra_candidates_by_tile=extra_candidates_by_tile,
        )

    for i, rec in enumerate(records):
        if rec.primitives:
            continue
        model = int(active_models[int(np.argmin(fit.model_mse[i, active_models] + (lam_model * rate_bits_all[active_models])))])
        rec.primitives = [
            _quantize_primitive(
                _primitive_from_model(
                    model=model,
                    c0=fit.candidate_c0[i, model, :],
                    c1=fit.candidate_c1[i, model, :],
                    alpha=1.0,
                    candidate_bank=flags.candidate_bank,
                )
            )
        ]
        records[i] = rec
    prim_raw, toc = encode_tiles(records)
    prim_payload, chunk_index = build_prim_chunks(prim_raw=prim_raw, toc=toc, chunk_tiles=cfg.chunk_tiles)

    residuals = [(0, 0, 0)] * len(records) if cfg.enable_res0 else None
    residual_maps_bytes = [np.zeros((4, 4, 3), dtype=np.int8).reshape(-1).tobytes()] * len(records) if cfg.enable_res1 else None
    res2_atoms = 8
    if flags.res2_basis_blocks_level == "mid":
        res2_atoms = 16
    elif flags.res2_basis_blocks_level == "max":
        res2_atoms = 24
    if flags.residual_patch_borrow_level == "max":
        res2_atoms = max(res2_atoms, 28)
    res2_sparse = _encode_res2_sparse(tiles=tile_images, records=records, max_atoms_per_tile=res2_atoms) if flags.enable_res2 else None
    res2_payload = json.dumps(res2_sparse, separators=(",", ":"), sort_keys=True).encode("utf-8") if (flags.enable_res2 and flags.container_v2_blocks and res2_sparse is not None) else None
    pstr_payload, pdel_payload = build_primitive_side_streams(
        prim_raw=prim_raw,
        toc=toc,
        enable_split_streams=bool(flags.split_entropy_streams and flags.container_v2_blocks),
        enable_neighbor_delta=bool(flags.neighbor_delta_coding and flags.container_v2_blocks),
    )

    head_flags = FLAG_CHUNKED_PRIM
    if cfg.deterministic:
        head_flags |= FLAG_DETERMINISTIC
    if residuals is not None:
        head_flags |= FLAG_HAS_RES0
    if residual_maps_bytes is not None:
        head_flags |= FLAG_HAS_RES1

    head = HeadBlock(
        width=width,
        height=height,
        tile_size=cfg.tile_size,
        max_primitives=cfg.max_primitives,
        color_space=1,
        quant_mode=2,
        flags=head_flags,
        tile_cols=fit.tile_cols,
        tile_rows=fit.tile_rows,
        quality=quality,
        preset_id=preset_id,
    )

    gpu_stack = detect_gpu_stack()
    meta = {
        "preset": cfg.preset,
        "quality": quality,
        "entropy": cfg.entropy,
        "chunk_tiles": cfg.chunk_tiles,
        "chunked_prim": True,
        "prim_chunks": len(chunk_index),
        "head_flags": head_flags,
        "res1_grid_size": 4 if cfg.enable_res1 else 0,
        "gpu_direct_layout": True,
        "cccl_ready": bool(gpu_stack.cccl.available),
        "gpu_stack_encode": {
            "cuda": gpu_stack.cuda.available,
            "cccl": gpu_stack.cccl.available,
        },
        "deterministic": cfg.deterministic,
        "encoder_backend": "gpu-stage3-rd-multicandidate",
        "encoder_kernel_ms": fit.kernel_ms,
        "encoder_fit_passes": int(flags.maxcompute_fit_passes),
        "encoder_model_lambda": lam_model,
        "feature_flags": asdict(flags),
        "encoder_rounds": rounds,
        "encoder_search_mode": flags.search_mode,
        "encoder_beam_width": int(flags.beam_width),
        "encoder_mcmc_steps": int(flags.mcmc_steps),
        "encoder_stochastic_restarts": int(flags.stochastic_restarts),
        "encoder_early_exit_patience": int(flags.early_exit_patience),
        "encoder_budget_cap_per_tile": per_tile_cap,
        "encoder_budget_target_bpp": flags.target_bpp,
        "encoder_budget_total_primitives": int(sum(len(r.primitives) for r in records)),
        "hierarchical_tiling_level": flags.hierarchical_tiling_level,
        "hierarchical_active_tiles": int(len(hierarchical_tile_ids)),
        "hierarchical_extra_candidates_total": int(sum(len(v) for v in extra_candidates_by_tile.values())),
        "res2_enabled": bool(flags.enable_res2),
        "res2_sparse": res2_sparse,
        "container_v2_blocks": bool(flags.container_v2_blocks),
        "pstr_bytes": int(len(pstr_payload)) if pstr_payload is not None else 0,
        "pdel_bytes": int(len(pdel_payload)) if pdel_payload is not None else 0,
        "res2_bytes": int(len(res2_payload)) if res2_payload is not None else 0,
        "encoder_model_counts": {
            "const": int(sum(1 for r in records for p in r.primitives if p.kind == 0)),
            "linear": int(sum(1 for r in records for p in r.primitives if p.kind == 1)),
            "line": int(sum(1 for r in records for p in r.primitives if p.kind == 2)),
            "curve": int(sum(1 for r in records for p in r.primitives if p.kind == 3)),
            "triangle": int(sum(1 for r in records for p in r.primitives if p.kind == 4)),
        },
    }

    bitstream = encode_weft(
        head=head,
        toc=toc,
        prim_payload=prim_payload,
        residuals=residuals,
        residual_maps=residual_maps_bytes,
        res2_payload=res2_payload,
        res1_grid_size=4,
        pstr_payload=pstr_payload,
        pdel_payload=pdel_payload,
        chunk_index=chunk_index,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)

    bytes_written = os.path.getsize(output_path)
    bpp = bytes_written * 8.0 / float(width * height)
    return EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        tile_count=len(records),
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=float("nan"),
        ssim=None,
        decode_hash="gpu-stage3-rd-multicandidate",
        metadata={"config": asdict(cfg), "encoder_kernel_ms": fit.kernel_ms, "backend": fit.backend},
    )


def _encode_image_legacy(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    if cfg.preset not in {DEFAULT_PRESET, "stylized-v1"}:
        raise EncodeError(f"unsupported preset: {cfg.preset}")

    quality = _clamp_quality(cfg.quality)
    if cfg.tile_size != TILE_SIZE:
        raise EncodeError(f"v1 requires tile_size={TILE_SIZE}")

    if cfg.preset == "rtx-heavy-v2":
        cfg.max_primitives = max(cfg.max_primitives, 48)
        preset_id = 2
    else:
        preset_id = 1

    image, width, height = load_image_linear(input_path)
    tile_images, cols, rows = _extract_tiles(image, cfg.tile_size)

    t0 = time.perf_counter()
    records: list[TileRecord] = []
    residual_maps_f32: list[np.ndarray | None] = []
    for tile in tile_images:
        rec, res1 = _fit_tile(
            tile,
            quality=quality,
            max_primitives=cfg.max_primitives,
            enable_res0=cfg.enable_res0,
            enable_res1=cfg.enable_res1,
            res1_grid_size=4,
        )
        records.append(rec)
        residual_maps_f32.append(res1)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    prim_raw, toc = encode_tiles(records)
    chunk_index = None
    if cfg.entropy == "chunked-rans":
        prim_payload, chunk_index = build_prim_chunks(prim_raw=prim_raw, toc=toc, chunk_tiles=cfg.chunk_tiles)
    elif cfg.entropy == "rans":
        prim_payload = encode_bytes(prim_raw)
    else:
        prim_payload = prim_raw

    residuals = [r.residual_rgb for r in records] if cfg.enable_res0 else None
    residual_maps_bytes = [_res1_map_bytes(rm if rm is not None else np.zeros((4, 4, 3), dtype=np.float32)) for rm in residual_maps_f32] if cfg.enable_res1 else None

    head_flags = 0
    if cfg.deterministic:
        head_flags |= FLAG_DETERMINISTIC
    if residuals is not None:
        head_flags |= FLAG_HAS_RES0
    if residual_maps_bytes is not None:
        head_flags |= FLAG_HAS_RES1
    if chunk_index is not None:
        head_flags |= FLAG_CHUNKED_PRIM

    head = HeadBlock(
        width=width,
        height=height,
        tile_size=cfg.tile_size,
        max_primitives=cfg.max_primitives,
        color_space=1,
        quant_mode=2,
        flags=head_flags,
        tile_cols=cols,
        tile_rows=rows,
        quality=quality,
        preset_id=preset_id,
    )

    gpu_stack = detect_gpu_stack()
    meta = {
        "preset": cfg.preset,
        "quality": quality,
        "entropy": cfg.entropy,
        "chunk_tiles": cfg.chunk_tiles,
        "chunked_prim": bool(chunk_index),
        "prim_chunks": len(chunk_index) if chunk_index else 0,
        "head_flags": head_flags,
        "res1_grid_size": 4 if cfg.enable_res1 else 0,
        "gpu_direct_layout": True,
        "cccl_ready": True,
        "gpu_stack_encode": {
            "cuda": gpu_stack.cuda.available,
            "cccl": gpu_stack.cccl.available,
        },
        "deterministic": cfg.deterministic,
        "encoder_ms": encode_ms,
    }

    bitstream = encode_weft(
        head=head,
        toc=toc,
        prim_payload=prim_payload,
        residuals=residuals,
        residual_maps=residual_maps_bytes,
        res1_grid_size=4,
        chunk_index=chunk_index,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)

    # Validate reconstruction from quantized stream representation.
    roundtrip_tiles = decode_tiles(prim_raw, toc)
    if residuals is not None:
        for tile, res in zip(roundtrip_tiles, residuals):
            tile.residual_rgb = res
    recon_res_maps = [rm if rm is not None else np.zeros((4, 4, 3), dtype=np.float32) for rm in residual_maps_f32] if cfg.enable_res1 else None
    recon = render_scene_tiled(
        roundtrip_tiles,
        width=width,
        height=height,
        tile_size=cfg.tile_size,
        deblock=True,
        residual_maps=recon_res_maps,
        res1_grid_size=4,
    )

    bytes_written = os.path.getsize(output_path)
    bpp = bytes_written * 8.0 / float(width * height)
    report = EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        tile_count=len(records),
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=psnr(image, recon),
        ssim=ssim(image, recon),
        decode_hash=decode_hash(recon),
        metadata={"config": asdict(cfg), "encoder_ms": encode_ms},
    )
    return report


_PERSISTENT_WORKER_POOL = None  # Lazy-initialized ProcessPoolExecutor
_PERSISTENT_WORKER_POOL_WORKERS = 0


def _get_persistent_worker_pool():
    """Return a process-wide ProcessPoolExecutor, creating it lazily.

    Spawning 8 workers takes ~500 ms per encode; for sweep loops that
    encode many images in a single process this is pure waste. A
    module-level singleton amortizes the cost — the pool is created
    on the first encode and reused across all subsequent encodes in
    the same Python process.

    Returns ``None`` on any error (multiprocessing unavailable, etc.)
    so callers can fall back to the serial path.
    """
    global _PERSISTENT_WORKER_POOL, _PERSISTENT_WORKER_POOL_WORKERS
    if _PERSISTENT_WORKER_POOL is not None:
        return _PERSISTENT_WORKER_POOL
    try:
        from concurrent.futures import ProcessPoolExecutor
        n_workers = min(os.cpu_count() or 4, 8)
        _PERSISTENT_WORKER_POOL = ProcessPoolExecutor(max_workers=n_workers)
        _PERSISTENT_WORKER_POOL_WORKERS = n_workers
        return _PERSISTENT_WORKER_POOL
    except Exception:
        return None


def shutdown_worker_pool() -> None:
    """Shut down the persistent worker pool if one is running.

    Callers in long-lived processes (notebooks, servers) can invoke
    this to release worker processes and their memory. Tests and
    sweep scripts don't need to call this — Python will clean up
    at interpreter exit.
    """
    global _PERSISTENT_WORKER_POOL, _PERSISTENT_WORKER_POOL_WORKERS
    if _PERSISTENT_WORKER_POOL is not None:
        try:
            _PERSISTENT_WORKER_POOL.shutdown(wait=True)
        except Exception:
            pass
        _PERSISTENT_WORKER_POOL = None
        _PERSISTENT_WORKER_POOL_WORKERS = 0


def _worker_gen_cands(args: tuple) -> "tuple[list[Primitive], np.ndarray]":
    """Module-level worker for parallel candidate generation + packing.

    Must live at module scope (not as a nested closure inside
    ``_fit_adaptive_state``) so ``ProcessPoolExecutor`` can pickle it
    when dispatching to child workers. A closure silently fails to
    pickle and triggers the ``except Exception`` fallback, which
    runs candidate generation serially on one core — that was the
    latent bug this function exists to fix.

    Also runs ``_pack_prims`` in the worker (instead of the parent)
    so the ~10 s the parent used to spend per encode on per-tile
    struct packing now happens in parallel across the worker pool
    and rides home on the already-pickled return value.

    Takes a ``(patch, quality)`` tuple because process-pool workers
    can't close over the encoder's ``quality`` local. Returns
    ``(candidate_list, packed_ndarray)`` — the list is used for
    bookkeeping (primitive selection, rate estimation) and the
    packed array is used for GPU batch scoring.
    """
    from .gpu_render import _pack_prims, _PRIM_DTYPE
    patch, quality = args
    ce = generate_edge_driven_candidates(patch, quality)
    ct = _generate_candidates(patch, quality)
    cands = _dedup_candidates(ce + ct)
    packed = _pack_prims(cands) if cands else np.zeros(1, dtype=_PRIM_DTYPE)
    return cands, packed


def _fit_adaptive_state(input_path: str, cfg: EncodeConfig) -> _AdaptiveFitState:
    """Run the slow primitive-fit phase of the adaptive encoder.

    Output is a frozen ``_AdaptiveFitState`` containing the greedy
    primitive search result, the bicubic-replaced version (eagerly
    computed because 3/4 primitive-family auto-select variants need
    it), and all the geometry needed by ``_build_bitstream_from_fit``.

    Variants in auto_select that share the same image + fit-relevant
    config (quality, encode_scale, decompose_lighting) can call this
    once and pass the result to ``_encode_image_adaptive`` repeatedly,
    cutting auto_select wall time roughly in half.
    """
    from .quadtree import decompose_quadtree, extract_adaptive_tiles

    quality = _clamp_quality(cfg.quality)
    base_split_threshold = 0.12 if quality < 70 else 0.08 if quality < 90 else 0.03
    res1_grid = 4  # 4x4 is enough — budget goes to more primitives instead
    image, width, height = load_image_linear(input_path)

    # Keep the original source around for end-to-end PSNR measurement.
    # When encode_scale<1.0 we downscale ``image`` in place for the fitting
    # pipeline, but the user-facing PSNR (and the decoder's default output
    # resolution) should be computed against the ORIGINAL dimensions.
    source_image = image
    source_width, source_height = width, height
    encode_scale = max(0.1, min(1.0, float(cfg.encode_scale)))

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        enc_w = max(8, int(round(width * encode_scale)))
        enc_h = max(8, int(round(height * encode_scale)))

        # Bug 4: reject pathological scale × dimension combinations. A tile
        # grid with fewer than one 32×32 macro tile on either axis produces
        # a degenerate encode where the entire image collapses to at most a
        # few constant patches. Raise a clear error instead of silently
        # producing a useless bitstream.
        if enc_w < 32 or enc_h < 32:
            raise EncodeError(
                f"encode_scale={encode_scale} on {width}x{height} input "
                f"would produce a {enc_w}x{enc_h} encoded image, which is "
                f"smaller than a single macro tile (32x32). Use a larger "
                f"encode_scale or a larger source image."
            )
        # Soft warning: small but usable. Emit to stderr so batch runs
        # aren't silent about degenerate cases.
        n_macro_tiles = ((enc_w + 31) // 32) * ((enc_h + 31) // 32)
        if n_macro_tiles < 4:
            import sys
            print(
                f"weft: warning: encode_scale={encode_scale} on "
                f"{width}x{height} produces only {n_macro_tiles} macro "
                f"tile(s); expect very low reconstruction quality.",
                file=sys.stderr,
            )

        img_pil = PILImage.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        img_pil = img_pil.resize((enc_w, enc_h), PILImage.LANCZOS)
        image = np.array(img_pil).astype(np.float32) / 255.0
        width, height = enc_w, enc_h

    # Bug 3: scale split_threshold by encode_scale. LANCZOS downscaling
    # smooths fine detail so the unscaled threshold produces proportionally
    # fewer splits on a downscaled image than on a full-res one. Multiplying
    # the threshold by encode_scale keeps the splitting behavior roughly
    # proportional to the amount of structure in the downscaled content.
    split_threshold = base_split_threshold * encode_scale

    # Phase 2 #17: albedo+lighting decomposition.
    # When enabled, split the image into smooth-lighting and detail-rich
    # albedo via Retinex. The primitive pipeline runs against the albedo
    # (which has less structure to capture) while the lighting is stored
    # as a small low-res grid in BLOCK_LITE for the decoder to multiply
    # back at output time.
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    lighting_grid: np.ndarray | None = None
    lite_payload: bytes | None = None
    if flags.decompose_lighting:
        from .intrinsic import decompose_retinex, downsample_lighting
        from .bitstream import pack_lite as _pack_lite
        # The decomposition operates on the (possibly downscaled) image
        # the encoder is going to fit primitives against. The lighting
        # grid we store is in encoded-image space; the decoder upsamples
        # to source dimensions when applying it.
        albedo, lighting_full = decompose_retinex(image)
        image = albedo  # primitives now fit against the smoother albedo
        gh = max(4, min(int(flags.lighting_grid_size), min(image.shape[:2]) // 2))
        gw = gh  # square grid
        lighting_grid = downsample_lighting(lighting_full, gh, gw)
        lite_payload = _pack_lite(lighting_grid)

    # Phase 1: Adaptive quadtree decomposition.
    quad_tiles = decompose_quadtree(
        image,
        split_threshold=split_threshold,
        min_tile=8,
        max_tile=32,
    )
    tile_patches = extract_adaptive_tiles(image, quad_tiles)

    # Phase 2: Tile fitting — GPU batch path with CPU fallback.
    t0 = time.perf_counter()

    # Generate candidates for all tiles using a persistent process
    # pool. The worker is ``_worker_gen_cands`` at module level (see
    # note there) — it MUST NOT be a closure or pickling fails and
    # the whole thing runs serially on one core. Pass (patch, quality)
    # tuples so the worker doesn't need to close over the encoder's
    # locals. The pool is shared across all encodes in this process
    # so sweep loops don't pay the ~500 ms worker-spawn cost per call.
    worker_args = [(p, quality) for p in tile_patches]
    pool = _get_persistent_worker_pool()
    if pool is not None:
        try:
            worker_results = list(pool.map(_worker_gen_cands, worker_args, chunksize=64))
        except Exception:
            # If the pool dies (worker crash, broken state), tear it
            # down and fall back to serial for this encode.
            shutdown_worker_pool()
            worker_results = [_worker_gen_cands(a) for a in worker_args]
    else:
        worker_results = [_worker_gen_cands(a) for a in worker_args]

    # Worker returns (candidates_list, packed_ndarray) per tile.
    all_candidates: list[list[Primitive]] = [wr[0] for wr in worker_results]
    packed_cands: list[np.ndarray] = [wr[1] for wr in worker_results]

    cand_gen_ms = (time.perf_counter() - t0) * 1000.0

    # GPU-accelerated batch tile fitting.
    # Uses gpu_batch_objectives for scoring, processes all tiles per greedy round.
    lam = _tile_lambda(quality)
    n_tiles = len(quad_tiles)

    from .gpu_render import _pack_prims, _PRIM_DTYPE, gpu_batch_score_prepacked

    # Pre-pack all tile pixels into one flat buffer.
    tile_pixel_offsets_map: dict[int, int] = {}
    pixel_parts: list[np.ndarray] = []
    running_px = 0
    for i in range(n_tiles):
        flat = tile_patches[i].astype(np.float32, order="C").ravel()
        tile_pixel_offsets_map[i] = running_px
        pixel_parts.append(flat)
        running_px += len(flat)
    all_tile_pixels = np.concatenate(pixel_parts) if pixel_parts else np.zeros(1, dtype=np.float32)

    # Initialize: pick the best single candidate per tile — ONE batched launch.
    selected_per_tile: list[list[Primitive]] = [[] for _ in range(n_tiles)]
    remaining_per_tile: list[list[Primitive]] = [[] for _ in range(n_tiles)]
    remaining_packed: list[np.ndarray | None] = [None] * n_tiles  # Pre-packed remaining candidates.
    selected_packed: list[np.ndarray] = [np.zeros(0, dtype=_PRIM_DTYPE)] * n_tiles

    tiles_with_cands: list[int] = []
    init_eval_arrays: list[np.ndarray] = []
    init_eval_counts: list[np.ndarray] = []
    init_tile_sizes: list[int] = []
    for i in range(n_tiles):
        cands = all_candidates[i]
        if not cands:
            avg = tile_patches[i].mean(axis=(0, 1))
            selected_per_tile[i] = [Primitive(kind=0, geom=(), color0=_ct(avg), alpha=1.0)]
            selected_packed[i] = _pack_prims(selected_per_tile[i])
        else:
            tiles_with_cands.append(i)
            # Each candidate is scored as a 1-primitive tile.
            init_eval_arrays.append(packed_cands[i][:len(cands)])
            init_eval_counts.append(np.ones(len(cands), dtype=np.int32))
            init_tile_sizes.append(quad_tiles[i].size)

    if tiles_with_cands:
        init_mses = gpu_batch_score_prepacked(
            all_tile_pixels, tile_pixel_offsets_map,
            tiles_with_cands, init_tile_sizes,
            init_eval_arrays, init_eval_counts,
        )

        for j, i in enumerate(tiles_with_cands):
            cands = all_candidates[i]
            if init_mses is not None:
                # Vectorized rate computation: index _PRIM_BYTE_SIZES_LUT
                # by the packed-array's 'kind' column instead of a
                # Python list comprehension over the dataclass list.
                kinds = packed_cands[i]["kind"][:len(cands)]
                rates = (8 + _PRIM_BYTE_SIZES_LUT[kinds] * 8).astype(np.float32)
                objs = init_mses[j] + lam * rates
                best_idx = int(np.argmin(objs))
            else:
                best_idx = 0
            selected_per_tile[i] = [cands[best_idx]]
            selected_packed[i] = packed_cands[i][best_idx:best_idx + 1].copy()
            # Remaining = all packed candidates except the selected one.
            pc = packed_cands[i]
            remaining_packed[i] = np.concatenate([pc[:best_idx], pc[best_idx + 1:]]) if len(cands) > 1 else None
            remaining_per_tile[i] = cands[:best_idx] + cands[best_idx + 1:]

    # Compute initial objectives — prepacked single launch.
    current_objs = np.zeros(n_tiles, dtype=np.float64)
    try:
        all_indices = list(range(n_tiles))
        obj_eval_arrays = [selected_packed[i] for i in all_indices]
        obj_eval_counts = [np.array([len(selected_per_tile[i])], dtype=np.int32) for i in all_indices]
        obj_tile_sizes = [quad_tiles[i].size for i in all_indices]

        obj_mses = gpu_batch_score_prepacked(
            all_tile_pixels, tile_pixel_offsets_map,
            all_indices, obj_tile_sizes,
            obj_eval_arrays, obj_eval_counts,
        )
        if obj_mses is not None:
            for i in range(n_tiles):
                current_objs[i] = float(obj_mses[i][0]) + lam * _estimate_bits(selected_per_tile[i], False)
    except Exception:
        pass

    # Threshold: tiles where adding primitives could help.
    # At high quality, include more tiles for greedy refinement.
    complexity_threshold = 0.00005 if quality >= 90 else 0.0005
    complex_tiles = [i for i in range(n_tiles)
                     if current_objs[i] > complexity_threshold and remaining_per_tile[i]]

    # Greedy rounds with residual re-seeding.
    # When greedy stalls, generate new candidates targeting the remaining error,
    # then resume. This lets tiles accumulate many more primitives.
    max_rounds = 20 if quality >= 90 else 8 if quality >= 70 else 4
    stall_count = 0
    max_stalls = 3  # Re-seed up to 3 times before giving up.

    for _round in range(max_rounds):
        active_tiles: list[int] = []
        eval_arrays: list[np.ndarray] = []
        eval_counts: list[np.ndarray] = []
        active_tile_sizes: list[int] = []

        for i in complex_tiles:
            rp = remaining_packed[i]
            sp = selected_packed[i]
            pool = remaining_per_tile[i]
            if rp is None or len(pool) == 0 or len(selected_per_tile[i]) >= quad_tiles[i].max_primitives:
                continue
            active_tiles.append(i)
            active_tile_sizes.append(quad_tiles[i].size)

            n_pool = len(pool)
            n_sel = len(sp)
            stride = n_sel + 1
            trial_arr = np.zeros(n_pool * stride, dtype=_PRIM_DTYPE)
            for s in range(n_sel):
                trial_arr[s::stride] = sp[s]
            trial_arr[n_sel::stride] = rp[:n_pool]
            eval_arrays.append(trial_arr)
            eval_counts.append(np.full(n_pool, stride, dtype=np.int32))

        if not active_tiles:
            break

        batch_mses = gpu_batch_score_prepacked(
            all_tile_pixels, tile_pixel_offsets_map,
            active_tiles, active_tile_sizes,
            eval_arrays, eval_counts,
        )

        if batch_mses is None:
            break

        improved_any = False
        still_active: list[int] = []
        for j, i in enumerate(active_tiles):
            mses = batch_mses[j]
            pool = remaining_per_tile[i]
            sel = selected_per_tile[i]

            base_rate = _estimate_bits(sel, include_residual=False)
            # Vectorized marginal rate via kind lookup on the packed
            # remaining-candidates array.
            rp_for_marginal = remaining_packed[i]
            kinds = rp_for_marginal["kind"][:len(pool)]
            marginal = (_PRIM_BYTE_SIZES_LUT[kinds] * 8).astype(np.float32)
            rates = base_rate + marginal
            objs = mses + lam * rates

            best_idx = int(np.argmin(objs))
            if float(objs[best_idx]) < current_objs[i]:
                selected_per_tile[i] = sel + [pool[best_idx]]
                rp = remaining_packed[i]
                selected_packed[i] = np.concatenate([selected_packed[i], rp[best_idx:best_idx + 1]])
                remaining_packed[i] = np.concatenate([rp[:best_idx], rp[best_idx + 1:]]) if len(pool) > 1 else None
                remaining_per_tile[i] = pool[:best_idx] + pool[best_idx + 1:]
                current_objs[i] = float(objs[best_idx])
                improved_any = True
                still_active.append(i)

        if not improved_any:
            stall_count += 1
            if stall_count > max_stalls:
                break
            # Re-seed: generate residual-targeting candidates for stalled tiles.
            for i in complex_tiles:
                if len(selected_per_tile[i]) >= quad_tiles[i].max_primitives:
                    continue
                sel = selected_per_tile[i]
                pred = render_tile(sel, tile_size=quad_tiles[i].size)
                resid_cands = _generate_residual_candidates(tile_patches[i], pred, quality)
                if resid_cands:
                    new_packed = _pack_prims(resid_cands)
                    old_rp = remaining_packed[i]
                    remaining_packed[i] = np.concatenate([old_rp, new_packed]) if old_rp is not None else new_packed
                    remaining_per_tile[i] = remaining_per_tile[i] + resid_cands
        else:
            stall_count = 0
            complex_tiles = still_active

    # Per-tile hybrid pre-pass: fit a 4×4 bicubic Bézier patch per
    # tile and produce an alternative ``bicubic_replaced_per_tile``
    # where each tile is the better of {primitive list, single
    # PRIM_BICUBIC primitive} by R-D score. We compute it
    # unconditionally so the cached fit state covers both the
    # baseline path (which ignores it) and the hybrid* paths (which
    # use it). The cost is one bicubic fit + render + score per tile,
    # which is small compared to the greedy primitive search.
    from .bicubic import fit_tile as _bic_fit_tile, eval_tile as _bic_eval_tile
    bicubic_replaced_per_tile: list[list[Primitive]] = [list(p) for p in selected_per_tile]
    n_bicubic_tiles = 0
    # Per-tile bicubic costs: 1 prim_count byte + 1 kind byte + 1
    # length byte + 48 payload bytes = 51 bytes / 408 bits.
    bic_bits = 51 * 8
    for i, (qt, patch) in enumerate(zip(quad_tiles, tile_patches)):
        cp = _bic_fit_tile(patch)
        # Round-trip through u8 quantization so the score reflects
        # what the decoder will actually see (encode_primitive
        # quantizes the 48 control points to u8).
        cp_q = np.round(np.clip(cp, 0.0, 1.0) * 255.0) / 255.0
        bic_recon = _bic_eval_tile(cp_q, qt.size)
        bic_mse = float(np.mean((patch - bic_recon) ** 2))

        prim_recon = render_tile(selected_per_tile[i], tile_size=qt.size)
        prim_mse = float(np.mean((patch - prim_recon) ** 2))
        prim_bits = 8 + sum(_PRIM_BYTE_SIZES.get(p.kind, 20) for p in selected_per_tile[i]) * 8

        # R-D score: lower is better. ``lam`` is the same one the
        # greedy selector used, so the comparison is consistent.
        prim_score = prim_mse + lam * prim_bits
        bic_score = bic_mse + lam * bic_bits

        if bic_score < prim_score:
            bicubic_replaced_per_tile[i] = [Primitive(
                kind=5,                                    # PRIM_BICUBIC
                geom=tuple(float(v) for v in cp_q.flatten()),
                color0=(0.0, 0.0, 0.0),
                color1=None,
                alpha=1.0,
            )]
            n_bicubic_tiles += 1

    fit_ms = (time.perf_counter() - t0) * 1000.0

    return _AdaptiveFitState(
        source_image=source_image,
        source_width=source_width,
        source_height=source_height,
        image=image,
        width=width,
        height=height,
        encode_scale=encode_scale,
        quad_tiles=quad_tiles,
        tile_patches=tile_patches,
        selected_per_tile=selected_per_tile,
        bicubic_replaced_per_tile=bicubic_replaced_per_tile,
        n_bicubic_tiles=n_bicubic_tiles,
        quality=quality,
        lam=lam,
        split_threshold=split_threshold,
        res1_grid=res1_grid,
        lighting_grid=lighting_grid,
        lite_payload=lite_payload,
        cand_gen_ms=cand_gen_ms,
        fit_ms=fit_ms,
    )


def _fit_adaptive_state_cached(
    input_path: str, cfg: EncodeConfig,
) -> _AdaptiveFitState:
    """Disk-cache wrapper around ``_fit_adaptive_state``.

    When ``cfg.fit_cache_dir`` is set, look up a saved fit state by
    cache key (sha256 of input bytes + fit-relevant cfg) and return it
    on hit. On miss, compute the fit normally and persist it. When
    ``cfg.fit_cache_dir`` is None (the default) this is a thin
    pass-through to ``_fit_adaptive_state``.
    """
    if not cfg.fit_cache_dir:
        return _fit_adaptive_state(input_path, cfg)
    cache_dir = Path(cfg.fit_cache_dir)
    cache_path = cache_dir / f"{_fit_cache_key(input_path, cfg)}.pkl"
    cached = _load_fit_state(cache_path)
    if cached is not None:
        return cached
    state = _fit_adaptive_state(input_path, cfg)
    _save_fit_state(state, cache_path)
    return state


def _encode_image_adaptive(
    input_path: str,
    output_path: str,
    cfg: EncodeConfig,
    *,
    fit_state: _AdaptiveFitState | None = None,
) -> EncodeReport:
    """Adaptive encode pipeline: build phase on top of a cached fit.

    Uses quadtree tile decomposition, edge-driven candidates, least-
    squares color optimization, and cross-tile boundary refinement.

    The slow primitive-fit phase is factored into ``_fit_adaptive_state``;
    pass a precomputed ``fit_state`` to skip the fit and reuse a
    sibling variant's work (auto_select does this for the four
    primitive-family variants). When ``fit_state`` is None we compute
    it locally — preserving behavior for direct callers.
    """
    from .quadtree import pack_qtree
    from .render import render_scene_adaptive

    if fit_state is None:
        fit_state = _fit_adaptive_state_cached(input_path, cfg)

    t0 = time.perf_counter()  # build-phase start; used to extend encode_ms below

    # Unpack frozen state into locals so the existing build phase
    # below can stay structurally unchanged.
    source_image = fit_state.source_image
    source_width = fit_state.source_width
    source_height = fit_state.source_height
    image = fit_state.image
    width = fit_state.width
    height = fit_state.height
    encode_scale = fit_state.encode_scale
    quad_tiles = fit_state.quad_tiles
    tile_patches = fit_state.tile_patches
    quality = fit_state.quality
    lam = fit_state.lam
    split_threshold = fit_state.split_threshold
    res1_grid = fit_state.res1_grid
    lighting_grid = fit_state.lighting_grid
    lite_payload = fit_state.lite_payload
    cand_gen_ms = fit_state.cand_gen_ms
    encode_ms = fit_state.fit_ms

    flags = FeatureFlags.from_dict(cfg.feature_flags)
    # Pick the bicubic-replaced or pure-greedy version of the
    # primitive list per the variant's flag. ``selected_per_tile`` is
    # mutated downstream by the records-build loop, so always work on
    # a per-call copy to avoid corrupting the cached state across
    # auto_select variants.
    if flags.hybrid_bicubic_per_tile:
        selected_per_tile = [list(p) for p in fit_state.bicubic_replaced_per_tile]
        n_bicubic_tiles = fit_state.n_bicubic_tiles
    else:
        selected_per_tile = [list(p) for p in fit_state.selected_per_tile]
        n_bicubic_tiles = 0

    # Build records with RES0/RES1.
    records: list[TileRecord] = []
    residual_maps_f32: list[np.ndarray | None] = []
    for i, (qt, patch) in enumerate(zip(quad_tiles, tile_patches)):
        prims = selected_per_tile[i]
        rec = TileRecord(primitives=prims, residual_rgb=(0, 0, 0))

        if cfg.enable_res0:
            pred = render_tile(prims, tile_size=qt.size)
            bias = (patch - pred).mean(axis=(0, 1))
            rb = int(np.clip(np.round(float(bias[0]) * 255.0), -127, 127))
            gb = int(np.clip(np.round(float(bias[1]) * 255.0), -127, 127))
            bb = int(np.clip(np.round(float(bias[2]) * 255.0), -127, 127))
            rec.residual_rgb = (rb, gb, bb)

        res1 = None
        if cfg.enable_res1:
            pred = render_tile(prims, tile_size=qt.size, residual_rgb=rec.residual_rgb)
            err = patch - pred
            res1 = _quantize_residual_map(err, grid_size=res1_grid)

        records.append(rec)
        residual_maps_f32.append(res1)

    # Phase 3: Cross-tile boundary refinement (for uniform-size tile groups).
    # Group tiles by size and refine boundaries within each uniform block.
    size_groups: dict[int, list[int]] = {}
    for i, qt in enumerate(quad_tiles):
        size_groups.setdefault(qt.size, []).append(i)

    # encoder_ms reports the *total* primitive-fit + records-build time:
    # cached fit phase from the FitState plus the records-build pass we
    # just finished. Auto-select shares the fit cost across multiple
    # build calls, so the per-variant encoder_ms is mostly the cheap
    # build phase after the first variant pays the fit.
    encode_ms = fit_state.fit_ms + (time.perf_counter() - t0) * 1000.0

    # Phase 3.5: DCT residual layer (brainstorm #16). When enabled,
    # render each tile's primitive reconstruction (including RES0 +
    # RES1), compute the residual against the source patch, DCT-II
    # the residual per channel, and quantize to int16. Stored in a
    # separate BLOCK_DCT and added back to the primitive output at
    # decode time. Designed to capture what the primitive bases can't
    # represent — the dense high-frequency content that JPEG/WebP win
    # on. Quality knob controls the quant step.
    dct_payload: bytes | None = None
    dct_coeffs: np.ndarray | None = None
    dct_step: float = 0.0
    dct_freq_alpha: float = 0.0
    dct_chroma_mode: int = 0
    dct_skip_threshold: float = 0.0
    dct_presence_bitmask: bytes = b""
    dct_layout_used: int = 0
    dct_per_tile_scales: list[float] | None = None
    dct_per_tile_scales_u8: bytes = b""
    dct_residual_ms = 0.0
    n_dct_tiles = 0
    n_dct_tiles_present = 0
    n_dct_tiles_empty_mode = 0
    if flags.dct_residual:
        from .dct_residual import (
            encode_tile_residuals as _dct_encode,
            quant_step_for_quality as _dct_step_for_q,
            _present_indices_to_bitmask,
            permute_tile_to_band,
            encode_tile_scale_u8,
            decode_tile_scale_u8,
            rgb_to_ycbcr_residual,
        )
        from .bitstream import pack_dct as _pack_dct
        from scipy.fft import dctn as _dctn_fast
        t_dct = time.perf_counter()
        dct_step = (
            float(flags.dct_residual_step)
            if flags.dct_residual_step is not None
            else _dct_step_for_q(quality)
        )
        dct_freq_alpha = float(flags.dct_residual_freq_alpha)
        dct_chroma_mode = int(flags.dct_residual_chroma_mode)
        dct_skip_threshold = float(flags.dct_residual_skip_threshold)
        # Per-tile residual = source patch − full primitive reconstruction
        # (primitives + RES0 bias + RES1 spatial residual map).
        all_tile_residuals: list[np.ndarray] = []
        # Per-tile mode selection (feature flag ``dct_residual_per_tile_mode``):
        # for each tile we also compute the residual that would result if
        # we emptied the primitive list entirely, and keep whichever mode
        # has a lower R-D cost (``primitive_bytes + dct_L1_cost`` for
        # prim mode vs ``dct_L1_cost`` for empty mode). Tiles that flip
        # to empty mode get their primitives zeroed, their residual_rgb
        # cleared, and their RES1 grid dropped — the DCT layer carries
        # the full signal for those tiles. No format change: the
        # decoder renders an empty primitive list as zeros, then adds
        # the DCT residual on top, which reconstructs the source.
        per_tile_mode = bool(flags.dct_residual_per_tile_mode)

        def _tile_dct_cost_bits(residual: np.ndarray) -> float:
            """Proxy for the compressed bit cost of a residual.

            Counts nonzero quantized coefficients after freq-weighted
            quantization (matching the real encoder) and assigns each
            ~4 bits of compressed cost — that's the zstd ballpark for
            a literal int16 value after the block-level compression
            pass. Not a perfect model, but it tracks the real bitstream
            cost much better than sum-of-absolute-values (which
            double-counts large coefficients that zstd easily packs
            into 2-4 bytes).

            Uses the luma channel only — the primary signal — at the
            tile's native DCT size. Chroma modes 1/2 only change the
            ABSOLUTE cost of both modes uniformly, so the decision
            (``prim mode vs empty mode``) is stable.
            """
            if dct_chroma_mode == 0:
                y = residual[..., 0]
            else:
                y = rgb_to_ycbcr_residual(residual)[..., 0]
            n = y.shape[0]
            coeffs = _dctn_fast(y.astype(np.float64), type=2, norm="ortho")
            if dct_freq_alpha > 0 and n == y.shape[1]:
                from .dct_residual import freq_weights
                step_grid = dct_step * freq_weights(n, dct_freq_alpha)
            else:
                step_grid = dct_step
            q = np.round(coeffs / step_grid).astype(np.int32)
            # 4 bits per nonzero coefficient: empirical match to the
            # zstd-compressed int16 buffer on natural-photo tiles.
            return float(np.count_nonzero(q) * 4)

        for i, (qt, patch) in enumerate(zip(quad_tiles, tile_patches)):
            rec = records[i]
            res1_map = residual_maps_f32[i] if cfg.enable_res1 else None
            tile_recon = render_tile(
                rec.primitives, tile_size=qt.size, residual_rgb=rec.residual_rgb,
            )
            if res1_map is not None:
                from .render import upsample_residual_map
                tile_recon = np.clip(
                    tile_recon + upsample_residual_map(res1_map, tile_size=qt.size, grid_size=res1_grid),
                    0.0, 1.0,
                )
            residual_prim = (patch - tile_recon).astype(np.float32)

            if per_tile_mode and rec.primitives:
                # Cost-estimate both modes. Primitive mode pays the
                # stored primitive bytes plus the bit cost of its DCT
                # residual; empty mode pays only the DCT cost (which
                # is larger because the DC coefficient absorbs the
                # tile mean).
                residual_empty = patch.astype(np.float32)
                prim_dct_bits = _tile_dct_cost_bits(residual_prim)
                empty_dct_bits = _tile_dct_cost_bits(residual_empty)
                prim_bits = _estimate_bits(rec.primitives, include_residual=False)
                prim_mode_cost = prim_bits + prim_dct_bits
                empty_mode_cost = empty_dct_bits
                if empty_mode_cost < prim_mode_cost:
                    rec.primitives = []
                    rec.residual_rgb = (0, 0, 0)
                    residual_maps_f32[i] = None
                    all_tile_residuals.append(residual_empty)
                    n_dct_tiles_empty_mode += 1
                    continue
            all_tile_residuals.append(residual_prim)
        # Iteration 3: per-tile skip — drop tiles whose residual RMS is
        # below the perceptual noise floor. The presence bitmask flags
        # the kept tiles; absent tiles contribute zero coefficients to
        # the bitstream and are skipped at decode time.
        if dct_skip_threshold > 0:
            present_flags = [
                bool(np.sqrt(float(np.mean(res * res))) >= dct_skip_threshold)
                for res in all_tile_residuals
            ]
        else:
            present_flags = [True] * len(all_tile_residuals)
        per_tile_residuals = [
            res for res, p in zip(all_tile_residuals, present_flags) if p
        ]
        dct_presence_bitmask = _present_indices_to_bitmask(present_flags)
        n_dct_tiles = len(all_tile_residuals)
        n_dct_tiles_present = len(per_tile_residuals)

        # Iteration 4 (BLOCK_DCT v5): per-tile adaptive quant scale.
        #
        # ⚠ Experimental: as of 2026-04-07, neither the energy-based
        # nor the activity-based heuristic actually beats uniform-finer
        # at iso-bytes on the natural-photo corpus. Uniform is
        # approximately the R-D optimum for PSNR distortion on
        # roughly-IID-Gaussian residuals. The format support is in
        # place (BLOCK_DCT v5 carries one u8 scale per present tile,
        # log-spaced [0.25, 4.0]) so future heuristics — perceptual
        # masking, R-D-greedy water-filling, learned masks — can drop
        # in without a format break. The current heuristic below uses
        # source-patch activity gated at the 75th percentile so only
        # the top quartile of busy tiles get finer quant; this keeps
        # the byte overhead small while exposing the format end-to-end.
        if (
            flags.dct_residual_adaptive_quant
            and len(per_tile_residuals) > 0
        ):
            present_patches = [
                patch for patch, p in zip(tile_patches, present_flags) if p
            ]
            tile_act = np.array(
                [float(np.std(p)) for p in present_patches], dtype=np.float64,
            )
            if np.any(tile_act > 1e-9):
                ref_act = float(np.percentile(tile_act[tile_act > 1e-9], 75))
            else:
                ref_act = 1.0
            ref_act = max(ref_act, 1e-9)
            raw_scales = np.clip(ref_act / np.maximum(tile_act, 1e-9), 0.5, 1.0)
            scales_u8 = bytes(encode_tile_scale_u8(float(s)) for s in raw_scales)
            dct_per_tile_scales = [decode_tile_scale_u8(b) for b in scales_u8]
            dct_per_tile_scales_u8 = scales_u8
        else:
            dct_per_tile_scales = None
            dct_per_tile_scales_u8 = b""

        dct_coeffs, _offsets = _dct_encode(
            per_tile_residuals, quant_step=dct_step, freq_alpha=dct_freq_alpha,
            chroma_mode=dct_chroma_mode, per_tile_scales=dct_per_tile_scales,
        )
        # Pack the DCT coefficients twice — once tile-major (v5
        # layout=0) and once band-major zigzag (v5 layout=1) — then
        # keep whichever gives a smaller compressed on-disk block.
        # Band-major usually wins on natural photos with many tiles
        # per size group (−30% bytes on landscape.jpg), but tile-major
        # wins on small images where each size group only has a
        # handful of tiles and band clustering has no statistical
        # benefit. The zstd compression cost here is a small fraction
        # of total encoder time (the primitive fit dominates).
        present_sizes = [
            int(qt.size) for qt, p in zip(quad_tiles, present_flags) if p
        ]
        payload_tile = _pack_dct(
            dct_coeffs, quant_step=dct_step, channels=3,
            freq_alpha=dct_freq_alpha, chroma_mode=dct_chroma_mode,
            presence_bitmask=dct_presence_bitmask, n_tiles=n_dct_tiles,
            layout=0,
            per_tile_scales_u8=dct_per_tile_scales_u8,
        )
        dct_coeffs_band = permute_tile_to_band(
            dct_coeffs, present_sizes, dct_chroma_mode,
        )
        payload_band = _pack_dct(
            dct_coeffs_band, quant_step=dct_step, channels=3,
            freq_alpha=dct_freq_alpha, chroma_mode=dct_chroma_mode,
            presence_bitmask=dct_presence_bitmask, n_tiles=n_dct_tiles,
            layout=1,
            per_tile_scales_u8=dct_per_tile_scales_u8,
        )
        # Approximate "compressed size" via the block-level zstd pass
        # that will run during bitstream serialization. Since both
        # payloads only differ in the coefficient-ordering region,
        # comparing their zstd outputs directly is a faithful proxy
        # for which one the final bitstream will prefer.
        from .bitstream import _zstd_compress as _zc
        dct_payload = (
            payload_band
            if len(_zc(payload_band)) < len(_zc(payload_tile))
            else payload_tile
        )
        dct_layout_used = 1 if dct_payload is payload_band else 0
        dct_residual_ms = (time.perf_counter() - t_dct) * 1000.0

    # Phase 4: Serialize to bitstream.
    # For the adaptive path, we use the max tile_size in the HEAD but include
    # a QTREE block for the decoder to know the actual per-tile layout.
    prim_raw, toc = encode_tiles(records)
    chunk_index = None
    if cfg.entropy == "chunked-rans":
        prim_payload, chunk_index = build_prim_chunks(
            prim_raw=prim_raw, toc=toc, chunk_tiles=cfg.chunk_tiles,
        )
    elif cfg.entropy == "rans":
        prim_payload = encode_bytes(prim_raw)
    else:
        prim_payload = prim_raw

    residuals = [r.residual_rgb for r in records] if cfg.enable_res0 else None
    zero_grid = np.zeros((res1_grid, res1_grid, 3), dtype=np.float32)
    residual_maps_bytes = (
        [_res1_map_bytes(rm if rm is not None else zero_grid)
         for rm in residual_maps_f32]
        if cfg.enable_res1 else None
    )

    head_flags = 0
    if cfg.deterministic:
        head_flags |= FLAG_DETERMINISTIC
    if residuals is not None:
        head_flags |= FLAG_HAS_RES0
    if residual_maps_bytes is not None:
        head_flags |= FLAG_HAS_RES1
    if chunk_index is not None:
        head_flags |= FLAG_CHUNKED_PRIM
    if lite_payload is not None:
        head_flags |= FLAG_HAS_LITE
    if dct_payload is not None:
        head_flags |= FLAG_HAS_DCT

    # HeadBlock stores USER-VISIBLE (source) dimensions so the decoder's
    # default output resolution matches what the user started with. The
    # encoded (downscaled) dimensions live in meta.encode_width / encode_height
    # and are what the renderer uses as the primitive coordinate space.
    #
    # tile_cols=0, tile_rows=0 remains the adaptive-mode sentinel — the
    # decoder reads per-tile layout from the QTREE block in that case.
    head = HeadBlock(
        width=source_width,
        height=source_height,
        tile_size=16,  # nominal; actual sizes in QTREE block
        max_primitives=max(qt.max_primitives for qt in quad_tiles),
        color_space=1,
        quant_mode=2,
        flags=head_flags,
        tile_cols=0,  # not a regular grid
        tile_rows=0,
        quality=quality,
        preset_id=2,
    )

    qtree_payload = pack_qtree(quad_tiles)
    gpu_stack = detect_gpu_stack()

    # Tile size distribution for metadata.
    size_counts = {}
    for qt in quad_tiles:
        size_counts[qt.size] = size_counts.get(qt.size, 0) + 1

    meta = {
        "preset": cfg.preset,
        "quality": quality,
        "entropy": cfg.entropy,
        "chunk_tiles": cfg.chunk_tiles,
        "chunked_prim": bool(chunk_index),
        "prim_chunks": len(chunk_index) if chunk_index else 0,
        "head_flags": head_flags,
        "res1_grid_size": 4 if cfg.enable_res1 else 0,
        "hybrid_bicubic_per_tile": bool(flags.hybrid_bicubic_per_tile),
        "hybrid_bicubic_tile_count": n_bicubic_tiles,
        "dct_residual": bool(flags.dct_residual),
        "dct_residual_step": dct_step if dct_payload is not None else 0.0,
        "dct_residual_tile_count": n_dct_tiles,
        "dct_residual_tiles_present": n_dct_tiles_present,
        "dct_residual_tiles_empty_mode": n_dct_tiles_empty_mode if dct_payload is not None else 0,
        "dct_residual_skip_threshold": dct_skip_threshold if dct_payload is not None else 0.0,
        "dct_residual_payload_bytes": len(dct_payload) if dct_payload is not None else 0,
        "dct_residual_ms": dct_residual_ms,
        "dct_residual_chroma_mode": dct_chroma_mode if dct_payload is not None else 0,
        "dct_residual_layout": dct_layout_used if dct_payload is not None else 0,
        "gpu_direct_layout": True,
        "cccl_ready": True,
        "gpu_stack_encode": {
            "cuda": gpu_stack.cuda.available,
            "cccl": gpu_stack.cccl.available,
        },
        "deterministic": cfg.deterministic,
        "encoder_ms": encode_ms,
        "adaptive_tiling": True,
        "tile_count": len(quad_tiles),
        "tile_size_distribution": size_counts,
        "split_threshold": split_threshold,
        "qtree_payload_bytes": len(qtree_payload),
        "encode_scale": encode_scale,
        "source_width": source_width,
        "source_height": source_height,
        "encode_width": width,
        "encode_height": height,
        # Phase 2 #17: lighting decomposition metadata.
        "decompose_lighting": flags.decompose_lighting,
        "lighting_grid_h": int(lighting_grid.shape[0]) if lighting_grid is not None else 0,
        "lighting_grid_w": int(lighting_grid.shape[1]) if lighting_grid is not None else 0,
    }

    bitstream = encode_weft(
        head=head,
        toc=toc,
        prim_payload=prim_payload,
        residuals=residuals,
        residual_maps=residual_maps_bytes,
        res1_grid_size=res1_grid,
        qtree_payload=qtree_payload,
        lite_payload=lite_payload,
        dct_payload=dct_payload,
        chunk_index=chunk_index,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)

    # Validate reconstruction (skip if WEFT_FAST_ENCODE is set).
    bytes_written = os.path.getsize(output_path)
    recon_psnr = float("nan")
    recon_psnr_encoded = float("nan")
    recon_ssim: float | None = None

    if not os.environ.get("WEFT_FAST_ENCODE"):
        roundtrip_tiles = decode_tiles(prim_raw, toc)
        if residuals is not None:
            for tile, res in zip(roundtrip_tiles, residuals):
                tile.residual_rgb = res
        recon_res_maps = (
            [rm if rm is not None else zero_grid
             for rm in residual_maps_f32]
            if cfg.enable_res1 else None
        )
        recon = render_scene_adaptive(
            records=roundtrip_tiles,
            quad_tiles=quad_tiles,
            width=width,
            height=height,
            residual_maps=recon_res_maps,
            res1_grid_size=res1_grid,
        )
        # DCT residual layer (brainstorm #16) — additive on top of the
        # primitive + RES0 + RES1 reconstruction. Applied here so the
        # encoder's self-consistency PSNR matches what the decoder will
        # produce, keeping verify drift ~0.
        if dct_coeffs is not None:
            from .dct_residual import apply_residual_to_image
            recon = apply_residual_to_image(
                recon, dct_coeffs, quad_tiles,
                quant_step=dct_step, channels=3,
                freq_alpha=dct_freq_alpha,
                chroma_mode=dct_chroma_mode,
                presence_bitmask=dct_presence_bitmask,
                per_tile_scales=dct_per_tile_scales,
            )

        # Self-consistency PSNR: reconstruction vs. the (downscaled +
        # decomposed) target the encoder was actually fitting against.
        # This measures how well the primitive-fitting pipeline
        # approximates its working copy. When decomposition is on, this
        # measures fidelity to the ALBEDO, not to the source image.
        recon_psnr_encoded = psnr(image, recon)
        recon_ssim_encoded = ssim(image, recon)

        # User-facing PSNR (Bug 2): upscale the reconstruction back to
        # source resolution and compare against the ORIGINAL image. This
        # is what an end user actually sees after ``weft decode``.
        #
        # Pipeline: recon (encoded-space albedo if decomposing else
        # encoded-space full image) → multiply by upsampled lighting if
        # decomposition is on → upscale to source dims → compare to source.
        if lite_payload is not None and lighting_grid is not None:
            from .intrinsic import upsample_lighting
            lighting_full = upsample_lighting(lighting_grid, height, width)
            recon_lit = np.clip(recon * lighting_full, 0.0, 1.0)
        else:
            recon_lit = recon

        if encode_scale < 1.0:
            from PIL import Image as PILImage
            recon_u8 = np.clip(recon_lit * 255.0, 0.0, 255.0).astype(np.uint8)
            upscaled = PILImage.fromarray(recon_u8).resize(
                (source_width, source_height), PILImage.BILINEAR,
            )
            recon_source = np.asarray(upscaled, dtype=np.float32) / 255.0
            recon_psnr = psnr(source_image, recon_source)
            recon_ssim = ssim(source_image, recon_source)
            recon_for_hash = recon_source
        else:
            recon_psnr = psnr(source_image, recon_lit)
            recon_ssim = ssim(source_image, recon_lit)
            recon_for_hash = recon_lit

    # BPP relative to source (original) dimensions.
    src_pixels = source_width * source_height
    bpp = bytes_written * 8.0 / float(src_pixels)

    # Decode-in-the-loop verification (Phase 1 idea #30).
    #
    # Round-trip the just-written bitstream through the production
    # decoder and compute end-to-end PSNR/SSIM/hash against the source
    # image. The verified numbers replace the encoder's internal estimate
    # in the top-level EncodeReport so users get honest, end-to-end
    # quality. The encoder's own estimate is preserved in metadata as
    # ``psnr_software`` and the difference is exposed as
    # ``verify_drift_db`` for catching encoder/decoder divergence.
    verify_psnr = float("nan")
    verify_ssim_value: float | None = None
    verify_hash = ""
    verify_drift = float("nan")
    verify_failure: str | None = None
    psnr_software = recon_psnr
    if cfg.verify_decode and not os.environ.get("WEFT_FAST_ENCODE"):
        try:
            from .decoder import decode_to_array
            verify_array = decode_to_array(output_path)
            verify_psnr = psnr(source_image, verify_array)
            verify_ssim_value = ssim(source_image, verify_array)
            verify_hash = decode_hash(verify_array)
            if psnr_software == psnr_software:  # not NaN
                verify_drift = abs(psnr_software - verify_psnr)
        except Exception as exc:
            # The verify pass should never break encoding outright. Record
            # the failure in the report and continue with the encoder's
            # internal estimate.
            verify_psnr = float("nan")
            verify_drift = float("nan")
            verify_failure = f"{type(exc).__name__}: {exc}"

        if (
            verify_drift == verify_drift  # not NaN
            and verify_drift > cfg.verify_drift_threshold_db
        ):
            msg = (
                f"verify drift {verify_drift:.2f} dB exceeds threshold "
                f"{cfg.verify_drift_threshold_db:.2f} dB "
                f"(encoder estimate={psnr_software:.2f}, decoded={verify_psnr:.2f})"
            )
            if cfg.verify_strict:
                raise EncodeError(msg)
            import sys
            print(f"weft: warning: {msg}", file=sys.stderr)

    # Decide which numbers go in the user-facing report fields.
    # Verified results take precedence when available — they're what the
    # user actually sees after a real decode.
    if verify_psnr == verify_psnr:  # not NaN
        report_psnr = verify_psnr
        report_ssim = verify_ssim_value
        report_hash = verify_hash
    else:
        report_psnr = recon_psnr
        report_ssim = recon_ssim
        report_hash = decode_hash(recon_for_hash) if recon_for_hash is not None else ""

    if os.environ.get("WEFT_FAST_ENCODE"):
        report_ssim = None
        report_hash = ""

    report = EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=source_width,
        height=source_height,
        tile_count=len(records),
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=report_psnr,
        ssim=report_ssim,
        decode_hash=report_hash,
        metadata={
            "config": asdict(cfg),
            "encoder_ms": encode_ms,
            "adaptive_tiling": True,
            "tile_size_distribution": size_counts,
            "encode_scale": encode_scale,
            "source_width": source_width,
            "source_height": source_height,
            "encode_width": width,
            "encode_height": height,
            # Phase 2 #17: lighting decomposition.
            "decompose_lighting": flags.decompose_lighting,
            "lighting_grid_h": int(lighting_grid.shape[0]) if lighting_grid is not None else 0,
            "lighting_grid_w": int(lighting_grid.shape[1]) if lighting_grid is not None else 0,
            "lite_payload_bytes": len(lite_payload) if lite_payload is not None else 0,
            # Per-tile hybrid (PRIM_BICUBIC).
            "hybrid_bicubic_per_tile": bool(flags.hybrid_bicubic_per_tile),
            "hybrid_bicubic_tile_count": n_bicubic_tiles,
            # DCT residual layer (brainstorm #16).
            "dct_residual": bool(flags.dct_residual),
            "dct_residual_step": dct_step if dct_payload is not None else 0.0,
            "dct_residual_tile_count": n_dct_tiles,
            "dct_residual_tiles_present": n_dct_tiles_present,
            "dct_residual_tiles_empty_mode": n_dct_tiles_empty_mode if dct_payload is not None else 0,
            "dct_residual_skip_threshold": dct_skip_threshold if dct_payload is not None else 0.0,
            "dct_residual_payload_bytes": len(dct_payload) if dct_payload is not None else 0,
            "dct_residual_ms": dct_residual_ms,
            "dct_residual_chroma_mode": dct_chroma_mode if dct_payload is not None else 0,
            "dct_residual_layout": dct_layout_used if dct_payload is not None else 0,
            # psnr_encoded: encoder's own fidelity to its downscaled working copy.
            # psnr_software: encoder's bilinear-upscaled estimate vs source.
            #                Equals report.psnr when verify is off; differs
            #                when verify is on (then report.psnr = verify_psnr).
            "psnr_encoded": recon_psnr_encoded,
            "psnr_software": psnr_software,
            # Decode-in-the-loop verification (#30).
            "verify_psnr": verify_psnr,
            "verify_ssim": verify_ssim_value,
            "verify_decode_hash": verify_hash,
            "verify_drift_db": verify_drift,
            "verify_failure": verify_failure,
        },
    )
    return report


def encode_image(input_path: str, output_path: str, config: EncodeConfig | None = None) -> EncodeReport:
    """Encode an image to .weft.

    Dispatches to the adaptive-quadtree encoder by default. This path runs a
    multi-round greedy with residual re-seeding, produces bitstreams with a
    QTREE block, and gives 2-4 dB higher PSNR than the old single-pass
    ``_encode_image_gpu_baseline`` path on the ACCEPT-OURS test fixture.

    Escape hatches:
      * ``WEFT_BASELINE_ENCODE=1`` — force the single-pass baseline encoder
        (the previous default). Useful for comparing new research techniques
        against a faster/cruder baseline.
      * ``WEFT_ALLOW_LEGACY_CPU_ENCODE=1`` — force the deprecated CPU path.

    Feature-flag dispatch:
      * ``feature_flags.bicubic_patch_tiles=True`` selects the bicubic-patch
        encoder (brainstorm #11) — closed-form per-tile Bézier fit instead
        of greedy primitive search.
    """
    cfg = config or EncodeConfig()
    if os.environ.get("WEFT_BASELINE_ENCODE", "") == "1":
        return _encode_image_gpu_baseline(input_path=input_path, output_path=output_path, cfg=cfg)
    if os.environ.get("WEFT_ALLOW_LEGACY_CPU_ENCODE", "") == "1":
        return _encode_image_legacy(input_path=input_path, output_path=output_path, cfg=cfg)
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    if flags.auto_select:
        return _encode_image_auto(input_path=input_path, output_path=output_path, cfg=cfg)
    if flags.gradient_field:
        return _encode_image_gradient(input_path=input_path, output_path=output_path, cfg=cfg)
    if flags.palette_planes_k > 0:
        return _encode_image_palette(input_path=input_path, output_path=output_path, cfg=cfg)
    if flags.bicubic_patch_tiles:
        return _encode_image_bicubic(input_path=input_path, output_path=output_path, cfg=cfg)
    return _encode_image_adaptive(input_path=input_path, output_path=output_path, cfg=cfg)


def _encode_image_gradient(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    """Brainstorm #1: gradient-field + Poisson-decode encoder.

    The image is encoded as ``(∂I/∂x, ∂I/∂y)`` quantized to int8 plus
    three per-channel mean values. The decoder solves the Poisson
    equation ``∇²I = div(grad)`` via the discrete cosine transform —
    a single closed-form linear pass per channel.

    Wins on hard-edge / large-flat-region content where the gradient
    field is ~99% sparse and the bitstream-level zstd crushes the
    dense int8 maps. Loses on smooth-varying content where every pixel
    has a small but nonzero gradient that quantizes below the noise
    floor (~scale-dependent PSNR ceiling).
    """
    from .gradient_field import encode as grd_encode, decode as grd_decode

    quality = _clamp_quality(cfg.quality)
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    grd_scale = max(1, min(65535, int(flags.gradient_field_scale)))
    grd_threshold = float(flags.gradient_field_threshold)

    image, width, height = load_image_linear(input_path)
    source_image = image
    source_width, source_height = width, height
    encode_scale = max(0.1, min(1.0, float(cfg.encode_scale)))

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        enc_w = max(8, int(round(width * encode_scale)))
        enc_h = max(8, int(round(height * encode_scale)))
        if enc_w < 8 or enc_h < 8:
            raise EncodeError(
                f"encode_scale={encode_scale} on {width}x{height} input "
                f"would produce a {enc_w}x{enc_h} encoded image, smaller "
                f"than the 8x8 minimum supported by gradient mode."
            )
        img_pil = PILImage.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        img_pil = img_pil.resize((enc_w, enc_h), PILImage.LANCZOS)
        image = np.array(img_pil).astype(np.float32) / 255.0
        width, height = enc_w, enc_h

    t0 = time.perf_counter()
    gx_q, gy_q, means = grd_encode(image, scale=grd_scale, threshold=grd_threshold)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    from .bitstream import pack_grd
    grd_payload = pack_grd(gx_q, gy_q, means, scale=grd_scale)
    nonzero = int((gx_q != 0).sum() + (gy_q != 0).sum())
    total_grad_entries = gx_q.size + gy_q.size

    head = HeadBlock(
        width=source_width,
        height=source_height,
        tile_size=16,
        max_primitives=1,
        color_space=1,
        quant_mode=1,
        flags=FLAG_HAS_GRD,
        tile_cols=0,
        tile_rows=0,
        quality=quality,
        preset_id=1,            # brainstorm idea number
    )

    meta = {
        "preset": "gradient-field",
        "quality": quality,
        "encoder_ms": encode_ms,
        "adaptive_tiling": False,
        "gradient_field": True,
        "gradient_field_scale": grd_scale,
        "gradient_field_threshold": grd_threshold,
        "gradient_field_nonzero": nonzero,
        "gradient_field_total_entries": total_grad_entries,
        "gradient_field_sparsity_pct": (1.0 - nonzero / max(total_grad_entries, 1)) * 100.0,
        "tile_count": 0,
        "encode_scale": encode_scale,
        "source_width": source_width,
        "source_height": source_height,
        "encode_width": width,
        "encode_height": height,
        "grd_payload_bytes": len(grd_payload),
        "feature_flags": asdict(flags),
    }

    bitstream = encode_weft(
        head=head,
        toc=[0],
        prim_payload=b"",
        residuals=None,
        grd_payload=grd_payload,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)
    bytes_written = os.path.getsize(output_path)
    bpp = bytes_written * 8.0 / float(source_width * source_height)

    # Encoder-side reconstruction (matches what the decoder will produce
    # because both call gradient_field.decode on the same quantized data).
    recon_encoded = grd_decode(gx_q, gy_q, means, scale=grd_scale)
    recon_psnr_encoded = psnr(image, recon_encoded)

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        u8 = np.clip(recon_encoded * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize(
            (source_width, source_height), PILImage.BILINEAR,
        )
        recon_source = np.asarray(upscaled, dtype=np.float32) / 255.0
    else:
        recon_source = recon_encoded
    recon_psnr = psnr(source_image, recon_source)
    recon_ssim = ssim(source_image, recon_source)

    # Decode-in-the-loop verification.
    verify_psnr = float("nan")
    verify_ssim_value: float | None = None
    verify_hash = ""
    verify_drift = float("nan")
    verify_failure: str | None = None
    psnr_software = recon_psnr
    if cfg.verify_decode and not os.environ.get("WEFT_FAST_ENCODE"):
        try:
            from .decoder import decode_to_array
            verify_array = decode_to_array(output_path)
            verify_psnr = psnr(source_image, verify_array)
            verify_ssim_value = ssim(source_image, verify_array)
            verify_hash = decode_hash(verify_array)
            if psnr_software == psnr_software:
                verify_drift = abs(psnr_software - verify_psnr)
        except Exception as exc:
            verify_failure = f"{type(exc).__name__}: {exc}"
        if (
            verify_drift == verify_drift
            and verify_drift > cfg.verify_drift_threshold_db
        ):
            msg = (
                f"verify drift {verify_drift:.2f} dB exceeds threshold "
                f"{cfg.verify_drift_threshold_db:.2f} dB "
                f"(encoder estimate={psnr_software:.2f}, decoded={verify_psnr:.2f})"
            )
            if cfg.verify_strict:
                raise EncodeError(msg)
            import sys
            print(f"weft: warning: {msg}", file=sys.stderr)

    if verify_psnr == verify_psnr:
        report_psnr = verify_psnr
        report_ssim = verify_ssim_value
        report_hash = verify_hash
    else:
        report_psnr = recon_psnr
        report_ssim = recon_ssim
        report_hash = decode_hash(recon_source)

    return EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=source_width,
        height=source_height,
        tile_count=0,
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=report_psnr,
        ssim=report_ssim,
        decode_hash=report_hash,
        metadata={
            "config": asdict(cfg),
            "encoder_ms": encode_ms,
            "preset": "gradient-field",
            "gradient_field": True,
            "gradient_field_scale": grd_scale,
            "gradient_field_threshold": grd_threshold,
            "gradient_field_nonzero": nonzero,
            "gradient_field_sparsity_pct": (1.0 - nonzero / max(total_grad_entries, 1)) * 100.0,
            "adaptive_tiling": False,
            "encode_scale": encode_scale,
            "source_width": source_width,
            "source_height": source_height,
            "encode_width": width,
            "encode_height": height,
            "grd_payload_bytes": len(grd_payload),
            "psnr_encoded": recon_psnr_encoded,
            "psnr_software": psnr_software,
            "verify_psnr": verify_psnr,
            "verify_ssim": verify_ssim_value,
            "verify_decode_hash": verify_hash,
            "verify_drift_db": verify_drift,
            "verify_failure": verify_failure,
            "feature_flags": asdict(flags),
        },
    )


# Candidate registry for auto-select. Each entry is a (name, encoder_fn,
# feature_flag_overrides) tuple. The encoder runs each candidate against
# the input and picks the winner by R-D score.
# Each candidate is (name, encoder_fn, feature_flag_overrides, cfg_overrides).
# cfg_overrides is an optional dict of EncodeConfig field overrides — used
# by hybrid-dct-tight to drop the now-redundant RES1 residual-map pass when
# the DCT residual is on.
_AUTO_CANDIDATES: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
    # Baseline (primitive-stack adaptive encoder) is included as of the
    # OptiX-removal day; with the CPU renderer it decisively wins
    # synthetic-render and is competitive on natural photos at higher
    # rate-distortion lambdas. The empty overrides dict means "leave
    # all variant flags off, fall through the dispatcher to
    # _encode_image_adaptive". Listed first so it's the deterministic
    # tiebreaker when scores match exactly.
    ("baseline",   "_encode_image_adaptive", {}, {}),
    # Hybrid: baseline + per-tile bicubic R-D pick. Strictly >= baseline
    # PSNR (the per-tile selector only swaps when bicubic wins locally),
    # at the cost of a small bytes overhead. Wins over baseline at λ=4
    # on most natural photos.
    ("hybrid",     "_encode_image_adaptive", {"hybrid_bicubic_per_tile": True}, {}),
    # Hybrid + DCT residual (brainstorm #16). Adds a per-tile DCT
    # frequency-domain residual on top of the hybrid primitive layer.
    # Wins on natural-photo / dense-frequency content where the
    # primitive bases plateau in quality — auto picks this when the
    # R-D math says the extra bytes earn enough PSNR to overtake the
    # other candidates.
    ("hybrid-dct", "_encode_image_adaptive",
     {"hybrid_bicubic_per_tile": True, "dct_residual": True,
      "dct_residual_per_tile_mode": True}, {}),
    # Hybrid + DCT, RES1 disabled. RES1 (the per-tile 4×4 low-resolution
    # residual grid) is structurally redundant with the DCT residual —
    # the DCT's low-frequency coefficients express the same signal at
    # finer granularity. Dropping RES1 saves 5–25% bytes depending on
    # content, at the cost of 0.4–2 dB PSNR where the DCT quantization
    # can't quite cover the gap. At λ=4 this is a strict R-D win on
    # the committed corpus; auto selects it when bytes dominate.
    ("hybrid-dct-tight", "_encode_image_adaptive",
     {"hybrid_bicubic_per_tile": True, "dct_residual": True,
      "dct_residual_per_tile_mode": True}, {"enable_res1": False}),
    ("bicubic",    "_encode_image_bicubic",  {"bicubic_patch_tiles": True}, {}),
    ("palette-16", "_encode_image_palette",  {"palette_planes_k": 16}, {}),
    ("palette-64", "_encode_image_palette",  {"palette_planes_k": 64}, {}),
    # Gradient-field encoder (brainstorm #1). PDE-based reconstruction
    # via DCT Poisson solve. Wins highest absolute PSNR on hard-edge /
    # large-flat-region content (shapes, charts, screenshots) at
    # higher byte cost than palette; sometimes beats palette on R-D
    # when the per-bit gain from finer color resolution outweighs the
    # byte overhead.
    ("gradient",   "_encode_image_gradient", {"gradient_field": True}, {}),
]


def _encode_image_auto(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    """Auto-select encoder: try each candidate, keep the best by R-D score.

    Encodes the image with each variant in ``_AUTO_CANDIDATES``
    (baseline, bicubic, palette-16, palette-64), scores each result by
    ``PSNR - lambda * BPP``, and writes only the winning bitstream to
    ``output_path``. The bitstream is byte-identical to whichever
    single-variant encode won, so the standard decoder dispatches
    correctly without auto-aware changes.

    **Cost**: dominated by the baseline encoder which is the slowest
    candidate (greedy primitive search, 10-100s on the corpus). The
    bicubic and palette candidates each finish in ~1-3s, so total
    auto-select cost ≈ baseline cost. If you need faster encodes
    where baseline isn't competitive (small natural photos, screenshots),
    set ``feature_flags={"bicubic_patch_tiles": True}`` or pick palette
    directly.
    """
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    base_lam = float(flags.auto_select_lambda)

    # Scalable variants are the ones that render crisply at any
    # target resolution:
    #   * primitive-family paths without the raster DCT residual
    #     (baseline, hybrid), which use analytic primitives (const,
    #     linear, line, curve, triangle, bicubic patches)
    #   * bicubic (pure Bernstein basis)
    #   * palette-16 / palette-64 (nearest-neighbor upsample of the
    #     label grid — the right thing for hard-edge content)
    # These are excluded when ``prefer_scalable`` is set:
    #   * hybrid-dct / hybrid-dct-tight (DCT residual is per-pixel
    #     raster, blurs on upscale)
    #   * gradient (gradient fields are per-pixel raster)
    _NON_SCALABLE_VARIANTS = {"hybrid-dct", "hybrid-dct-tight", "gradient"}

    # Quality-aware R-D scoring.
    #
    # The user's ``cfg.quality`` knob is supposed to mean "I want this
    # much quality" — q=95 should produce higher PSNR than q=75. But
    # ``auto_select_lambda`` at a fixed value (default 4.0) gives a
    # fixed rate-distortion balance regardless of the user's intent,
    # which means at q=90+ on dense natural-photo content the math
    # picks ``palette-*`` (low PSNR, low BPP) over ``hybrid-dct-tight``
    # or ``gradient`` (high PSNR, high BPP) because the byte penalty
    # crushes the quality gain.
    #
    # Fix: scale lambda DOWN above q=75, leave it alone below. This
    # preserves the q=75 default (no behavior change at default) and
    # also q≤75 (where reducing rate pressure further would just push
    # the encoder onto strictly-worse rate-distortion points and
    # surprised users by collapsing PSNR for small byte savings).
    # Above q=75, the scale decays quadratically toward zero so the
    # encoder gets progressively more PSNR-greedy.
    #
    #   q ≤ 75: scale=1.00, lam=4    (preserved — no q-knob effect on lam)
    #   q=80:   scale=0.64, lam=2.56
    #   q=85:   scale=0.36, lam=1.44
    #   q=90:   scale=0.16, lam=0.64
    #   q=95:   scale=0.04, lam=0.16
    #   q=99:   scale=0.0016, lam≈0.006 (essentially "max PSNR")
    #
    # Asymmetric on purpose: at low quality the user already gets a
    # smaller bitstream from the encoder choosing simpler variants
    # (palette/bicubic) at the same lambda; we don't need to crank
    # rate pressure up further. Empirically, scaling lam UP at q=50
    # caused mixed.png to drop from 28.23 → 23.07 dB at q=50 because
    # palette-16 won the R-D race over hybrid-dct-tight, which is the
    # exact opposite of what we want.
    quality = _clamp_quality(cfg.quality)
    if quality > 75:
        lam_scale = ((100 - quality) / 25.0) ** 2
    else:
        lam_scale = 1.0
    lam = base_lam * lam_scale

    # Strip auto_select before recursing so we don't loop back here.
    base_flags = dict(cfg.feature_flags or {})
    base_flags.pop("auto_select", None)
    base_flags.pop("auto_select_lambda", None)

    # Encoder lookup — avoids forward references; all five candidate
    # encoders live in this same module.
    encoders = {
        "_encode_image_adaptive": _encode_image_adaptive,
        "_encode_image_bicubic":  _encode_image_bicubic,
        "_encode_image_palette":  _encode_image_palette,
        "_encode_image_gradient": _encode_image_gradient,
    }

    out_path_obj = Path(output_path)
    candidate_reports: list[tuple[str, float, EncodeReport, Path]] = []

    # Pre-compute the shared adaptive fit once. The four primitive-
    # family variants (baseline, hybrid, hybrid-dct, hybrid-dct-tight)
    # all start from the same greedy primitive search and only differ
    # in what they layer on top (per-tile bicubic R-D, RES1 grid, DCT
    # residual). Sharing the fit cuts auto-select wall time roughly
    # in half on adaptive-encoder-heavy fixtures.
    shared_fit: _AdaptiveFitState | None = None
    if any(fn_name == "_encode_image_adaptive" for _, fn_name, _, _ in _AUTO_CANDIDATES):
        # Build a "fit cfg" with variant-specific flags cleared so the
        # cache key (when fit_cache_dir is set) is stable across all
        # variant calls. The fit phase only reads quality, encode_scale,
        # decompose_lighting, and lighting_grid_size; everything else
        # (hybrid_bicubic_per_tile, dct_residual, enable_res0/1,
        # entropy, etc.) is build-only and doesn't affect the fit.
        fit_flags = dict(base_flags)
        for key in (
            "bicubic_patch_tiles", "palette_planes_k", "gradient_field",
            "hybrid_bicubic_per_tile", "dct_residual",
        ):
            fit_flags.pop(key, None)
        fit_cfg = EncodeConfig(**{**asdict(cfg), "feature_flags": fit_flags})
        shared_fit = _fit_adaptive_state_cached(input_path, fit_cfg)

    # Disable per-variant verify_decode: only the chosen winner is
    # re-verified after auto-select picks it, saving ~1 s per
    # loser variant on natural-photo fixtures.
    auto_verify_final = bool(cfg.verify_decode)

    for name, fn_name, overrides, cfg_overrides in _AUTO_CANDIDATES:
        if flags.prefer_scalable and name in _NON_SCALABLE_VARIANTS:
            continue
        sub_flags = dict(base_flags)
        # Clear other variant flags so each candidate runs cleanly even
        # if the user passed a stale combination (e.g. bicubic + palette).
        sub_flags["bicubic_patch_tiles"] = False
        sub_flags["palette_planes_k"] = 0
        sub_flags["gradient_field"] = False
        sub_flags["hybrid_bicubic_per_tile"] = False
        sub_flags["dct_residual"] = False
        sub_flags.update(overrides)
        sub_cfg = EncodeConfig(
            **{**asdict(cfg), **cfg_overrides, "feature_flags": sub_flags,
               "verify_decode": False},
        )
        tmp_path = out_path_obj.with_suffix(out_path_obj.suffix + f".{name}.tmp")
        if fn_name == "_encode_image_adaptive":
            rep = _encode_image_adaptive(
                input_path=input_path,
                output_path=str(tmp_path),
                cfg=sub_cfg,
                fit_state=shared_fit,
            )
        else:
            rep = encoders[fn_name](
                input_path=input_path,
                output_path=str(tmp_path),
                cfg=sub_cfg,
            )
        # Higher score is better. Guard against NaN PSNR.
        psnr_val = float(rep.psnr) if rep.psnr is not None and rep.psnr == rep.psnr else -1e9
        score = psnr_val - lam * float(rep.bits_per_pixel)
        candidate_reports.append((name, score, rep, tmp_path))

    # Pick the winner.
    candidate_reports.sort(key=lambda r: r[1], reverse=True)

    # Quality tiebreak: when the top R-D winner is significantly
    # behind a near-tied competitor on PSNR, prefer the
    # higher-quality variant. The rule:
    #
    #   1. Look at all candidates whose R-D score is within
    #      ``PSNR_TIEBREAK_WINDOW`` (5.0) of the top score.
    #   2. Find the highest-PSNR candidate in that set.
    #   3. If that candidate's PSNR is at least
    #      ``PSNR_TIEBREAK_MIN_GAIN`` (2.0 dB) above the R-D top
    #      winner's PSNR, the encoder picks it instead.
    #   4. Otherwise, the R-D winner stands.
    #
    # Why: at q=75 on dense saturated content (cyberpunk, mixed),
    # ``hybrid-dct-tight`` produces ~4 dB more PSNR than
    # ``palette-64`` but uses ~80% more bytes, so the R-D math at
    # λ=4 picks palette-64. The user perceives this as "q=75
    # produced lower PSNR than q=50". The 2 dB gain threshold
    # ensures the override only fires when there's a meaningful
    # quality jump available — close PSNR calls (e.g.
    # hybrid-dct-tight vs hybrid-dct, where RES1 adds ~0.7 dB) keep
    # the R-D winner and stay byte-efficient.
    PSNR_TIEBREAK_WINDOW = 5.0
    PSNR_TIEBREAK_MIN_GAIN = 2.0
    top_score = candidate_reports[0][1]
    top_psnr = (
        float(candidate_reports[0][2].psnr)
        if candidate_reports[0][2].psnr == candidate_reports[0][2].psnr
        else -1e9
    )
    candidates_in_window = [
        r for r in candidate_reports
        if r[1] >= top_score - PSNR_TIEBREAK_WINDOW
    ]
    best_in_window = max(
        candidates_in_window,
        key=lambda r: float(r[2].psnr) if r[2].psnr == r[2].psnr else -1e9,
    )
    best_in_window_psnr = (
        float(best_in_window[2].psnr)
        if best_in_window[2].psnr == best_in_window[2].psnr
        else -1e9
    )
    if best_in_window_psnr >= top_psnr + PSNR_TIEBREAK_MIN_GAIN:
        winner_name, winner_score, winner_rep, winner_path = best_in_window
    else:
        winner_name, winner_score, winner_rep, winner_path = candidate_reports[0]

    # Move winner into the requested output path; clean up losers.
    # The PSNR tiebreak above can move a non-top-R-D candidate into
    # the winner slot, so iterate over the full report list and skip
    # whichever path matches the chosen winner.
    if out_path_obj.exists():
        out_path_obj.unlink()
    winner_path.replace(out_path_obj)
    for _, _, _, loser_path in candidate_reports:
        if loser_path == winner_path:
            continue
        if loser_path.exists():
            loser_path.unlink()

    # Annotate the winner's report with the auto-select decision so
    # downstream tooling (sweep, experiments, R-D plots) can see which
    # variant won and why.
    winner_meta = dict(winner_rep.metadata)
    winner_meta["auto_select"] = True
    winner_meta["auto_select_lambda"] = lam  # effective (post-quality-scale)
    winner_meta["auto_select_lambda_base"] = base_lam
    winner_meta["auto_select_lambda_scale"] = lam_scale
    winner_meta["auto_selected_variant"] = winner_name
    winner_meta["auto_select_winner_score"] = winner_score
    winner_meta["auto_select_candidates"] = [
        {
            "name": n,
            "score": s,
            "psnr": float(r.psnr) if r.psnr is not None else float("nan"),
            "bpp": float(r.bits_per_pixel),
            "bytes": int(r.bytes_written),
        }
        for (n, s, r, _) in candidate_reports
    ]

    # Re-verify the winner (sub-variants were encoded with
    # verify_decode disabled to avoid paying ~1 s of decode per
    # losing variant). This is a drop-in replacement for the verify
    # block inside _encode_image_adaptive / _encode_image_* — we
    # only pay the decode cost once, on the file we're keeping.
    report_psnr = winner_rep.psnr
    report_ssim = winner_rep.ssim
    report_hash = winner_rep.decode_hash
    if auto_verify_final and not os.environ.get("WEFT_FAST_ENCODE"):
        try:
            from .decoder import decode_to_array
            source_image_for_verify, _sw, _sh = load_image_linear(input_path)
            verify_array = decode_to_array(str(out_path_obj))
            verify_psnr = float(psnr(source_image_for_verify, verify_array))
            verify_ssim_value = float(ssim(source_image_for_verify, verify_array))
            from .render import decode_hash as _decode_hash
            verify_hash = _decode_hash(verify_array)
            psnr_sw = winner_meta.get("psnr_software", float("nan"))
            if psnr_sw == psnr_sw:  # not NaN
                verify_drift = abs(psnr_sw - verify_psnr)
            else:
                verify_drift = float("nan")
            winner_meta["verify_psnr"] = verify_psnr
            winner_meta["verify_ssim"] = verify_ssim_value
            winner_meta["verify_decode_hash"] = verify_hash
            winner_meta["verify_drift_db"] = verify_drift
            winner_meta["verify_failure"] = None
            # The verified numbers take over the top-level report
            # fields — matching what the non-auto path does.
            report_psnr = verify_psnr
            report_ssim = verify_ssim_value
            report_hash = verify_hash
            if (
                verify_drift == verify_drift  # not NaN
                and verify_drift > cfg.verify_drift_threshold_db
            ):
                msg = (
                    f"auto-select winner verify drift {verify_drift:.2f} dB "
                    f"exceeds threshold {cfg.verify_drift_threshold_db:.2f} dB"
                )
                if cfg.verify_strict:
                    raise EncodeError(msg)
                import sys
                print(f"weft: warning: {msg}", file=sys.stderr)
        except EncodeError:
            raise
        except Exception as exc:
            winner_meta["verify_failure"] = f"{type(exc).__name__}: {exc}"

    return EncodeReport(
        input_path=winner_rep.input_path,
        output_path=str(out_path_obj),
        width=winner_rep.width,
        height=winner_rep.height,
        tile_count=winner_rep.tile_count,
        bits_per_pixel=winner_rep.bits_per_pixel,
        bytes_written=winner_rep.bytes_written,
        psnr=report_psnr,
        ssim=report_ssim,
        decode_hash=report_hash,
        metadata=winner_meta,
    )


def _encode_image_palette(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    """Brainstorm #20: K-color palette + per-pixel index encoder.

    No tiles, no quadtree, no per-region basis. The whole image is
    encoded as ``(palette[K], labels[H][W])``. Hard-edged content
    (screenshots, vector art, text, pixel art) is the regime this is
    built for — every step discontinuity is captured exactly up to the
    palette quantization. Smooth gradients suffer because there are
    only K colors, so this should LOSE on natural photos vs bicubic.

    The bitstream layout reuses the same envelope as bicubic: empty
    PRIM/TOC sentinel, no QTREE block, BLOCK_PAL carries the payload,
    HEAD flag FLAG_HAS_PAL marks the file.
    """
    from .palette import fit_palette, render_palette

    quality = _clamp_quality(cfg.quality)
    flags = FeatureFlags.from_dict(cfg.feature_flags)
    k = max(1, min(256, int(flags.palette_planes_k)))

    image, width, height = load_image_linear(input_path)
    source_image = image
    source_width, source_height = width, height
    encode_scale = max(0.1, min(1.0, float(cfg.encode_scale)))

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        enc_w = max(8, int(round(width * encode_scale)))
        enc_h = max(8, int(round(height * encode_scale)))
        if enc_w < 8 or enc_h < 8:
            raise EncodeError(
                f"encode_scale={encode_scale} on {width}x{height} input "
                f"would produce a {enc_w}x{enc_h} encoded image, smaller "
                f"than the 8x8 minimum supported by palette mode."
            )
        img_pil = PILImage.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        img_pil = img_pil.resize((enc_w, enc_h), PILImage.LANCZOS)
        image = np.array(img_pil).astype(np.float32) / 255.0
        width, height = enc_w, enc_h

    t0 = time.perf_counter()
    palette, labels = fit_palette(image, k)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    from .bitstream import pack_pal
    pal_payload = pack_pal(palette, labels)

    head = HeadBlock(
        width=source_width,
        height=source_height,
        tile_size=16,         # nominal; not used in PAL mode
        max_primitives=1,     # not used in PAL mode but must be > 0
        color_space=1,
        quant_mode=1,
        flags=FLAG_HAS_PAL,
        tile_cols=0,
        tile_rows=0,
        quality=quality,
        preset_id=20,         # brainstorm idea number
    )

    meta = {
        "preset": "palette-planes",
        "quality": quality,
        "encoder_ms": encode_ms,
        "adaptive_tiling": False,
        "palette_planes": True,
        "palette_planes_k": int(palette.shape[0]),
        "palette_planes_k_requested": int(k),
        "tile_count": 0,
        "encode_scale": encode_scale,
        "source_width": source_width,
        "source_height": source_height,
        "encode_width": width,
        "encode_height": height,
        "pal_payload_bytes": len(pal_payload),
        "feature_flags": asdict(flags),
    }

    bitstream = encode_weft(
        head=head,
        toc=[0],
        prim_payload=b"",
        residuals=None,
        pal_payload=pal_payload,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)
    bytes_written = os.path.getsize(output_path)
    bpp = bytes_written * 8.0 / float(source_width * source_height)

    # Encoder-side reconstruction.
    recon_encoded = render_palette(palette, labels)
    recon_psnr_encoded = psnr(image, recon_encoded)

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        u8 = np.clip(recon_encoded * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize(
            (source_width, source_height), PILImage.BILINEAR,
        )
        recon_source = np.asarray(upscaled, dtype=np.float32) / 255.0
    else:
        recon_source = recon_encoded
    recon_psnr = psnr(source_image, recon_source)
    recon_ssim = ssim(source_image, recon_source)

    # Decode-in-the-loop verification.
    verify_psnr = float("nan")
    verify_ssim_value: float | None = None
    verify_hash = ""
    verify_drift = float("nan")
    verify_failure: str | None = None
    psnr_software = recon_psnr
    if cfg.verify_decode and not os.environ.get("WEFT_FAST_ENCODE"):
        try:
            from .decoder import decode_to_array
            verify_array = decode_to_array(output_path)
            verify_psnr = psnr(source_image, verify_array)
            verify_ssim_value = ssim(source_image, verify_array)
            verify_hash = decode_hash(verify_array)
            if psnr_software == psnr_software:
                verify_drift = abs(psnr_software - verify_psnr)
        except Exception as exc:
            verify_failure = f"{type(exc).__name__}: {exc}"
        if (
            verify_drift == verify_drift
            and verify_drift > cfg.verify_drift_threshold_db
        ):
            msg = (
                f"verify drift {verify_drift:.2f} dB exceeds threshold "
                f"{cfg.verify_drift_threshold_db:.2f} dB "
                f"(encoder estimate={psnr_software:.2f}, decoded={verify_psnr:.2f})"
            )
            if cfg.verify_strict:
                raise EncodeError(msg)
            import sys
            print(f"weft: warning: {msg}", file=sys.stderr)

    if verify_psnr == verify_psnr:
        report_psnr = verify_psnr
        report_ssim = verify_ssim_value
        report_hash = verify_hash
    else:
        report_psnr = recon_psnr
        report_ssim = recon_ssim
        report_hash = decode_hash(recon_source)

    return EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=source_width,
        height=source_height,
        tile_count=0,
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=report_psnr,
        ssim=report_ssim,
        decode_hash=report_hash,
        metadata={
            "config": asdict(cfg),
            "encoder_ms": encode_ms,
            "preset": "palette-planes",
            "palette_planes": True,
            "palette_planes_k": int(palette.shape[0]),
            "palette_planes_k_requested": int(k),
            "adaptive_tiling": False,
            "encode_scale": encode_scale,
            "source_width": source_width,
            "source_height": source_height,
            "encode_width": width,
            "encode_height": height,
            "pal_payload_bytes": len(pal_payload),
            "psnr_encoded": recon_psnr_encoded,
            "psnr_software": psnr_software,
            "verify_psnr": verify_psnr,
            "verify_ssim": verify_ssim_value,
            "verify_decode_hash": verify_hash,
            "verify_drift_db": verify_drift,
            "verify_failure": verify_failure,
            "feature_flags": asdict(flags),
        },
    )


def _encode_image_bicubic(input_path: str, output_path: str, cfg: EncodeConfig) -> EncodeReport:
    """Brainstorm #11: closed-form bicubic-patch tile encoder.

    Each quadtree tile is approximated by a single 4×4 Bézier control grid
    fit by linear least squares — no greedy primitive search, no GPU
    fitting kernels, no residual blocks. The bitstream carries an empty
    PRIM/TOC (sentinel for "BIC mode") plus a BLOCK_BIC payload of all
    control grids and a QTREE block of tile layout. Decoder evaluates the
    bicubic in NumPy and skips OptiX entirely.
    """
    from .bicubic import fit_tiles, render_image as bicubic_render
    from .quadtree import decompose_quadtree, pack_qtree

    quality = _clamp_quality(cfg.quality)
    base_split_threshold = 0.12 if quality < 70 else 0.08 if quality < 90 else 0.03
    image, width, height = load_image_linear(input_path)
    source_image = image
    source_width, source_height = width, height
    encode_scale = max(0.1, min(1.0, float(cfg.encode_scale)))

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        enc_w = max(8, int(round(width * encode_scale)))
        enc_h = max(8, int(round(height * encode_scale)))
        if enc_w < 32 or enc_h < 32:
            raise EncodeError(
                f"encode_scale={encode_scale} on {width}x{height} input "
                f"would produce a {enc_w}x{enc_h} encoded image, smaller "
                f"than a single 32x32 macro tile."
            )
        img_pil = PILImage.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        img_pil = img_pil.resize((enc_w, enc_h), PILImage.LANCZOS)
        image = np.array(img_pil).astype(np.float32) / 255.0
        width, height = enc_w, enc_h

    split_threshold = base_split_threshold * encode_scale
    flags = FeatureFlags.from_dict(cfg.feature_flags)

    t0 = time.perf_counter()
    quad_tiles = decompose_quadtree(image, split_threshold=split_threshold, min_tile=8, max_tile=32)
    control_grids = fit_tiles(image, quad_tiles)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    qtree_payload = pack_qtree(quad_tiles)
    from .bitstream import pack_bic, unpack_bic
    bic_payload = pack_bic(control_grids)
    # Use the dequantized grids for the encoder-side reconstruction so the
    # software estimate matches what the decoder will actually produce.
    control_grids_q = unpack_bic(bic_payload)

    size_counts: dict[int, int] = {}
    for qt in quad_tiles:
        size_counts[qt.size] = size_counts.get(qt.size, 0) + 1

    head = HeadBlock(
        width=source_width,
        height=source_height,
        tile_size=16,
        max_primitives=1,  # not used in BIC mode but must be > 0
        color_space=1,
        quant_mode=1,
        flags=FLAG_HAS_BIC,
        tile_cols=0,
        tile_rows=0,
        quality=quality,
        preset_id=11,  # brainstorm idea number — easy to grep for in dumps
    )

    meta = {
        "preset": "bicubic-patch",
        "quality": quality,
        "encoder_ms": encode_ms,
        "adaptive_tiling": True,
        "bicubic_patch_tiles": True,
        "tile_count": len(quad_tiles),
        "tile_size_distribution": size_counts,
        "split_threshold": split_threshold,
        "qtree_payload_bytes": len(qtree_payload),
        "bic_payload_bytes": len(bic_payload),
        "encode_scale": encode_scale,
        "source_width": source_width,
        "source_height": source_height,
        "encode_width": width,
        "encode_height": height,
        "feature_flags": asdict(flags),
    }

    bitstream = encode_weft(
        head=head,
        toc=[0],          # empty PRIM TOC sentinel
        prim_payload=b"",
        residuals=None,
        qtree_payload=qtree_payload,
        bic_payload=bic_payload,
        block_alignment=cfg.block_alignment,
        meta=meta,
    )
    with open(output_path, "wb") as f:
        f.write(bitstream)
    bytes_written = os.path.getsize(output_path)
    bpp = bytes_written * 8.0 / float(source_width * source_height)

    # Encoder-side reconstruction (working-copy / encoded-space PSNR).
    recon_encoded = bicubic_render(control_grids_q, quad_tiles, width=width, height=height)
    recon_psnr_encoded = psnr(image, recon_encoded)

    if encode_scale < 1.0:
        from PIL import Image as PILImage
        u8 = np.clip(recon_encoded * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize(
            (source_width, source_height), PILImage.BILINEAR,
        )
        recon_source = np.asarray(upscaled, dtype=np.float32) / 255.0
    else:
        recon_source = recon_encoded
    recon_psnr = psnr(source_image, recon_source)
    recon_ssim = ssim(source_image, recon_source)

    # Decode-in-the-loop verification.
    verify_psnr = float("nan")
    verify_ssim_value: float | None = None
    verify_hash = ""
    verify_drift = float("nan")
    verify_failure: str | None = None
    psnr_software = recon_psnr
    if cfg.verify_decode and not os.environ.get("WEFT_FAST_ENCODE"):
        try:
            from .decoder import decode_to_array
            verify_array = decode_to_array(output_path)
            verify_psnr = psnr(source_image, verify_array)
            verify_ssim_value = ssim(source_image, verify_array)
            verify_hash = decode_hash(verify_array)
            if psnr_software == psnr_software:
                verify_drift = abs(psnr_software - verify_psnr)
        except Exception as exc:
            verify_failure = f"{type(exc).__name__}: {exc}"
        if (
            verify_drift == verify_drift
            and verify_drift > cfg.verify_drift_threshold_db
        ):
            msg = (
                f"verify drift {verify_drift:.2f} dB exceeds threshold "
                f"{cfg.verify_drift_threshold_db:.2f} dB "
                f"(encoder estimate={psnr_software:.2f}, decoded={verify_psnr:.2f})"
            )
            if cfg.verify_strict:
                raise EncodeError(msg)
            import sys
            print(f"weft: warning: {msg}", file=sys.stderr)

    if verify_psnr == verify_psnr:
        report_psnr = verify_psnr
        report_ssim = verify_ssim_value
        report_hash = verify_hash
    else:
        report_psnr = recon_psnr
        report_ssim = recon_ssim
        report_hash = decode_hash(recon_source)

    return EncodeReport(
        input_path=input_path,
        output_path=output_path,
        width=source_width,
        height=source_height,
        tile_count=len(quad_tiles),
        bits_per_pixel=bpp,
        bytes_written=bytes_written,
        psnr=report_psnr,
        ssim=report_ssim,
        decode_hash=report_hash,
        metadata={
            "config": asdict(cfg),
            "encoder_ms": encode_ms,
            "preset": "bicubic-patch",
            "bicubic_patch_tiles": True,
            "adaptive_tiling": True,
            "tile_size_distribution": size_counts,
            "encode_scale": encode_scale,
            "source_width": source_width,
            "source_height": source_height,
            "encode_width": width,
            "encode_height": height,
            "bic_payload_bytes": len(bic_payload),
            "qtree_payload_bytes": len(qtree_payload),
            "psnr_encoded": recon_psnr_encoded,
            "psnr_software": psnr_software,
            "verify_psnr": verify_psnr,
            "verify_ssim": verify_ssim_value,
            "verify_decode_hash": verify_hash,
            "verify_drift_db": verify_drift,
            "verify_failure": verify_failure,
            "feature_flags": asdict(flags),
        },
    )
