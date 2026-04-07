"""DCT-based per-tile residual encoder + decoder — brainstorm idea #16.

A frequency-domain residual layer that sits on top of any tile-based
encoder. After the primitive layer captures the image structure, the
residual (source minus reconstruction) is DCT-transformed per tile,
quantized to int8, and stored. The decoder unpacks the residual,
inverse-DCTs each tile, and adds it to the primitive reconstruction.

Why bother:

* The primitive / bicubic / palette bases all struggle on dense
  high-frequency content (natural photos, fine textures). The
  iso-bytes JPEG/WebP benchmark on the committed corpus shows WEFT
  losing by 6-11 dB on those regimes — and JPEG IS a DCT codec, so
  the gap is exactly the value of the missing basis.
* A DCT residual is *additive* — it doesn't replace any existing
  basis, so it can't make the encoder strictly worse. The encoder
  decides how aggressively to quantize based on a quality knob.

Implementation:

* Per quadtree tile: full-tile DCT-II via ``scipy.fft.dctn`` per
  channel. Tile sizes are 8/16/32 in adaptive mode and the basis
  matrices are computed implicitly by scipy.
* Uniform scalar quantization with a step derived from the encoder
  quality knob (lower quality = coarser step = fewer bytes).
* Coefficients stored as int16 in tile-row-major order. Deferring
  the entropy coding to the bitstream-level zstd pass (on hard-edge
  content the high-frequency residual coefficients are mostly zero,
  so zstd's RLE finds them efficiently).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.fft import dctn, idctn


# ── BT.601 RGB ↔ YCbCr matrices for residual data ──────────────────────
#
# Standard JPEG/JFIF colorspace conversion. We work with RESIDUALS
# (centered around zero), so we use the linear matrix without the
# 128-offset that JFIF applies for absolute pixel values. The Cb and
# Cr channels for residuals are still ~10× less perceptually important
# than Y, so subsampling them 2× per axis is nearly free visually.

_RGB_TO_YCBCR = np.array([
    [ 0.299,     0.587,     0.114    ],
    [-0.168736, -0.331264,  0.5      ],
    [ 0.5,      -0.418688, -0.081312],
], dtype=np.float64)

_YCBCR_TO_RGB = np.array([
    [1.0,  0.0,        1.402    ],
    [1.0, -0.344136,  -0.714136],
    [1.0,  1.772,      0.0     ],
], dtype=np.float64)


def rgb_to_ycbcr_residual(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB residual (h, w, 3) to YCbCr residual via BT.601."""
    return rgb.astype(np.float64) @ _RGB_TO_YCBCR.T


def ycbcr_to_rgb_residual(ycbcr: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_ycbcr_residual."""
    return ycbcr.astype(np.float64) @ _YCBCR_TO_RGB.T


def subsample_2x(arr: np.ndarray) -> np.ndarray:
    """2× box-average downsample of an (h, w) array. Both dims must be even."""
    h, w = arr.shape
    if h % 2 or w % 2:
        raise ValueError(f"subsample_2x requires even dims, got {arr.shape}")
    return arr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def upsample_2x(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """2× nearest-neighbor upsample of an (h, w) array to target dims.

    Nearest-neighbor (rather than bilinear) so the upsample is the
    exact inverse of the box-average downsample for constant blocks.
    For non-constant blocks the reconstruction has half-pixel
    aliasing artifacts on the chroma channel, but that's perceptually
    invisible because the eye doesn't resolve chroma at the pixel
    level.
    """
    return np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)[:target_h, :target_w]


@lru_cache(maxsize=16)
def zigzag_flat_indices(N: int) -> np.ndarray:
    """Return the zigzag-order flat indices for an N×N coefficient block.

    The result is a ``(N*N,)`` int array where ``result[k]`` is the
    row-major flat index of the k-th zigzag coefficient. DC (0, 0) is
    first and the highest-frequency corner (N-1, N-1) is last.

    JPEG uses this ordering on 8×8 blocks to cluster low-frequency
    coefficients (typically nonzero) at the start of the stream and
    high-frequency coefficients (typically zero) at the end. For
    WEFT's band-major layout the zigzag gives a similar benefit:
    after transposing tiles × coefficients, each "band row" now
    contains coefficients of similar perceptual significance, which
    makes the overall value distribution more uniform and improves
    zstd's match-finding on adjacent tiles.
    """
    if N < 1:
        raise ValueError(f"zigzag_flat_indices N must be >= 1, got {N}")
    order: list[int] = []
    for s in range(2 * N - 1):
        if s % 2 == 0:
            # Anti-diagonal going up-right: (s, 0) → (0, s)
            for i in range(min(s, N - 1), max(-1, s - N), -1):
                j = s - i
                if 0 <= j < N:
                    order.append(i * N + j)
        else:
            for i in range(max(0, s - N + 1), min(s, N - 1) + 1):
                j = s - i
                if 0 <= j < N:
                    order.append(i * N + j)
    return np.asarray(order, dtype=np.int64)


@lru_cache(maxsize=16)
def _inverse_zigzag_flat_indices(N: int) -> np.ndarray:
    """Inverse of ``zigzag_flat_indices``: row-major → zigzag position."""
    fwd = zigzag_flat_indices(N)
    inv = np.empty_like(fwd)
    inv[fwd] = np.arange(fwd.size)
    return inv


def _tile_channel_sizes(size: int, chroma_mode: int) -> list[int]:
    """Return the per-channel block size list for a tile of the given
    size under a given chroma_mode.

    * ``chroma_mode=0`` (RGB): three full-size channels.
    * ``chroma_mode=1`` (YCbCr 4:4:4): three full-size channels.
    * ``chroma_mode=2`` (YCbCr 4:2:0): Y full size, Cb and Cr at N/2.
    """
    if chroma_mode in (0, 1):
        return [size, size, size]
    if chroma_mode == 2:
        return [size, size // 2, size // 2]
    raise ValueError(f"bad chroma_mode: {chroma_mode}")


def permute_tile_to_band(
    coeffs: np.ndarray,
    tile_sizes: list[int],
    chroma_mode: int,
) -> np.ndarray:
    """Reorder a tile-major DCT coefficient stream to band-major zigzag.

    Input layout (tile-major): per-tile channel blocks concatenated in
    QTREE iteration order (what ``encode_tile_residuals`` emits).

    Output layout (band-major): tiles grouped by (tile_size, channel).
    Within each group, coefficients are interleaved by zigzag position
    — i.e. "DC across all tiles, then (1,0) across all tiles, …".

    Empirically this roughly halves the zstd-compressed DCT payload on
    natural-photo content because coefficients at the same frequency
    position have a much more uniform value distribution across tiles
    than the mixed distribution in a single tile.
    """
    if chroma_mode not in (0, 1, 2):
        raise ValueError(f"bad chroma_mode: {chroma_mode}")

    # 1) Slice the tile-major buffer into per-(tile, channel) blocks.
    per_tile_blocks: list[list[np.ndarray]] = []
    cursor = 0
    for size in tile_sizes:
        ch_sizes = _tile_channel_sizes(size, chroma_mode)
        blocks = []
        for n in ch_sizes:
            count = n * n
            blocks.append(coeffs[cursor:cursor + count].reshape(n, n))
            cursor += count
        per_tile_blocks.append(blocks)
    if cursor != coeffs.size:
        raise ValueError(
            f"tile-major coefficient buffer size mismatch: "
            f"used {cursor}, got {coeffs.size}"
        )

    # 2) Group tiles by (tile_size, channel_index, channel_size).
    # For chroma modes 0/1 every channel has the same size as the tile;
    # for mode 2 channels 1,2 are half-size. We group by (tile_size,
    # channel_index) so tiles of different sizes stay in separate bands.
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for (size, blocks) in zip(tile_sizes, per_tile_blocks):
        for ch_idx, block in enumerate(blocks):
            groups.setdefault((size, ch_idx), []).append(block)

    # 3) For each group, stack → zigzag → transpose → flatten.
    out_chunks: list[np.ndarray] = []
    for (tile_size, ch_idx) in sorted(groups.keys()):
        stacked = np.stack(groups[(tile_size, ch_idx)], axis=0)  # (n_tiles, n, n)
        n = stacked.shape[1]
        flat = stacked.reshape(stacked.shape[0], n * n)           # (n_tiles, n*n)
        zz = zigzag_flat_indices(n)
        zigged = flat[:, zz]                                      # (n_tiles, n*n) in zigzag order
        # Transpose to (n*n, n_tiles) so "all DCs" come first, then
        # "all (1,0)s", etc.
        band = zigged.T.reshape(-1)
        out_chunks.append(band)

    if not out_chunks:
        # Empty input — every tile was filtered by the presence bitmask
        # before reaching us, so there are no coefficients to permute.
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(out_chunks).astype(np.int16)


def permute_band_to_tile(
    coeffs: np.ndarray,
    tile_sizes: list[int],
    chroma_mode: int,
) -> np.ndarray:
    """Inverse of ``permute_tile_to_band``.

    Vectorized: precomputes the per-tile-channel output offsets, then
    for each (size, channel) group does the un-zigzag and scatters
    every member's block into the tile-major output array via a
    single fancy-index assignment. Avoids the per-tile Python loops
    of the previous list-of-lists version.
    """
    if chroma_mode not in (0, 1, 2):
        raise ValueError(f"bad chroma_mode: {chroma_mode}")
    n_tiles = len(tile_sizes)

    if n_tiles == 0:
        return np.zeros(0, dtype=np.int16)

    # Precompute per-tile, per-channel output offsets in the
    # tile-major flat output buffer.
    per_tile_ch_offsets: list[list[int]] = []
    cur = 0
    for size in tile_sizes:
        ch_sizes = _tile_channel_sizes(size, chroma_mode)
        offs = []
        for n in ch_sizes:
            offs.append(cur)
            cur += n * n
        per_tile_ch_offsets.append(offs)
    total_size = cur

    out = np.empty(total_size, dtype=np.int16)

    # Group tile indices by (tile_size, channel_index).
    group_members: dict[tuple[int, int], list[int]] = {}
    for tile_idx, size in enumerate(tile_sizes):
        for ch_idx in range(3):
            group_members.setdefault((size, ch_idx), []).append(tile_idx)

    cursor = 0
    for (tile_size, ch_idx) in sorted(group_members.keys()):
        ch_sizes = _tile_channel_sizes(tile_size, chroma_mode)
        n = ch_sizes[ch_idx]
        n_per = n * n
        members = group_members[(tile_size, ch_idx)]
        n_members = len(members)
        n_vals = n_members * n_per
        band = coeffs[cursor:cursor + n_vals]
        cursor += n_vals
        # Undo the transpose + zigzag.
        zigged = band.reshape(n_per, n_members).T  # (n_members, n_per)
        inv_zz = _inverse_zigzag_flat_indices(n)
        flat = zigged[:, inv_zz]                   # (n_members, n_per)
        # Vectorized scatter: build a (n_members * n_per,) flat index
        # array and write the entire group in one assignment.
        offset_arr = np.fromiter(
            (per_tile_ch_offsets[m][ch_idx] for m in members),
            dtype=np.int64,
            count=n_members,
        )
        scatter_idx = (
            offset_arr[:, None] + np.arange(n_per, dtype=np.int64)[None, :]
        ).reshape(-1)
        out[scatter_idx] = flat.reshape(-1).astype(np.int16, copy=False)

    if cursor != coeffs.size:
        raise ValueError(
            f"band-major coefficient buffer size mismatch: "
            f"used {cursor}, got {coeffs.size}"
        )

    return out


@lru_cache(maxsize=16)
def freq_weights(N: int, alpha: float) -> np.ndarray:
    """Per-coefficient frequency weight matrix for an N×N DCT block.

    Returns ``(N, N)`` float64 with ``weights[0, 0] = 1.0`` (DC, no
    extra quantization) growing linearly to ``weights[N-1, N-1] =
    1 + alpha`` (highest frequency, alpha× more aggressive
    quantization). The effective quantization step at coefficient
    (i, j) is ``base_step * weights[i, j]``.

    This is the simplest "perceptual" weighting scheme — JPEG uses a
    hand-tuned 8×8 lookup table whose effect is similar (~6× variation
    between low- and high-freq). Linear weighting works well enough for
    natural-photo content and generalizes to any tile size, which is
    what we need for the variable-size quadtree.

    Cached because tile sizes (8, 16, 32) and alpha (encoder-fixed)
    take few distinct values during a single encode/decode pass.
    """
    if alpha < 0:
        raise ValueError(f"freq_weights alpha must be >= 0, got {alpha}")
    if N < 1:
        raise ValueError(f"freq_weights N must be >= 1, got {N}")
    i, j = np.mgrid[0:N, 0:N].astype(np.float64)
    # Linear ramp from 1.0 (DC) to 1+alpha (highest freq), via Manhattan
    # distance scaled to [0, 1].
    if N == 1:
        return np.array([[1.0]], dtype=np.float64)
    ramp = (i + j) / (2.0 * (N - 1))
    return 1.0 + alpha * ramp


def quant_step_for_quality(quality: int) -> float:
    """Map a quality knob in [0, 100] to a DCT-coefficient quantization step.

    Lower step → finer quantization → higher PSNR / more bytes.

    The schedule is hand-tuned so q=75 gives a step ~0.025 (about
    1/40 of the [0, 1] residual range), q=95 gives ~0.008, q=50 gives
    ~0.05. Outside [10, 99] the formula clamps to a minimum step that
    keeps int16 coefficients in range for the worst-case full-tile
    DCT magnitude.
    """
    q = max(10, min(99, int(quality)))
    # Smooth quadratic: at q=99 step ≈ 0.004; at q=50 step ≈ 0.05.
    raw = ((101 - q) / 100.0) ** 2 * 0.20
    return float(max(0.003, min(0.20, raw)))


def _quant_channel(
    block: np.ndarray, *, base_step: float, freq_alpha: float,
) -> np.ndarray:
    """DCT a single (h, w) channel and quantize to int16."""
    h, w = block.shape
    coeffs = dctn(block.astype(np.float64), type=2, norm="ortho")
    if freq_alpha > 0 and h == w:
        step_grid = base_step * freq_weights(h, freq_alpha)
    else:
        step_grid = base_step
    return np.clip(np.round(coeffs / step_grid), -32768, 32767).astype(np.int16)


def _dequant_channel(
    coeffs: np.ndarray, *, base_step: float, freq_alpha: float,
) -> np.ndarray:
    """Dequantize and IDCT a single (h, w) channel back to float32."""
    h, w = coeffs.shape
    if freq_alpha > 0 and h == w:
        step_grid = base_step * freq_weights(h, freq_alpha)
    else:
        step_grid = base_step
    block = coeffs.astype(np.float64) * step_grid
    return idctn(block, type=2, norm="ortho").astype(np.float32)


# Chroma quantization scale (chroma steps are this many times the
# luma step). Matches the JPEG default ratio (~1.5-2×).
CHROMA_QUANT_SCALE = 1.5


# ── Per-tile adaptive quantization scale (BLOCK_DCT v5) ────────────────
#
# Each tile gets a multiplicative scale factor in the log-spaced range
# [0.25, 4.0]. The effective DCT quant step for a tile is
# ``base_quant_step * tile_scale``, so:
#
#   * scale < 1.0 → finer quantization → more bits, lower error
#     (assigned to high-energy / detail-rich tiles)
#   * scale = 1.0 → unchanged (legacy v4 behavior)
#   * scale > 1.0 → coarser quantization → fewer bits, higher error
#     (assigned to low-energy / smooth tiles)
#
# 256 levels (u8) of resolution, log-spaced so doubling/halving the
# scale corresponds to a fixed number of u8 steps. The encoder
# normally writes scales in roughly [0.5, 2.0]; the wider [0.25, 4.0]
# range exists so future heuristics can be more aggressive without
# format breaks.

_TILE_SCALE_LOG_RANGE = 4.0  # log2(4.0 / 0.25) = log2(16)
_TILE_SCALE_LOG_MIN = -2.0   # log2(0.25)


def encode_tile_scale_u8(scale: float) -> int:
    """Quantize a tile-scale float to a u8 (log-spaced [0.25, 4.0])."""
    if scale <= 0:
        raise ValueError(f"tile scale must be > 0, got {scale}")
    log_scale = float(np.log2(scale))
    norm = (log_scale - _TILE_SCALE_LOG_MIN) / _TILE_SCALE_LOG_RANGE
    norm = max(0.0, min(1.0, norm))
    return int(round(norm * 255))


def decode_tile_scale_u8(u: int) -> float:
    """Inverse of ``encode_tile_scale_u8``."""
    norm = max(0, min(255, int(u))) / 255.0
    log_scale = _TILE_SCALE_LOG_MIN + norm * _TILE_SCALE_LOG_RANGE
    return float(2.0 ** log_scale)


def encode_tile_residuals(
    residuals: list[np.ndarray],
    *,
    quant_step: float,
    freq_alpha: float = 0.0,
    chroma_mode: int = 0,
    per_tile_scales: list[float] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """DCT-encode and quantize a list of per-tile residual arrays.

    Each entry in ``residuals`` must be ``(tile_size, tile_size, c)``
    float32 in roughly [-0.5, 0.5]. Returns a flat int16 buffer
    containing the quantized DCT coefficients in tile order, channel-
    interleaved per tile, plus a list of byte offsets for the start
    of each tile's coefficients.

    ``chroma_mode`` selects the colorspace and chroma layout:

      0 (RGB)         3 channels at full tile size each
      1 (YCbCr 4:4:4) Y, Cb, Cr at full tile size; chroma uses
                      ``CHROMA_QUANT_SCALE × quant_step``
      2 (YCbCr 4:2:0) Y at full tile size, Cb and Cr box-averaged
                      to half size per axis (75% chroma byte savings)

    ``per_tile_scales`` (BLOCK_DCT v5) is an optional list of
    multiplicative quant-step scales, one per tile, in roughly
    [0.25, 4.0]. The effective step for tile *i* is
    ``quant_step * per_tile_scales[i]`` — smaller scales give finer
    quantization (more bits, lower error) on detail-heavy tiles, and
    larger scales give coarser quantization (fewer bits) on smooth
    tiles. ``None`` means "all 1.0" (legacy uniform quantization).

    Per-frequency weighting (``freq_alpha``) is applied to each
    channel using the channel's own DCT block size — so the chroma
    quantization is doubly adaptive (subsampled and per-frequency
    weighted at half the resolution).
    """
    if quant_step <= 0:
        raise ValueError("quant_step must be > 0")
    if freq_alpha < 0:
        raise ValueError("freq_alpha must be >= 0")
    if chroma_mode not in (0, 1, 2):
        raise ValueError(f"chroma_mode must be 0/1/2, got {chroma_mode}")
    if per_tile_scales is not None and len(per_tile_scales) != len(residuals):
        raise ValueError(
            f"per_tile_scales length {len(per_tile_scales)} does not match "
            f"residuals length {len(residuals)}"
        )

    coeff_blocks: list[np.ndarray] = []
    offsets: list[int] = []
    cursor = 0

    for i, tile in enumerate(residuals):
        h, w, c = tile.shape
        offsets.append(cursor)
        tile_scale = float(per_tile_scales[i]) if per_tile_scales is not None else 1.0
        tile_step = quant_step * tile_scale

        if chroma_mode == 0:
            # RGB: per-channel DCT at full size, uniform step.
            for ch in range(c):
                q = _quant_channel(tile[..., ch], base_step=tile_step, freq_alpha=freq_alpha)
                flat = q.reshape(-1)
                coeff_blocks.append(flat)
                cursor += int(flat.size)
            continue

        # YCbCr modes: convert RGB → YCbCr, then encode each channel.
        ycbcr = rgb_to_ycbcr_residual(tile)  # (h, w, 3)
        Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]

        # Y: full size, luma step
        q_y = _quant_channel(Y, base_step=tile_step, freq_alpha=freq_alpha)
        coeff_blocks.append(q_y.reshape(-1))
        cursor += int(q_y.size)

        chroma_step = tile_step * CHROMA_QUANT_SCALE
        if chroma_mode == 1:
            # 4:4:4 — Cb/Cr at full size, chroma step
            q_cb = _quant_channel(Cb, base_step=chroma_step, freq_alpha=freq_alpha)
            q_cr = _quant_channel(Cr, base_step=chroma_step, freq_alpha=freq_alpha)
            coeff_blocks.append(q_cb.reshape(-1))
            coeff_blocks.append(q_cr.reshape(-1))
            cursor += int(q_cb.size) + int(q_cr.size)
        else:
            # 4:2:0 — Cb/Cr 2× box-average then DCT at half size, chroma step
            if h % 2 or w % 2:
                raise ValueError(
                    f"4:2:0 chroma subsampling requires even tile dims, got {tile.shape}"
                )
            Cb_sub = subsample_2x(Cb)
            Cr_sub = subsample_2x(Cr)
            q_cb = _quant_channel(Cb_sub, base_step=chroma_step, freq_alpha=freq_alpha)
            q_cr = _quant_channel(Cr_sub, base_step=chroma_step, freq_alpha=freq_alpha)
            coeff_blocks.append(q_cb.reshape(-1))
            coeff_blocks.append(q_cr.reshape(-1))
            cursor += int(q_cb.size) + int(q_cr.size)

    offsets.append(cursor)
    if coeff_blocks:
        all_coeffs = np.concatenate(coeff_blocks)
    else:
        all_coeffs = np.zeros(0, dtype=np.int16)
    return all_coeffs, offsets


def _batch_dequant_idct(
    q_batch: np.ndarray,
    *,
    base_step: float,
    freq_alpha: float,
    per_tile_step_scales: np.ndarray | None,
) -> np.ndarray:
    """Batched dequant + inverse DCT for a (n, h, w) int16 coefficient stack.

    All blocks must share the same (h, w) shape — the caller groups
    tiles by size before stacking. ``per_tile_step_scales`` is an
    optional (n,) float array applied per-tile to the base step.
    """
    n, h, w = q_batch.shape
    if freq_alpha > 0 and h == w:
        step_grid = base_step * freq_weights(h, freq_alpha)  # (h, w)
    else:
        step_grid = np.full((h, w), base_step, dtype=np.float64)
    if per_tile_step_scales is not None:
        # Broadcast per-tile scale across the (h, w) step grid: (n, h, w)
        step_3d = step_grid[None, :, :] * per_tile_step_scales[:, None, None]
    else:
        step_3d = step_grid[None, :, :]
    block = q_batch.astype(np.float64) * step_3d
    return idctn(block, type=2, norm="ortho", axes=(-2, -1)).astype(np.float32)


def decode_tile_residuals(
    coeffs: np.ndarray,
    tile_sizes: Iterable[int],
    *,
    quant_step: float,
    channels: int = 3,
    freq_alpha: float = 0.0,
    chroma_mode: int = 0,
    per_tile_scales: list[float] | None = None,
) -> list[np.ndarray]:
    """Inverse-DCT each tile's quantized residual back to a float array.

    ``chroma_mode`` must match the encoder's value (carried in the
    BLOCK_DCT header). ``per_tile_scales`` (BLOCK_DCT v5) is the same
    list of multiplicative quant-step scales the encoder used; pass
    ``None`` (the default) for legacy uniform quantization. Returns a
    list of (size, size, 3) float32 RGB residual arrays.

    Batched: tiles are grouped by ``size`` and each group is processed
    as a single 3D ``(n_tiles_at_size, h, w)`` IDCT call instead of
    one IDCT per tile per channel. On landscape.jpg this cuts the
    DCT-residual decode from ~840 ms to ~75 ms (≈11×) — the per-call
    scipy overhead and Python-level loop bookkeeping dominate over
    the actual IDCT cost on small (16, 16) blocks.
    """
    sizes_list = list(tile_sizes)
    n_tiles = len(sizes_list)
    if per_tile_scales is not None and len(per_tile_scales) != n_tiles:
        raise ValueError(
            f"per_tile_scales length {len(per_tile_scales)} does not match "
            f"tile_sizes length {n_tiles}"
        )

    if n_tiles == 0:
        return []

    if chroma_mode not in (0, 1, 2):
        raise ValueError(f"bad chroma_mode: {chroma_mode}")

    # ── Pass 1: walk the coefficient buffer once to populate per-size
    # batches. We precompute the cursor offsets so we can use numpy
    # slicing instead of repeated reshapes inside the hot loop.
    n_channels_full = channels if chroma_mode == 0 else 1   # Y in YCbCr
    chroma_subsampled = chroma_mode == 2

    # buckets[size] = (idx_list, q_y_blocks, q_cb_blocks, q_cr_blocks)
    buckets: dict[int, tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}

    cursor = 0
    for tile_idx, size in enumerate(sizes_list):
        bucket = buckets.setdefault(size, ([], [], [], []))
        bucket[0].append(tile_idx)

        if chroma_mode == 0:
            # RGB: 3 channels at full size, no chroma split.
            n = size * size
            for ch in range(channels):
                bucket[1 + ch].append(coeffs[cursor:cursor + n].reshape(size, size))
                cursor += n
            continue

        # YCbCr modes — Y at full size.
        n_y = size * size
        bucket[1].append(coeffs[cursor:cursor + n_y].reshape(size, size))
        cursor += n_y

        if chroma_subsampled:
            half = size // 2
            n_c = half * half
        else:
            n_c = size * size
        bucket[2].append(coeffs[cursor:cursor + n_c].reshape(
            half if chroma_subsampled else size, half if chroma_subsampled else size,
        ))
        cursor += n_c
        bucket[3].append(coeffs[cursor:cursor + n_c].reshape(
            half if chroma_subsampled else size, half if chroma_subsampled else size,
        ))
        cursor += n_c

    if per_tile_scales is not None:
        per_tile_scales_arr = np.asarray(per_tile_scales, dtype=np.float64)
    else:
        per_tile_scales_arr = None

    # ── Pass 2: process each (size) bucket as a batched IDCT.
    out: list[np.ndarray] = [None] * n_tiles  # type: ignore

    for size, (indices, c0_blocks, c1_blocks, c2_blocks) in buckets.items():
        idx_arr = np.asarray(indices, dtype=np.int64)
        if per_tile_scales_arr is not None:
            scales_for_bucket = per_tile_scales_arr[idx_arr]
        else:
            scales_for_bucket = None

        if chroma_mode == 0:
            # RGB: dequant each channel as a (n, size, size) batch, then
            # stack into the final (n, size, size, 3) output.
            r_batch = _batch_dequant_idct(
                np.stack(c0_blocks, axis=0),
                base_step=quant_step, freq_alpha=freq_alpha,
                per_tile_step_scales=scales_for_bucket,
            )
            g_batch = _batch_dequant_idct(
                np.stack(c1_blocks, axis=0),
                base_step=quant_step, freq_alpha=freq_alpha,
                per_tile_step_scales=scales_for_bucket,
            )
            b_batch = _batch_dequant_idct(
                np.stack(c2_blocks, axis=0),
                base_step=quant_step, freq_alpha=freq_alpha,
                per_tile_step_scales=scales_for_bucket,
            )
            tiles_3d = np.stack([r_batch, g_batch, b_batch], axis=-1)
            for j, tile_idx in enumerate(indices):
                out[tile_idx] = tiles_3d[j]
            continue

        # YCbCr modes — Y at base step; chroma at base * CHROMA_QUANT_SCALE.
        Y_batch = _batch_dequant_idct(
            np.stack(c0_blocks, axis=0),
            base_step=quant_step, freq_alpha=freq_alpha,
            per_tile_step_scales=scales_for_bucket,
        )
        chroma_base_step = quant_step * CHROMA_QUANT_SCALE
        Cb_batch_small = _batch_dequant_idct(
            np.stack(c1_blocks, axis=0),
            base_step=chroma_base_step, freq_alpha=freq_alpha,
            per_tile_step_scales=scales_for_bucket,
        )
        Cr_batch_small = _batch_dequant_idct(
            np.stack(c2_blocks, axis=0),
            base_step=chroma_base_step, freq_alpha=freq_alpha,
            per_tile_step_scales=scales_for_bucket,
        )

        if chroma_subsampled:
            # Nearest-neighbour 2× upsample on the batched chroma:
            # repeat both spatial axes. This is exactly the inverse of
            # the encoder's 2× box-average for constant blocks (and
            # perceptually invisible for non-constant blocks because
            # the eye doesn't resolve chroma at the pixel level).
            Cb_batch = np.repeat(np.repeat(Cb_batch_small, 2, axis=-2), 2, axis=-1)
            Cr_batch = np.repeat(np.repeat(Cr_batch_small, 2, axis=-2), 2, axis=-1)
        else:
            Cb_batch = Cb_batch_small
            Cr_batch = Cr_batch_small

        # Batched YCbCr → RGB. Stack to (n, h, w, 3) and matmul.
        ycbcr_batch = np.stack([Y_batch, Cb_batch, Cr_batch], axis=-1)
        rgb_batch = (ycbcr_batch.astype(np.float64) @ _YCBCR_TO_RGB.T).astype(np.float32)

        for j, tile_idx in enumerate(indices):
            out[tile_idx] = rgb_batch[j]

    return out  # type: ignore


def _bitmask_to_present_indices(bitmask: bytes, n_tiles: int) -> list[bool]:
    """Expand a packed presence bitmask to a list of n_tiles booleans."""
    out = [False] * n_tiles
    for i in range(n_tiles):
        if bitmask[i >> 3] & (1 << (i & 7)):
            out[i] = True
    return out


def _present_indices_to_bitmask(present: list[bool]) -> bytes:
    """Pack a list of booleans into a presence bitmask (LSB-first per byte)."""
    n = len(present)
    out = bytearray((n + 7) // 8)
    for i, p in enumerate(present):
        if p:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def apply_residual_to_image(
    recon: np.ndarray,
    coeffs: np.ndarray,
    quad_tiles,
    *,
    quant_step: float,
    channels: int = 3,
    freq_alpha: float = 0.0,
    chroma_mode: int = 0,
    presence_bitmask: bytes | None = None,
    per_tile_scales: list[float] | None = None,
) -> np.ndarray:
    """Add per-tile IDCT residuals to a full-image reconstruction in place.

    ``recon`` is the (h, w, c) primitive-stack reconstruction (already
    has RES0/RES1 applied). ``quad_tiles`` is the same QuadTile list the
    encoder used (each has ``.x``, ``.y``, ``.size``). Returns a new
    image with the residual added and clipped to [0, 1].

    ``presence_bitmask`` (iteration 3): when supplied, only the tiles
    whose bit is set carry coefficients in ``coeffs``. Tiles whose bit
    is clear are skipped (their primitive reconstruction is left as-is).
    ``None`` means "every tile is present" (legacy v2 behavior).

    ``per_tile_scales`` (BLOCK_DCT v5): one float per *present* tile,
    a multiplicative factor applied to ``quant_step``. ``None`` means
    "all 1.0" (uniform quantization, legacy behavior).
    """
    n_total = len(quad_tiles)
    if presence_bitmask is not None and len(presence_bitmask) > 0:
        present = _bitmask_to_present_indices(presence_bitmask, n_total)
    else:
        present = [True] * n_total

    # Only the present tiles' sizes need to be passed to decode_tile_residuals
    # — the decoder walks the coefficient stream in lockstep with this
    # subset.
    present_quad_tiles = [qt for qt, p in zip(quad_tiles, present) if p]
    sizes = [int(qt.size) for qt in present_quad_tiles]
    residuals = decode_tile_residuals(
        coeffs, sizes, quant_step=quant_step, channels=channels,
        freq_alpha=freq_alpha, chroma_mode=chroma_mode,
        per_tile_scales=per_tile_scales,
    )
    out = recon.copy()
    h, w = out.shape[:2]
    for qt, res in zip(present_quad_tiles, residuals):
        x0, y0, s = int(qt.x), int(qt.y), int(qt.size)
        # Tiles may extend past the image right/bottom edges (the
        # encoder pads the source up to a multiple of TILE_SIZE_MAX
        # before splitting). Clip the residual to the actual image area.
        x1 = min(x0 + s, w)
        y1 = min(y0 + s, h)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] += res[: y1 - y0, : x1 - x0]
    return np.clip(out, 0.0, 1.0)
