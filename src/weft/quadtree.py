"""Adaptive quadtree tile decomposition for WEFT.

Splits the image into variable-size tiles (32x32, 16x16, 8x8) based on
local complexity.  Flat regions get large tiles (fewer seams, fewer
primitives wasted); complex regions get small tiles (more primitives
per pixel, better precision).
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence

import numpy as np

from .constants import MAX_PRIMITIVES_BY_TILE_SIZE, TILE_SIZE_MAX, TILE_SIZE_MIN


@dataclass(slots=True)
class QuadTile:
    """One tile in the adaptive quadtree decomposition."""
    x: int          # pixel x-origin in the image
    y: int          # pixel y-origin in the image
    size: int       # tile side length (8, 16, or 32)
    index: int      # sequential index in the tile list

    @property
    def max_primitives(self) -> int:
        return MAX_PRIMITIVES_BY_TILE_SIZE.get(self.size, 48)


def _tile_complexity(patch: np.ndarray) -> float:
    """Gradient-magnitude + variance metric for a tile patch."""
    lum = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
    gy, gx = np.gradient(lum)
    grad_mag = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    variance = float(np.var(lum))
    return grad_mag + variance * 4.0


def decompose_quadtree(
    image: np.ndarray,
    *,
    split_threshold: float = 0.08,
    min_tile: int = TILE_SIZE_MIN,
    max_tile: int = TILE_SIZE_MAX,
) -> list[QuadTile]:
    """Decompose an image into an adaptive quadtree of tiles.

    Starts with max_tile-sized macro-tiles, then recursively splits
    tiles whose complexity exceeds *split_threshold* until min_tile
    is reached.

    Returns tiles in row-major order (top-to-bottom, left-to-right
    within each macro-tile group).
    """
    h, w = image.shape[:2]

    # Pad image to be divisible by max_tile.
    pad_h = (max_tile - h % max_tile) % max_tile
    pad_w = (max_tile - w % max_tile) % max_tile
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    ph, pw = image.shape[:2]

    tiles: list[QuadTile] = []

    def _split(x: int, y: int, size: int) -> None:
        patch = image[y:y + size, x:x + size]
        complexity = _tile_complexity(patch)

        if size <= min_tile or complexity < split_threshold:
            tiles.append(QuadTile(x=x, y=y, size=size, index=0))
            return

        half = size // 2
        _split(x, y, half)
        _split(x + half, y, half)
        _split(x, y + half, half)
        _split(x + half, y + half, half)

    for my in range(0, ph, max_tile):
        for mx in range(0, pw, max_tile):
            _split(mx, my, max_tile)

    # Assign sequential indices.
    for i, tile in enumerate(tiles):
        tile.index = i

    return tiles


def extract_adaptive_tiles(
    image: np.ndarray,
    tiles: list[QuadTile],
) -> list[np.ndarray]:
    """Extract pixel data for each QuadTile."""
    h, w = image.shape[:2]
    pad_h = (TILE_SIZE_MAX - h % TILE_SIZE_MAX) % TILE_SIZE_MAX
    pad_w = (TILE_SIZE_MAX - w % TILE_SIZE_MAX) % TILE_SIZE_MAX
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    patches: list[np.ndarray] = []
    for tile in tiles:
        patches.append(image[tile.y:tile.y + tile.size, tile.x:tile.x + tile.size].copy())
    return patches


def pack_qtree(tiles: Sequence[QuadTile]) -> bytes:
    """Serialize quadtree tile layout to QTREE block payload.

    Format: u32 count, then per tile: u16 x, u16 y, u8 size.
    """
    out = struct.pack("<I", len(tiles))
    for t in tiles:
        out += struct.pack("<HHB", t.x, t.y, t.size)
    return out


def unpack_qtree(blob: bytes) -> list[QuadTile]:
    """Deserialize QTREE block payload."""
    if len(blob) < 4:
        raise ValueError("truncated QTREE")
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * 5
    if len(blob) < expected:
        raise ValueError("truncated QTREE entries")
    tiles: list[QuadTile] = []
    off = 4
    for i in range(n):
        x, y, size = struct.unpack_from("<HHB", blob, off)
        tiles.append(QuadTile(x=x, y=y, size=size, index=i))
        off += 5
    return tiles
