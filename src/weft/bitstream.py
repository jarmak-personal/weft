"""WEFT bitstream read/write utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Any

from .constants import (
    BLOCK_BIC,
    BLOCK_CIDX,
    BLOCK_DCT,
    BLOCK_GRD,
    BLOCK_HEAD,
    BLOCK_LITE,
    BLOCK_META,
    BLOCK_PAL,
    BLOCK_PDEL,
    BLOCK_PSTR,
    BLOCK_PRIM,
    BLOCK_QTREE,
    BLOCK_RES0,
    BLOCK_RES1,
    BLOCK_RES2,
    BLOCK_TOC,
    MAGIC,
    MAJOR_VERSION,
    MAX_PRIMITIVES_PER_TILE,
    MINOR_VERSION,
    TILE_SIZE,
    FLAG_CHUNKED_PRIM,
    FLAG_HAS_BIC,
    FLAG_HAS_DCT,
    FLAG_HAS_GRD,
    FLAG_HAS_LITE,
    FLAG_HAS_PAL,
    FLAG_HAS_RES0,
    FLAG_HAS_RES1,
)


class BitstreamError(ValueError):
    pass


# ── Block-level zstd compression ────────────────────────────────────

# Blocks that benefit from compression (large, redundant payloads).
_COMPRESSIBLE_BLOCKS = {BLOCK_PRIM, BLOCK_RES0, BLOCK_RES1, BLOCK_RES2, BLOCK_QTREE, BLOCK_BIC, BLOCK_PAL, BLOCK_GRD, BLOCK_DCT}
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"  # zstd frame magic (little-endian 0xFD2FB528)


def _zstd_compress(data: bytes, level: int = 19) -> bytes:
    """Compress data with zstd. Returns original data if compression doesn't help.

    Default level 19 is tuned for the DCT coefficient stream on natural
    photos: the Shannon entropy bound is only ~4% below what level 19
    achieves on a typical ~1 MB DCT block, and bumping from level 16 to
    19 gets ~half of that remaining headroom (−4% bytes on landscape.jpg's
    DCT block). Level 22 is not measurably better and is ~2× slower.
    """
    try:
        import zstandard
        compressed = zstandard.ZstdCompressor(level=level).compress(data)
        if len(compressed) < len(data):
            return compressed
    except ImportError:
        pass
    return data


def _zstd_decompress_if_needed(data: bytes) -> bytes:
    """Decompress data if it has a zstd frame header, otherwise return as-is."""
    if len(data) >= 4 and data[:4] == _ZSTD_MAGIC:
        try:
            import zstandard
            return zstandard.ZstdDecompressor().decompress(data)
        except ImportError:
            raise BitstreamError("block is zstd-compressed but zstandard module is not installed")
    return data


@dataclass(slots=True)
class BlockEntry:
    tag: bytes
    offset: int
    length: int


@dataclass(slots=True)
class PrimChunkEntry:
    start_tile: int
    tile_count: int
    raw_offset: int
    raw_length: int
    coded_offset: int
    coded_length: int


@dataclass(slots=True)
class HeadBlock:
    width: int
    height: int
    tile_size: int = TILE_SIZE
    max_primitives: int = MAX_PRIMITIVES_PER_TILE
    color_space: int = 1
    quant_mode: int = 1
    flags: int = 0
    tile_cols: int = 0
    tile_rows: int = 0
    quality: int = 75
    preset_id: int = 1


@dataclass(slots=True)
class WeftFile:
    major: int
    minor: int
    head: HeadBlock
    toc: list[int]
    prim_payload: bytes
    res0_payload: bytes | None
    res1_payload: bytes | None
    res2_payload: bytes | None
    pstr_payload: bytes | None
    pdel_payload: bytes | None
    cidx_payload: bytes | None
    chunk_index: list[PrimChunkEntry] | None
    qtree_payload: bytes | None
    lite_payload: bytes | None  # Phase 2 #17: low-resolution lighting grid
    bic_payload: bytes | None   # Brainstorm #11: bicubic-patch control grids
    pal_payload: bytes | None   # Brainstorm #20: palette + per-pixel labels
    grd_payload: bytes | None   # Brainstorm #1: gradient field + means
    dct_payload: bytes | None   # Brainstorm #16: per-tile DCT residual
    meta: dict[str, Any]
    block_entries: list[BlockEntry]


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    rem = value % alignment
    if rem == 0:
        return value
    return value + (alignment - rem)


def _pack_head(head: HeadBlock) -> bytes:
    # Layout: width(I) height(I) tile_size(H) max_primitives(H)
    # color_space(B) quant_mode(B) flags(H) tile_cols(I) tile_rows(I)
    # quality(B) preset_id(B) trailing_padding(H) = 28 bytes total.
    #
    # The flags field used to be (B, B) = (flags8, padding8) but the
    # padding byte was always 0; merging into a single H is wire-
    # compatible with the old layout (a 16-bit little-endian load of
    # (flag_byte, 0) gives the same numeric value as the old B). The
    # extra 8 bits hold flags >= 1 << 8 (FLAG_HAS_GRD and beyond).
    return struct.pack(
        "<IIHHBBHIIBBH",
        int(head.width),
        int(head.height),
        int(head.tile_size),
        int(head.max_primitives),
        int(head.color_space),
        int(head.quant_mode),
        int(head.flags) & 0xFFFF,
        int(head.tile_cols),
        int(head.tile_rows),
        int(head.quality),
        int(head.preset_id),
        0,
    )


def _unpack_head(blob: bytes) -> HeadBlock:
    expected = struct.calcsize("<IIHHBBHIIBBH")
    if len(blob) != expected:
        raise BitstreamError("invalid HEAD size")
    (
        width,
        height,
        tile_size,
        max_prims,
        color_space,
        quant_mode,
        flags,
        tile_cols,
        tile_rows,
        quality,
        preset_id,
        __,
    ) = struct.unpack("<IIHHBBHIIBBH", blob)
    if width <= 0 or height <= 0:
        raise BitstreamError("invalid dimensions")
    if tile_size <= 0 or tile_size > 64:
        raise BitstreamError("invalid tile size")
    if max_prims <= 0 or max_prims > 255:
        raise BitstreamError("invalid max primitives")
    return HeadBlock(
        width=width,
        height=height,
        tile_size=tile_size,
        max_primitives=max_prims,
        color_space=color_space,
        quant_mode=quant_mode,
        flags=flags,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        quality=quality,
        preset_id=preset_id,
    )


def _pack_toc(toc: list[int]) -> bytes:
    if not toc:
        return struct.pack("<I", 0)
    return struct.pack("<I", len(toc)) + b"".join(struct.pack("<I", int(v)) for v in toc)


def _unpack_toc(blob: bytes) -> list[int]:
    if len(blob) < 4:
        raise BitstreamError("truncated TOC")
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * 4
    if len(blob) != expected:
        raise BitstreamError("invalid TOC length")
    toc = list(struct.unpack_from(f"<{n}I", blob, 4)) if n > 0 else []
    for i in range(len(toc) - 1):
        if toc[i + 1] < toc[i]:
            raise BitstreamError("TOC offsets must be monotonic")
    return toc


def _pack_res0(residuals: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    out += struct.pack("<I", len(residuals))
    for r, g, b in residuals:
        if not (-127 <= r <= 127 and -127 <= g <= 127 and -127 <= b <= 127):
            raise BitstreamError("residual out of range")
        out += struct.pack("<bbb", int(r), int(g), int(b))
    return bytes(out)


def _unpack_res0(blob: bytes) -> list[tuple[int, int, int]]:
    if len(blob) < 4:
        raise BitstreamError("truncated RES0")
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * 3
    if len(blob) != expected:
        raise BitstreamError("invalid RES0 length")
    res: list[tuple[int, int, int]] = []
    off = 4
    for _ in range(n):
        r, g, b = struct.unpack_from("<bbb", blob, off)
        res.append((int(r), int(g), int(b)))
        off += 3
    return res


def _pack_res1(residual_maps: list[bytes], grid_size: int = 4) -> bytes:
    if grid_size <= 0 or grid_size > 16:
        raise BitstreamError("invalid RES1 grid size")
    expected_len = grid_size * grid_size * 3
    out = bytearray()
    out += struct.pack("<IBBH", len(residual_maps), grid_size, grid_size, 0)
    for m in residual_maps:
        if len(m) != expected_len:
            raise BitstreamError("invalid RES1 map length")
        out += m
    return bytes(out)


def _unpack_res1(blob: bytes) -> tuple[int, list[bytes]]:
    if len(blob) < struct.calcsize("<IBBH"):
        raise BitstreamError("truncated RES1")
    n, gw, gh, _ = struct.unpack_from("<IBBH", blob, 0)
    if gw <= 0 or gh <= 0:
        raise BitstreamError("invalid RES1 grid")
    per = gw * gh * 3
    expected = struct.calcsize("<IBBH") + n * per
    if len(blob) != expected:
        raise BitstreamError("invalid RES1 length")
    maps: list[bytes] = []
    off = struct.calcsize("<IBBH")
    for _ in range(n):
        maps.append(blob[off : off + per])
        off += per
    return int(gw), maps


def _pack_meta(meta: dict[str, Any]) -> bytes:
    body = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def _unpack_meta(blob: bytes) -> dict[str, Any]:
    if len(blob) < 4:
        raise BitstreamError("truncated META")
    n = struct.unpack_from("<I", blob, 0)[0]
    if 4 + n != len(blob):
        raise BitstreamError("invalid META length")
    raw = blob[4 : 4 + n]
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BitstreamError("invalid META JSON") from exc


def pack_chunk_index(chunks: list[PrimChunkEntry]) -> bytes:
    out = bytearray()
    out += struct.pack("<I", len(chunks))
    for c in chunks:
        out += struct.pack(
            "<IIIIII",
            int(c.start_tile),
            int(c.tile_count),
            int(c.raw_offset),
            int(c.raw_length),
            int(c.coded_offset),
            int(c.coded_length),
        )
    return bytes(out)


def unpack_chunk_index(blob: bytes) -> list[PrimChunkEntry]:
    if len(blob) < 4:
        raise BitstreamError("truncated CIDX")
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * struct.calcsize("<IIIIII")
    if len(blob) != expected:
        raise BitstreamError("invalid CIDX length")
    out: list[PrimChunkEntry] = []
    off = 4
    for _ in range(n):
        start_tile, tile_count, raw_offset, raw_length, coded_offset, coded_length = struct.unpack_from("<IIIIII", blob, off)
        off += struct.calcsize("<IIIIII")
        out.append(
            PrimChunkEntry(
                start_tile=int(start_tile),
                tile_count=int(tile_count),
                raw_offset=int(raw_offset),
                raw_length=int(raw_length),
                coded_offset=int(coded_offset),
                coded_length=int(coded_length),
            )
        )
    return out


def pack_lite(grid: Any) -> bytes:
    """Serialize a (grid_h, grid_w, 3) float lighting grid to BLOCK_LITE bytes.

    Format:
        u16 grid_w
        u16 grid_h
        grid_h * grid_w * 3 int8 values, each = clamp(round((v - 1.0) * 64), -127, 127)

    The int8 channel encodes a multiplicative factor centered on 1.0 (no
    change), with ±127 representing roughly ±2.0 around the identity.
    Lighting at the center (0) means "multiply by 1.0", positive means
    brighter, negative means darker.
    """
    import numpy as np
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise BitstreamError("lighting grid must be (h, w, 3)")
    h, w, _ = arr.shape
    # Map linear lighting [~0, ~3] to int8 around 0 with scale=64.
    quantized = np.clip(np.round((arr - 1.0) * 64.0), -127.0, 127.0).astype(np.int8)
    return struct.pack("<HH", int(w), int(h)) + quantized.tobytes()


def pack_bic(control_grids: Any) -> bytes:
    """Serialize a list/array of (4, 4, 3) bicubic control grids to BLOCK_BIC bytes.

    Format:
        u32 n_tiles
        n_tiles * 4 * 4 * 3 = 48*n_tiles bytes (u8 control point values, [0, 1] linear RGB)

    The control points are quantized to 8 bits per channel. With ~3400 tiles
    on a 1.5MP image that's ~163 KB raw → typically ~60-80 KB after the
    block-level zstd pass on natural content (the grids are spatially smooth).
    """
    import numpy as np
    arr = np.asarray(control_grids, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[1:] != (4, 4, 3):
        raise BitstreamError(
            f"bicubic control grids must be (n_tiles, 4, 4, 3), got {arr.shape}"
        )
    n = int(arr.shape[0])
    quantized = np.clip(np.round(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    return struct.pack("<I", n) + quantized.tobytes()


def unpack_bic(blob: bytes) -> Any:
    """Deserialize BLOCK_BIC bytes back to (n_tiles, 4, 4, 3) float32 in [0, 1]."""
    import numpy as np
    if len(blob) < 4:
        raise BitstreamError("truncated BIC block header")
    n = struct.unpack_from("<I", blob, 0)[0]
    expected = 4 + n * 48
    if len(blob) != expected:
        raise BitstreamError(f"BIC block size mismatch: got {len(blob)}, expected {expected}")
    raw = np.frombuffer(blob, dtype=np.uint8, offset=4, count=n * 48).reshape((n, 4, 4, 3))
    return raw.astype(np.float32) / 255.0


def pack_pal(palette: Any, labels: Any) -> bytes:
    """Serialize a (palette, labels) pair to BLOCK_PAL bytes.

    Format:
        u16 K           — palette size, 1..256
        u32 width       — labels width in pixels
        u32 height      — labels height in pixels
        K * 3 bytes     — palette RGB (u8)
        width*height    — labels (u8) per pixel, indexing into the palette

    The labels grid is stored uncompressed; the bitstream-level zstd pass
    on BLOCK_PAL handles the actual compression. Adjacent pixels share
    labels for the vast majority of natural and screenshot content, so
    zstd ratios of 5-15× are typical.
    """
    import numpy as np
    pal = np.asarray(palette, dtype=np.uint8)
    lab = np.asarray(labels, dtype=np.uint8)
    if pal.ndim != 2 or pal.shape[1] != 3:
        raise BitstreamError(f"palette must be (K, 3), got {pal.shape}")
    K = int(pal.shape[0])
    if K < 1 or K > 256:
        raise BitstreamError(f"palette size must be in [1, 256], got {K}")
    if lab.ndim != 2:
        raise BitstreamError(f"labels must be (h, w), got {lab.shape}")
    h, w = int(lab.shape[0]), int(lab.shape[1])
    if int(lab.max(initial=0)) >= K:
        raise BitstreamError("label out of range for palette size")
    return struct.pack("<HII", K, w, h) + pal.tobytes() + lab.tobytes()


def unpack_pal(blob: bytes) -> tuple:
    """Deserialize BLOCK_PAL bytes to (palette, labels).

    Returns ``(palette: ndarray (K, 3) uint8, labels: ndarray (h, w) uint8)``.
    """
    import numpy as np
    header_size = struct.calcsize("<HII")
    if len(blob) < header_size:
        raise BitstreamError("truncated PAL block header")
    K, w, h = struct.unpack_from("<HII", blob, 0)
    pal_size = K * 3
    expected = header_size + pal_size + w * h
    if len(blob) != expected:
        raise BitstreamError(
            f"PAL block size mismatch: got {len(blob)}, expected {expected}"
        )
    pal_off = header_size
    lab_off = pal_off + pal_size
    palette = np.frombuffer(blob, dtype=np.uint8, offset=pal_off, count=pal_size).reshape((K, 3))
    labels = np.frombuffer(blob, dtype=np.uint8, offset=lab_off, count=w * h).reshape((h, w))
    return palette.copy(), labels.copy()


def pack_grd(gx_q: Any, gy_q: Any, means: Any, scale: int) -> bytes:
    """Serialize a gradient-field payload to BLOCK_GRD bytes.

    Format:
        u32 width
        u32 height
        u16 scale         — quantization scale used by the encoder
        u16 channels
        c * float32       — per-channel image means
        h*w*c int8        — gx_q (forward x-gradient, last column = 0)
        h*w*c int8        — gy_q (forward y-gradient, last row = 0)

    The two gradient blocks are stored uncompressed; the bitstream-level
    zstd pass on BLOCK_GRD handles the actual compression. On hard-edge
    content the gradient field is ~99% sparse and zstd ratios approach
    100×; on smooth or noisy content the field is dense and the block
    pays its full ~6 bytes/pixel cost.
    """
    import numpy as np
    gx = np.ascontiguousarray(np.asarray(gx_q, dtype=np.int8))
    gy = np.ascontiguousarray(np.asarray(gy_q, dtype=np.int8))
    if gx.ndim != 3 or gy.shape != gx.shape:
        raise BitstreamError(f"gradient maps must be (h, w, c) and matching, got {gx.shape} / {gy.shape}")
    h, w, c = gx.shape
    means_arr = np.asarray(means, dtype=np.float32).reshape(-1)
    if means_arr.shape[0] != c:
        raise BitstreamError(f"means must have {c} entries, got {means_arr.shape[0]}")
    if scale < 1 or scale > 65535:
        raise BitstreamError(f"scale must be in [1, 65535], got {scale}")
    header = struct.pack("<IIHH", int(w), int(h), int(scale), int(c))
    return header + means_arr.tobytes() + gx.tobytes() + gy.tobytes()


def unpack_grd(blob: bytes) -> tuple:
    """Deserialize BLOCK_GRD bytes to ``(gx_q, gy_q, means, scale)``."""
    import numpy as np
    header_size = struct.calcsize("<IIHH")
    if len(blob) < header_size:
        raise BitstreamError("truncated GRD block header")
    w, h, scale, c = struct.unpack_from("<IIHH", blob, 0)
    means_size = c * 4
    grad_size = h * w * c
    expected = header_size + means_size + 2 * grad_size
    if len(blob) != expected:
        raise BitstreamError(
            f"GRD block size mismatch: got {len(blob)}, expected {expected}"
        )
    off = header_size
    means = np.frombuffer(blob, dtype=np.float32, offset=off, count=c).copy()
    off += means_size
    gx = np.frombuffer(blob, dtype=np.int8, offset=off, count=grad_size).reshape((h, w, c)).copy()
    off += grad_size
    gy = np.frombuffer(blob, dtype=np.int8, offset=off, count=grad_size).reshape((h, w, c)).copy()
    return gx, gy, means, int(scale)


def pack_dct(
    coeffs: Any,
    quant_step: float,
    channels: int = 3,
    *,
    freq_alpha: float = 0.0,
    chroma_mode: int = 0,
    presence_bitmask: bytes | None = None,
    n_tiles: int = 0,
    layout: int = 0,
    per_tile_scales_u8: bytes | None = None,
) -> bytes:
    """Serialize a per-tile DCT residual block (format v5).

    Format::

        u8  version              (= 5)
        u8  chroma_mode          (0 = RGB, 1 = YCbCr 4:4:4, 2 = YCbCr 4:2:0)
        u8  layout               (0 = tile-major row-major, 1 = band-major zigzag)
        u8  _reserved            (= 0)
        f32 quant_step           (base step before per-frequency weighting)
        f32 freq_alpha           (per-frequency weight ramp; 0 = uniform)
        u16 channels
        u32 n_tiles              (total tile count for the presence bitmask)
        u32 n_int16              (length of the coefficient buffer)
        u32 n_scales             (length of the per-tile scales array; 0 = uniform)
        bytes ceil(n_tiles/8)    (presence bitmask: bit i = 1 → tile i has coeffs)
        bytes n_scales           (u8 per-tile quant-step scales, log-spaced [0.25, 4.0])
        n_int16 * int16          (DCT coefficients for present tiles only)

    v5 (2026-04-07) adds a per-tile quantization scale stream so the
    encoder can spend bits where they matter — finer quant on
    detail-rich tiles, coarser quant on smooth ones. ``n_scales`` is
    the count of u8 scales that follow the presence bitmask; it
    typically equals the number of present tiles (popcount of the
    bitmask) but ``n_scales == 0`` is the legacy "uniform quant"
    mode where every tile uses ``quant_step`` directly.

    v4 (with layout byte), v3 (with bitmask), and v2 (no bitmask)
    are still readable for backwards compatibility.
    """
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(coeffs, dtype=np.int16))
    if arr.ndim != 1:
        raise BitstreamError(f"DCT coefficients must be 1-D int16, got {arr.shape}")
    if chroma_mode not in (0, 1, 2):
        raise BitstreamError(f"chroma_mode must be 0/1/2, got {chroma_mode}")
    if freq_alpha < 0:
        raise BitstreamError(f"freq_alpha must be >= 0, got {freq_alpha}")
    if n_tiles < 0:
        raise BitstreamError(f"n_tiles must be >= 0, got {n_tiles}")
    if layout not in (0, 1):
        raise BitstreamError(f"layout must be 0 or 1, got {layout}")
    if presence_bitmask is None:
        presence_bitmask = b""
    expected_mask_len = (n_tiles + 7) // 8
    if len(presence_bitmask) != expected_mask_len:
        raise BitstreamError(
            f"presence_bitmask size {len(presence_bitmask)} does not match "
            f"n_tiles={n_tiles} (expected {expected_mask_len} bytes)"
        )
    if per_tile_scales_u8 is None:
        per_tile_scales_u8 = b""
    header = struct.pack(
        "<BBBBffHIII",
        5,                       # version
        int(chroma_mode),
        int(layout),
        0,                       # reserved
        float(quant_step),
        float(freq_alpha),
        int(channels),
        int(n_tiles),
        int(arr.size),
        int(len(per_tile_scales_u8)),
    )
    return header + bytes(presence_bitmask) + bytes(per_tile_scales_u8) + arr.tobytes()


def unpack_dct(blob: bytes) -> tuple:
    """Deserialize BLOCK_DCT bytes (v5 format, with v4/v3/v2 fallback).

    Returns ``(coeffs, quant_step, channels, freq_alpha, chroma_mode,
    presence_bitmask, n_tiles, layout, per_tile_scales_u8)``. When
    ``n_tiles == 0`` the presence bitmask is empty and every tile is
    treated as present. When ``per_tile_scales_u8`` is empty, every
    tile uses the uniform ``quant_step`` (legacy v2/v3/v4 behavior).
    ``layout`` is 0 (tile-major row-major) for v2/v3 and may be 0 or
    1 (band-major zigzag) for v4/v5.
    """
    import numpy as np
    if len(blob) < 1:
        raise BitstreamError("truncated DCT block header")
    version = blob[0]
    if version == 5:
        header_size = struct.calcsize("<BBBBffHIII")
        if len(blob) < header_size:
            raise BitstreamError("truncated DCT v5 block header")
        (
            _v, chroma_mode, layout, _res,
            quant_step, freq_alpha, channels, n_tiles, n_coeffs, n_scales,
        ) = struct.unpack_from("<BBBBffHIII", blob, 0)
        mask_len = (n_tiles + 7) // 8
        expected = header_size + mask_len + n_scales + n_coeffs * 2
        if len(blob) != expected:
            raise BitstreamError(
                f"DCT v5 block size mismatch: got {len(blob)}, expected {expected}"
            )
        off = header_size
        presence_bitmask = bytes(blob[off:off + mask_len])
        off += mask_len
        per_tile_scales_u8 = bytes(blob[off:off + n_scales])
        off += n_scales
        coeffs = np.frombuffer(
            blob, dtype=np.int16, offset=off, count=n_coeffs,
        ).copy()
        return (
            coeffs, float(quant_step), int(channels), float(freq_alpha),
            int(chroma_mode), presence_bitmask, int(n_tiles), int(layout),
            per_tile_scales_u8,
        )
    if version == 4:
        header_size = struct.calcsize("<BBBBffHII")
        if len(blob) < header_size:
            raise BitstreamError("truncated DCT v4 block header")
        (
            _v, chroma_mode, layout, _res,
            quant_step, freq_alpha, channels, n_tiles, n_coeffs,
        ) = struct.unpack_from("<BBBBffHII", blob, 0)
        mask_len = (n_tiles + 7) // 8
        expected = header_size + mask_len + n_coeffs * 2
        if len(blob) != expected:
            raise BitstreamError(
                f"DCT v4 block size mismatch: got {len(blob)}, expected {expected}"
            )
        presence_bitmask = bytes(blob[header_size:header_size + mask_len])
        coeffs = np.frombuffer(
            blob, dtype=np.int16, offset=header_size + mask_len, count=n_coeffs,
        ).copy()
        return (
            coeffs, float(quant_step), int(channels), float(freq_alpha),
            int(chroma_mode), presence_bitmask, int(n_tiles), int(layout),
            b"",
        )
    if version == 3:
        header_size = struct.calcsize("<BBffHII")
        if len(blob) < header_size:
            raise BitstreamError("truncated DCT v3 block header")
        (
            _v, chroma_mode, quant_step, freq_alpha, channels, n_tiles, n_coeffs,
        ) = struct.unpack_from("<BBffHII", blob, 0)
        mask_len = (n_tiles + 7) // 8
        expected = header_size + mask_len + n_coeffs * 2
        if len(blob) != expected:
            raise BitstreamError(
                f"DCT v3 block size mismatch: got {len(blob)}, expected {expected}"
            )
        presence_bitmask = bytes(blob[header_size:header_size + mask_len])
        coeffs = np.frombuffer(
            blob, dtype=np.int16, offset=header_size + mask_len, count=n_coeffs,
        ).copy()
        return (
            coeffs, float(quant_step), int(channels), float(freq_alpha),
            int(chroma_mode), presence_bitmask, int(n_tiles), 0, b"",
        )
    if version == 2:
        # Backwards-compat: v2 had no presence bitmask (every tile present).
        header_size = struct.calcsize("<BBffHI")
        if len(blob) < header_size:
            raise BitstreamError("truncated DCT v2 block header")
        _v, chroma_mode, quant_step, freq_alpha, channels, n_coeffs = struct.unpack_from(
            "<BBffHI", blob, 0,
        )
        expected = header_size + n_coeffs * 2
        if len(blob) != expected:
            raise BitstreamError(
                f"DCT v2 block size mismatch: got {len(blob)}, expected {expected}"
            )
        coeffs = np.frombuffer(blob, dtype=np.int16, offset=header_size, count=n_coeffs).copy()
        return (
            coeffs, float(quant_step), int(channels), float(freq_alpha),
            int(chroma_mode), b"", 0, 0, b"",
        )
    raise BitstreamError(f"unsupported DCT block version {version}")


def unpack_lite(blob: bytes) -> Any:
    """Deserialize BLOCK_LITE bytes back to a (grid_h, grid_w, 3) float32 lighting grid.

    Returns the inverse of ``pack_lite``: a multiplicative factor centered
    on 1.0. Note this is lossy due to int8 quantization (~0.016 step in
    linear lighting space).
    """
    import numpy as np
    if len(blob) < 4:
        raise BitstreamError("truncated LITE block header")
    w, h = struct.unpack_from("<HH", blob, 0)
    expected = 4 + h * w * 3
    if len(blob) != expected:
        raise BitstreamError(f"LITE block size mismatch: got {len(blob)}, expected {expected}")
    raw = np.frombuffer(blob, dtype=np.int8, offset=4, count=h * w * 3).reshape((h, w, 3))
    return raw.astype(np.float32) / 64.0 + 1.0


def encode_weft(
    *,
    head: HeadBlock,
    toc: list[int],
    prim_payload: bytes,
    residuals: list[tuple[int, int, int]] | None,
    residual_maps: list[bytes] | None = None,
    res2_payload: bytes | None = None,
    res1_grid_size: int = 4,
    pstr_payload: bytes | None = None,
    pdel_payload: bytes | None = None,
    qtree_payload: bytes | None = None,
    lite_payload: bytes | None = None,
    bic_payload: bytes | None = None,
    pal_payload: bytes | None = None,
    grd_payload: bytes | None = None,
    dct_payload: bytes | None = None,
    meta: dict[str, Any],
    chunk_index: list[PrimChunkEntry] | None = None,
    block_alignment: int = 64,
    major: int = MAJOR_VERSION,
    minor: int = MINOR_VERSION,
) -> bytes:
    if block_alignment <= 0 or block_alignment > 4096:
        raise BitstreamError("invalid block alignment")

    head_blob = _pack_head(head)
    toc_blob = _pack_toc(toc)
    res0_blob = _pack_res0(residuals) if residuals is not None else None
    res1_blob = _pack_res1(residual_maps, grid_size=res1_grid_size) if residual_maps is not None else None
    cidx_blob = pack_chunk_index(chunk_index) if chunk_index else None

    checksum_src = (
        head_blob
        + toc_blob
        + prim_payload
        + (res0_blob or b"")
        + (res1_blob or b"")
        + (res2_payload or b"")
        + (pstr_payload or b"")
        + (pdel_payload or b"")
        + (cidx_blob or b"")
        + (qtree_payload or b"")
        + (lite_payload or b"")
        + (bic_payload or b"")
        + (pal_payload or b"")
        + (grd_payload or b"")
        + (dct_payload or b"")
    )
    checksum = hashlib.sha256(checksum_src).hexdigest()
    meta_blob = _pack_meta({**meta, "sha256": checksum, "block_alignment": block_alignment})

    blocks: list[tuple[bytes, bytes]] = [
        (BLOCK_HEAD, head_blob),
        (BLOCK_TOC, toc_blob),
        (BLOCK_PRIM, prim_payload),
    ]
    if cidx_blob is not None:
        blocks.append((BLOCK_CIDX, cidx_blob))
    if res0_blob is not None:
        blocks.append((BLOCK_RES0, res0_blob))
    if res1_blob is not None:
        blocks.append((BLOCK_RES1, res1_blob))
    if res2_payload is not None:
        blocks.append((BLOCK_RES2, res2_payload))
    if pstr_payload is not None:
        blocks.append((BLOCK_PSTR, pstr_payload))
    if pdel_payload is not None:
        blocks.append((BLOCK_PDEL, pdel_payload))
    if qtree_payload is not None:
        blocks.append((BLOCK_QTREE, qtree_payload))
    if lite_payload is not None:
        blocks.append((BLOCK_LITE, lite_payload))
    if bic_payload is not None:
        blocks.append((BLOCK_BIC, bic_payload))
    if pal_payload is not None:
        blocks.append((BLOCK_PAL, pal_payload))
    if grd_payload is not None:
        blocks.append((BLOCK_GRD, grd_payload))
    if dct_payload is not None:
        blocks.append((BLOCK_DCT, dct_payload))
    blocks.append((BLOCK_META, meta_blob))

    # Compress large blocks with zstd.
    blocks = [
        (tag, _zstd_compress(body) if tag in _COMPRESSIBLE_BLOCKS else body)
        for tag, body in blocks
    ]

    block_count = len(blocks)
    header = struct.pack("<4sBBHI", MAGIC, major, minor, block_count, 0)
    dir_size = block_count * struct.calcsize("<4sQQ")
    start = _align_up(len(header) + dir_size, block_alignment)

    dir_entries = bytearray()
    payload = bytearray()
    running = 0

    if start > (len(header) + dir_size):
        payload += b"\x00" * (start - (len(header) + dir_size))
        running += start - (len(header) + dir_size)

    for tag, body in blocks:
        aligned_running = _align_up(running, block_alignment)
        if aligned_running > running:
            payload += b"\x00" * (aligned_running - running)
            running = aligned_running
        offset = len(header) + dir_size + running
        dir_entries += struct.pack("<4sQQ", tag, offset, len(body))
        payload += body
        running += len(body)

    return bytes(header + dir_entries + payload)


def decode_weft(blob: bytes) -> WeftFile:
    if len(blob) < struct.calcsize("<4sBBHI"):
        raise BitstreamError("truncated file header")
    magic, major, minor, block_count, _ = struct.unpack_from("<4sBBHI", blob, 0)
    if magic != MAGIC:
        raise BitstreamError("invalid magic")
    if major != MAJOR_VERSION:
        raise BitstreamError(f"unsupported major version: {major}")

    off = struct.calcsize("<4sBBHI")
    dir_size = block_count * struct.calcsize("<4sQQ")
    if off + dir_size > len(blob):
        raise BitstreamError("truncated block directory")

    block_map: dict[bytes, bytes] = {}
    block_entries: list[BlockEntry] = []
    for _ in range(block_count):
        tag, b_off, b_len = struct.unpack_from("<4sQQ", blob, off)
        off += struct.calcsize("<4sQQ")
        end = b_off + b_len
        if end > len(blob):
            raise BitstreamError("block extends past EOF")
        block_entries.append(BlockEntry(tag=tag, offset=int(b_off), length=int(b_len)))
        block_map[tag] = blob[b_off:end]

    for tag in (BLOCK_HEAD, BLOCK_TOC, BLOCK_PRIM, BLOCK_META):
        if tag not in block_map:
            raise BitstreamError(f"missing required block: {tag.decode('ascii', errors='ignore').strip()}")

    # Decompress any zstd-compressed blocks.
    for tag in _COMPRESSIBLE_BLOCKS:
        if tag in block_map:
            block_map[tag] = _zstd_decompress_if_needed(block_map[tag])

    head = _unpack_head(block_map[BLOCK_HEAD])
    toc = _unpack_toc(block_map[BLOCK_TOC])
    prim_payload = block_map[BLOCK_PRIM]
    meta = _unpack_meta(block_map[BLOCK_META])

    # Derive tile count.  For fixed-grid mode, validate against HEAD dimensions.
    # For adaptive mode (tile_cols == 0), infer from TOC.
    if head.tile_cols > 0 and head.tile_rows > 0:
        tile_count = head.tile_cols * head.tile_rows
        if toc and len(toc) != tile_count + 1:
            raise BitstreamError("TOC tile count mismatch")
    else:
        tile_count = max(0, len(toc) - 1) if toc else 0

    cidx_payload = block_map.get(BLOCK_CIDX)
    if cidx_payload is not None:
        chunk_index = unpack_chunk_index(cidx_payload)
    else:
        chunk_index = None

    if (head.flags & FLAG_CHUNKED_PRIM) and chunk_index is None:
        raise BitstreamError("HEAD indicates chunked PRIM but CIDX block is missing")

    residual_payload = block_map.get(BLOCK_RES0)
    if residual_payload is not None:
        residuals = _unpack_res0(residual_payload)
        if len(residuals) != tile_count:
            raise BitstreamError("RES0 tile count mismatch")
    if (head.flags & FLAG_HAS_RES0) and residual_payload is None:
        raise BitstreamError("HEAD indicates RES0 but RES0 block is missing")

    res1_payload = block_map.get(BLOCK_RES1)
    if res1_payload is not None:
        _, res1_maps = _unpack_res1(res1_payload)
        if len(res1_maps) != tile_count:
            raise BitstreamError("RES1 tile count mismatch")
    if (head.flags & FLAG_HAS_RES1) and res1_payload is None:
        raise BitstreamError("HEAD indicates RES1 but RES1 block is missing")

    res2_payload = block_map.get(BLOCK_RES2)
    pstr_payload = block_map.get(BLOCK_PSTR)
    pdel_payload = block_map.get(BLOCK_PDEL)
    qtree_payload = block_map.get(BLOCK_QTREE)
    lite_payload = block_map.get(BLOCK_LITE)
    bic_payload = block_map.get(BLOCK_BIC)
    pal_payload = block_map.get(BLOCK_PAL)
    grd_payload = block_map.get(BLOCK_GRD)
    dct_payload = block_map.get(BLOCK_DCT)
    if (head.flags & FLAG_HAS_LITE) and lite_payload is None:
        raise BitstreamError("HEAD indicates LITE but LITE block is missing")
    if (head.flags & FLAG_HAS_BIC) and bic_payload is None:
        raise BitstreamError("HEAD indicates BIC but BIC block is missing")
    if (head.flags & FLAG_HAS_PAL) and pal_payload is None:
        raise BitstreamError("HEAD indicates PAL but PAL block is missing")
    if (head.flags & FLAG_HAS_GRD) and grd_payload is None:
        raise BitstreamError("HEAD indicates GRD but GRD block is missing")
    if (head.flags & FLAG_HAS_DCT) and dct_payload is None:
        raise BitstreamError("HEAD indicates DCT but DCT block is missing")

    checksum_src = (
        block_map[BLOCK_HEAD]
        + block_map[BLOCK_TOC]
        + prim_payload
        + (residual_payload or b"")
        + (res1_payload or b"")
        + (res2_payload or b"")
        + (pstr_payload or b"")
        + (pdel_payload or b"")
        + (cidx_payload or b"")
        + (qtree_payload or b"")
        + (lite_payload or b"")
        + (bic_payload or b"")
        + (pal_payload or b"")
        + (grd_payload or b"")
        + (dct_payload or b"")
    )
    expected = hashlib.sha256(checksum_src).hexdigest()
    found = meta.get("sha256")
    if isinstance(found, str) and found != expected:
        raise BitstreamError("checksum mismatch")

    return WeftFile(
        major=major,
        minor=minor,
        head=head,
        toc=toc,
        prim_payload=prim_payload,
        res0_payload=residual_payload,
        res1_payload=res1_payload,
        res2_payload=res2_payload,
        pstr_payload=pstr_payload,
        pdel_payload=pdel_payload,
        cidx_payload=cidx_payload,
        chunk_index=chunk_index,
        qtree_payload=qtree_payload,
        lite_payload=lite_payload,
        bic_payload=bic_payload,
        pal_payload=pal_payload,
        grd_payload=grd_payload,
        dct_payload=dct_payload,
        meta=meta,
        block_entries=block_entries,
    )


def decode_residuals(blob: bytes | None) -> list[tuple[int, int, int]] | None:
    if blob is None:
        return None
    return _unpack_res0(blob)


def decode_residual_maps(blob: bytes | None) -> tuple[int, list[bytes]] | None:
    if blob is None:
        return None
    return _unpack_res1(blob)
