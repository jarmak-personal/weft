"""Constants for WEFT bitstream and codec defaults."""

from __future__ import annotations

MAGIC = b"WEFT"
MAJOR_VERSION = 1
MINOR_VERSION = 4  # bumped at the RTIC → Weft rename

BLOCK_HEAD = b"HEAD"
BLOCK_TOC = b"TOC "
BLOCK_PRIM = b"PRIM"
BLOCK_CIDX = b"CIDX"
BLOCK_RES0 = b"RES0"
BLOCK_RES1 = b"RES1"
BLOCK_RES2 = b"RES2"
BLOCK_PSTR = b"PSTR"
BLOCK_PDEL = b"PDEL"
BLOCK_META = b"META"
BLOCK_QTREE = b"QTRE"
BLOCK_LITE = b"LITE"  # Phase 2 #17: low-resolution lighting grid for albedo×lighting decomposition
BLOCK_BIC = b"BIC "  # Brainstorm #11: per-tile bicubic Bézier control grids (closed-form fit, no greedy search)
BLOCK_PAL = b"PAL "  # Brainstorm #20: K-color palette + per-pixel index grid (no per-tile structure)
BLOCK_GRD = b"GRD "  # Brainstorm #1: int8 gradient field + per-channel means (DCT Poisson decode)
BLOCK_DCT = b"DCT "  # Brainstorm #16: per-tile DCT-quantized residual (additive layer on primitive reconstruction)

TILE_SIZE = 16
TILE_SIZE_MIN = 8
TILE_SIZE_MAX = 32
TILE_SIZES = (8, 16, 32)
MAX_PRIMITIVES_PER_TILE = 48
MAX_PRIMITIVES_BY_TILE_SIZE = {8: 24, 16: 48, 32: 96}
COLOR_SPACE_LINEAR_RGB = 1
QUANT_MODE_UNORM8 = 1
QUANT_MODE_UNORM16 = 2

DEFAULT_PRESET = "rtx-heavy-v2"

PRIM_CONST_PATCH = 0
PRIM_LINEAR_PATCH = 1
PRIM_LINE = 2
PRIM_QUAD_CURVE = 3
PRIM_POLYGON = 4
# Per-tile hybrid: a single 4×4 bicubic Bézier control grid (48 bytes
# of u8 RGB control points) standing in for an entire tile's primitive
# stack. The encoder fits both a primitive stack AND a bicubic per
# tile, then picks whichever has the better R-D score, storing the
# winner as the tile's primitive list. Bicubic primitives are opaque
# (alpha = 1) and replace whatever was in the framebuffer below them,
# so when present they're always the only primitive in their tile.
PRIM_BICUBIC = 5

PRIMITIVE_NAME_TO_ID = {
    "const_patch": PRIM_CONST_PATCH,
    "linear_patch": PRIM_LINEAR_PATCH,
    "line": PRIM_LINE,
    "quad_curve": PRIM_QUAD_CURVE,
    "polygon": PRIM_POLYGON,
    "bicubic_tile": PRIM_BICUBIC,
}

PRIMITIVE_ID_TO_NAME = {v: k for k, v in PRIMITIVE_NAME_TO_ID.items()}

QUALITY_MIN = 0
QUALITY_MAX = 100

# Head flags.
FLAG_DETERMINISTIC = 1 << 0
FLAG_HAS_RES0 = 1 << 1
FLAG_HAS_RES1 = 1 << 2
FLAG_CHUNKED_PRIM = 1 << 3
FLAG_LAYERED_GEOMETRY = 1 << 4
FLAG_HAS_LITE = 1 << 5  # bitstream contains a BLOCK_LITE lighting grid (#17)
FLAG_HAS_BIC = 1 << 6   # bitstream is a bicubic-patch encode (no PRIM stack; #11)
FLAG_HAS_PAL = 1 << 7   # bitstream is a palette-planes encode (no PRIM stack, no QTREE; #20)
FLAG_HAS_GRD = 1 << 8   # bitstream is a gradient-field encode (no PRIM stack, no QTREE; #1)
FLAG_HAS_DCT = 1 << 9   # bitstream carries an additive per-tile DCT residual layer (#16)
