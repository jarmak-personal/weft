# Weft Bitstream Spec (v1.4)

This document describes the Weft `.weft` container format as of the v1.4 bitstream revision. The format is a small fixed header followed by a directory of tagged blocks, each aligned to a 64-byte boundary. Different auto-select variants produce different block combinations in the same container.

## Container Layout

Binary little-endian file:

1. **File header** (12 bytes):
   - `magic(4)` — `b"WEFT"`
   - `major(1)` — `1`
   - `minor(1)` — `4` (current)
   - `block_count(2)` — number of entries in the block directory
   - `reserved(4)` — zero

2. **Block directory** — `block_count` entries, each:
   - `tag(4)` — 4-byte block type tag (e.g. `b"HEAD"`, `b"PRIM"`, `b"PAL "`)
   - `offset(8)` — byte offset from start of file to the block's payload
   - `length(8)` — length of the payload in bytes

3. **Block payloads** — concatenated, each aligned to a 64-byte boundary.

All multi-byte integers are little-endian. The 64-byte alignment lets a GPU-resident decoder map payloads with cache-line alignment; the gaps between payloads are zero-padded.

## Required Blocks

### HEAD

The header block is always present and always first. Format: `<IIHHBBBBIIBBH>`

- `width(u32)`, `height(u32)` — source image dimensions in pixels
- `tile_size(u16)` — nominal tile size (16 for uniform grids; adaptive variants set this as a hint)
- `max_primitives(u16)` — per-tile primitive count cap
- `color_space(u8)` — 1 = linear RGB
- `quant_mode(u8)` — 1 = UNORM8
- `flags(u8)` — see flag bits below
- `reserved(u8)` — zero
- `tile_cols(u32)`, `tile_rows(u32)` — grid dimensions, or (0, 0) for adaptive quadtree bitstreams (layout is in QTREE instead)
- `quality(u8)`, `preset_id(u8)` — encoder quality setting, preset identifier
- `padding(u16)` — zero

**Flag bits:**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `DETERMINISTIC` | Encoded with deterministic branches enabled |
| 1 | `HAS_RES0` | RES0 scalar residual block is present |
| 2 | `HAS_RES1` | RES1 raster residual block is present (deprecated, superseded by DCT) |
| 3 | `CHUNKED_PRIM` | PRIM is chunk-entropy-coded, CIDX is present |
| 4 | `LAYERED_GEOMETRY` | Primitive stack uses multi-layer z-ordering |
| 5 | `HAS_LITE` | Albedo/lighting decomposition with low-res lighting grid |
| 6 | `HAS_BIC` | Whole-image bicubic patch encode (no PRIM stack) |
| 7 | `HAS_PAL` | Palette + labels encode (no PRIM, no QTREE) |

The v1.4 header has two additional flag bits beyond the 8 listed above, carried in the high byte of the reserved region for gradient and DCT layers:

| Bit | Name | Meaning |
|---|---|---|
| 8 | `HAS_GRD` | Gradient-field encode (no PRIM, no QTREE) |
| 9 | `HAS_DCT` | Additive DCT residual layer is present (stacks on top of any primitive basis) |

### META

The META block is always present. Payload: UTF-8 JSON object with a trailing SHA-256 checksum over the JSON bytes. The JSON carries:

- `auto_selected_variant` — which variant won the R-D tournament (`baseline`, `hybrid`, `hybrid-dct`, `hybrid-dct-tight`, `bicubic`, `palette-16`, `palette-64`, `gradient`)
- `encode_width`, `encode_height` — dimensions the encoder actually fit against (may be smaller than HEAD dimensions if `encode_scale < 1` was used)
- `feature_flags` — the full FeatureFlags dict used to encode
- `quality` — numeric quality setting
- encoder-internal telemetry (tile counts, bytes per block, variant scores)

Decoders should rely on the auto-select variant string + HEAD flag bits to dispatch; JSON contents beyond those are informational.

## Primitive-Family Blocks

The primitive-family variants (`baseline`, `hybrid`, `hybrid-dct`, `hybrid-dct-tight`) write the following blocks:

### PRIM

Entropy-coded byte stream of tile-concatenated primitive payloads. Each tile's primitive list is encoded as:

```
u8 prim_count
for each primitive:
    u8 kind      # 0 = const, 1 = linear, 2 = line, 3 = curve, 4 = triangle, 5 = bicubic
    u8 payload_length
    payload      # see primitive payloads below
```

The concatenated tile bytes are entropy-coded in either whole-stream mode (`rans`) or chunked mode (`chunked-rans` + CIDX).

### TOC

Per-tile index into the decoded PRIM byte stream. Format: `u32 count` followed by `count * u32` offset entries. For N tiles, TOC contains N+1 entries (the last is the total decoded length).

### CIDX (optional)

Chunk directory for chunked-rANS PRIM mode. Format: `u32 chunk_count` followed by `chunk_count * <IIIIII>` entries:

- `start_tile(u32)` — first tile in this chunk
- `tile_count(u32)` — number of tiles in this chunk
- `raw_offset(u32)` — offset into the decoded byte stream
- `raw_length(u32)` — decoded length in bytes
- `coded_offset(u32)` — offset into the PRIM block (encoded bytes)
- `coded_length(u32)` — encoded length in bytes

CIDX enables chunk-local entropy decode and lets a GPU scheduler process chunks in parallel.

### QTREE

Adaptive quadtree tile layout. Format: `u32 tile_count` followed by `tile_count * <HHHH>` entries:

- `x(u16)`, `y(u16)` — tile top-left corner in encoded-image coordinates
- `size(u16)` — tile side length in pixels (legal values are 8, 16, 32)
- `index(u16)` — linearization index used by PRIM/TOC/residual blocks

QTREE is required for primitive-family bitstreams produced by the adaptive encoder. Legacy uniform-grid bitstreams (`tile_cols` and `tile_rows` both non-zero in HEAD) synthesize the quadtree from the grid at decode time.

## Primitive Payloads

Primitive geometry is **tile-local** — coordinates are expressed in the tile's own `[0, tile_size)` space, not image-global. Positions are quantized to u16 in `[0, 15]` for 16×16 tiles; colors are quantized to u8 sRGB in `[0, 255]`; alpha is quantized to u16 in `[0, 1]`.

| Kind | Payload size | Layout |
|---|---:|---|
| `const_patch` (0) | 5 | `color(3) alpha(2)` |
| `linear_patch` (1) | 16 | `x0 y0 x1 y1(4*u16) color0(3) color1(3) alpha(u16)` |
| `line` (2) | 15 | `x0 y0 x1 y1 thickness(5*u16) color(3) alpha(u16)` |
| `quad_curve` (3) | 19 | `x0 y0 cx cy x1 y1 thickness(7*u16) color(3) alpha(u16)` |
| `polygon` (4) | 17 | `x0 y0 x1 y1 x2 y2(6*u16) color(3) alpha(u16)` (triangle) |
| `bicubic_tile` (5) | 48 | `control_grid(48 u8)` — 16 control points × 3 channels |

The bicubic primitive (kind 5) is a whole-tile renderer: when a tile's primitive list contains a bicubic, it's always the only primitive in that tile and it replaces whatever was below it. Control points are stored in row-major `(j, i, ch)` order.

## Alt-Basis Blocks

The non-primitive variants (`bicubic`, `palette-16`, `palette-64`, `gradient`) write a single basis block instead of PRIM/TOC/QTREE. HEAD sets the corresponding `HAS_*` flag bit.

### BIC — Whole-Image Bicubic

Payload: per-tile 4×4 bicubic Bernstein control grids, concatenated. Each tile contributes `16 control points × 3 channels × 1 byte = 48 bytes`. The tile layout is in QTREE (which IS present for BIC — the whole-image bicubic is really per-adaptive-tile bicubic).

### PAL — Palette + Labels

Payload layout:

```
u8 k                      # palette size (16 or 64 typical)
u8 bits_per_label         # log2(k), typically 4 or 6
u16 reserved
k * 3 * u8 palette        # RGB triples in linear [0, 255]
u32 label_bytes           # compressed label grid byte count
label_bytes               # zstd-compressed packed label grid
```

The label grid is laid out in row-major order at the encoded image resolution. No QTREE is used — palette encodes the whole image as a single dense label grid.

### GRD — Gradient Field

Payload layout:

```
u16 width, height         # gradient field dimensions
u8 scale                  # gradient quantization scale factor
u8 reserved
3 * f32 means             # per-channel image means
width * height * 3 * i8 gx    # quantized ∂I/∂x
width * height * 3 * i8 gy    # quantized ∂I/∂y
```

The decoder reconstructs I by Poisson-solving ∇²I = div(∇I) via a DCT-II decomposition, then adds the per-channel means.

### DCT — Additive Residual Layer (optional, stacks on top of any primitive basis)

The DCT residual layer is an optional *additive* block that can accompany any primitive-family bitstream. When present, the decoder:

1. Renders the primitive / bicubic / palette / gradient basis as usual
2. Decodes the DCT residual and adds it on top
3. Clamps the result to `[0, 1]`

Payload carries:

- Per-tile quantized DCT coefficients (band-major zigzag layout)
- Quantization step + per-frequency weighting alpha
- Chroma mode (0 = RGB, 1 = YCbCr 4:4:4, 2 = YCbCr 4:2:0)
- Presence bitmask (tiles whose residual RMS is below threshold are skipped)
- Optional per-tile adaptive quantization scales (v5 feature, off by default)

The DCT residual supports YCbCr 4:2:0 chroma subsampling, so chroma planes are transmitted at 1/4 the spatial density of luminance. This is the JPEG-compatible colorspace choice and matches how JPEG handles chroma.

### RES0 — Scalar Residual (optional)

Per-tile scalar RGB bias applied after primitive compositing. Format: `tile_count * 3 * i8` (values in `[-127, 127]`). The decoder adds `residual_rgb / 255.0` to each tile's output before the final clamp.

RES0 is a legacy narrow residual layer that closes small color-shift errors the primitive search can't quite hit. It's typically a few bytes per tile and is included by the `baseline` and `hybrid` variants when `enable_res0=True`.

### RES1 — Raster Residual Maps (deprecated)

Per-tile low-resolution raster residual maps, intended as a bridge between primitive rasterization and DCT residual layers. Format:

```
u32 tile_count
u8 grid_w, grid_h        # residual grid dimensions per tile
u16 reserved
tile_count * grid_w * grid_h * 3 i8 samples
```

RES1 is deprecated as of v1.3 and disabled in the `hybrid-dct-tight` auto-select variant — the DCT residual layer structurally subsumes it (DCT's low-frequency coefficients express the same signal at finer granularity). New bitstreams should not produce RES1.

### LITE — Lighting Grid (experimental)

Optional low-resolution lighting grid for albedo×lighting decomposition. When HAS_LITE is set, the decoder multiplies the primitive reconstruction (treated as an albedo image) by the bilinearly-upsampled lighting grid. Experimental — not used by any current auto-select variant.

## Decode Determinism

- The primitive search, auto-select tournament, and R-D scoring are all deterministic at a given feature-flag config
- Sample positions are at pixel centers (half-integer coordinates) — this is bit-equivalent between the CPU reference renderer and the optional GPU primitive decode kernel
- The decoder emits a SHA-256 decode hash over the float32 output buffer (see `decode_hash` in `render.py`) that's stable across CPU / GPU decode paths

## Version History

- **v1.0** — initial prototype with primitive stack + RES0/RES1
- **v1.1** — adaptive quadtree (QTREE block), chunked rANS (CIDX)
- **v1.2** — alt-basis blocks (BIC, PAL), variant dispatch via HEAD flags
- **v1.3** — GRD gradient-field basis, DCT residual layer (first cut), hybrid auto-select
- **v1.4** — DCT v5 layout (band-major zigzag, per-tile adaptive quant scaffold), cleanup pass, rename from RTIC to Weft, bitstream magic `b"WEFT"`

Forward compatibility is intentionally limited: a minor version bump may change block semantics. A `v1.3` decoder will not read a `v1.4` file and vice versa.
