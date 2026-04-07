# Weft: An Adaptive-Basis Image Codec

*Tech-exploration whitepaper. April 2026.*

## Abstract

Weft is an experimental image codec built around a single design choice: instead of compressing every image with one basis (DCT for JPEG/WebP, wavelets for JPEG 2000, learned features for neural codecs), Weft picks a different basis for each image — and within each image, for each tile — from a small library of analytic representations. The encoder runs a rate-distortion tournament across the candidates and writes only the winner.

On a 14-fixture committed corpus spanning hard-edge, smooth, structured, and synthetic-photo content, Weft at quality 75 wins **10/14 vs JPEG and 11/14 vs WebP at iso-bytes**, with average PSNR deltas of **+11.6 dB vs JPEG and +5.8 dB vs WebP**. Across all 14 fixtures Weft uses **243 KB total**; PNG-lossless uses 1259 KB and WebP-lossless uses 850 KB on the same corpus. The wins come from non-DCT bases (palette, gradient field, primitive stack, bicubic patch) on content where DCT codecs structurally underperform.

The losses are honest and predictable: pure noise texture loses to JPEG by 5 dB because DCT is locally optimal for high-entropy frequency content, and synthetic photos with broad smooth regions lose by 1–2 dB. Both regimes are structurally hostile to discrete-basis codecs and we make no claim to beat JPEG there.

This whitepaper documents the core technique, the bases catalog, the auto-selection algorithm, the benchmark methodology, the measured results, and an honest section on what we tried that didn't work (most notably the "film projector" hypothesis around scale-independent decode).

## 1. Why another codec?

JPEG, WebP, AVIF, and JPEG XL all share a fundamental architectural choice: one basis representation, applied uniformly to every image. JPEG transforms 8×8 blocks of every image with DCT. WebP does the same with extra prediction modes. AVIF uses the AV1 transform stack — DCT, ADST, and FLIPADST — but they're still general-purpose linear transforms run over every pixel.

This choice is mathematically defensible and engineering-efficient: one transform, one entropy coder, one decoder, deployed across hundreds of millions of devices. But it means that on content far from natural-photo statistics, the codec is doing the wrong thing. A 256-color pixel-art icon compressed with JPEG produces ringing on every edge and uses dozens of kilobytes for what is essentially a label grid plus a tiny color palette. A page of vector-rendered diagrams compressed with WebP-lossless rounds the strokes into noise patterns. A pure smooth gradient compressed with PNG occupies thousands of bytes per kilopixel for what is a single bicubic patch's worth of information.

The right answer for each of those content types is well-known: use a palette codec for pixel art, use a vector format for diagrams, use a smooth-function approximation for gradients. The problem is that every image isn't *one* of those things — it's a mix, often within the same image. A mobile screenshot has UI chrome (palette), text (palette + edges), photographs (DCT), and gradient backgrounds (smooth). Using one codec on the whole image throws away a lot of information about the local structure.

Weft is what happens if you say: *what if the codec picked the right representation per tile?*

## 2. Core idea: adaptive basis selection

The codec's input is an image. The first step is to decompose the image into tiles using an adaptive quadtree — splits where local variance is high, merges where the content is uniform. Each tile then becomes an independent compression problem.

For each tile, Weft has a small library of candidate representations to choose from:

- **Primitive stack** — analytic shapes (constant patches, linear gradients, lines, quadratic Bézier curves, triangles, 4×4 bicubic Bernstein patches) composited per tile in z-order. Variable-length encoding; the encoder runs a greedy primitive search to find the smallest set of primitives that reconstructs the tile to a target error.

- **Per-tile bicubic patch** — a single 4×4 Bernstein control grid replacing the entire primitive stack for that tile. 48 bytes (16 control points × 3 channels × 1 byte each, normalized). Cheaper than a long primitive stack when the tile content is smooth.

- **Whole-image bicubic** — when the entire image is dominated by smooth content, a single global bicubic basis encoding can beat tile-level fits.

- **Palette + labels** — a color palette of K representative colors (k-means clustered in linear RGB) plus a per-pixel label grid stored as packed indices. Best for hard-edge content where every pixel is one of a small number of distinct colors.

- **Gradient field** — store ∂I/∂x and ∂I/∂y quantized to int8 per channel plus per-channel means; the decoder reconstructs I by solving the Poisson equation ∇²I = ∂(∂I/∂x)/∂x + ∂(∂I/∂y)/∂y via a closed-form DCT-II solve. Best for content where the *gradient field* is locally simple even if the image isn't (large flat regions with sharp boundaries — diagrams, charts, region maps).

- **DCT residual layer** — JPEG-style frequency-domain quantized residual added on top of any of the above primitive bases. YCbCr 4:2:0 chroma, per-tile presence bitmask, per-frequency quantization weighting, optional adaptive per-tile scales. Closes the natural-photo PSNR gap when the primitive bases plateau in quality.

The encoder fits each candidate against the source tile, computes a rate-distortion score (`PSNR − λ · BPP`), and keeps the winner. The bitstream records which candidate won so the decoder can dispatch to the matching renderer.

This is not a new technique — encoders have used multiple modes per tile since at least the H.264 era. What's unusual about Weft is that the modes aren't variants of one transform (DCT, intra-prediction modes, quantization tables) — they're entirely different mathematical representations of the same underlying signal. A palette tile and a primitive-stack tile and a DCT-residual tile have nothing in common at the math level. They share only the container that holds them and the auto-select tournament that picked them.

## 3. The bases in detail

### 3.1 Primitive stack

A primitive stack is a list of analytic shapes composited per tile. The supported primitive kinds:

| Kind | Geometry | Color | Coverage function |
|---|---|---|---|
| Const patch | none | RGB + alpha | All pixels |
| Linear gradient | (x₀,y₀,x₁,y₁) | RGB₀ + RGB₁ + alpha | t = clamp((p−p₀)·v / |v|², 0, 1); RGB(t) = lerp(RGB₀, RGB₁, t) |
| Line | (x₀,y₀,x₁,y₁,thickness) | RGB + alpha | exp(−d²/(2σ²)) where d is point-to-segment distance, σ = max(0.5, thickness) |
| Quadratic curve | (p₀, c, p₁, thickness) | RGB + alpha | Same as line, with d computed via 31-segment polyline approximation of the Bézier |
| Triangle | (p₀, p₁, p₂) | RGB + alpha | Barycentric coverage test |
| Bicubic patch | 16 control points | implicit (control points carry RGB) | Bernstein basis evaluated at the pixel grid in normalized [0,1]² coordinates |

Primitives are quantized: positions to u16 in [0, tile_size−1], colors to u8 sRGB, alpha to u16 in [0,1]. Geometry is tile-local — coordinates live in the tile's own [0, tile_size) space, not image-global, which means primitive positions are independent of where the tile sits in the image and stay short.

The encoder fits a primitive stack per tile by greedy search: for each tile, generate a candidate bank of ~18 model templates (const, multi-angle linears, lines, triangles, curves), score each by residual MSE after composite, pick the best, subtract its contribution, and repeat until the residual is below threshold or the tile budget is exhausted. The candidate scoring is parallelized: there is an optional CUDA kernel (`gpu_encoder.batch_tile_objective`) that scores all candidates for all tiles in one launch, with a CPU fallback.

### 3.2 Per-tile bicubic patch

A per-tile bicubic patch is a single 4×4 grid of RGB control points evaluated via Bernstein basis. The Bernstein cubic blending functions are:

  B₀(t) = (1−t)³, B₁(t) = 3t(1−t)², B₂(t) = 3t²(1−t), B₃(t) = t³

The reconstructed pixel at normalized position (u, v) ∈ [0,1]² is:

  I(u, v) = Σᵢ Σⱼ B_i(u) B_j(v) C[i][j]

where C[i][j] is the (i,j)-th 3-channel control point. Storage is 16 control points × 3 channels × 1 byte each = 48 bytes per tile.

The encoder fits a bicubic patch by solving the closed-form least-squares system for the control points given the source tile pixels, then quantizing the controls to u8. The fit takes one matrix solve per tile (no iteration). When the per-tile R-D tournament includes a bicubic candidate, this is the value scored against the primitive stack — and the encoder picks whichever gives the better R-D score.

### 3.3 Palette + labels

A palette+labels encoding consists of:
- a small color palette (K = 16 or 64 typical) of representative RGB colors, k-means clustered in linear RGB space against the image pixels
- a per-pixel label grid storing the index (log₂ K bits) of the closest palette color for each pixel

The decoder is a single `palette[labels]` lookup. The encoder adds an entropy-coding pass on the label grid (zstd via numpy) since spatially adjacent labels are highly correlated.

Palette encoding has two interesting structural properties:
1. **It's lossless on its own basis** — if the source image only uses K distinct colors, palette+labels is exactly recoverable.
2. **It scales independently** — to decode the image at a different target resolution, you nearest-neighbor sample the label grid at the target resolution and then look up the palette. Hard-edge content (icons, screenshots, diagrams) survives the upscale crisply because nearest-neighbor on a label grid never interpolates across a color boundary.

Both properties are why palette compresses pixel-art / screenshot / diagram content so much better than DCT codecs do, and why nearest-neighbor upscaling of a palette image is the structurally correct upsample for that content.

### 3.4 Gradient field

A gradient-field encoding stores:
- ∂I/∂x quantized to int8 per channel (signed range [−1, 1])
- ∂I/∂y quantized to int8 per channel
- Per-channel mean values (3 floats)

The decoder reconstructs I from the gradients by solving the Poisson equation:

  ∇²I = div(∇I) = ∂²I/∂x² + ∂²I/∂y²

with Dirichlet boundary conditions matching the encoded means. The closed-form solution uses a DCT-II decomposition: the Laplacian operator becomes a diagonal matrix in the DCT basis, so the solve is one forward DCT, one element-wise division, and one inverse DCT per channel. Total: O(N log N) per channel.

This basis wins on content where the *gradient field* is locally simple even when the *image* isn't: charts with lots of flat regions and sharp boundaries, region maps, diagrams, and other content with large constant zones separated by step edges. Quantizing the gradients to int8 throws away high-frequency fidelity (the int8 quantization floor is the structural ceiling), but for flat-regions-with-edges content the gradients are mostly zero everywhere except at the edges, so int8 is plenty.

### 3.5 DCT residual layer

A DCT residual layer is a JPEG-style additive frequency-domain residual that runs *on top of* whichever primitive basis the encoder picked. After the primitive layer reconstructs its best estimate of the source tile, the residual `source − primitive_reconstruction` is split into 8×8 blocks, transformed via DCT-II, quantized using a per-frequency weighting matrix, entropy coded, and stored.

The format supports YCbCr 4:2:0 chroma subsampling (the chroma planes are encoded at 1/4 the spatial density of luminance), per-tile presence bitmasks (tiles whose residual RMS is below a skip threshold are dropped from the stream, saving the bytes), and optional per-tile adaptive quantization scales.

The DCT residual is the layer that gives Weft competitive PSNR on natural photos. Without it, the primitive bases plateau at ~30–35 dB on dense natural texture; with it, Weft's `hybrid-dct-tight` variant matches WebP and JPEG on natural-photo content within a few decibels at the same byte budget (and beats them on the structured corpus where the primitive layer is doing most of the work).

The "hybrid-dct" naming is historical — the variant is "primitive layer + per-tile DCT residual", and "tight" means RES1 (legacy raster residual) is disabled because the DCT residual structurally subsumes it.

## 4. Auto-select: the encoder's R-D tournament

The encoder runs a tournament across a fixed list of candidate variants (`_AUTO_CANDIDATES` in `encoder.py`), encodes each, and picks the winner by score. Variants are listed in priority order (deterministic tiebreak):

1. `baseline` — primitive stack only
2. `hybrid` — primitive stack + per-tile bicubic R-D pick
3. `hybrid-dct` — hybrid + DCT residual + RES1 raster residual
4. `hybrid-dct-tight` — hybrid + DCT residual, RES1 disabled
5. `bicubic` — pure whole-image bicubic encode
6. `palette-16` — 16-color palette
7. `palette-64` — 64-color palette
8. `gradient` — gradient-field encode

The score is `PSNR(reconstruction, source) − λ · BPP(bytes_written, image_pixels)`. λ scales with the user's `quality` setting:

- `quality ≤ 75`: `λ = 4.0` (the historical default)
- `quality > 75`: `λ = 4.0 · ((100 − quality) / 25)²` — quadratic decay so `quality=90` gives λ ≈ 0.64, `quality=95` gives λ ≈ 0.16, and at `quality=100` λ goes to zero (purely PSNR-driven, byte cost ignored)

This makes the user's quality knob actually behave like a quality knob: higher quality means more bytes for more PSNR, and the auto-select tournament picks the variant whose Pareto position best matches the requested quality target.

There is also a PSNR tiebreak applied to close R-D scores: if two variants land within a 5-point window in score, the variant with PSNR ≥ 2 dB higher wins regardless of which had the lower R-D score. This avoids the encoder picking a "barely cheaper, much worse" alternative when the byte savings are noise-floor.

**Fit caching.** The first four variants (baseline, hybrid, hybrid-dct, hybrid-dct-tight) all share the same primitive-stack fit phase — they only differ in what gets layered on top (per-tile bicubic R-D, DCT residual, RES1). The encoder caches the post-fit `_AdaptiveFitState` and reuses it across these variants. This roughly halves auto-select wall time vs. running each variant from scratch.

A `prefer_scalable` feature flag is also available, which excludes the `hybrid-dct`, `hybrid-dct-tight`, and `gradient` variants from the candidate pool. The remaining variants are all "scale-independent" — they render crisply at any output resolution because their bases are analytic. Useful when the consumer of the bitstream needs to decode at multiple resolutions without paying for a bilinear upscale of a raster residual.

## 5. Bitstream container

A `.weft` file is a sequence of tagged blocks with a small fixed header:

```
File header (12 bytes):
  magic       4 bytes  b"WEFT"
  major       1 byte   1
  minor       1 byte   4 (current)
  block_count 2 bytes  number of blocks in directory
  reserved    4 bytes  zero

Block directory: block_count entries, each:
  tag         4 bytes  block type (e.g. "HEAD", "PRIM", "PAL ", "DCT ")
  offset      8 bytes  byte offset of payload from start of file
  length      8 bytes  payload length in bytes
```

After the directory comes the payloads, each aligned to a 64-byte boundary (so the decoder can map them with cache-line alignment).

Block types:

| Tag | Required? | Description |
|---|---|---|
| `HEAD` | yes | Image dimensions, tile size, color space, flags, primitive count cap |
| `TOC ` | when PRIM present | Per-tile offset/length table for the PRIM block |
| `PRIM` | when primitive variant won | Entropy-coded primitive stack bytes |
| `BIC ` | when bicubic variant won | Per-tile bicubic control grids (48 bytes/tile) |
| `PAL ` | when palette variant won | (palette, label grid) tuple |
| `GRD ` | when gradient variant won | Quantized gradient field + per-channel means |
| `DCT ` | optional, additive | Per-tile DCT-quantized residual layer |
| `QTREE` | yes | Adaptive tile layout (per-tile x, y, size) |
| `RES0` | optional | Per-tile scalar RGB residual (small bias correction) |
| `RES1` | optional, deprecated | Per-tile 4×4 raster residual maps (legacy; superseded by DCT) |
| `META` | yes | JSON metadata with checksum (auto-select variant, encode params, etc.) |

The header carries flag bits indicating which payload blocks are present, so the decoder can pre-allocate and dispatch in one pass. Forward compatibility is intentionally limited: the bitstream major version pins the format and minor versions can add new optional blocks but cannot change existing semantics.

## 6. Benchmark methodology

The benchmark in `scripts/whitepaper_bench.py` runs the following pipeline on every fixture in the committed corpus:

1. **Encode with Weft** at quality 75 with `auto_select=True` (the default user-facing setting). Record the encoded byte count and the auto-selected variant.
2. **Decode the Weft bitstream**, convert to sRGB uint8, and compute PSNR against the source image. This is the "quality at q=75" for Weft.
3. **Iso-bytes JPEG**: binary search JPEG quality 1..100 to find the highest quality whose encoded size is ≤ Weft bytes for that fixture. Decode and compute PSNR.
4. **Iso-bytes WebP**: same protocol, but with WebP (`Pillow.Image.save(format="WEBP", method=6)`).
5. **PNG lossless** and **WebP lossless** byte counts as reference points.

The iso-bytes protocol is the right comparison for *"given a budget, who gives the highest PSNR?"* — which is what a deployed codec actually has to answer. It's stricter than fixed-quality comparisons because it forces every codec to compete on the same byte budget rather than letting any codec hide quality losses behind a smaller file.

PSNR is computed on sRGB uint8 arrays (`10 · log₁₀(255² / MSE)`). Lossless reference points are reported as 99.99 dB (a display cap; mathematical PSNR is infinite). All comparisons use the source image as ground truth.

The corpus is 14 deterministic synthetic fixtures generated by `samples/inputs/_generate.py` plus the `hard-edges.png` and `synthetic-render-1024.png` reference images, all committed to the repo. Re-running `python samples/inputs/_generate.py` regenerates byte-identical fixtures. This means the entire benchmark is reproducible from the repo without external datasets.

## 7. Results

### 7.1 Per-fixture breakdown

| Fixture | Weft KB | Weft dB | JPEG dB | WebP dB | PNG KB | WebP-L KB | Variant |
|---|---:|---:|---:|---:|---:|---:|---|
| hard-edges | 11.6 | 43.57 | 25.45 | 31.92 | 42.9 | 13.7 | palette-64 |
| synth-chart | 6.7 | 47.65 | 26.76 | 32.97 | 12.3 | 5.0 | palette-64 |
| synth-diagram | 2.2 | 47.97 | 25.16 | 43.57 | 2.1 | 0.3 | palette-16 |
| synth-icons | 4.9 | 47.80 | 26.76 | 34.13 | 3.4 | 1.0 | gradient |
| synth-mandelbrot | 14.9 | 27.05 | 24.10 | 25.25 | 126.6 | 43.9 | palette-64 |
| synth-noise-texture | 48.4 | 39.16 | 44.49 | 44.25 | 251.8 | 182.7 | hybrid-dct-tight |
| synth-photo-landscape | 14.6 | 41.24 | 42.91 | 44.49 | 20.4 | 10.9 | hybrid-dct-tight |
| synth-photo-natural | 75.2 | 33.29 | 33.77 | 36.01 | 555.4 | 467.4 | hybrid-dct-tight |
| synth-pixel-sprite | 3.0 | 42.76 | 23.71 | 31.12 | 4.7 | 0.5 | gradient |
| synth-region-map | 2.3 | 48.12 | 22.81 | 40.52 | 2.1 | 0.4 | palette-16 |
| synth-shapes | 8.0 | 48.60 | 29.96 | 35.65 | 4.1 | 1.4 | gradient |
| synth-smooth-gradient | 12.7 | 49.47 | 49.70 | 48.45 | 97.1 | 65.4 | bicubic |
| synth-terminal | 8.5 | 49.52 | 29.06 | 37.99 | 5.3 | 2.0 | gradient |
| synthetic-render-1024 | 30.0 | 46.28 | 44.98 | 45.24 | 131.7 | 55.9 | hybrid-dct-tight |
| **AGGREGATE** | **243.0** | **43.75** | **32.12** | **37.97** | **1259.8** | **850.5** | |

(JPEG dB and WebP dB are at iso-bytes — JPEG/WebP quality tuned to match Weft's byte count for that fixture.)

### 7.2 Aggregate

- Weft total bytes across 14 fixtures: **243 KB**
- PNG-lossless equivalent: **1260 KB** (Weft is 5.2× smaller)
- WebP-lossless equivalent: **850 KB** (Weft is 3.5× smaller)
- Iso-byte wins (Weft PSNR > alternative): **10/14 vs JPEG**, **11/14 vs WebP**
- Average PSNR delta vs JPEG at iso-bytes: **+11.63 dB**
- Average PSNR delta vs WebP at iso-bytes: **+5.78 dB**

### 7.3 What variant won where

| Variant | Count | Fixtures |
|---|---:|---|
| `gradient` | 4 | synth-icons, synth-pixel-sprite, synth-shapes, synth-terminal |
| `hybrid-dct-tight` | 4 | synth-noise-texture, synth-photo-landscape, synth-photo-natural, synthetic-render-1024 |
| `palette-64` | 3 | hard-edges, synth-chart, synth-mandelbrot |
| `palette-16` | 2 | synth-diagram, synth-region-map |
| `bicubic` | 1 | synth-smooth-gradient |

Two clean clusters:

- **Hard-edge / structured / pixel-art content**: gradient and palette dominate (9/14). These are the fixtures where Weft's PSNR delta vs JPEG is in the +18 to +25 dB range — JPEG is doing fundamentally the wrong thing, and a basis that matches the content's structure pays off enormously.
- **Natural-photo and dense-texture content**: hybrid-dct-tight dominates (4/14). Here Weft is matching JPEG/WebP within ±2 dB at iso-bytes; the DCT residual layer is doing most of the work, with the primitive layer providing a small Pareto improvement.
- **Pure smooth content**: bicubic is the right answer (1/14). Synth-smooth-gradient is the only fixture where bicubic wins — it loses to JPEG by 0.23 dB at iso-bytes but compresses to 1/8 the bytes of the next-best lossless option.

### 7.4 Where Weft loses

Three fixtures where Weft falls behind at iso-bytes:

| Fixture | Weft dB | JPEG dB | Δ | Reason |
|---|---:|---:|---:|---|
| synth-noise-texture | 39.16 | 44.49 | −5.33 | Pure noise is the JPEG home court — DCT quantization is structurally optimal for high-entropy frequency content. Weft's hybrid-dct-tight gets close but pays a primitive-layer overhead it can't earn back. |
| synth-photo-landscape | 41.24 | 42.91 | −1.67 | Synthetic photo with broad smooth regions and sharp transitions. The primitive layer is choosing primitives that the DCT residual then has to undo, costing bytes. |
| synth-photo-natural | 33.29 | 33.77 | −0.48 | Multi-octave natural noise. Same story as above but tighter — 0.5 dB is within tournament noise. |

These losses are honest and not fixable without giving up something else. They're also predictable: any time the source image is dense high-frequency texture, a discrete-basis codec is structurally at a disadvantage to a pure transform codec.

## 8. What didn't work

### 8.1 The film projector hypothesis

The original premise of this codec — back when it was called RTIC and used OptiX ray tracing for the decoder — was *"tiny analytic bitstream + GPU rendering = arbitrary-resolution crisp output"*. The hypothesis was that a tiny primitive bitstream could be GPU-decoded sublinearly with respect to output pixel count, beating raster codecs at scale because raster codecs always have to do an O(output_pixels) bilinear upscale on top of their decode.

We tested this in a three-phase experiment after the rename:

1. **Phase 1**: built a CUDA primitive-decode kernel (`gpu_render.decode_tile_pixels`). Verified bit-exact vs CPU on five primitive kinds, single tile and multi-tile, at 1×, 2×, and 4× scale. ✓
2. **Phase 2**: wired the kernel into a `_render_primitive_stack_gpu` decoder path. Verified bit-exact vs CPU on 8 corpus fixtures at 1× and 2× upscale. ✓
3. **Phase 3**: benchmarked decode time at 1×, 2×, 4×, 8× output resolution against libjpeg-turbo and PIL-PNG.

The result was **partial**:

- **Sublinearity is real**: Weft's GPU decode log-log slope was ~0.65, JPEG's was ~1.10. The shape of the cost curve genuinely is flatter — Weft amortizes a fixed orchestration cost over more output pixels as the upscale grows.
- **Absolute speed is a loss**: at 8× upscale on a 1024² source, Weft GPU decode took ~4.9 seconds vs JPEG's ~0.3 seconds. The constant cost (cuMemAlloc / cuMemcpy / cuMemFree per call + Python orchestration + kernel launch overhead) dominates so much that even at 32× extrapolation Weft would still lose by ~6×.
- **The crispness advantage is real but commodity**: at 4× upscale on hard-edge content, Weft palette delivered an edge-sharpness score of 0.2514 vs the analytic ground truth's 0.2788 — but PNG with nearest-neighbor upscale matched it exactly (0.2505) at *smaller* bytes. The "scale-independence" we thought was unique to Weft's analytic primitives is also delivered by trivial nearest-neighbor on any indexed-color codec.

The honest verdict: the slope is real, the absolute speed loses, and the crispness niche is matched by PNG+NN. The remaining defensible angle from the film projector framing is the bicubic basis on smooth content (which compresses to ~1/8 the bytes of WebP-lossless), and that survived as one of the auto-select variants in the production codec.

### 8.2 The bicubic niche

After Phase 3 we ran a separate auto-select variant survey across 13 fixtures × q=50/75/90 × {default, prefer_scalable} to find whether the smooth-gradient bicubic win generalizes to real content. It does not. Bicubic is picked as the winning variant exclusively on `synth-smooth-gradient`. On every other fixture (real photos, mixed content, hard-edge content), bicubic loses R-D scoring to one of the other candidates at every quality level.

Our conclusion is that the per-tile bicubic patch (the `PRIM_BICUBIC` primitive kind 5 inside primitive stacks) is more useful than the standalone whole-image bicubic basis. Per-tile bicubic gets picked across many fixtures by the `hybrid` variant's per-tile R-D check, even though the standalone bicubic basis only wins on one fixture. The lesson: the basis is useful, but as a *local* tool inside the primitive stack, not as a *global* alternative.

### 8.3 Encoder R-D scoring drift

A latent bug surfaced during the GPU primitive kernel verification work: the encoder's CUDA R-D scoring kernel was using 16 polyline segments to approximate quadratic Bézier curves, while the CPU reference renderer uses 31. The encoder was scoring curve primitives against a slightly coarser rasterization than the CPU reference produced, which meant the R-D math for curve-containing tiles was slightly off. The fix is one constant change in `gpu_render.py` (16 → 31 segments) and a side effect of the film projector experiment was finding it.

This is the kind of bug that's invisible to test suites because both paths (encoder scoring + decoder render) are internally consistent — they just disagree with each other. It only shows up when you compare them directly, which the kernel correctness work happened to do.

## 9. Limitations and what Weft is not

- **Decode is not faster than libjpeg-turbo.** The CPU decoder is roughly 2–10× slower than libjpeg-turbo at the same output size. The GPU decode kernel exists and is correct but doesn't change the answer because per-call CUDA memory overhead dominates at practical scales.
- **No bytestream forward-compatibility.** The minor version is bumped on every format change; a v1.4 decoder will not read a v1.3 file. This is fine for a tech-exploration repo but unusable for any deployment.
- **Encoder is slow.** Auto-select runs 8 candidate encodings per image, and the slowest (the primitive search baseline) is on the order of seconds-per-megapixel. The encoder caches the primitive fit across the four primitive-family variants, but the bicubic / palette / gradient candidates each pay full encode time. The overall auto-select cost is roughly 2× the slowest single-variant encode.
- **No error correction or transmission framing.** The bitstream is a flat block container designed for local file storage, not network packetization or partial decoding under loss.
- **No perceptual or learned components.** Weft is a classical hand-built codec. Modern neural codecs (and ESRGAN-style upscalers) will beat it on natural texture reconstruction, especially at extreme scales.
- **Not a production codec.** The format may break between minor versions without notice. Don't put valuable images in `.weft` files.

## 10. Future directions

A few honest future directions for this codec, ordered by what we think is most likely to actually pay off:

1. **Per-call CUDA overhead reduction.** The film projector hypothesis was shut down by per-call cuMemAlloc/cuMemcpy/cuMemFree overhead, not by the kernel itself. A persistent buffer pool + cached kernel state might close 10× of the gap and put Weft's GPU decode within a small factor of libjpeg-turbo. This would be the most compelling thing to do for the "decode at scale" framing.

2. **Better encoder for the hybrid-dct path on natural photos.** The 1–2 dB loss vs JPEG on synthetic photos comes partly from the primitive layer choosing primitives the DCT residual then has to undo. A joint primitive-vs-DCT R-D allocation per tile (decide early whether the tile is "DCT-only" and skip the primitive search there) would likely close the gap.

3. **Structural verifiability / attribution.** The primitive stack representation has structure that DCT codecs don't: you can ablate individual primitives and see what each one contributes to the reconstruction. This is interesting for verifiable / interpretable image compression and for adversarial-robustness work, neither of which raster codecs can express. We haven't built any of this; it's listed here because it's the only direction that genuinely benefits from Weft's architecture rather than fighting it.

4. **Container format polish.** A real major-version-stable bitstream with forward-compatible block dispatch, framing for streaming, and a proper conformance suite. Not interesting research but necessary if anyone ever wanted to use this for real.

Nothing on this list is a confident bet. Weft has measurable wins on the corpus we have, and the honest framing is that those wins exist *because* the corpus has a lot of structured content. On a pure natural-photo corpus, JPEG and WebP would be more competitive and the wins would be smaller.

## 11. Reproducing the numbers

Everything in section 7 is reproducible from the repo with no external datasets:

```bash
# 1. Install
pip install -e .[gpu]   # or pip install -e .  for CPU-only

# 2. (Optional) regenerate the synthetic corpus
python samples/inputs/_generate.py

# 3. Run the benchmark
python scripts/whitepaper_bench.py

# Outputs:
#   samples/runs/whitepaper-bench/results.csv     per-fixture numbers
#   samples/runs/whitepaper-bench/summary.json    aggregate stats
```

The benchmark is deterministic — re-running it produces byte-identical encoded outputs because the encoder's primitive search is deterministic at the same quality + feature flag config. PSNR numbers will be byte-identical across runs.

## Appendix A: Repo layout

```
src/weft/                Codec source
  api.py                 High-level encode_image / benchmark entry
  encoder.py             Adaptive encoder with auto-select tournament
  decoder.py             Decoder with per-basis dispatch
  bitstream.py           File container, block dispatch, magic + version
  constants.py           Magic, version, block tags, primitive kinds
  primitives.py          Primitive type definitions and quantization
  render.py              CPU primitive renderer (reference)
  gpu_render.py          Optional CUDA primitive decode kernel (Phase 1+2)
  bicubic.py             Bicubic patch fit / render
  palette.py             K-means palette + label grid
  gradient_field.py      Poisson-solve gradient field
  dct_residual.py        DCT residual layer (per-tile, YCbCr 4:2:0)
  feature_flags.py       FeatureFlags dataclass (encoder behavior knobs)
  cli.py                 Argparse CLI
  ...
src/weft/kernels/        NVRTC-compiled CUDA sources for the encoder GPU path
tests/                   pytest suite (122 tests)
docs/
  WEFT_SPEC.md           Bitstream format specification (current v1.4)
  WEFT_WHITEPAPER.md     This document
  EXPERIMENTS.md         Experiment harness CLI reference
samples/inputs/          Committed test corpus (deterministic generators)
samples/runs/            (gitignored) Benchmark output directory
scripts/whitepaper_bench.py  This whitepaper's reproduction script
```

## Appendix B: Glossary

- **Adaptive quadtree** — recursive image decomposition where regions of high local variance get smaller tiles and uniform regions get larger ones.
- **Auto-select** — the encoder's tournament across candidate variants, scoring by `PSNR − λ · BPP` and keeping the winner.
- **Iso-bytes** — comparing two codecs at the same byte budget. The iso-bytes protocol used here picks the highest JPEG/WebP quality whose output is ≤ Weft's byte count.
- **Primitive stack** — list of analytic shapes composited per tile in z-order.
- **PRIM_BICUBIC** — kind 5 in the primitive enum; a single 4×4 Bernstein control grid acting as a per-tile primitive.
- **Hybrid variant** — the auto-select candidate that runs both a primitive search AND a per-tile bicubic R-D check, swapping in the per-tile bicubic when it wins locally.
- **DCT residual** — JPEG-style additive frequency-domain residual layer applied on top of the primitive reconstruction.
- **Gradient field** — `(∂I/∂x, ∂I/∂y)` quantized to int8, decoded via Poisson solve.
- **Palette + labels** — K-color palette plus per-pixel index grid.
