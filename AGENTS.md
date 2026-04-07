# AGENTS.md

## What This Is

Weft is an experimental image codec built around **adaptive basis selection**: for each image — and for each tile within the image — the encoder picks the representation from a small library of bases that best fits the content under a rate-distortion score. The bases are:

- **Primitive stack** — analytic shapes (constant patches, linear gradients, lines, quadratic curves, triangles, 4×4 bicubic Bernstein patches) composited per tile. Crisp at any decode resolution because each primitive is evaluated analytically.
- **Bicubic patch** — a single 4×4 Bernstein control grid per tile, for smooth content.
- **Palette + labels** — a quantized color palette plus per-pixel label indices, for hard-edge content (pixel art, screenshots, diagrams).
- **Gradient field** — `∂I/∂x` and `∂I/∂y` quantized to int8 per channel; the decoder reconstructs via a DCT Poisson solve.
- **DCT residual layer** — a JPEG-style frequency-domain residual added on top of any of the above. Closes the natural-photo PSNR gap.

The encoder runs `auto_select`: it tries each variant, scores by `PSNR − λ·BPP`, keeps the winner, and writes the winning bitstream. The decoder dispatches based on which block types are present.

The codec's historical name was RTIC ("Ray Traced Image Codec") back when the decoder used OptiX ray tracing to render primitives on the GPU. The ray tracing path was removed; the CPU renderer is now primary and bit-equivalent to the encoder's self-consistency scoring. A CUDA primitive-decode kernel exists for experimentation but is not faster than CPU at practical scales (per-call overhead dominates).

## Repo Layout

```
src/weft/              Codec package (encoder, decoder, CLI, entropy, metrics, gpu)
src/weft/kernels/      NVRTC-compiled CUDA source for the optional GPU encoder path
tests/                 Roundtrip, conformance, entropy, bitstream, auto-select tests
docs/                  Spec (WEFT_SPEC.md), GPU integration notes, experiments log
samples/inputs/        Committed test corpus (natural photos + synthetic fixtures)
samples/runs/          (gitignored) Transient benchmark outputs
```

## Execution Model

- **CPU-first.** The production encoder and decoder are pure CPU + NumPy. Everything that matters for correctness runs on CPU.
- **GPU is opt-in.** `cuda-python` + NVRTC is used for two accelerated paths: (1) the encoder's R-D scoring tile-fit kernel (2× encode speed on supported hardware) and (2) an experimental primitive-decode kernel. Both degrade gracefully to CPU if CUDA isn't available.
- **No compiled C extensions.** All device code is JIT-compiled at runtime through NVRTC.

## Encode / Decode Flow

**Encode** (`weft encode`):
1. Load image → linear RGB
2. Adaptive quadtree decompose (variable tile sizes)
3. For each auto-select variant: fit the appropriate basis (primitive search / bicubic fit / k-means palette / Poisson gradient / DCT transform), compute R-D score
4. Pick the winner, optionally add a DCT residual layer, write the `.weft` container (`HEAD`, `TOC`, `PRIM`/`BIC`/`PAL`/`GRD`, optional `RES0`/`RES1`/`DCT`, `QTREE`, `META`)

**Decode** (`weft decode`):
1. Read container, parse blocks
2. Dispatch on present blocks to the matching renderer (primitive stack / bicubic / palette / gradient field)
3. Apply DCT residual on top if present
4. Apply any lighting / refinement layers
5. Save output image

## Key Design Notes

- Bitstream magic is `b"WEFT"` (4 bytes) followed by version + block directory.
- Tile layout is adaptive quadtree (variable-size), stored in a `QTREE` block.
- Auto-select uses quality-aware lambda scaling: `((100 - q) / 25.0)²` for `q > 75`, flat otherwise, so `quality` actually behaves like a knob the user expects.
- The encoder caches the adaptive-primitive fit across auto-select variants that share the same fit-relevant config — a single `_AdaptiveFitState` is built once and reused across baseline / hybrid / hybrid-dct / hybrid-dct-tight.
