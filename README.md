# Weft

**An adaptive-basis image codec.** Weft encodes an image by picking the best representation from a small library of bases, per tile and per image — primitive stacks, bicubic patches, color palettes, gradient fields, and DCT frequency residuals. For each image the encoder runs a rate-distortion tournament over the candidates and writes only the winner.

This is a tech-exploration repo, not a production codec. The measurements below are real; the engineering is not production-grade. Expect rough edges.

## What it does well

On the committed 14-fixture test corpus (see `scripts/whitepaper_bench.py` for the reproduction), Weft at quality 75 wins **10/14 vs JPEG and 11/14 vs WebP at iso-bytes**, averaging **+11.6 dB vs JPEG** and **+5.8 dB vs WebP**. Across all fixtures Weft uses 243 KB total; PNG-lossless uses 1260 KB and WebP-lossless uses 850 KB on the same corpus. The most striking wins are on structured / mixed content where classical DCT codecs plateau:

- Natural photos (landscape, cartoon-family, baseball): primitive stack + DCT residual beats JPEG by 3–6 dB at the same byte budget
- Pixel art and diagrams: palette basis wins at ~10× smaller bytes than PNG
- Synthetic rendered scenes: primitive stack wins decisively (single-digit KB for crisp reconstruction)
- Smooth analytic content: bicubic basis wins PSNR/byte against everything including WebP-lossless

## What it doesn't do

- **Decode is not faster than libjpeg.** Weft's CPU decoder is roughly 2–10× slower than libjpeg-turbo at the same output size. The primitive walk is analytic but has real constant overhead per tile. An experimental CUDA decode kernel exists (`src/weft/gpu_render.py`) and is correct, but per-call GPU memory overhead dominates at practical scales, so it's not actually faster than CPU either.
- **The "crisp at any scale" property is real but commodity.** Weft's primitive bases render analytically at any target resolution, but for hard-edge content PNG + nearest-neighbor upscale delivers the same edge sharpness for smaller bytes. The exception is smooth analytic content where Weft's bicubic path genuinely wins.
- **No perceptual / neural components.** This is a classical codec. Modern neural codecs (and ESRGAN-style upscalers) will beat Weft on natural texture reconstruction at scale.

## Core idea

For each tile (adaptive quadtree, variable sizes) the encoder fits every candidate basis:

| Basis | Best for | Bitstream block |
|---|---|---|
| Primitive stack | Mixed content; analytic shapes | `PRIM` |
| Per-tile bicubic patch | Smooth gradients | `PRIM` (as `PRIM_BICUBIC`) |
| Full-image bicubic | Entirely smooth images | `BIC` |
| Palette + labels | Hard-edge pixel art / screenshots | `PAL` |
| Gradient field (Poisson solve) | Large flat regions with sharp edges | `GRD` |
| DCT residual (additive) | Everything above + natural texture fidelity | `DCT` |

Auto-select runs all candidates, scores by `PSNR − λ · BPP`, keeps the winner. Lambda scales with the quality knob so `quality=90` actually means "spend bytes for quality".

The decoder reads the block directory and dispatches to the renderer matching the bases present in the bitstream.

## Install

```bash
pip install -e .
```

Optional GPU extras (used by the encoder's tile-fit scoring kernel; degrades to CPU automatically if missing):

```bash
pip install -e .[gpu]
```

## CLI

```bash
# Encode with auto-select (recommended)
weft encode input.png output.weft --quality 75

# Decode
weft decode output.weft decoded.png

# Decode at a different resolution (analytic bases render crisply at any scale)
weft decode output.weft upscaled.png --width 1920 --height 1080

# Benchmark against a fixture set
weft bench ./dataset report.json --quality 75
```

## Python API

```python
from weft.api import encode_image
from weft.decoder import decode_image, decode_to_array
from weft.types import EncodeConfig

cfg = EncodeConfig(
    quality=75,
    feature_flags={"auto_select": True},
)
report = encode_image("input.png", "output.weft", cfg)
print(f"{report.bytes_written} bytes, variant={report.metadata['auto_selected_variant']}")

# Decode to a file
decode_image("output.weft", "decoded.png")

# Or decode directly to a NumPy array (linear RGB float32)
array = decode_to_array("output.weft", width=1920, height=1080)
```

## Read more

- **Project site** — `docs/index.html` is a GitHub Pages landing page with interactive image-comparison sliders for all 14 fixtures. To preview locally: `cd docs && python -m http.server 8000` then open `http://localhost:8000/`. To deploy: enable GitHub Pages in the repo settings and point it at `/docs` on `main`.
- [`docs/WEFT_WHITEPAPER.md`](docs/WEFT_WHITEPAPER.md) — full technical whitepaper: core idea, bases catalog, auto-select algorithm, benchmark methodology + results, honest limitations, future directions
- [`docs/WEFT_SPEC.md`](docs/WEFT_SPEC.md) — bitstream format specification (current v1.4)
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — experiment harness CLI reference
- [`scripts/whitepaper_bench.py`](scripts/whitepaper_bench.py) — one-command reproduction of the whitepaper numbers (no external datasets required)
- [`scripts/generate_site_assets.py`](scripts/generate_site_assets.py) — rebuilds the comparison PNGs + JSON the site loads

## Repo layout

```
src/weft/              Codec source (encoder, decoder, CLI, bases, GPU helpers)
src/weft/kernels/      NVRTC-compiled CUDA sources
tests/                 pytest suite (122 tests — bitstream, encoder, decoder, bases)
docs/WEFT_SPEC.md      Bitstream format specification
docs/WEFT_WHITEPAPER.md   Technical whitepaper
samples/inputs/        Committed test corpus (deterministic generators)
scripts/whitepaper_bench.py   Benchmark reproduction script
```

## Naming

The codec was originally called RTIC ("Ray Traced Image Codec") when the decoder used OptiX to render primitives on GPU ray-tracing cores. That path was removed; "ray traced" stopped being true long before the name got updated. The new name comes from weaving: each tile picks a different *weft* — a different thread — and the tiles together form the fabric of the image. Optional backronym: **W**eighted **E**ncoding of **F**requency **T**iles.

## Status

Active tech exploration. Not recommended for production use. Bitstream format may break between minor versions without notice.
