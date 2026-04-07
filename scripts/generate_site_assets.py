"""Generate comparison image assets for the Weft GitHub Pages site.

For each corpus fixture:
1. Copy the source PNG into docs/assets/compare/
2. Encode with Weft at quality 75 auto-select, decode, save as PNG
3. Binary-search JPEG quality to find iso-bytes match, decode, save as PNG
4. (Optional) Same for WebP at iso-bytes

Also writes docs/assets/bench_data.json with per-fixture metrics so the
site's JavaScript can populate the gallery and benchmark table without
duplicating the numbers in HTML.

Usage:
    python scripts/generate_site_assets.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weft.api import encode_image  # noqa: E402
from weft.decoder import decode_to_array  # noqa: E402
from weft.image_io import linear_to_srgb  # noqa: E402
from weft.types import EncodeConfig  # noqa: E402


CORPUS = [
    "samples/inputs/hard-edges.png",
    "samples/inputs/synth-chart.png",
    "samples/inputs/synth-diagram.png",
    "samples/inputs/synth-icons.png",
    "samples/inputs/synth-mandelbrot.png",
    "samples/inputs/synth-noise-texture.png",
    "samples/inputs/synth-photo-landscape.png",
    "samples/inputs/synth-photo-natural.png",
    "samples/inputs/synth-pixel-sprite.png",
    "samples/inputs/synth-region-map.png",
    "samples/inputs/synth-shapes.png",
    "samples/inputs/synth-smooth-gradient.png",
    "samples/inputs/synth-terminal.png",
    "samples/inputs/synthetic-render-1024.png",
]

# Human-friendly labels and descriptions for the gallery cards.
FIXTURE_META = {
    "hard-edges": {
        "title": "Hard edges",
        "note": "Scanned-document style hard-edge content with text and geometric shapes. Palette-64 wins by capturing the discrete color structure.",
    },
    "synth-chart": {
        "title": "Line chart",
        "note": "Vector-rendered line chart with grid, axes, data points, labels. Palette captures the discrete colors; JPEG rings at every edge.",
    },
    "synth-diagram": {
        "title": "Flowchart diagram",
        "note": "Axis-aligned arrows and discrete-fill boxes — pure vector-art content. Palette-16 fits in 2.2 KB.",
    },
    "synth-icons": {
        "title": "Icon grid",
        "note": "4×4 grid of 64×64 icons with discrete colors and hard edges. Gradient field wins by capturing the large flat regions + sharp boundaries cheaply.",
    },
    "synth-mandelbrot": {
        "title": "Mandelbrot set",
        "note": "Mandelbrot fractal with cyclic color mapping. Palette-64 captures the color cycling; absolute PSNR is modest because the detailed boundary region is structurally hard.",
    },
    "synth-noise-texture": {
        "title": "Band-pass noise",
        "note": "Perlin-like band-pass-filtered Gaussian noise. This is JPEG's home court — DCT quantization is locally optimal for high-entropy frequency content. Weft loses by ~5 dB.",
    },
    "synth-photo-landscape": {
        "title": "Synthetic landscape",
        "note": "Gradient sky + gradient ground + sun + mountain. Mixed smooth/structural content; hybrid-dct-tight picks up the primitive layer with DCT on top.",
    },
    "synth-photo-natural": {
        "title": "Natural photo proxy",
        "note": "Multi-octave correlated noise approximating natural photo statistics. Weft matches JPEG within 0.5 dB here — close but not a win.",
    },
    "synth-pixel-sprite": {
        "title": "Pixel art sprite",
        "note": "Tiled 16×16 pixel-art sprite scaled up. Gradient field captures the discrete pixel boundaries in 3 KB.",
    },
    "synth-region-map": {
        "title": "Region map",
        "note": "Choropleth-style discrete-color region map. Palette-16 captures everything in 2.3 KB — JPEG at the same budget can't resolve the boundaries.",
    },
    "synth-shapes": {
        "title": "Geometric shapes",
        "note": "Solid-fill circles, rectangles, triangles with hard edges. Gradient field wins; JPEG rings every boundary.",
    },
    "synth-smooth-gradient": {
        "title": "Smooth gradient",
        "note": "Pure radial + linear gradients, no hard edges. Bicubic wins — a single 4×4 control grid per tile captures smooth content almost perfectly.",
    },
    "synth-terminal": {
        "title": "Terminal screen",
        "note": "Bitmap-font text on discrete palette colors, like a terminal session. Gradient field handles the hard glyph edges better than DCT ringing.",
    },
    "synthetic-render-1024": {
        "title": "Rendered scene",
        "note": "Synthetic rendered scene with mixed smooth and structural features. Hybrid-dct-tight wins narrowly; primitive layer + DCT residual cover both regimes.",
    },
}


OUT_DIR = "docs/assets/compare"
DATA_PATH = "docs/assets/bench_data.json"
WEFT_QUALITY = 75


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.99
    return 10.0 * math.log10(1.0 / mse)


def iso_bytes_jpeg(src_img: Image.Image, target_bytes: int) -> tuple[np.ndarray, int, int, float]:
    """Return (decoded array, bytes, quality, psnr) at the highest JPEG
    quality whose encoded size is ≤ target_bytes."""
    source_arr = np.asarray(src_img)
    best = None
    lo, hi = 1, 100
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = BytesIO()
        src_img.save(buf, "JPEG", quality=mid, optimize=True)
        b = buf.getvalue()
        if len(b) <= target_bytes:
            decoded = np.asarray(Image.open(BytesIO(b)).convert("RGB"))
            best = (decoded, len(b), mid, psnr(source_arr, decoded))
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        buf = BytesIO()
        src_img.save(buf, "JPEG", quality=1, optimize=True)
        b = buf.getvalue()
        decoded = np.asarray(Image.open(BytesIO(b)).convert("RGB"))
        return decoded, len(b), 1, psnr(source_arr, decoded)
    return best


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = "samples/runs/site-assets-tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    fixtures = []
    totals = {"weft": 0, "jpeg": 0, "png_lossless": 0}

    for src_path in CORPUS:
        if not os.path.exists(src_path):
            print(f"SKIP {src_path}")
            continue
        name = os.path.splitext(os.path.basename(src_path))[0]
        meta = FIXTURE_META.get(name, {"title": name, "note": ""})

        src_img = Image.open(src_path).convert("RGB")
        source_arr = np.asarray(src_img)
        h, w = source_arr.shape[:2]
        # Match the whitepaper bench: Pillow-reencoded PNG size is the
        # honest "PNG lossless baseline" number since on-disk source
        # files may have been produced with different tooling.
        _png_buf = BytesIO()
        src_img.save(_png_buf, "PNG", optimize=True)
        png_lossless_bytes = len(_png_buf.getvalue())

        # 1. Copy source
        source_out = os.path.join(OUT_DIR, f"{name}_source.png")
        shutil.copyfile(src_path, source_out)

        # 2. Weft encode + decode
        weft_path = os.path.join(tmp_dir, f"{name}.weft")
        cfg = EncodeConfig(
            quality=WEFT_QUALITY,
            feature_flags={"auto_select": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(src_path, weft_path, cfg)
        variant = rep.metadata.get("auto_selected_variant", "?")
        weft_bytes = rep.bytes_written

        weft_linear = decode_to_array(weft_path)
        weft_srgb = (np.clip(linear_to_srgb(weft_linear), 0.0, 1.0) * 255.0).astype(np.uint8)
        if weft_srgb.shape != source_arr.shape:
            # Defensive: encode_scale<1 would downsample; resize source to match
            # for consistent display and PSNR.
            src_img_match = src_img.resize(
                (weft_srgb.shape[1], weft_srgb.shape[0]), Image.LANCZOS
            )
            source_arr_match = np.asarray(src_img_match)
            weft_out = Image.fromarray(weft_srgb)
        else:
            source_arr_match = source_arr
            weft_out = Image.fromarray(weft_srgb)
        weft_psnr = psnr(source_arr_match, weft_srgb)
        weft_out.save(os.path.join(OUT_DIR, f"{name}_weft.png"))

        # 3. JPEG at iso-bytes
        jpeg_arr, jpeg_bytes, jpeg_q, jpeg_psnr = iso_bytes_jpeg(src_img, weft_bytes)
        Image.fromarray(jpeg_arr).save(os.path.join(OUT_DIR, f"{name}_jpeg.png"))

        fixture_data = {
            "name": name,
            "title": meta["title"],
            "note": meta["note"],
            "width": int(w),
            "height": int(h),
            "variant": variant,
            "source_bytes": int(png_lossless_bytes),
            "weft": {
                "bytes": int(weft_bytes),
                "psnr_db": round(weft_psnr, 2),
            },
            "jpeg_iso": {
                "bytes": int(jpeg_bytes),
                "psnr_db": round(jpeg_psnr, 2),
                "quality": int(jpeg_q),
            },
            "delta_db": round(weft_psnr - jpeg_psnr, 2),
            "assets": {
                "source": f"assets/compare/{name}_source.png",
                "weft": f"assets/compare/{name}_weft.png",
                "jpeg": f"assets/compare/{name}_jpeg.png",
            },
        }
        fixtures.append(fixture_data)

        totals["weft"] += weft_bytes
        totals["jpeg"] += jpeg_bytes
        totals["png_lossless"] += png_lossless_bytes

        print(
            f"{name:<26} {variant:<18} "
            f"W={weft_bytes/1024:>6.1f}K/{weft_psnr:5.2f}dB  "
            f"J={jpeg_bytes/1024:>6.1f}K/{jpeg_psnr:5.2f}dB  "
            f"Δ={weft_psnr - jpeg_psnr:+5.2f}dB"
        )

    # Aggregate summary
    n = len(fixtures)
    wins_jpeg = sum(1 for f in fixtures if f["delta_db"] > 0)
    avg_delta_jpeg = sum(f["delta_db"] for f in fixtures) / max(1, n)
    avg_weft_psnr = sum(f["weft"]["psnr_db"] for f in fixtures) / max(1, n)
    avg_jpeg_psnr = sum(f["jpeg_iso"]["psnr_db"] for f in fixtures) / max(1, n)

    data = {
        "weft_quality": WEFT_QUALITY,
        "fixture_count": n,
        "wins_jpeg": wins_jpeg,
        "avg_weft_psnr_db": round(avg_weft_psnr, 2),
        "avg_jpeg_psnr_iso_db": round(avg_jpeg_psnr, 2),
        "avg_delta_db_vs_jpeg": round(avg_delta_jpeg, 2),
        "totals_kb": {
            "weft": round(totals["weft"] / 1024, 1),
            "jpeg_iso": round(totals["jpeg"] / 1024, 1),
            "png_lossless": round(totals["png_lossless"] / 1024, 1),
        },
        "fixtures": fixtures,
    }

    os.makedirs("docs/assets", exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print()
    print(f"Summary: {wins_jpeg}/{n} iso-byte wins vs JPEG  "
          f"(avg +{avg_delta_jpeg:.2f} dB)")
    print(f"Total bytes: Weft={totals['weft']/1024:.1f}K  "
          f"JPEG={totals['jpeg']/1024:.1f}K  "
          f"PNG-lossless={totals['png_lossless']/1024:.1f}K")
    print(f"Assets: {OUT_DIR}/ ({n} fixtures × 3 variants)")
    print(f"Data: {DATA_PATH}")


if __name__ == "__main__":
    main()
