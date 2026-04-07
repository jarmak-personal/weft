"""Generate synthetic test fixtures for WEFT encoder evaluation.

Each image targets a specific content regime so the corpus exercises
the encoder's basis-selection logic across the full spectrum:

* synth-smooth-gradient.png  — purely smooth radial+linear gradients
                                (bicubic-friendly regime)
* synth-shapes.png            — solid-fill geometric shapes with hard
                                outlines (palette / primitive-stack
                                regime; no antialiasing on edges)
* synth-mandelbrot.png        — Mandelbrot fractal with smooth color
                                cycling (mixed regime — large flat
                                interior + high-detail boundary)
* synth-noise-texture.png     — band-pass-filtered noise (Perlin-like)
                                — high-frequency texture, no edges
                                (stress test for any smooth basis)
* synth-chart.png             — line chart with axes / gridlines /
                                text (vector-graphics + text mix,
                                similar to hard-edges.png but more
                                quantitative)
* synth-photo-landscape.png   — synthetic landscape: gradient sky,
                                gradient ground, sun disk, mountain
                                triangle (mixed smooth + hard-edge
                                content typical of rendered scenes)
* synth-photo-natural.png     — multi-octave correlated noise with
                                soft color gradient and a few
                                structural features. Approximates
                                natural photo statistics (correlated
                                mid-frequency variation). The regime
                                where DCT-based codecs (JPEG/WebP)
                                beat primitive/palette/bicubic by
                                5-10 dB; useful as a stress test for
                                non-DCT encoders.

Run:
    python3 samples/inputs/_generate.py

The script is deterministic — every invocation produces byte-identical
output files, so the committed corpus is reproducible from source.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_HERE = Path(__file__).parent


# ── Helpers ────────────────────────────────────────────────────────────

def _save(arr: np.ndarray, name: str) -> None:
    out_path = _HERE / name
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(out_path)
    size = out_path.stat().st_size
    print(f"  wrote {name:<32} {arr.shape[1]}×{arr.shape[0]}  {size:>6,} bytes")


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load DejaVuSans at the requested size, falling back to PIL default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


# ── 1. Smooth gradient ────────────────────────────────────────────────

def make_smooth_gradient(h: int = 512, w: int = 512) -> np.ndarray:
    """Radial red + linear green/blue gradients. Pure smooth content."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(w, h)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = np.clip(1.0 - r * 1.5, 0.0, 1.0)
    img[..., 1] = xx / w
    img[..., 2] = yy / h
    # Add a subtle diagonal modulation so the gradient isn't perfectly
    # axis-aligned (more representative of natural smooth content).
    img[..., 1] += 0.15 * np.sin((xx + yy) * 0.02)
    img[..., 2] += 0.10 * np.cos((xx - yy) * 0.015)
    return np.clip(img * 255.0, 0.0, 255.0)


# ── 2. Geometric shapes ───────────────────────────────────────────────

def make_shapes(h: int = 512, w: int = 512) -> np.ndarray:
    """Solid-fill geometric shapes with hard edges and no antialiasing."""
    img = Image.new("RGB", (w, h), (245, 246, 250))
    draw = ImageDraw.Draw(img)

    # Background dividing line
    draw.line([(0, h * 3 // 4), (w, h * 3 // 4)], fill=(180, 180, 195), width=2)

    # Big red circle
    draw.ellipse([60, 60, 260, 260], fill=(220, 60, 60))
    # Green rectangle
    draw.rectangle([300, 80, 480, 220], fill=(60, 200, 80))
    # Blue triangle
    draw.polygon([(120, 320), (340, 320), (230, 460)],
                 fill=(60, 100, 220))
    # Yellow ring
    draw.ellipse([320, 300, 470, 450], fill=(220, 200, 60))
    draw.ellipse([350, 330, 440, 420], fill=(245, 246, 250))
    # Diagonal line
    draw.line([(20, 480), (490, 20)], fill=(40, 40, 50), width=4)
    # Small accent rectangles
    for i, c in enumerate([(180, 80, 200), (40, 180, 180), (240, 140, 60)]):
        draw.rectangle([20 + i * 40, 290, 50 + i * 40, 320], fill=c)

    return np.array(img)


# ── 3. Mandelbrot ──────────────────────────────────────────────────────

def make_mandelbrot(h: int = 512, w: int = 512, max_iter: int = 128) -> np.ndarray:
    """Mandelbrot set with cyclic color mapping. Vectorized escape-time."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = (xx / w) * 3.0 - 2.1
    cy = (yy / h) * 2.4 - 1.2
    c = cx + 1j * cy
    z = np.zeros_like(c)
    iters = np.full((h, w), max_iter, dtype=np.float64)

    # Vectorized escape: only iterate cells still bounded.
    active = np.ones((h, w), dtype=bool)
    for i in range(max_iter):
        z[active] = z[active] * z[active] + c[active]
        diverged = active & (np.abs(z) > 2.0)
        iters[diverged] = i
        active &= ~diverged
        if not active.any():
            break

    # Smooth iteration count via continuous escape formula.
    norm_iters = iters.copy()
    escaped = iters < max_iter
    norm_iters[escaped] = iters[escaped] + 1 - np.log(np.log(np.abs(z[escaped]) + 1e-9)) / np.log(2)
    norm_iters /= max_iter

    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 0] = np.sin(norm_iters * np.pi * 3.0 + 0.0) * 0.5 + 0.5
    img[..., 1] = np.sin(norm_iters * np.pi * 5.0 + 1.0) * 0.5 + 0.5
    img[..., 2] = np.sin(norm_iters * np.pi * 7.0 + 2.0) * 0.5 + 0.5
    img[~escaped] = 0.0  # interior is black
    return np.clip(img * 255.0, 0.0, 255.0)


# ── 4. Noise texture ───────────────────────────────────────────────────

def make_noise_texture(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    """FFT band-pass filtered Gaussian noise — Perlin-like natural texture.

    The band-pass keeps mid frequencies (peak at ~1/16 cycle/pixel) and
    rejects DC and high-frequency noise, producing smooth-but-organic
    blob patterns.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    F = np.fft.fft2(noise)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(ky * ky + kx * kx)
    # Bandpass centered at freq ≈ 0.04 (period ≈ 25 px)
    spectrum = np.exp(-((freq - 0.04) ** 2) / 0.0008)
    filtered = np.real(np.fft.ifft2(F * spectrum))
    # Normalize to [0, 1]
    fmin, fmax = filtered.min(), filtered.max()
    norm = (filtered - fmin) / (fmax - fmin)
    # Map to a warm color palette (orange-ish)
    img = np.stack([
        np.clip(norm * 1.10 + 0.05, 0.0, 1.0),
        np.clip(norm * 0.85 + 0.15, 0.0, 1.0),
        np.clip(norm * 0.55 + 0.20, 0.0, 1.0),
    ], axis=-1)
    return img * 255.0


# ── 5. Chart with text + grid ──────────────────────────────────────────

def make_chart(h: int = 480, w: int = 640) -> np.ndarray:
    """Line chart with grid, axes, data points, and labels.

    Vector-graphics-style content: solid background, hard-edged lines,
    rasterized text. Very different from photographic content.
    """
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_title = _load_font(18)
    font_label = _load_font(13)
    font_axis = _load_font(11)

    # Plot area
    px0, py0, px1, py1 = 80, 60, w - 30, h - 60

    # Grid
    for x in range(px0, px1 + 1, (px1 - px0) // 10):
        draw.line([(x, py0), (x, py1)], fill=(220, 222, 230), width=1)
    for y in range(py0, py1 + 1, (py1 - py0) // 6):
        draw.line([(px0, y), (px1, y)], fill=(220, 222, 230), width=1)

    # Axes (drawn last so they're on top of grid)
    draw.line([(px0, py1), (px1, py1)], fill=(40, 40, 50), width=2)
    draw.line([(px0, py0), (px0, py1)], fill=(40, 40, 50), width=2)

    # Data series — two damped sinusoids of different colors
    series = [
        ((220, 60, 60), 0.020, 0.005, 0.0),
        ((60, 100, 220), 0.015, 0.008, 1.5),
    ]
    for color, fa, fb, phase in series:
        pts = []
        for px in range(px0, px1 + 1, 4):
            t = (px - px0) / (px1 - px0)
            v = math.sin(t * 2 * math.pi * 2 + phase) * math.exp(-t * 1.5)
            py = int(py0 + (py1 - py0) * (0.5 - 0.4 * v))
            pts.append((px, py))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=2)
        # Marker dots every 8 samples
        for p in pts[::8]:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)

    # Labels
    draw.text((px0 + 80, 22), "WEFT encoder PSNR vs quality",
              fill=(20, 20, 30), font=font_title)
    draw.text((w // 2 - 30, h - 30), "quality →", fill=(40, 40, 50), font=font_label)
    draw.text((10, h // 2 - 8), "PSNR", fill=(40, 40, 50), font=font_label)
    # Y-axis tick labels
    for i, v in enumerate(("40", "30", "20", "10")):
        y = py0 + (py1 - py0) * (i + 1) // 4 - 6
        draw.text((50, y), v, fill=(60, 60, 70), font=font_axis)

    return np.array(img)


# ── 6. Synthetic photo (landscape) ─────────────────────────────────────

def make_photo_landscape(h: int = 512, w: int = 768) -> np.ndarray:
    """Synthetic landscape: smooth sky/ground gradients + sun + mountain.

    A controlled "photo-like" image with smooth gradient regions and a
    few hard-edged objects (sun disc, mountain triangle). Mixed regime
    where the per-tile hybrid encoder should decisively win.
    """
    img = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Sky gradient (top half) — blue to warm orange near horizon
    sky_t = np.clip(yy / (h * 0.55), 0.0, 1.0)
    img[..., 0] = 0.3 + sky_t * 0.7      # blue→orange
    img[..., 1] = 0.5 + sky_t * 0.4
    img[..., 2] = 0.95 - sky_t * 0.55

    # Ground (bottom half) — green to dark green
    ground_mask = yy >= h * 0.55
    gt = np.clip((yy[ground_mask] - h * 0.55) / (h * 0.45), 0.0, 1.0)
    img[ground_mask, 0] = 0.20 + (1 - gt) * 0.30
    img[ground_mask, 1] = 0.55 - gt * 0.25
    img[ground_mask, 2] = 0.15 + (1 - gt) * 0.20

    # Sun disc (hard-edged)
    sun_cx, sun_cy, sun_r = w * 0.72, h * 0.30, 38.0
    sun_dist = np.sqrt((xx - sun_cx) ** 2 + (yy - sun_cy) ** 2)
    sun_mask = sun_dist < sun_r
    img[sun_mask, 0] = 1.00
    img[sun_mask, 1] = 0.92
    img[sun_mask, 2] = 0.55
    # Soft halo (smooth blend)
    halo = np.exp(-((sun_dist - sun_r) ** 2) / 600.0) * (sun_dist >= sun_r)
    img[..., 0] = np.minimum(1.0, img[..., 0] + halo * 0.35)
    img[..., 1] = np.minimum(1.0, img[..., 1] + halo * 0.25)
    img[..., 2] = np.minimum(1.0, img[..., 2] + halo * 0.10)

    # Mountain (triangle, hard-edged)
    peak = (w * 0.30, h * 0.38)
    left = (w * 0.05, h * 0.58)
    right = (w * 0.55, h * 0.58)
    # Barycentric rasterization
    def _tri_mask(p0, p1, p2):
        v0x, v0y = p1[0] - p0[0], p1[1] - p0[1]
        v1x, v1y = p2[0] - p0[0], p2[1] - p0[1]
        v2x, v2y = xx - p0[0], yy - p0[1]
        d00 = v0x * v0x + v0y * v0y
        d01 = v0x * v1x + v0y * v1y
        d11 = v1x * v1x + v1y * v1y
        d20 = v2x * v0x + v2y * v0y
        d21 = v2x * v1x + v2y * v1y
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-9:
            return np.zeros((h, w), dtype=bool)
        v = (d11 * d20 - d01 * d21) / denom
        u = (d00 * d21 - d01 * d20) / denom
        return (v >= 0) & (u >= 0) & (v + u <= 1)

    mountain = _tri_mask(peak, left, right)
    # Mountain gradient: dark at peak → lighter at base
    mt = np.clip((yy - peak[1]) / (h * 0.20), 0.0, 1.0)
    img[mountain, 0] = 0.18 + mt[mountain] * 0.30
    img[mountain, 1] = 0.20 + mt[mountain] * 0.25
    img[mountain, 2] = 0.30 + mt[mountain] * 0.20
    # Snow cap (small triangle near peak)
    snow = _tri_mask(peak, (peak[0] - 25, peak[1] + 35), (peak[0] + 25, peak[1] + 35))
    img[snow, 0] = 0.95
    img[snow, 1] = 0.96
    img[snow, 2] = 0.98

    return np.clip(img * 255.0, 0.0, 255.0)


# ── 7. Natural photo (multi-octave correlated noise + structure) ──────

def _bandpass_noise(h: int, w: int, freq_peak: float, bandwidth: float, seed: int) -> np.ndarray:
    """FFT band-pass-filtered Gaussian noise normalized to zero mean / unit std."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))
    F = np.fft.fft2(noise)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(ky * ky + kx * kx)
    spectrum = np.exp(-((freq - freq_peak) ** 2) / bandwidth)
    filtered = np.real(np.fft.ifft2(F * spectrum))
    filtered -= filtered.mean()
    s = filtered.std()
    if s > 0:
        filtered /= s
    return filtered.astype(np.float32)


def make_photo_natural(h: int = 512, w: int = 768) -> np.ndarray:
    """Multi-octave correlated noise approximating natural photo statistics.

    Composes three octaves of band-pass-filtered noise at descending
    weights, adds a soft directional color gradient, and overlays a
    pair of soft structural features (a darker blob and a brighter
    rim) so the image isn't pure noise. The result has the
    correlated-mid-frequency-variation look of a real photo without
    needing copyrighted content.

    The regime: dense, mostly mid-frequency variation with no clean
    palette, no large flat regions, and no clean edges. JPEG/WebP win
    here by 5-10 dB on iso-bytes vs the current WEFT encoders — this
    fixture is the stress test for whatever DCT-class basis we add.
    """
    # Per-channel multi-octave noise (different seeds per channel/octave
    # so the channels aren't perfectly correlated).
    img = np.zeros((h, w, 3), dtype=np.float32)
    octaves = [
        # (peak_freq, bandwidth, weight)
        (0.020, 0.0008, 0.55),  # large-scale variation
        (0.060, 0.0015, 0.30),  # mid-scale texture
        (0.150, 0.0030, 0.15),  # fine texture
    ]
    for ch in range(3):
        for oct_idx, (fp, bw, wt) in enumerate(octaves):
            img[..., ch] += wt * _bandpass_noise(h, w, fp, bw, seed=11 + ch * 7 + oct_idx)

    # Bring into [0, 1] roughly via tanh + offset (more photo-like than
    # linear normalization, which would create clipping artifacts).
    img = 0.5 + 0.18 * np.tanh(img)

    # Soft directional color gradient (warm sunlight from upper-left).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = (xx / w) * 0.18 + (1.0 - yy / h) * 0.12
    img[..., 0] += light * 0.18      # red gain in lit region
    img[..., 1] += light * 0.12      # green
    img[..., 2] += light * 0.04      # very little blue
    # Cool shadow tint in the lower-right
    shadow = (1.0 - xx / w) * 0.15 + (yy / h) * 0.10
    img[..., 0] -= shadow * 0.06
    img[..., 2] += shadow * 0.10

    # Two structural features so the image isn't pure noise:
    # 1. A soft dark blob (foreground object)
    yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)
    blob_cy, blob_cx, blob_r = h * 0.62, w * 0.30, 75.0
    blob_dist = np.sqrt((xx_f - blob_cx) ** 2 + (yy_f - blob_cy) ** 2)
    blob = np.exp(-(blob_dist ** 2) / (2 * blob_r ** 2))
    img -= blob[..., None] * 0.18

    # 2. A bright rim along the top-right (sky / specular highlight)
    rim_dist = np.minimum(yy_f, w - xx_f) / 60.0
    rim = np.exp(-rim_dist) * (xx_f > w * 0.55) * (yy_f < h * 0.40)
    img += rim[..., None] * np.array([0.18, 0.14, 0.06], dtype=np.float32)

    return np.clip(img * 255.0, 0.0, 255.0)


# ── Main ───────────────────────────────────────────────────────────────

# ── 8. Pixel-art / icon set ────────────────────────────────────────────
#
# True discrete-color content with no antialiasing — every pixel is a
# palette color. Used to evaluate WEFT's "scale-independent rendering"
# story against PNG / WebP-lossless on content that the encoder's
# palette path can capture exactly. Drawing primitives are inlined
# below (not via PIL.ImageDraw, which antialiases by default).

_PA_PAL = {
    "bg":     (40, 44, 52),
    "panel":  (33, 37, 43),
    "border": (60, 64, 72),
    "text":   (220, 223, 228),
    "accent": (97, 175, 239),
    "ok":     (152, 195, 121),
    "warn":   (229, 192, 123),
    "err":    (224, 108, 117),
    "purple": (198, 120, 221),
    "cyan":   (86, 182, 194),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
}


def _pa_box(arr, x0, y0, x1, y1, color):
    arr[y0:y1, x0:x1] = color


def _pa_circle(arr, cx, cy, r, color):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    arr[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color


def _pa_ring(arr, cx, cy, r_out, r_in, color):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    arr[(d2 <= r_out * r_out) & (d2 > r_in * r_in)] = color


def make_icons(grid_w: int = 4, grid_h: int = 4, icon: int = 64, gap: int = 16) -> np.ndarray:
    """4×4 grid of 64×64 icons with discrete colors and hard edges."""
    pad = gap
    w = grid_w * icon + (grid_w + 1) * gap
    h = grid_h * icon + (grid_h + 1) * gap
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = _PA_PAL["panel"]
    arr[:2, :] = _PA_PAL["border"]
    arr[-2:, :] = _PA_PAL["border"]
    arr[:, :2] = _PA_PAL["border"]
    arr[:, -2:] = _PA_PAL["border"]

    def cell(col, row):
        x0 = pad + col * (icon + gap)
        y0 = pad + row * (icon + gap)
        return x0, y0, x0 + icon, y0 + icon

    for col, color in enumerate([_PA_PAL["accent"], _PA_PAL["ok"], _PA_PAL["warn"], _PA_PAL["err"]]):
        x0, y0, x1, y1 = cell(col, 0)
        _pa_box(arr, x0, y0, x1, y1, _PA_PAL["bg"])
        _pa_circle(arr, (x0 + x1) // 2, (y0 + y1) // 2, 24, color)
    for col, color in enumerate([_PA_PAL["purple"], _PA_PAL["cyan"], _PA_PAL["accent"], _PA_PAL["ok"]]):
        x0, y0, x1, y1 = cell(col, 1)
        _pa_box(arr, x0, y0, x1, y1, _PA_PAL["bg"])
        _pa_ring(arr, (x0 + x1) // 2, (y0 + y1) // 2, 26, 18, color)
    for col, color in enumerate([_PA_PAL["text"], _PA_PAL["warn"], _PA_PAL["err"], _PA_PAL["cyan"]]):
        x0, y0, x1, y1 = cell(col, 2)
        _pa_box(arr, x0, y0, x1, y1, _PA_PAL["bg"])
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        arr[cy - 4:cy + 4, x0 + 12:x1 - 12] = color
        arr[y0 + 12:y1 - 12, cx - 4:cx + 4] = color
    for col, color in enumerate([_PA_PAL["accent"], _PA_PAL["ok"], _PA_PAL["purple"], _PA_PAL["err"]]):
        x0, y0, x1, y1 = cell(col, 3)
        _pa_box(arr, x0, y0, x1, y1, _PA_PAL["bg"])
        for k in range(8):
            arr[y0 + 16 + k * 4:y0 + 20 + k * 4, x0 + 14 + k * 4:x0 + 22 + k * 4] = color
        for k in range(8):
            arr[y0 + 48 - k * 4:y0 + 52 - k * 4, x0 + 14 + k * 4:x0 + 22 + k * 4] = color
    return arr


def make_pixel_sprite(scale: int = 12, pad: int = 32) -> np.ndarray:
    """Tiled 16×16 pixel-art sprite, nearest-neighbor scaled — every
    output pixel is a discrete palette color."""
    SPRITE = [
        "   xxxxxxxx     ",
        "  xRRRRRRRRx    ",
        " xRRwwRRwwRRx   ",
        " xRwwwRwwwwRx   ",
        " xRRRRRRRRRRx   ",
        "  xxxKKKKxx     ",
        "    xKKKKx      ",
        "    xKKKKx      ",
        "    xK..Kx      ",
        "    xK..Kx      ",
        "    xKKKKx      ",
        "    xKKKKx      ",
        "    xxxxxx      ",
        "                ",
        "                ",
        "                ",
    ]
    palette = {
        "x": _PA_PAL["black"],
        "R": _PA_PAL["err"],
        "w": _PA_PAL["white"],
        "K": (180, 130, 80),
        ".": _PA_PAL["text"],
    }
    base_h, base_w = len(SPRITE), len(SPRITE[0])
    base = np.zeros((base_h, base_w, 3), dtype=np.uint8)
    bg = (45, 50, 60)
    base[:] = bg
    for y, row in enumerate(SPRITE):
        for x, ch in enumerate(row):
            c = palette.get(ch)
            if c is not None:
                base[y, x] = c
    big = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)
    gh, gw = 4, 4
    cell_h = base_h * scale
    cell_w = base_w * scale
    out_h = gh * cell_h + (gh + 1) * pad
    out_w = gw * cell_w + (gw + 1) * pad
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[:] = bg
    for r in range(gh):
        for c in range(gw):
            y0 = pad + r * (cell_h + pad)
            x0 = pad + c * (cell_w + pad)
            out[y0:y0 + cell_h, x0:x0 + cell_w] = big
    return out


def make_terminal(w: int = 720, h: int = 432) -> np.ndarray:
    """Simulated terminal screen with bitmap-font text composited from
    a 1-bit mask onto discrete palette colors."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = _PA_PAL["bg"]
    font = ImageFont.load_default()
    lines = [
        ("$ weft encode samples/icons.png out.weft", _PA_PAL["text"]),
        ("encoding...", _PA_PAL["warn"]),
        ("auto-select winner: palette-64", _PA_PAL["ok"]),
        ("bytes:   3.8 KB", _PA_PAL["text"]),
        ("psnr:    inf  (lossless on discrete colors)", _PA_PAL["text"]),
        ("", _PA_PAL["text"]),
        ("$ weft decode out.weft --width 4096 --height 4096", _PA_PAL["text"]),
        ("decoding at 8x source resolution...", _PA_PAL["warn"]),
        ("decode_backend: palette-cpu (nearest-neighbor)", _PA_PAL["ok"]),
        ("output: 4096x4096 PNG, 12 ms", _PA_PAL["text"]),
        ("", _PA_PAL["text"]),
        ("$ ls -la out.weft", _PA_PAL["text"]),
        ("-rw-r--r--  1 user  staff  3892 Apr  7 10:42 out.weft", _PA_PAL["text"]),
        ("$ identify -format '%w x %h\\n' decoded.png", _PA_PAL["text"]),
        ("4096 x 4096", _PA_PAL["accent"]),
    ]
    y = 16
    line_h = 22
    for text, color in lines:
        mask_img = Image.new("1", (w, h), 0)
        ImageDraw.Draw(mask_img).text((20, y), text, font=font, fill=1)
        arr[np.array(mask_img, dtype=bool)] = color
        y += line_h
    arr[:32] = _PA_PAL["panel"]
    for i, color in enumerate([_PA_PAL["err"], _PA_PAL["warn"], _PA_PAL["ok"]]):
        cx = 16 + i * 20
        cy = 16
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dx * dx + dy * dy <= 25:
                    arr[cy + dy, cx + dx] = color
    return arr


def make_region_map(w: int = 640, h: int = 480) -> np.ndarray:
    """Choropleth-style discrete-color region map."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = (220, 230, 240)
    region_colors = [
        (250, 220, 180), (190, 220, 180), (220, 180, 200),
        (180, 200, 220), (220, 200, 220), (200, 220, 200),
    ]
    rng = np.random.default_rng(7)
    rows, cols = 4, 5
    cell_h = h // rows
    cell_w = w // cols
    for r in range(rows):
        for c in range(cols):
            color = region_colors[(r * cols + c) % len(region_colors)]
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < cols - 1 else w
            jitter_x = int(rng.integers(-12, 13))
            jitter_y = int(rng.integers(-12, 13))
            yy0 = max(0, y0 + (jitter_y if r > 0 else 0))
            yy1 = min(h, y1)
            xx0 = max(0, x0 + (jitter_x if c > 0 else 0))
            xx1 = min(w, x1)
            arr[yy0:yy1, xx0:xx1] = color
    border = (80, 90, 100)
    for r in range(1, rows):
        arr[r * cell_h:r * cell_h + 1] = border
    for c in range(1, cols):
        arr[:, c * cell_w:c * cell_w + 1] = border
    for cx, cy in [(100, 90), (380, 200), (520, 150), (220, 350), (480, 380)]:
        arr[cy - 4:cy + 5, cx - 4:cx + 5] = (255, 255, 255)
        arr[cy - 3:cy + 4, cx - 3:cx + 4] = (40, 40, 50)
    return arr


def make_diagram(w: int = 720, h: int = 480) -> np.ndarray:
    """Vector flowchart diagram with axis-aligned arrows and discrete-fill boxes."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = (245, 247, 252)

    def draw_box(x0, y0, x1, y1, fill, border):
        arr[y0:y1, x0:x1] = border
        arr[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = fill

    def draw_arrow(x0, y0, x1, y1, color):
        if y0 == y1:
            xa, xb = sorted([x0, x1])
            arr[y0 - 1:y0 + 1, xa:xb] = color
            for k in range(6):
                arr[y0 - 1 - k:y0 + 1 + k, x1 - 6 + k:x1 - 5 + k] = color
        elif x0 == x1:
            ya, yb = sorted([y0, y1])
            arr[ya:yb, x0 - 1:x0 + 1] = color
            for k in range(6):
                arr[y1 - 6 + k:y1 - 5 + k, x0 - 1 - k:x0 + 1 + k] = color

    draw_box(40, 40, 200, 100, (200, 220, 240), (50, 80, 120))
    draw_box(280, 40, 440, 100, (210, 230, 200), (60, 100, 60))
    draw_box(520, 40, 680, 100, (240, 220, 200), (120, 80, 50))
    draw_box(40, 200, 200, 280, (220, 200, 230), (100, 60, 120))
    draw_box(280, 200, 440, 280, (200, 210, 230), (60, 70, 130))
    draw_box(520, 200, 680, 280, (230, 200, 200), (130, 60, 60))
    draw_box(160, 360, 560, 440, (240, 210, 180), (140, 90, 30))
    arrow_color = (40, 40, 50)
    draw_arrow(120, 100, 120, 200, arrow_color)
    draw_arrow(360, 100, 360, 200, arrow_color)
    draw_arrow(600, 100, 600, 200, arrow_color)
    draw_arrow(360, 280, 360, 360, arrow_color)
    draw_arrow(120, 280, 200, 360, arrow_color)
    draw_arrow(600, 280, 520, 360, arrow_color)
    return arr


def main() -> None:
    print("Generating synthetic test fixtures into samples/inputs/")
    _save(make_smooth_gradient(),    "synth-smooth-gradient.png")
    _save(make_shapes(),             "synth-shapes.png")
    _save(make_mandelbrot(),         "synth-mandelbrot.png")
    _save(make_noise_texture(),      "synth-noise-texture.png")
    _save(make_chart(),              "synth-chart.png")
    _save(make_photo_landscape(),    "synth-photo-landscape.png")
    _save(make_photo_natural(),      "synth-photo-natural.png")
    # Pixel-art / icon corpus (no antialiasing — every pixel is a discrete
    # palette color, used to evaluate scale-independent rendering).
    _save(make_icons(),              "synth-icons.png")
    _save(make_pixel_sprite(),       "synth-pixel-sprite.png")
    _save(make_terminal(),           "synth-terminal.png")
    _save(make_region_map(),         "synth-region-map.png")
    _save(make_diagram(),            "synth-diagram.png")
    print("Done.")


if __name__ == "__main__":
    main()
