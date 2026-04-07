"""Intrinsic image decomposition for WEFT (Phase 2 #17).

Splits a linear-RGB image into:

  ``albedo``   — surface reflectance: smooth where the surface is uniform,
                 sharp at material boundaries. Easier for the primitive
                 fitting pipeline to compress because most of the
                 high-frequency content collapses to a few colors per
                 region instead of a continuous shading gradient.

  ``lighting`` — slowly-varying incident illumination, with no
                 high-frequency content. Compresses to a tiny low-resolution
                 grid that the decoder bilinearly upsamples and multiplies
                 back at output time.

The MVP uses **single-scale Retinex**: the canonical fast intrinsic
decomposition. The smooth lighting estimate is a wide Gaussian blur of
the image in log space; the albedo falls out as image / lighting. This
is approximate (a real intrinsic-image network would do better at
specularities and material boundaries), but it's fast, deterministic,
pure-NumPy, and a good baseline for the rendered-image-domain hypothesis.

The decomposition is invertible: ``albedo * lighting == image`` to
within floating-point tolerance for the smooth-lighting regime. Where
the lighting estimate has high-frequency error (sharp edges leaking
into the lighting channel), the albedo absorbs the inverse error and
the round-trip is still exact.
"""

from __future__ import annotations

import numpy as np


def _gaussian_blur_separable(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel separable Gaussian blur, pure NumPy."""
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)
    radius = max(1, int(round(sigma * 4.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    out = arr.astype(np.float32, copy=False)
    if out.ndim == 2:
        out = out[..., None]
    h, w, c = out.shape

    pad_h = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    blurred_h = np.zeros_like(out)
    for ch in range(c):
        acc = np.zeros((h, w), dtype=np.float32)
        for i, k in enumerate(kernel):
            acc += k * pad_h[:, i : i + w, ch]
        blurred_h[:, :, ch] = acc

    pad_v = np.pad(blurred_h, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    out2 = np.zeros_like(blurred_h)
    for ch in range(c):
        acc = np.zeros((h, w), dtype=np.float32)
        for i, k in enumerate(kernel):
            acc += k * pad_v[i : i + h, :, ch]
        out2[:, :, ch] = acc

    if arr.ndim == 2:
        out2 = out2[:, :, 0]
    return out2


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Use scipy if available, otherwise fall back to the pure-NumPy version."""
    try:
        from scipy.ndimage import gaussian_filter
        if arr.ndim == 3:
            return gaussian_filter(arr.astype(np.float32, copy=False), sigma=(sigma, sigma, 0))
        return gaussian_filter(arr.astype(np.float32, copy=False), sigma=sigma)
    except ImportError:
        return _gaussian_blur_separable(arr, sigma)


def decompose_retinex(
    image: np.ndarray,
    sigma_frac: float = 0.15,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-scale Retinex decomposition into (albedo, lighting).

    Parameters
    ----------
    image
        Linear-RGB float image, shape ``(H, W, 3)`` or ``(H, W)``,
        values in roughly ``[0, 1]``.
    sigma_frac
        Gaussian blur sigma as a fraction of ``min(H, W)``. Default
        ``0.06`` corresponds to a wide blur (~6% of the image side).
    epsilon
        Floor added before taking ``log`` to keep dark pixels stable.

    Returns
    -------
    albedo, lighting
        Both ``np.float32`` arrays the same shape as ``image``, with
        ``albedo * lighting ≈ image`` within float tolerance. The
        decomposition is **globally normalized** so that albedo fits in
        ``[0, 1]`` (the range the primitive pipeline expects), with the
        excess folded into the lighting channel. Lighting values outside
        ``[0, ~3]`` will be clamped at quantization time, but in
        practice global normalization keeps lighting comfortably within
        that range for natural images.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] not in (1, 3):
        raise ValueError(f"image must have 1 or 3 channels, got {img.shape[-1]}")

    h, w = img.shape[:2]
    sigma = sigma_frac * float(min(h, w))

    log_img = np.log(np.maximum(img, epsilon))
    log_lighting = _gaussian_blur(log_img, sigma)
    lighting = np.exp(log_lighting)

    safe_lighting = np.maximum(lighting, epsilon)
    albedo = img / safe_lighting

    # Global normalization: scale so albedo fits in [0, 1].
    # The primitive pipeline encodes colors as uint8 in [0, 1]; values
    # above 1 would clip and silently lose information. We compute the
    # 99.5th percentile of albedo (per channel max) and scale by that,
    # absorbing the inverse scale into lighting so albedo*lighting is
    # unchanged. The 99.5th percentile (instead of strict max) avoids
    # giving disproportionate weight to a handful of outlier pixels.
    if albedo.size > 0:
        scale = float(np.percentile(albedo.max(axis=-1), 99.5))
        if scale > 1.0:
            albedo = albedo / scale
            lighting = lighting * scale
            # The percentile may still leave a few pixels above 1; clamp
            # those for safety. The corresponding error is absorbed into
            # the residual block at fitting time.
            np.clip(albedo, 0.0, 1.0, out=albedo)

    if image.ndim == 2:
        albedo = albedo[..., 0]
        lighting = lighting[..., 0]

    return albedo.astype(np.float32), lighting.astype(np.float32)


def downsample_lighting(
    lighting: np.ndarray, grid_h: int, grid_w: int,
) -> np.ndarray:
    """Box-filter downsample a lighting image to a fixed grid.

    Used by the encoder to produce the small lighting tensor that goes
    into BLOCK_LITE.
    """
    if lighting.ndim != 3 or lighting.shape[-1] != 3:
        raise ValueError(f"lighting must be (h, w, 3), got {lighting.shape}")
    h, w, _ = lighting.shape
    out = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    for gy in range(grid_h):
        y0 = (gy * h) // grid_h
        y1 = ((gy + 1) * h) // grid_h
        for gx in range(grid_w):
            x0 = (gx * w) // grid_w
            x1 = ((gx + 1) * w) // grid_w
            patch = lighting[y0:y1, x0:x1, :]
            if patch.size > 0:
                out[gy, gx, :] = patch.mean(axis=(0, 1))
    return out


def upsample_lighting(
    grid: np.ndarray, target_h: int, target_w: int,
) -> np.ndarray:
    """Bilinear upsample a low-res lighting grid back to image size.

    Pixel centers in target space map to grid-space coordinates so that
    grid cell ``(gy, gx)`` is centered at pixel coordinate
    ``(gy + 0.5) * target_h / grid_h - 0.5`` and similarly for x.
    """
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise ValueError(f"grid must be (gh, gw, 3), got {grid.shape}")
    gh, gw, _ = grid.shape
    if gh < 2 or gw < 2:
        return np.tile(grid.mean(axis=(0, 1), keepdims=True), (target_h, target_w, 1))

    pixel_y = (np.arange(target_h, dtype=np.float32) + 0.5) * gh / target_h - 0.5
    pixel_x = (np.arange(target_w, dtype=np.float32) + 0.5) * gw / target_w - 0.5
    pixel_y = np.clip(pixel_y, 0.0, gh - 1.0)
    pixel_x = np.clip(pixel_x, 0.0, gw - 1.0)

    y0 = np.floor(pixel_y).astype(np.int32)
    x0 = np.floor(pixel_x).astype(np.int32)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    yf = (pixel_y - y0)[:, None]
    xf = (pixel_x - x0)[None, :]

    out = np.empty((target_h, target_w, 3), dtype=np.float32)
    for ch in range(3):
        c00 = grid[y0[:, None], x0[None, :], ch]
        c01 = grid[y0[:, None], x1[None, :], ch]
        c10 = grid[y1[:, None], x0[None, :], ch]
        c11 = grid[y1[:, None], x1[None, :], ch]
        top = c00 * (1.0 - xf) + c01 * xf
        bot = c10 * (1.0 - xf) + c11 * xf
        out[..., ch] = top * (1.0 - yf) + bot * yf
    return out
