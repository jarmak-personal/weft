"""Gradient-field encoder + Poisson-solve decoder — brainstorm idea #1.

Structurally distinct from every other encoder in the codebase: there
are no tiles, no primitives, no basis fitting, no palette. The image
is stored as the (∂I/∂x, ∂I/∂y) gradient field plus three per-channel
mean values. The decoder solves Poisson's equation ∇²I = div(grad)
via the discrete cosine transform (Neumann boundary conditions) — a
single closed-form linear solve in O(N log N) per channel.

Why bother:

* Hard-edge / flat-region content (UI screenshots, vector art, charts)
  produces a near-zero gradient field everywhere except at edges,
  which makes the int8 gradient maps crush under bitstream-level zstd.
  On synthetic hard-edge content the field is ~99% sparse and the
  effective compression ratio approaches palette-64.
* The bitstream is a *PDE source term*, not a picture or a function
  approximation. This is the only encoder in the codebase that decodes
  by solving a differential equation rather than evaluating a basis.

Why it can lose:

* Smooth gradients at int8 quantization clip to the noise floor.
  Each gradient pixel has resolution 1/scale, and a smooth ramp with
  per-pixel delta of 1/(image_dim) quantizes to 0/1 at scale=64,
  introducing stair-stepping artifacts at the Poisson solve.
* Storage cost is 6*H*W bytes pre-compression (3 channels × 2 axes ×
  int8) regardless of content, so high-frequency / dense-gradient
  content costs the same as flat content but reconstructs poorly.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


# ── Quantization defaults ──────────────────────────────────────────────

# scale=128 → each int8 unit represents 1/128 ≈ 0.0078 of the [0, 1]
# linear RGB range. Representable range: ±127/128 ≈ ±0.992 — clips on
# pure black-to-white step edges (gradient = 1.0) by ~1%, which is
# below the perceptual noise floor on quantized natural images. The
# higher scale (vs the original 64) gives 2× finer resolution on
# smooth content for the same byte cost. The structural limitation
# (smooth ramps with per-pixel delta < 1/128 quantize to zero) still
# applies — gradient-field is a hard-edge / sparse-gradient encoder,
# not a smooth-content encoder.
DEFAULT_SCALE = 128

# Soft deadzone for noise rejection: zero out gradient values smaller
# than this threshold (in [0, 1] linear units). 0.005 ≈ 1/200, just
# below the int8 quantization step at scale=64.
DEFAULT_THRESHOLD = 0.005


# ── Encoder ────────────────────────────────────────────────────────────

def encode(
    image: np.ndarray,
    *,
    scale: int = DEFAULT_SCALE,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-difference each channel, threshold, and quantize to int8.

    Returns ``(gx_q, gy_q, means)`` where each ``g?_q`` has shape
    ``(h, w, c)`` int8 and ``means`` is shape ``(c,)`` float32. The
    last column of ``gx_q`` and the last row of ``gy_q`` are zero
    (no forward difference past the image edge — Neumann no-flux BC).
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)
    h, w, c = image.shape
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, : w - 1, :] = image[:, 1:, :] - image[:, : w - 1, :]
    gy[: h - 1, :, :] = image[1:, :, :] - image[: h - 1, :, :]
    if threshold > 0.0:
        gx[np.abs(gx) < threshold] = 0.0
        gy[np.abs(gy) < threshold] = 0.0
    gx_q = np.clip(np.round(gx * scale), -127, 127).astype(np.int8)
    gy_q = np.clip(np.round(gy * scale), -127, 127).astype(np.int8)
    means = image.mean(axis=(0, 1)).astype(np.float32)
    return gx_q, gy_q, means


# ── Decoder (DCT-based Poisson solve) ──────────────────────────────────

def _laplacian_eigenvalues(h: int, w: int) -> np.ndarray:
    """DCT-II eigenvalues for the discrete 5-point Laplacian on (h, w).

    The discrete Laplacian under Neumann boundaries has eigenvalues
    ``2*cos(π*i/h) + 2*cos(π*j/w) - 4`` (negative-semidefinite).
    The DC component λ[0, 0] = 0 corresponds to the constant null
    space; we replace it with 1 to avoid division-by-zero and zero
    out the DC after the divide.
    """
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    eig = 2.0 * np.cos(np.pi * i / h) + 2.0 * np.cos(np.pi * j / w) - 4.0
    eig[0, 0] = 1.0
    return eig.astype(np.float64)


def decode(
    gx_q: np.ndarray,
    gy_q: np.ndarray,
    means: np.ndarray,
    *,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Reconstruct the image from quantized gradients via DCT Poisson solve.

    Returns ``(h, w, c)`` float32 in [0, 1].
    """
    h, w, c = gx_q.shape
    gx = gx_q.astype(np.float32) / float(scale)
    gy = gy_q.astype(np.float32) / float(scale)

    # Discrete divergence (backward difference of forward gradients):
    # div[i, j] = gx[i, j] - gx[i, j-1] + gy[i, j] - gy[i-1, j]
    # with the implicit gx[i, -1] = gy[-1, j] = 0 (Neumann BC).
    div = gx.copy()
    div[:, 1:, :] -= gx[:, : w - 1, :]
    div += gy
    div[1:, :, :] -= gy[: h - 1, :, :]

    eig = _laplacian_eigenvalues(h, w)
    out = np.empty_like(div)
    for ch in range(c):
        D = dctn(div[..., ch].astype(np.float64), type=2, norm="ortho")
        D /= eig
        D[0, 0] = 0.0  # DC handled by the per-channel mean offset
        recon = idctn(D, type=2, norm="ortho")
        out[..., ch] = (recon + float(means[ch])).astype(np.float32)
    return np.clip(out, 0.0, 1.0)
