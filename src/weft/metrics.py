"""Image quality metrics."""

from __future__ import annotations

import math

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(d * d))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / m))


def ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    try:
        return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(a, b, multichannel=True, data_range=1.0))


def lpips_score(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        import torch
        import lpips
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)

    def _to_tensor(x: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        # LPIPS expects -1..1 sRGB-like range.
        t = t * 2.0 - 1.0
        return t.to(device)

    with torch.no_grad():
        score = loss_fn(_to_tensor(a), _to_tensor(b))
    return float(score.item())
