"""GPU scoring helpers for encoder candidate evaluation."""

from __future__ import annotations

import numpy as np


class GpuScoreError(RuntimeError):
    pass


_KERNEL_SRC = r'''
extern "C" __global__ void weft_mse_batch(
    const float* src,
    const float* preds,
    float* out,
    int n_candidates,
    int n_values
) {
    int cid = blockIdx.x;
    if (cid >= n_candidates) return;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_values; i += blockDim.x) {
        float d = src[i] - preds[cid * n_values + i];
        acc += d * d;
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[cid] = smem[0] / (float)n_values;
    }
}
'''


def batch_mse(src_tile: np.ndarray, pred_tiles: list[np.ndarray]) -> np.ndarray:
    """Compute MSE for candidate predictions.

    Uses GPU kernel when available, falls back to vectorized NumPy.
    """
    if not pred_tiles:
        return np.empty((0,), dtype=np.float32)

    # Vectorized CPU path: stack all predictions, compute MSE in one shot.
    src_flat = src_tile.ravel().astype(np.float32)
    preds = np.array([p.ravel() for p in pred_tiles], dtype=np.float32)
    diff = preds - src_flat[None, :]
    return np.mean(diff * diff, axis=1).astype(np.float32)
