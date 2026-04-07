"""GPU encoder helpers for WEFT.

Stage-3 path computes per-tile model candidates on GPU:
- const patch
- multi-angle linear patches
- line templates
- triangle templates
- quadratic-curve templates

Uses cuda-python + NVRTC with RMM-backed allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Any

import numpy as np


class GpuEncodeError(RuntimeError):
    pass


@dataclass(slots=True)
class GpuTileFitResult:
    model_ids: np.ndarray
    color0: np.ndarray
    color1: np.ndarray
    model_mse: np.ndarray
    candidate_c0: np.ndarray
    candidate_c1: np.ndarray
    tile_cols: int
    tile_rows: int
    model_count: int
    kernel_ms: float
    backend: str


_KERNEL_SRC = r"""
#define MODEL_COUNT 18
#define PI_F 3.14159265358979323846f

__device__ inline float weft_absf(float x) { return x < 0.0f ? -x : x; }
__device__ inline float weft_minf(float a, float b) { return a < b ? a : b; }
__device__ inline float weft_maxf(float a, float b) { return a > b ? a : b; }

__device__ inline float weft_point_line_dist(float px, float py, float x0, float y0, float x1, float y1) {
    float vx = x1 - x0;
    float vy = y1 - y0;
    float den = vx * vx + vy * vy + 1e-8f;
    float t = ((px - x0) * vx + (py - y0) * vy) / den;
    t = weft_maxf(0.0f, weft_minf(1.0f, t));
    float cx = x0 + t * vx;
    float cy = y0 + t * vy;
    float dx = px - cx;
    float dy = py - cy;
    return sqrtf(dx * dx + dy * dy);
}

__device__ inline int weft_inside_tri(float px, float py, float x0, float y0, float x1, float y1, float x2, float y2) {
    float den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (weft_absf(den) < 1e-6f) return 0;
    float a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den;
    float b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den;
    float c = 1.0f - a - b;
    return (a >= 0.0f && b >= 0.0f && c >= 0.0f) ? 1 : 0;
}

__device__ inline float weft_curve_dist(
    float px, float py, float x0, float y0, float cx, float cy, float x1, float y1
) {
    float best = 1e9f;
    float px0 = x0;
    float py0 = y0;
    for (int i = 1; i <= 12; ++i) {
        float t = (float)i / 12.0f;
        float u = 1.0f - t;
        float qx = u * u * x0 + 2.0f * u * t * cx + t * t * x1;
        float qy = u * u * y0 + 2.0f * u * t * cy + t * t * y1;
        float d = weft_point_line_dist(px, py, px0, py0, qx, qy);
        best = weft_minf(best, d);
        px0 = qx;
        py0 = qy;
    }
    return best;
}

__device__ inline int weft_model_bucket(int model, int lx, int ly, int tile_size) {
    float x = (float)lx;
    float y = (float)ly;
    float ts = (float)(tile_size - 1);
    float cx = ts * 0.5f;
    float cy = ts * 0.5f;

    if (model == 0) return 0;

    // 1..8: angle-split linear models.
    if (model >= 1 && model <= 8) {
        float angle = ((float)(model - 1)) * (PI_F / 8.0f);
        float nx = cosf(angle);
        float ny = sinf(angle);
        float proj = (x - cx) * nx + (y - cy) * ny;
        return proj < 0.0f ? 0 : 1;
    }

    // 9..11: line templates
    if (model == 9) return (weft_absf(x - cx) <= 1.5f) ? 0 : 1;  // vertical
    if (model == 10) return (weft_absf(y - cy) <= 1.5f) ? 0 : 1; // horizontal
    if (model == 11) return (weft_absf(y - x) <= 1.5f) ? 0 : 1;  // main diag

    // 12..14: triangle templates
    if (model == 12) return weft_inside_tri(x, y, 0.0f, 0.0f, ts, 0.0f, 0.0f, ts) ? 0 : 1;
    if (model == 13) return weft_inside_tri(x, y, ts, 0.0f, ts, ts, 0.0f, 0.0f) ? 0 : 1;
    if (model == 14) return weft_inside_tri(x, y, 0.0f, ts, ts, ts, 0.0f, 0.0f) ? 0 : 1;

    // 15..17: quadratic-curve templates
    if (model == 15) {
        float d = weft_curve_dist(x, y, 0.0f, cy, cx, 0.0f, ts, cy);
        return d <= 1.25f ? 0 : 1;
    }
    if (model == 16) {
        float d = weft_curve_dist(x, y, 0.0f, cy, cx, ts, ts, cy);
        return d <= 1.25f ? 0 : 1;
    }
    float d = weft_curve_dist(x, y, cx, 0.0f, 0.0f, cy, cx, ts);
    return d <= 1.25f ? 0 : 1;
}

extern "C" __global__ void weft_tile_fit_candidates(
    const float* image,
    int width,
    int height,
    int tile_size,
    int tile_cols,
    int* out_model,
    float* out_c0,
    float* out_c1,
    float* out_mse,
    float* out_c0_all,
    float* out_c1_all
) {
    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    int tile_x = tile_idx % tile_cols;
    int tile_y = tile_idx / tile_cols;
    int x0 = tile_x * tile_size;
    int y0 = tile_y * tile_size;

    __shared__ float sum0[MODEL_COUNT][3];
    __shared__ float sum1[MODEL_COUNT][3];
    __shared__ float c0[MODEL_COUNT][3];
    __shared__ float c1[MODEL_COUNT][3];
    __shared__ int cnt0[MODEL_COUNT];
    __shared__ int cnt1[MODEL_COUNT];
    __shared__ float err[MODEL_COUNT];

    if (tid < MODEL_COUNT) {
        cnt0[tid] = 0;
        cnt1[tid] = 0;
        err[tid] = 0.0f;
        for (int ch = 0; ch < 3; ++ch) {
            sum0[tid][ch] = 0.0f;
            sum1[tid][ch] = 0.0f;
            c0[tid][ch] = 0.0f;
            c1[tid][ch] = 0.0f;
        }
    }
    __syncthreads();

    int n = tile_size * tile_size;
    for (int i = tid; i < n; i += blockDim.x) {
        int lx = i % tile_size;
        int ly = i / tile_size;
        int x = x0 + lx;
        int y = y0 + ly;
        if (x >= width) x = width - 1;
        if (y >= height) y = height - 1;
        int idx = (y * width + x) * 3;
        float r = image[idx + 0];
        float g = image[idx + 1];
        float b = image[idx + 2];

        for (int m = 0; m < MODEL_COUNT; ++m) {
            int bucket = weft_model_bucket(m, lx, ly, tile_size);
            if (bucket == 0) {
                atomicAdd(&cnt0[m], 1);
                atomicAdd(&sum0[m][0], r);
                atomicAdd(&sum0[m][1], g);
                atomicAdd(&sum0[m][2], b);
            } else {
                atomicAdd(&cnt1[m], 1);
                atomicAdd(&sum1[m][0], r);
                atomicAdd(&sum1[m][1], g);
                atomicAdd(&sum1[m][2], b);
            }
        }
    }
    __syncthreads();

    if (tid < MODEL_COUNT) {
        if (tid == 0) {
            float inv = 1.0f / (float)max(cnt0[0], 1);
            c0[0][0] = sum0[0][0] * inv;
            c0[0][1] = sum0[0][1] * inv;
            c0[0][2] = sum0[0][2] * inv;
            c1[0][0] = c0[0][0];
            c1[0][1] = c0[0][1];
            c1[0][2] = c0[0][2];
        } else {
            float inv0 = 1.0f / (float)max(cnt0[tid], 1);
            float inv1 = 1.0f / (float)max(cnt1[tid], 1);
            c0[tid][0] = sum0[tid][0] * inv0;
            c0[tid][1] = sum0[tid][1] * inv0;
            c0[tid][2] = sum0[tid][2] * inv0;
            c1[tid][0] = sum1[tid][0] * inv1;
            c1[tid][1] = sum1[tid][1] * inv1;
            c1[tid][2] = sum1[tid][2] * inv1;
        }
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        int lx = i % tile_size;
        int ly = i / tile_size;
        int x = x0 + lx;
        int y = y0 + ly;
        if (x >= width) x = width - 1;
        if (y >= height) y = height - 1;
        int idx = (y * width + x) * 3;
        float r = image[idx + 0];
        float g = image[idx + 1];
        float b = image[idx + 2];

        for (int m = 0; m < MODEL_COUNT; ++m) {
            int bucket = weft_model_bucket(m, lx, ly, tile_size);
            float pr = bucket == 0 ? c0[m][0] : c1[m][0];
            float pg = bucket == 0 ? c0[m][1] : c1[m][1];
            float pb = bucket == 0 ? c0[m][2] : c1[m][2];
            float dr = r - pr;
            float dg = g - pg;
            float db = b - pb;
            atomicAdd(&err[m], dr * dr + dg * dg + db * db);
        }
    }
    __syncthreads();

    if (tid == 0) {
        float invn = 1.0f / (float)n;
        int best_model = 0;
        float best_err = err[0] * invn;
        for (int m = 0; m < MODEL_COUNT; ++m) {
            float mse = err[m] * invn;
            out_mse[tile_idx * MODEL_COUNT + m] = mse;
            int moff = (tile_idx * MODEL_COUNT + m) * 3;
            out_c0_all[moff + 0] = c0[m][0];
            out_c0_all[moff + 1] = c0[m][1];
            out_c0_all[moff + 2] = c0[m][2];
            out_c1_all[moff + 0] = c1[m][0];
            out_c1_all[moff + 1] = c1[m][1];
            out_c1_all[moff + 2] = c1[m][2];
            if (mse < best_err) {
                best_err = mse;
                best_model = m;
            }
        }
        out_model[tile_idx] = best_model;
        int off = tile_idx * 3;
        out_c0[off + 0] = c0[best_model][0];
        out_c0[off + 1] = c0[best_model][1];
        out_c0[off + 2] = c0[best_model][2];
        out_c1[off + 0] = c1[best_model][0];
        out_c1[off + 1] = c1[best_model][1];
        out_c1[off + 2] = c1[best_model][2];
    }
}
"""


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return int(getattr(value, "value"))


def _import_cuda_bindings():
    try:
        from cuda.bindings import driver, nvrtc  # type: ignore[import-not-found]

        return driver, nvrtc, "cuda.bindings"
    except Exception:
        try:
            from cuda import cuda as driver  # type: ignore[import-not-found]
            from cuda import nvrtc  # type: ignore[import-not-found]

            return driver, nvrtc, "cuda.legacy"
        except Exception as exc:
            raise GpuEncodeError(f"cuda-python bindings unavailable: {exc}") from exc


def _unwrap_call(ret: Any) -> tuple[Any, tuple[Any, ...]]:
    if not isinstance(ret, tuple) or len(ret) == 0:
        raise GpuEncodeError("unexpected cuda-python call return shape")
    return ret[0], ret[1:]


def _driver_check(driver: Any, ret: Any):
    err, vals = _unwrap_call(ret)
    success = _as_int(getattr(driver.CUresult, "CUDA_SUCCESS", 0))
    if _as_int(err) != success:
        raise GpuEncodeError(f"CUDA driver error {_as_int(err)}")
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


def _nvrtc_check(nvrtc: Any, ret: Any):
    err, vals = _unwrap_call(ret)
    success = _as_int(getattr(nvrtc.nvrtcResult, "NVRTC_SUCCESS", 0))
    if _as_int(err) != success:
        raise GpuEncodeError(f"NVRTC error {_as_int(err)}")
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


_rmm_mr_set: bool = False


def _ensure_rmm_pool() -> None:
    """Idempotent: install a plain CudaMemoryResource the first time only.

    Previously created a ``PoolMemoryResource`` on every call, which both
    leaked pool instances (each call set a NEW pool as the current resource
    while the old one remained alive via dangling references) AND
    intermittently segfaulted during interpreter shutdown because the
    pool's C++ destructor races with CUDA teardown.
    """
    global _rmm_mr_set
    if _rmm_mr_set:
        return
    try:
        import rmm  # type: ignore[import-not-found]
        import rmm.mr  # type: ignore[import-not-found]
    except Exception as exc:
        raise GpuEncodeError(f"RMM is required for GPU encode: {exc}") from exc
    rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
    _rmm_mr_set = True


class _RmmBuffer:
    def __init__(self, nbytes: int) -> None:
        try:
            import rmm  # type: ignore[import-not-found]
        except Exception as exc:
            raise GpuEncodeError(f"RMM is required for GPU encode: {exc}") from exc
        _ensure_rmm_pool()
        self._buf = rmm.DeviceBuffer(size=nbytes)
        self.nbytes = nbytes

    @property
    def ptr(self) -> int:
        return int(self._buf.ptr)


@lru_cache(maxsize=1)
def _compile_ptx() -> tuple[bytes, str]:
    _driver, nvrtc, stack_name = _import_cuda_bindings()
    prog = None
    try:
        prog = _nvrtc_check(
            nvrtc,
            nvrtc.nvrtcCreateProgram(
                _KERNEL_SRC.encode("utf-8"),
                b"weft_tile_fit_candidates.cu",
                0,
                [],
                [],
            ),
        )
        options = [b"--std=c++14"]
        _nvrtc_check(nvrtc, nvrtc.nvrtcCompileProgram(prog, len(options), options))
        ptx_size = int(_nvrtc_check(nvrtc, nvrtc.nvrtcGetPTXSize(prog)))
        ptx_buf = bytearray(ptx_size)
        try:
            _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog, ptx_buf))
            ptx = bytes(ptx_buf)
        except TypeError:
            ptx = _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog))
            ptx = bytes(ptx)
        if ptx.endswith(b"\x00"):
            ptx = ptx[:-1]
        if not ptx:
            raise GpuEncodeError("empty PTX produced by NVRTC")
        return ptx, stack_name
    except Exception as exc:
        raise GpuEncodeError(f"failed to compile GPU tile-fit kernel: {exc}") from exc
    finally:
        if prog is not None:
            try:
                _nvrtc_check(nvrtc, nvrtc.nvrtcDestroyProgram(prog))
            except Exception:
                pass


def fit_tiles_gpu_constant(image: np.ndarray, tile_size: int, iterations: int = 1) -> GpuTileFitResult:
    if image.ndim != 3 or image.shape[2] != 3:
        raise GpuEncodeError("image must have shape (H, W, 3)")
    if int(iterations) < 1:
        raise GpuEncodeError("iterations must be >= 1")
    h, w, _ = image.shape
    if h <= 0 or w <= 0:
        raise GpuEncodeError("image dimensions must be > 0")
    img = np.asarray(image, dtype=np.float32, order="C")
    tile_cols = (w + tile_size - 1) // tile_size
    tile_rows = (h + tile_size - 1) // tile_size
    tile_count = tile_cols * tile_rows

    driver, _nvrtc, _stack = _import_cuda_bindings()
    ptx, stack_name = _compile_ptx()

    context = None
    stream = None
    module = None
    d_img = None
    d_model = None
    d_c0 = None
    d_c1 = None
    d_mse = None
    d_c0_all = None
    d_c1_all = None

    h_model = np.empty((tile_count,), dtype=np.int32)
    h_c0 = np.empty((tile_count, 3), dtype=np.float32)
    h_c1 = np.empty((tile_count, 3), dtype=np.float32)
    model_count = 18
    h_mse = np.empty((tile_count, model_count), dtype=np.float32)
    h_c0_all = np.empty((tile_count, model_count, 3), dtype=np.float32)
    h_c1_all = np.empty((tile_count, model_count, 3), dtype=np.float32)

    try:
        _driver_check(driver, driver.cuInit(0))
        cu_device = _driver_check(driver, driver.cuDeviceGet(0))
        context = _driver_check(driver, driver.cuCtxCreate(0, cu_device))
        stream = _driver_check(driver, driver.cuStreamCreate(0))
        try:
            module = _driver_check(driver, driver.cuModuleLoadData(ptx))
        except Exception:
            ptx_arr = np.frombuffer(ptx + b"\x00", dtype=np.uint8).copy()
            module = _driver_check(driver, driver.cuModuleLoadData(ptx_arr.ctypes.data))
        kernel = _driver_check(driver, driver.cuModuleGetFunction(module, b"weft_tile_fit_candidates"))

        d_img = _RmmBuffer(img.nbytes)
        d_model = _RmmBuffer(h_model.nbytes)
        d_c0 = _RmmBuffer(h_c0.nbytes)
        d_c1 = _RmmBuffer(h_c1.nbytes)
        d_mse = _RmmBuffer(h_mse.nbytes)
        d_c0_all = _RmmBuffer(h_c0_all.nbytes)
        d_c1_all = _RmmBuffer(h_c1_all.nbytes)
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_img.ptr, img.ctypes.data, img.nbytes, stream))

        arg_image = np.array([d_img.ptr], dtype=np.uint64)
        arg_w = np.array([w], dtype=np.int32)
        arg_h = np.array([h], dtype=np.int32)
        arg_tile_size = np.array([tile_size], dtype=np.int32)
        arg_tile_cols = np.array([tile_cols], dtype=np.int32)
        arg_model = np.array([d_model.ptr], dtype=np.uint64)
        arg_c0 = np.array([d_c0.ptr], dtype=np.uint64)
        arg_c1 = np.array([d_c1.ptr], dtype=np.uint64)
        arg_mse = np.array([d_mse.ptr], dtype=np.uint64)
        arg_c0_all = np.array([d_c0_all.ptr], dtype=np.uint64)
        arg_c1_all = np.array([d_c1_all.ptr], dtype=np.uint64)
        args = np.array(
            [
                arg_image.ctypes.data,
                arg_w.ctypes.data,
                arg_h.ctypes.data,
                arg_tile_size.ctypes.data,
                arg_tile_cols.ctypes.data,
                arg_model.ctypes.data,
                arg_c0.ctypes.data,
                arg_c1.ctypes.data,
                arg_mse.ctypes.data,
                arg_c0_all.ctypes.data,
                arg_c1_all.ctypes.data,
            ],
            dtype=np.uint64,
        )

        t0 = time.perf_counter()
        for _ in range(int(iterations)):
            _driver_check(
                driver,
                driver.cuLaunchKernel(
                    kernel,
                    tile_count,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    args.ctypes.data,
                    0,
                ),
            )
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_model.ctypes.data, d_model.ptr, h_model.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_c0.ctypes.data, d_c0.ptr, h_c0.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_c1.ctypes.data, d_c1.ptr, h_c1.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_mse.ctypes.data, d_mse.ptr, h_mse.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_c0_all.ctypes.data, d_c0_all.ptr, h_c0_all.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_c1_all.ctypes.data, d_c1_all.ptr, h_c1_all.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))
        kernel_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        raise GpuEncodeError(f"GPU tile fitting failed: {exc}") from exc
    finally:
        d_img = None
        d_model = None
        d_c0 = None
        d_c1 = None
        d_mse = None
        d_c0_all = None
        d_c1_all = None
        if stream is not None:
            try:
                _driver_check(driver, driver.cuStreamDestroy(stream))
            except Exception:
                pass
        if module is not None:
            try:
                _driver_check(driver, driver.cuModuleUnload(module))
            except Exception:
                pass
        if context is not None:
            try:
                _driver_check(driver, driver.cuCtxDestroy(context))
            except Exception:
                pass

    return GpuTileFitResult(
        model_ids=h_model.astype(np.int32),
        color0=np.clip(h_c0, 0.0, 1.0).astype(np.float32),
        color1=np.clip(h_c1, 0.0, 1.0).astype(np.float32),
        model_mse=h_mse.astype(np.float32),
        candidate_c0=np.clip(h_c0_all, 0.0, 1.0).astype(np.float32),
        candidate_c1=np.clip(h_c1_all, 0.0, 1.0).astype(np.float32),
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        model_count=model_count,
        kernel_ms=kernel_ms,
        backend=f"{stack_name}-nvrtc-rmm-stage3",
    )
