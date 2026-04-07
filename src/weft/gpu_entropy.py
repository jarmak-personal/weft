"""GPU chunk entropy decode for WEFT PRIM payloads.

Pure cuda-python (NVRTC + Driver API launch) with RMM-backed allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .bitstream import PrimChunkEntry


class GpuEntropyError(RuntimeError):
    pass


@dataclass(slots=True)
class GpuEntropyResult:
    raw: bytes
    chunk_count: int
    backend: str


_rmm_mr_set: bool = False


class _RmmBuffer:
    @staticmethod
    def _ensure_rmm() -> None:
        """Idempotent: install a plain CudaMemoryResource the first time only.

        Using a plain ``CudaMemoryResource`` (not a ``PoolMemoryResource``)
        avoids an intermittent segfault on interpreter shutdown: the pool's
        C++ destructor calls ``cuEventSynchronize`` on pending events and
        races with CUDA driver teardown. The pool's only performance
        advantage is amortizing allocation latency across many small
        requests; the decode path allocates a bounded number of buffers
        per call, so the pool is not worth the crash surface.
        """
        global _rmm_mr_set
        if _rmm_mr_set:
            return
        import rmm.mr  # type: ignore[import-not-found]
        rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
        _rmm_mr_set = True

    def __init__(self, nbytes: int) -> None:
        try:
            import rmm  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            raise GpuEntropyError(f"RMM is required for GPU entropy decode: {exc}") from exc
        self._ensure_rmm()
        self._buf = rmm.DeviceBuffer(size=nbytes)
        self.nbytes = nbytes

    @property
    def ptr(self) -> int:
        return int(self._buf.ptr)


_KERNEL_SRC = r'''
extern "C" {
#define MODE_RAW 0
#define MODE_RANS 1
#define PROB_BITS 12
#define TOTFREQ (1 << PROB_BITS)
#define RANS_L (1 << 23)

typedef struct {
    int start_tile;
    int tile_count;
    int raw_offset;
    int raw_length;
    int coded_offset;
    int coded_length;
} ChunkDesc;

__device__ __forceinline__ unsigned int rd_u32(const unsigned char* p) {
    return ((unsigned int)p[0]) | (((unsigned int)p[1]) << 8) | (((unsigned int)p[2]) << 16) | (((unsigned int)p[3]) << 24);
}

__device__ __forceinline__ unsigned short rd_u16(const unsigned char* p) {
    return (unsigned short)(p[0] | (p[1] << 8));
}

__global__ void weft_decode_chunks(
    const unsigned char* coded,
    const ChunkDesc* descs,
    int n_chunks,
    unsigned char* out_raw,
    int* status
) {
    int cid = (int)blockIdx.x;
    if (cid >= n_chunks) return;
    if (threadIdx.x != 0) return;

    ChunkDesc d = descs[cid];
    const unsigned char* ptr = coded + d.coded_offset;

    if (d.coded_length <= 0 || d.raw_length < 0) {
        status[cid] = 1;
        return;
    }

    int mode = (int)ptr[0];
    if (mode == MODE_RAW) {
        if (d.coded_length < 5) {
            status[cid] = 2;
            return;
        }
        unsigned int raw_len = rd_u32(ptr + 1);
        if ((int)raw_len != d.raw_length || ((int)raw_len + 5) != d.coded_length) {
            status[cid] = 3;
            return;
        }
        for (int i = 0; i < d.raw_length; ++i) {
            out_raw[d.raw_offset + i] = ptr[5 + i];
        }
        status[cid] = 0;
        return;
    }

    if (mode != MODE_RANS) {
        status[cid] = 4;
        return;
    }

    if (d.coded_length < 7) {
        status[cid] = 5;
        return;
    }

    unsigned int original_size = rd_u32(ptr + 1);
    if ((int)original_size != d.raw_length) {
        status[cid] = 6;
        return;
    }
    unsigned int n_syms = rd_u16(ptr + 5);
    int off = 7;

    int need = off + ((int)n_syms) * 3 + 8;
    if (need > d.coded_length) {
        status[cid] = 7;
        return;
    }

    int freqs[256];
    int cum[257];
    for (int i = 0; i < 256; ++i) freqs[i] = 0;

    for (unsigned int i = 0; i < n_syms; ++i) {
        int sym = (int)ptr[off];
        int freq = (int)rd_u16(ptr + off + 1);
        off += 3;
        if (sym < 0 || sym > 255 || freq <= 0) {
            status[cid] = 8;
            return;
        }
        freqs[sym] = freq;
    }

    int running = 0;
    for (int i = 0; i < 256; ++i) {
        cum[i] = running;
        running += freqs[i];
    }
    cum[256] = running;
    if (running != TOTFREQ) {
        status[cid] = 9;
        return;
    }

    unsigned int state = rd_u32(ptr + off);
    off += 4;
    unsigned int emitted_len = rd_u32(ptr + off);
    off += 4;
    if (off + (int)emitted_len != d.coded_length) {
        status[cid] = 10;
        return;
    }

    const unsigned char* emitted = ptr + off;
    int eptr = (int)emitted_len - 1;

    for (int i = 0; i < d.raw_length; ++i) {
        int slot = (int)(state & (TOTFREQ - 1));
        int sym = 0;
        while (sym < 256 && !(slot >= cum[sym] && slot < cum[sym + 1])) {
            sym += 1;
        }
        if (sym >= 256) {
            status[cid] = 11;
            return;
        }

        out_raw[d.raw_offset + i] = (unsigned char)sym;
        int freq = freqs[sym];
        state = (unsigned int)(freq * (int)(state >> PROB_BITS) + (slot - cum[sym]));

        while (state < RANS_L && eptr >= 0) {
            state = (state << 8) | (unsigned int)emitted[eptr];
            eptr -= 1;
        }
    }

    status[cid] = 0;
}
}
'''


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
            raise GpuEntropyError(f"cuda-python bindings unavailable: {exc}") from exc


def _driver_error_string(driver: Any, err: Any) -> str:
    try:
        ret = driver.cuGetErrorString(err)
        if isinstance(ret, tuple) and len(ret) >= 2:
            msg = ret[1]
            if isinstance(msg, (bytes, bytearray)):
                return msg.decode("utf-8", errors="ignore")
            return str(msg)
    except Exception:
        pass
    return f"CUresult={_as_int(err)}"


def _nvrtc_error_string(nvrtc: Any, err: Any) -> str:
    try:
        ret = nvrtc.nvrtcGetErrorString(err)
        if isinstance(ret, tuple) and len(ret) >= 2:
            msg = ret[1]
            if isinstance(msg, (bytes, bytearray)):
                return msg.decode("utf-8", errors="ignore")
            return str(msg)
        if isinstance(ret, (bytes, bytearray)):
            return ret.decode("utf-8", errors="ignore")
        return str(ret)
    except Exception:
        return f"NVRTC={_as_int(err)}"


def _unwrap_call(ret: Any) -> tuple[Any, tuple[Any, ...]]:
    if not isinstance(ret, tuple) or len(ret) == 0:
        raise GpuEntropyError("unexpected cuda-python call return shape")
    return ret[0], ret[1:]


def _driver_check(driver: Any, ret: Any):
    err, vals = _unwrap_call(ret)
    success = _as_int(getattr(driver.CUresult, "CUDA_SUCCESS", 0))
    if _as_int(err) != success:
        raise GpuEntropyError(f"Driver API error: {_driver_error_string(driver, err)}")
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


def _nvrtc_check(nvrtc: Any, ret: Any):
    err, vals = _unwrap_call(ret)
    success = _as_int(getattr(nvrtc.nvrtcResult, "NVRTC_SUCCESS", 0))
    if _as_int(err) != success:
        raise GpuEntropyError(f"NVRTC error: {_nvrtc_error_string(nvrtc, err)}")
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


def _nvrtc_log(nvrtc: Any, prog: Any) -> str:
    try:
        log_size = _nvrtc_check(nvrtc, nvrtc.nvrtcGetProgramLogSize(prog))
        if not log_size:
            return ""
        buf = bytearray(int(log_size))
        try:
            _nvrtc_check(nvrtc, nvrtc.nvrtcGetProgramLog(prog, buf))
            return bytes(buf).rstrip(b"\x00").decode("utf-8", errors="ignore")
        except TypeError:
            txt = _nvrtc_check(nvrtc, nvrtc.nvrtcGetProgramLog(prog))
            if isinstance(txt, (bytes, bytearray)):
                return bytes(txt).decode("utf-8", errors="ignore")
            return str(txt)
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _compile_ptx_cuda_python() -> tuple[bytes, str]:
    driver, nvrtc, stack_name = _import_cuda_bindings()
    prog = None
    try:
        prog = _nvrtc_check(
            nvrtc,
            nvrtc.nvrtcCreateProgram(
                _KERNEL_SRC.encode("utf-8"),
                b"weft_decode_chunks.cu",
                0,
                [],
                [],
            ),
        )

        options = [b"--std=c++14"]
        try:
            _nvrtc_check(nvrtc, nvrtc.nvrtcCompileProgram(prog, len(options), options))
        except GpuEntropyError as exc:
            log = _nvrtc_log(nvrtc, prog)
            if log:
                raise GpuEntropyError(f"NVRTC compile failed: {exc}; log: {log}") from exc
            raise

        try:
            ptx_size = int(_nvrtc_check(nvrtc, nvrtc.nvrtcGetPTXSize(prog)))
            ptx_buf = bytearray(ptx_size)
            try:
                _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog, ptx_buf))
                ptx = bytes(ptx_buf)
            except TypeError:
                ptx = _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog))
                if not isinstance(ptx, (bytes, bytearray)):
                    ptx = bytes(str(ptx), "utf-8")
                else:
                    ptx = bytes(ptx)
        except Exception as exc:
            raise GpuEntropyError(f"failed to retrieve NVRTC PTX: {exc}") from exc

        if ptx.endswith(b"\x00"):
            ptx = ptx[:-1]
        if not ptx:
            raise GpuEntropyError("empty PTX produced by NVRTC")
        return ptx, stack_name
    finally:
        if prog is not None:
            try:
                _nvrtc_check(nvrtc, nvrtc.nvrtcDestroyProgram(prog))
            except Exception:
                pass


def _decode_with_cuda_python(
    prim_payload: bytes,
    toc: list[int],
    chunk_index: list[PrimChunkEntry],
) -> GpuEntropyResult:
    driver, _nvrtc, _stack_name = _import_cuda_bindings()
    ptx, stack_name = _compile_ptx_cuda_python()

    raw_total = toc[-1]
    n_chunks = len(chunk_index)

    if n_chunks <= 0:
        return GpuEntropyResult(raw=b"", chunk_count=0, backend=f"{stack_name}-driver-nvrtc")

    desc = np.zeros((n_chunks, 6), dtype=np.int32)
    for i, c in enumerate(chunk_index):
        desc[i, 0] = int(c.start_tile)
        desc[i, 1] = int(c.tile_count)
        desc[i, 2] = int(c.raw_offset)
        desc[i, 3] = int(c.raw_length)
        desc[i, 4] = int(c.coded_offset)
        desc[i, 5] = int(c.coded_length)

    h_coded = np.frombuffer(prim_payload, dtype=np.uint8)
    h_raw = np.empty((raw_total,), dtype=np.uint8)
    h_status = np.empty((n_chunks,), dtype=np.int32)

    context = None
    stream = None
    module = None

    d_coded = None
    d_desc = None
    d_raw = None
    d_status = None

    try:
        _driver_check(driver, driver.cuInit(0))
        cu_device = _driver_check(driver, driver.cuDeviceGet(0))
        context = _driver_check(driver, driver.cuCtxCreate(0, cu_device))
        stream = _driver_check(driver, driver.cuStreamCreate(0))

        try:
            module = _driver_check(driver, driver.cuModuleLoadData(ptx))
        except Exception:
            # Some bindings/toolkits expect a pointer to null-terminated PTX.
            ptx_arr = np.char.array(ptx + b"\x00")
            module = _driver_check(driver, driver.cuModuleLoadData(ptx_arr.ctypes.data))

        kernel = _driver_check(driver, driver.cuModuleGetFunction(module, b"weft_decode_chunks"))

        d_coded = _RmmBuffer(h_coded.nbytes)
        d_desc = _RmmBuffer(desc.nbytes)
        d_raw = _RmmBuffer(h_raw.nbytes)
        d_status = _RmmBuffer(h_status.nbytes)

        if h_coded.nbytes > 0:
            _driver_check(driver, driver.cuMemcpyHtoDAsync(d_coded.ptr, h_coded.ctypes.data, h_coded.nbytes, stream))
        if desc.nbytes > 0:
            _driver_check(driver, driver.cuMemcpyHtoDAsync(d_desc.ptr, desc.ctypes.data, desc.nbytes, stream))

        arg_coded = np.array([d_coded.ptr], dtype=np.uint64)
        arg_desc = np.array([d_desc.ptr], dtype=np.uint64)
        arg_nchunks = np.array([n_chunks], dtype=np.int32)
        arg_raw = np.array([d_raw.ptr], dtype=np.uint64)
        arg_status = np.array([d_status.ptr], dtype=np.uint64)

        args = np.array(
            [
                arg_coded.ctypes.data,
                arg_desc.ctypes.data,
                arg_nchunks.ctypes.data,
                arg_raw.ctypes.data,
                arg_status.ctypes.data,
            ],
            dtype=np.uint64,
        )

        _driver_check(
            driver,
            driver.cuLaunchKernel(
                kernel,
                n_chunks,
                1,
                1,
                1,
                1,
                1,
                0,
                stream,
                args.ctypes.data,
                0,
            ),
        )

        if h_status.nbytes > 0:
            _driver_check(driver, driver.cuMemcpyDtoHAsync(h_status.ctypes.data, d_status.ptr, h_status.nbytes, stream))
        if h_raw.nbytes > 0:
            _driver_check(driver, driver.cuMemcpyDtoHAsync(h_raw.ctypes.data, d_raw.ptr, h_raw.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        if np.any(h_status != 0):
            first = int(np.where(h_status != 0)[0][0])
            code = int(h_status[first])
            raise GpuEntropyError(f"GPU entropy decode failed on chunk {first} with status {code}")

        return GpuEntropyResult(raw=h_raw.tobytes(), chunk_count=n_chunks, backend=f"{stack_name}-driver-nvrtc")
    finally:
        d_coded = None
        d_desc = None
        d_raw = None
        d_status = None
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


def decode_prim_payload_chunked_gpu(
    prim_payload: bytes,
    toc: list[int],
    chunk_index: list[PrimChunkEntry],
) -> GpuEntropyResult:
    if not chunk_index:
        raise GpuEntropyError("chunk index required for GPU chunk decode")
    if not toc:
        return GpuEntropyResult(raw=b"", chunk_count=0, backend="none")

    raw_total = toc[-1]
    if raw_total < 0:
        raise GpuEntropyError("invalid TOC raw total")

    res = _decode_with_cuda_python(prim_payload=prim_payload, toc=toc, chunk_index=chunk_index)
    if len(res.raw) != raw_total:
        raise GpuEntropyError("cuda-python decode produced invalid raw length")
    return res
