"""CUDA/CCCL backend detection and stubs for GPU-first pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Any

import numpy as np


@dataclass(slots=True)
class CudaRuntimeInfo:
    available: bool
    version: str | None
    detail: str


@dataclass(slots=True)
class CcclRuntimeInfo:
    available: bool
    version: str | None
    detail: str


@dataclass(slots=True)
class GpuStackInfo:
    cuda: CudaRuntimeInfo
    cccl: CcclRuntimeInfo


def _probe_module(name: str) -> tuple[bool, str | None, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", None)
        return True, ver, f"{name} import succeeded"
    except Exception as exc:
        return False, None, str(exc)


def detect_cuda() -> CudaRuntimeInfo:
    ok, ver, detail = _probe_module("cuda")
    return CudaRuntimeInfo(available=ok, version=ver, detail=detail)


def detect_cccl() -> CcclRuntimeInfo:
    # Check common CCCL Python entry points.
    for name in ("cuda.cccl", "cccl"):
        ok, ver, detail = _probe_module(name)
        if ok:
            return CcclRuntimeInfo(available=True, version=ver, detail=detail)
    ok, ver, detail = _probe_module("cuda")
    if ok:
        return CcclRuntimeInfo(
            available=False,
            version=None,
            detail="cuda present but CCCL Python module not found (expected cuda.cccl or cccl)",
        )
    return CcclRuntimeInfo(available=False, version=None, detail=detail)


def detect_gpu_stack() -> GpuStackInfo:
    return GpuStackInfo(cuda=detect_cuda(), cccl=detect_cccl())


def gpu_stack_dict() -> dict[str, Any]:
    return asdict(detect_gpu_stack())


def run_fast_tile_scoring_kernel(*_args, **_kwargs) -> None:
    """Placeholder for CUDA+CCCL raster scoring kernel.

    Planned production path:
    - CUDA kernel evaluates candidate tiles.
    - CCCL performs reductions/compaction/top-k selection.
    """
    return None


def cccl_argmin(values: list[float]) -> int:
    """Return argmin index from CCCL-backed implementations."""
    if not values:
        raise ValueError("values must not be empty")

    try:
        mod = importlib.import_module("cuda.cccl")
        fn = getattr(mod, "argmin", None)
        if callable(fn):
            return int(fn(values))

        alg = getattr(mod, "algorithms", None)
        fn2 = getattr(alg, "argmin", None) if alg is not None else None
        if callable(fn2):
            return int(fn2(values))
    except Exception:
        pass
    # CPU fallback.
    return int(np.argmin(np.asarray(values, dtype=np.float64)))


def cccl_exclusive_scan(values: list[int]) -> list[int]:
    """Exclusive prefix-sum with CCCL backend."""
    if not values:
        return []
    try:
        mod = importlib.import_module("cuda.cccl")
        fn = getattr(mod, "exclusive_scan", None)
        if callable(fn):
            out = fn(values)
            return [int(v) for v in out]
        alg = getattr(mod, "algorithms", None)
        fn2 = getattr(alg, "exclusive_scan", None) if alg is not None else None
        if callable(fn2):
            out = fn2(values)
            return [int(v) for v in out]
    except Exception:
        raise RuntimeError("CCCL exclusive_scan unavailable; CPU fallback disabled")
    raise RuntimeError("CCCL exclusive_scan unavailable; CPU fallback disabled")


def cccl_compact(flags: list[bool]) -> list[int]:
    """Return indices with True flags using CCCL-style compaction semantics."""
    if not flags:
        return []
    try:
        mod = importlib.import_module("cuda.cccl")
        fn = getattr(mod, "compact", None)
        if callable(fn):
            out = fn(flags)
            return [int(v) for v in out]
        alg = getattr(mod, "algorithms", None)
        fn2 = getattr(alg, "compact", None) if alg is not None else None
        if callable(fn2):
            out = fn2(flags)
            return [int(v) for v in out]
    except Exception:
        # Research fallback path until CCCL Python API exposes compact in all envs.
        pass
    return [i for i, f in enumerate(flags) if bool(f)]


def cccl_segmented_topk(
    *,
    scores: list[float],
    segment_ids: list[int],
    k: int,
    largest: bool = False,
) -> list[int]:
    """Return source indices of top-k scores per segment."""
    if len(scores) != len(segment_ids):
        raise ValueError("scores and segment_ids length mismatch")
    if k <= 0 or not scores:
        return []
    try:
        mod = importlib.import_module("cuda.cccl")
        fn = getattr(mod, "segmented_topk", None)
        if callable(fn):
            out = fn(scores=scores, segment_ids=segment_ids, k=int(k), largest=bool(largest))
            return [int(v) for v in out]
        alg = getattr(mod, "algorithms", None)
        fn2 = getattr(alg, "segmented_topk", None) if alg is not None else None
        if callable(fn2):
            out = fn2(scores=scores, segment_ids=segment_ids, k=int(k), largest=bool(largest))
            return [int(v) for v in out]
    except Exception:
        # Research fallback path until CCCL Python API exposes segmented_topk in all envs.
        pass

    order = np.argsort(np.asarray(scores, dtype=np.float64))
    if largest:
        order = order[::-1]
    per_seg: dict[int, int] = {}
    out: list[int] = []
    for idx in order.tolist():
        seg = int(segment_ids[idx])
        used = per_seg.get(seg, 0)
        if used >= k:
            continue
        out.append(int(idx))
        per_seg[seg] = used + 1
    return out
