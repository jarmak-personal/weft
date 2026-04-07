"""Shared CUDA context for Weft GPU modules.

All GPU modules should use this shared context rather than creating
their own. That avoids multi-context issues where device memory from
one context isn't accessible from another.

If a CUDA error poisons the context, ``reset_cuda_state()`` destroys
it and the next ``get_cuda_state()`` call creates a fresh one.
"""

from __future__ import annotations

_shared_state: dict | None = None


def get_cuda_state() -> dict:
    """Get or create the shared CUDA driver state (context + stream)."""
    global _shared_state
    if _shared_state is not None:
        return _shared_state

    from .gpu_encoder import _import_cuda_bindings, _driver_check

    driver, nvrtc, stack_name = _import_cuda_bindings()
    _driver_check(driver, driver.cuInit(0))
    device = _driver_check(driver, driver.cuDeviceGet(0))
    ctx = _driver_check(driver, driver.cuCtxCreate(0, device))
    stream = _driver_check(driver, driver.cuStreamCreate(0))

    _shared_state = {
        "driver": driver,
        "nvrtc": nvrtc,
        "ctx": ctx,
        "stream": stream,
        "stack_name": stack_name,
    }
    return _shared_state


def reset_cuda_state() -> None:
    """Destroy the current CUDA context and force re-creation on next use.

    Also resets all cached kernel state in GPU modules.
    """
    global _shared_state
    if _shared_state is not None:
        try:
            _shared_state["driver"].cuCtxDestroy(_shared_state["ctx"])
        except Exception:
            pass
        _shared_state = None

    # Reset GPU module caches so they recompile kernels next call.
    import importlib
    for mod_name in ("weft.gpu_render",):
        try:
            mod = importlib.import_module(mod_name)
            for attr in dir(mod):
                if attr.startswith("_") and attr.endswith("_state"):
                    setattr(mod, attr, None)
        except Exception:
            pass
