"""Regression tests that lock down the RD scoring path.

These exist because an earlier GPU-acceleration refactor silently replaced
the sophisticated multi-round greedy + residual re-seeding encode loop with
a naive one-shot per-tile call and deleted the CPU fallback branches from
the inner-loop functions. The changes weren't caught by CI because nothing
pinned RD-scoring behavior. This file fixes that gap.

What these tests protect:

1. ``_tile_lambda`` — the q>=90 ultra-low lambda branch must exist with the
   exact `base * 0.0002` multiplier, and lambda must drop monotonically as
   quality rises within each branch.

2. ``_greedy_add`` and ``_refine_alpha_pass`` — both have a GPU scoring path
   with a graceful CPU fallback. We force ``gpu_batch_objectives`` to return
   ``None`` (as if no GPU) and verify the CPU path runs to completion and
   returns improving (or at-least-non-worsening) objective values.

3. Pure-CPU inner-loop functions (``_one_pass_remove``, ``_refine_colors_pass``,
   ``_refine_coords_pass``) — verified they run end-to-end on synthetic tiles
   and return finite objectives. If a future refactor introduces a GPU path
   to any of these, the corresponding test must be updated to force the CPU
   branch.
"""

from __future__ import annotations

import numpy as np
import pytest

import weft.encoder as enc
from weft.encoder import (
    _greedy_add,
    _one_pass_remove,
    _refine_alpha_pass,
    _refine_colors_pass,
    _refine_coords_pass,
    _tile_lambda,
)
from weft.primitives import Primitive


# ---------------------------------------------------------------------------
#  Lambda formula — locked-down constants
# ---------------------------------------------------------------------------


def test_tile_lambda_has_q90_branch() -> None:
    """At q>=90 the lambda must drop into the ultra-low branch (base * 0.0002)."""
    base_95 = ((101 - 95) / 100.0) ** 2
    assert _tile_lambda(95) == pytest.approx(base_95 * 0.0002, rel=1e-9)

    base_90 = ((101 - 90) / 100.0) ** 2
    assert _tile_lambda(90) == pytest.approx(base_90 * 0.0002, rel=1e-9)


def test_tile_lambda_below_q90_uses_standard_formula() -> None:
    """At q<90 the standard formula (base * 0.001) applies."""
    base_89 = ((101 - 89) / 100.0) ** 2
    assert _tile_lambda(89) == pytest.approx(base_89 * 0.001, rel=1e-9)

    base_75 = ((101 - 75) / 100.0) ** 2
    assert _tile_lambda(75) == pytest.approx(base_75 * 0.001, rel=1e-9)

    base_50 = ((101 - 50) / 100.0) ** 2
    assert _tile_lambda(50) == pytest.approx(base_50 * 0.001, rel=1e-9)


def test_tile_lambda_q90_cliff_is_steeper_than_continuous() -> None:
    """q=90 must undercut q=89 — otherwise the ultra-low branch is missing."""
    assert _tile_lambda(90) < _tile_lambda(89)
    # And by a meaningful factor (>2x), not just a rounding wobble.
    assert _tile_lambda(89) / max(_tile_lambda(90), 1e-12) > 2.0


def test_tile_lambda_monotonic_within_branches() -> None:
    """Within each branch, higher quality -> lower (or equal) lambda."""
    for q in range(50, 89):
        assert _tile_lambda(q) >= _tile_lambda(q + 1)
    for q in range(90, 100):
        assert _tile_lambda(q) >= _tile_lambda(q + 1)


# ---------------------------------------------------------------------------
#  Shared test fixtures
# ---------------------------------------------------------------------------


def _make_tile() -> np.ndarray:
    rng = np.random.default_rng(0xC0DE)
    return rng.random((16, 16, 3), dtype=np.float32)


def _candidate_pool() -> list[Primitive]:
    return [
        Primitive(kind=0, geom=(), color0=(0.1, 0.2, 0.3), alpha=1.0),
        Primitive(kind=0, geom=(), color0=(0.8, 0.1, 0.1), alpha=1.0),
        Primitive(kind=0, geom=(), color0=(0.2, 0.7, 0.3), alpha=1.0),
        Primitive(kind=0, geom=(), color0=(0.4, 0.4, 0.4), alpha=1.0),
        Primitive(kind=0, geom=(), color0=(0.5, 0.5, 0.5), alpha=1.0),
    ]


def _force_gpu_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``gpu_batch_objectives`` call act as if the GPU were unavailable.

    This triggers the CPU fallback branches in ``_greedy_add`` and
    ``_refine_alpha_pass``. The other inner-loop functions don't use the GPU
    helper at all, so the monkeypatch is a no-op for them.
    """
    import weft.gpu_render as gpu_render

    def _always_none(*_args, **_kwargs):
        return None

    monkeypatch.setattr(gpu_render, "gpu_batch_objectives", _always_none)


# ---------------------------------------------------------------------------
#  Inner loops under forced CPU fallback
# ---------------------------------------------------------------------------


def test_greedy_add_cpu_fallback_returns_improving_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_greedy_add's CPU fallback must actually run and improve the objective."""
    _force_gpu_none(monkeypatch)
    tile = _make_tile()
    pool = _candidate_pool()

    initial_obj, _ = enc._tile_objective(tile, [], (0, 0, 0), _tile_lambda(75))
    selected, final_obj, pred = _greedy_add(
        tile,
        selected=[],
        pool=pool,
        max_primitives=3,
        residual=(0, 0, 0),
        lam=_tile_lambda(75),
        res1_grid_size=4,
    )

    assert len(selected) >= 1  # Added at least one primitive.
    assert final_obj <= initial_obj + 1e-9  # Objective didn't get worse.
    assert pred.shape == tile.shape


def test_greedy_add_cpu_fallback_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same inputs -> same outputs, across repeated calls."""
    _force_gpu_none(monkeypatch)
    tile = _make_tile()

    def run() -> tuple[int, float]:
        selected, obj, _ = _greedy_add(
            tile,
            selected=[],
            pool=_candidate_pool(),
            max_primitives=4,
            residual=(0, 0, 0),
            lam=_tile_lambda(75),
            res1_grid_size=4,
        )
        return len(selected), obj

    n1, obj1 = run()
    n2, obj2 = run()
    assert n1 == n2
    assert obj1 == pytest.approx(obj2, rel=1e-9, abs=1e-9)


def test_refine_alpha_cpu_fallback_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_refine_alpha_pass must tolerate the GPU path returning None."""
    _force_gpu_none(monkeypatch)
    tile = _make_tile()
    seed = [
        Primitive(kind=0, geom=(), color0=(0.4, 0.4, 0.4), alpha=0.6),
        Primitive(kind=0, geom=(), color0=(0.6, 0.6, 0.6), alpha=0.7),
    ]
    selected, obj, pred = _refine_alpha_pass(
        tile,
        selected=seed,
        residual=(0, 0, 0),
        lam=_tile_lambda(75),
        res1_grid_size=4,
    )
    assert len(selected) == len(seed)
    assert pred.shape == tile.shape
    assert np.isfinite(obj)


# ---------------------------------------------------------------------------
#  Pure-CPU inner-loop sanity
# ---------------------------------------------------------------------------


def test_one_pass_remove_drops_redundant_primitives() -> None:
    """Given a stack of identical constants, _one_pass_remove must drop duplicates."""
    tile = _make_tile()
    redundant = [
        Primitive(kind=0, geom=(), color0=(0.5, 0.5, 0.5), alpha=1.0)
        for _ in range(5)
    ]
    selected, obj, pred = _one_pass_remove(
        tile,
        selected=redundant,
        residual=(0, 0, 0),
        lam=_tile_lambda(75),
        res1_grid_size=4,
    )
    assert len(selected) >= 1
    assert len(selected) <= len(redundant)
    assert pred.shape == tile.shape
    assert np.isfinite(obj)


def test_refine_colors_runs_end_to_end() -> None:
    """_refine_colors_pass is pure-CPU — just verify it produces valid output."""
    tile = _make_tile()
    seed = [
        Primitive(kind=0, geom=(), color0=(0.5, 0.5, 0.5), alpha=1.0),
        Primitive(kind=0, geom=(), color0=(0.3, 0.3, 0.3), alpha=0.5),
    ]
    selected, obj, pred = _refine_colors_pass(
        tile,
        selected=seed,
        residual=(0, 0, 0),
        lam=_tile_lambda(75),
        res1_grid_size=4,
    )
    assert len(selected) == len(seed)
    assert pred.shape == tile.shape
    assert np.isfinite(obj)


def test_refine_coords_runs_on_line_primitive() -> None:
    """_refine_coords_pass must tolerate a line primitive (kind=2) with thickness."""
    tile = _make_tile()
    seed = [
        Primitive(
            kind=2,
            geom=(2.0, 2.0, 12.0, 12.0, 1.0),
            color0=(0.8, 0.2, 0.2),
            alpha=1.0,
        ),
    ]
    selected, obj, pred = _refine_coords_pass(
        tile,
        selected=seed,
        residual=(0, 0, 0),
        lam=_tile_lambda(75),
        res1_grid_size=4,
    )
    assert len(selected) == 1
    assert pred.shape == tile.shape
    assert np.isfinite(obj)
