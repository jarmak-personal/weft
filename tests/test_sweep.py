"""Regression tests for the sweep tool (weft.sweep / `weft sweep`).

These cover the structural contract — the sweep produces the expected
files with the expected schema — without depending on specific PSNR
numbers (which would brittle this test against legitimate encoder
improvements).
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.sweep import (
    BUILTIN_VARIANTS,
    SweepResult,
    run_sweep,
    _resolve_variants,
)


def _make_synth(path: Path, size: int = 96) -> None:
    """Tiny diagonal-gradient synthetic image with a bright square."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = (xx * 255).astype(np.uint8)
    img[..., 1] = (yy * 255).astype(np.uint8)
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4, 2] = 255
    Image.fromarray(img, mode="RGB").save(path)


def test_resolve_variants_includes_builtins() -> None:
    out = _resolve_variants(["baseline"], None)
    assert "baseline" in out
    assert out["baseline"] == BUILTIN_VARIANTS["baseline"]


def test_resolve_variants_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        _resolve_variants(["does-not-exist"], None)


def test_resolve_variants_custom_overrides_builtin() -> None:
    custom = {"baseline": {"quality": 99}}
    out = _resolve_variants(["baseline"], custom)
    assert out["baseline"] == {"quality": 99}



def test_run_sweep_writes_expected_outputs() -> None:
    """A 1-image × 1-variant × 1-scale sweep produces all the expected files."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        _make_synth(src)

        summary = run_sweep(
            images=[src],
            output_dir=out,
            scales=[1.0],
            quality=75,
            variants=["baseline"],
            write_html=True,
        )

        # Expected output files exist
        assert (out / "results.json").exists()
        assert (out / "results.csv").exists()
        assert (out / "index.html").exists()

        # Bitstream + decoded PNG were written
        weft_files = list(out.glob("*.weft"))
        assert len(weft_files) == 1
        decoded_files = list(out.glob("*_decoded.png"))
        assert len(decoded_files) == 1

        # Summary structure
        assert summary["n_results"] == 1
        assert summary["scales"] == [1.0]
        assert summary["variants"] == ["baseline"]
        assert summary["run_seconds"] >= 0
        rec = summary["results"][0]
        assert rec["image"] == "tiny.png"
        assert rec["scale"] == 1.0
        assert rec["error"] is None
        # All numeric fields populated.
        assert rec["weft_bytes"] > 0
        assert rec["bpp"] > 0
        assert rec["tile_count"] > 0



def test_run_sweep_records_failures_without_crashing() -> None:
    """A pathological encode_scale should be recorded as an error, not raise."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-fail-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        # Tiny 96px image at scale 0.25 → 24px → below the 32px floor → encoder raises EncodeError.
        _make_synth(src, size=96)

        summary = run_sweep(
            images=[src],
            output_dir=out,
            scales=[1.0, 0.25],  # 1.0 should succeed, 0.25 should be recorded as error
            quality=75,
            variants=["baseline"],
            write_html=True,
        )

        assert summary["n_results"] == 2
        ok = [r for r in summary["results"] if r["error"] is None]
        err = [r for r in summary["results"] if r["error"] is not None]
        assert len(ok) == 1
        assert len(err) == 1
        # The error row recorded the failure but didn't blow up the sweep.
        assert err[0]["scale"] == 0.25
        assert "encode" in err[0]["error"].lower()
        # HTML still got written (with the failure row visible).
        assert (out / "index.html").exists()



def test_run_sweep_csv_schema() -> None:
    """CSV output has the documented schema."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-csv-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        _make_synth(src)
        run_sweep(
            images=[src], output_dir=out, scales=[1.0], variants=["baseline"], write_html=False,
        )
        with (out / "results.csv").open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        for key in (
            "image", "variant", "scale", "weft_bytes", "bpp", "psnr",
            "drift_db", "ratio_vs_src", "ratio_vs_raw",
        ):
            assert key in row, f"missing CSV column: {key}"



def test_run_sweep_two_variants() -> None:
    """A two-variant sweep produces 2 encodes per (image × scale)."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-2v-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        _make_synth(src, size=128)
        summary = run_sweep(
            images=[src],
            output_dir=out,
            scales=[1.0],
            variants=["baseline", "decompose-16"],
            write_html=True,
        )
        assert summary["n_results"] == 2
        variants_seen = sorted({r["variant"] for r in summary["results"]})
        assert variants_seen == ["baseline", "decompose-16"]
        # Both should have produced a .weft file
        assert len(list(out.glob("*.weft"))) == 2


# ── bicubic / parallel-workers / flag-snapshot tests ──────────────────
# These do NOT need OptiX: the bicubic variant is the GPU-free path,
# which lets us exercise the new sweep features in CI environments
# where the OptiX backend isn't available.


def test_bicubic_variant_in_builtin_registry() -> None:
    """The 'bicubic' name is wired into the built-in registry."""
    assert "bicubic" in BUILTIN_VARIANTS
    assert BUILTIN_VARIANTS["bicubic"] == {
        "feature_flags": {"bicubic_patch_tiles": True}
    }


def test_run_sweep_bicubic_only_no_optix_required() -> None:
    """A bicubic-only sweep round-trips without touching OptiX."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-bic-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        _make_synth(src, size=128)
        summary = run_sweep(
            images=[src],
            output_dir=out,
            scales=[1.0],
            variants=["bicubic"],
            write_html=True,
        )
        assert summary["n_results"] == 1
        rec = summary["results"][0]
        assert rec["variant"] == "bicubic"
        assert rec["error"] is None
        assert rec["weft_bytes"] > 0
        # The flag snapshot is preserved on each result.
        assert rec["feature_flags"] == {"bicubic_patch_tiles": True}
        # Bitstream and decoded PNG were both written.
        assert len(list(out.glob("*.weft"))) == 1
        assert len(list(out.glob("*_decoded.png"))) == 1


def test_run_sweep_csv_carries_feature_flags_column() -> None:
    """The CSV must record per-row feature_flags JSON for honest A/B."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-csv-flags-") as td:
        root = Path(td)
        src = root / "tiny.png"
        out = root / "sweep_out"
        _make_synth(src, size=128)
        run_sweep(
            images=[src],
            output_dir=out,
            scales=[1.0],
            variants=["bicubic"],
            write_html=False,
        )
        with (out / "results.csv").open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "feature_flags" in rows[0]
        # Stable JSON dump (sorted keys), so two rows with the same flags
        # compare equal as plain strings.
        assert rows[0]["feature_flags"] == '{"bicubic_patch_tiles": true}'


def test_run_sweep_workers_2_matches_workers_1() -> None:
    """Parallel run with 2 workers produces the same per-cell results as serial."""
    with tempfile.TemporaryDirectory(prefix="weft-sweep-par-") as td:
        root = Path(td)
        src1 = root / "a.png"
        src2 = root / "b.png"
        _make_synth(src1, size=96)
        _make_synth(src2, size=96)
        out_serial = root / "serial"
        out_parallel = root / "parallel"
        kwargs = dict(
            images=[src1, src2],
            scales=[1.0],
            variants=["bicubic"],
            write_html=False,
        )
        s_summary = run_sweep(output_dir=out_serial, workers=1, **kwargs)
        p_summary = run_sweep(output_dir=out_parallel, workers=2, **kwargs)
        assert s_summary["n_results"] == p_summary["n_results"] == 2
        # Sorted by (image, variant, scale) — should be deterministic.
        # Compare on PSNR (content-deterministic) rather than weft_bytes:
        # the bitstream JSON-encodes encoder_ms, so wall-clock timing
        # variation leaks 1-2 bytes of length into the file size between
        # runs even though the rendered pixels are identical.
        s_keys = [(r["image"], r["variant"], r["scale"], round(r["psnr"], 4))
                  for r in s_summary["results"]]
        p_keys = [(r["image"], r["variant"], r["scale"], round(r["psnr"], 4))
                  for r in p_summary["results"]]
        assert s_keys == p_keys
        # The summary surfaces the worker count.
        assert s_summary["workers"] == 1
        assert p_summary["workers"] == 2
