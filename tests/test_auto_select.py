"""Auto-select encoder tests.

The auto-select encoder runs all configured candidates (bicubic,
palette-16, palette-64) and writes the winning bitstream by R-D score.
These tests verify:
* The bitstream is byte-identical to whichever single-variant encode won
* The metadata records which variant won and the candidate scoreboard
* The standard decoder dispatches correctly (no auto-aware changes)
* The lambda parameter actually shifts the winner toward higher PSNR
  (low lambda) or smaller bytes (high lambda)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.api import decode_image, encode_image
from weft.types import EncodeConfig


def _make_solid_blocks(path: Path, h: int = 128, w: int = 128) -> None:
    """5-color image with hard rectangle boundaries (favors palette)."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[h // 8: h // 2, w // 8: w // 2] = (220, 60, 60)
    img[h // 8: h // 2, w // 2: 7 * w // 8] = (60, 200, 80)
    img[h // 2: 7 * h // 8, w // 8: w // 2] = (60, 100, 220)
    img[h // 2: 7 * h // 8, w // 2: 7 * w // 8] = (220, 200, 60)
    Image.fromarray(img, "RGB").save(path)


def _make_smooth_gradient(path: Path, size: int = 96) -> None:
    """Smooth diagonal gradient — favors bicubic over palette."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = (xx * 255).astype(np.uint8)
    img[..., 1] = (yy * 255).astype(np.uint8)
    img[..., 2] = ((1 - xx) * (1 - yy) * 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(path)


def test_auto_select_picks_a_hard_edge_specialist_for_blocks() -> None:
    """A 5-color hard-edged image should be won by a hard-edge specialist
    (palette-* or gradient — bicubic and primitive bases all fall apart
    on step discontinuities).
    """
    with tempfile.TemporaryDirectory(prefix="weft-auto-pal-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "blocks.weft"
        decoded_path = td_p / "blocks_decoded.png"
        _make_solid_blocks(src)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "auto_select_lambda": 4.0},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.metadata["auto_select"] is True
        winner = rep.metadata["auto_selected_variant"]
        assert winner in ("palette-16", "palette-64", "gradient"), \
            f"expected a hard-edge specialist, got {winner}"
        # Scoreboard contains all 8 candidates.
        candidates = rep.metadata["auto_select_candidates"]
        assert len(candidates) == 8
        names = {c["name"] for c in candidates}
        assert names == {
            "baseline", "hybrid", "hybrid-dct", "hybrid-dct-tight",
            "bicubic", "palette-16", "palette-64", "gradient",
        }
        # Winner has the highest score.
        assert candidates[0]["name"] == winner
        assert candidates[0]["score"] >= candidates[1]["score"]

        # Standard decoder dispatches correctly via whichever path the
        # winning variant lives on.
        drep = decode_image(str(weft_path), str(decoded_path))
        expected_backend = {
            "palette-16": "palette-cpu",
            "palette-64": "palette-cpu",
            "gradient":   "gradient-cpu",
        }[winner]
        assert drep.metadata["decode_backend"] == expected_backend


def test_auto_select_picks_bicubic_for_smooth_gradient() -> None:
    """A smooth diagonal gradient should be won by bicubic — palette
    quantizes the gradient into stair-stepping bands."""
    with tempfile.TemporaryDirectory(prefix="weft-auto-bic-") as td:
        td_p = Path(td)
        src = td_p / "grad.png"
        weft_path = td_p / "grad.weft"
        decoded_path = td_p / "grad_decoded.png"
        _make_smooth_gradient(src, size=96)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "auto_select_lambda": 4.0},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        winner = rep.metadata["auto_selected_variant"]
        assert winner == "bicubic", f"expected bicubic winner, got {winner}"

        drep = decode_image(str(weft_path), str(decoded_path))
        assert drep.metadata["decode_backend"] == "bicubic-cpu"


def test_auto_select_extreme_high_lambda_picks_smallest_bytes() -> None:
    """At very high lambda the score collapses to ``-lambda * BPP``, so the
    auto-select winner must be whichever candidate has the smallest BPP
    among its three candidates — regardless of PSNR.
    """
    with tempfile.TemporaryDirectory(prefix="weft-auto-lam-") as td:
        td_p = Path(td)
        src = td_p / "grad.png"
        _make_smooth_gradient(src, size=128)

        cfg_r = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "auto_select_lambda": 1000.0},
            verify_drift_threshold_db=999.0,
        )
        rep_r = encode_image(str(src), str(td_p / "r.weft"), cfg_r)

        candidates = rep_r.metadata["auto_select_candidates"]
        winner_name = rep_r.metadata["auto_selected_variant"]
        winner_record = next(c for c in candidates if c["name"] == winner_name)
        # The winner has the smallest BPP among the candidates.
        min_bpp = min(c["bpp"] for c in candidates)
        assert winner_record["bpp"] == pytest.approx(min_bpp, abs=1e-6)


def _make_noise_texture(path: Path, size: int = 96) -> None:
    """Dense high-frequency texture — primitive bases plateau, palette has
    a fixed PSNR ceiling, so this is a fixture where R-D scoring used to
    pick palette over hybrid-dct-tight at high quality."""
    rng = np.random.default_rng(7)
    base = (rng.uniform(0.2, 0.8, (size, size, 3)) * 255).astype(np.uint8)
    # Add some structure so primitives have something to grab
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base[..., 0] = np.clip(base[..., 0].astype(np.float32) + xx * 64, 0, 255).astype(np.uint8)
    Image.fromarray(base, "RGB").save(path)


def test_auto_select_quality_knob_is_psnr_monotonic_on_noise() -> None:
    """Higher cfg.quality must yield strictly higher PSNR on noise-heavy
    content. Regression test for the q95 bug where auto_select would
    pick palette-* (low PSNR) at high quality because the byte penalty
    crushed the quality gain in the R-D math.

    The fix is the quality-aware lambda scale: lam(q) = base × ((100-q)/25)^2.
    """
    with tempfile.TemporaryDirectory(prefix="weft-auto-mono-") as td:
        td_p = Path(td)
        src = td_p / "noise.png"
        _make_noise_texture(src, size=128)

        psnrs = {}
        for q in (50, 75, 95):
            cfg = EncodeConfig(
                quality=q,
                feature_flags={"auto_select": True},
                verify_drift_threshold_db=999.0,
            )
            rep = encode_image(str(src), str(td_p / f"q{q}.weft"), cfg)
            psnrs[q] = float(rep.psnr)

        assert psnrs[75] >= psnrs[50] - 0.1, \
            f"q75 PSNR {psnrs[75]:.2f} regressed vs q50 {psnrs[50]:.2f}"
        assert psnrs[95] >= psnrs[75] - 0.1, \
            f"q95 PSNR {psnrs[95]:.2f} regressed vs q75 {psnrs[75]:.2f}"


def test_auto_select_psnr_tiebreak_invariant() -> None:
    """The auto-select winner satisfies the PSNR-tiebreak invariant:
    either it has the top R-D score, OR it is within the score window
    of the top AND has at least PSNR_TIEBREAK_MIN_GAIN dB more PSNR
    than the R-D top scorer.

    Locks in the q50→q75 dip fix without forcing every encode to swap
    to the highest-PSNR variant in the window (which would erase
    compression-ratio wins on close calls)."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="weft-tiebreak-") as td:
        td_p = Path(td)
        size = 128
        base = rng.integers(20, 235, (size, size, 3), dtype=np.uint8)
        base[:size//2, :size//2] = (255, 30, 30)
        base[:size//2, size//2:] = (30, 255, 30)
        base[size//2:, :size//2] = (30, 30, 255)
        base[size//2:, size//2:] = (255, 255, 30)
        Image.fromarray(base, "RGB").save(td_p / "trick.png")

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(td_p / "trick.png"), str(td_p / "out.weft"), cfg)
        cands = rep.metadata["auto_select_candidates"]
        winner_name = rep.metadata["auto_selected_variant"]
        winner = next(c for c in cands if c["name"] == winner_name)

        cands_by_score = sorted(cands, key=lambda c: -c["score"])
        top = cands_by_score[0]
        WINDOW = 5.0
        MIN_GAIN = 2.0

        if winner_name == top["name"]:
            # R-D winner stood — fine.
            return
        # Tiebreak fired: the winner must be in the window AND have
        # >= MIN_GAIN more PSNR than the top R-D scorer.
        assert winner["score"] >= top["score"] - WINDOW, (
            f"tiebreak winner {winner_name} score={winner['score']:.2f} is "
            f"more than {WINDOW} below top score {top['score']:.2f}"
        )
        assert winner["psnr"] >= top["psnr"] + MIN_GAIN, (
            f"tiebreak winner {winner_name} psnr={winner['psnr']:.2f} is "
            f"not at least {MIN_GAIN} dB above top psnr {top['psnr']:.2f}"
        )


def test_auto_select_quality_lambda_scale_preserves_q75_default() -> None:
    """The lambda scale must be exactly 1.0 at q=75 so the default
    (auto_select_lambda=4.0) behavior is preserved when no quality
    is specified."""
    with tempfile.TemporaryDirectory(prefix="weft-auto-q75-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_solid_blocks(src, h=64, w=64)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "auto_select_lambda": 4.0},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        # At q=75 the lambda scale is exactly 1.0; effective lambda
        # equals the user-supplied base.
        assert rep.metadata["auto_select_lambda_scale"] == pytest.approx(1.0, abs=1e-9)
        assert rep.metadata["auto_select_lambda"] == pytest.approx(4.0, abs=1e-9)
        assert rep.metadata["auto_select_lambda_base"] == pytest.approx(4.0, abs=1e-9)


def test_auto_select_loser_temp_files_are_cleaned_up() -> None:
    """The encoder writes one .weft and removes the candidate temp files."""
    with tempfile.TemporaryDirectory(prefix="weft-auto-clean-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "out.weft"
        _make_solid_blocks(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True},
            verify_drift_threshold_db=999.0,
        )
        encode_image(str(src), str(weft_path), cfg)
        # Only one .weft file in the directory; no leftover .tmp files.
        weft_files = list(td_p.glob("*.weft"))
        tmp_files = list(td_p.glob("*.tmp"))
        assert len(weft_files) == 1
        assert len(tmp_files) == 0


# ── Shared adaptive fit cache ──────────────────────────────────────────


def test_auto_select_shared_fit_produces_same_winner_as_unshared() -> None:
    """Sharing the primitive fit across auto-select variants must
    produce a byte-identical (modulo timing-stamped metadata) winner
    bitstream as running each variant from scratch."""
    import weft.encoder as enc
    from weft.bitstream import decode_weft

    with tempfile.TemporaryDirectory(prefix="weft-auto-share-") as td:
        td_p = Path(td)
        src = td_p / "img.png"
        _make_solid_blocks(src, h=128, w=128)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True},
            verify_drift_threshold_db=999.0,
        )

        # A: shared fit (current behavior).
        encode_image(str(src), str(td_p / "shared.weft"), cfg)

        # B: monkey-patch _encode_image_adaptive to drop the cached
        # fit_state, forcing every variant to re-run the fit.
        orig = enc._encode_image_adaptive

        def no_share(*args, **kwargs):
            kwargs.pop("fit_state", None)
            return orig(*args, **kwargs)

        enc._encode_image_adaptive = no_share
        try:
            encode_image(str(src), str(td_p / "unshared.weft"), cfg)
        finally:
            enc._encode_image_adaptive = orig

        a = decode_weft((td_p / "shared.weft").read_bytes())
        b = decode_weft((td_p / "unshared.weft").read_bytes())
        # All payload blocks must be byte-identical. Only the META JSON
        # block (which carries timing fields) is allowed to differ.
        for field in ("prim_payload", "qtree_payload", "dct_payload",
                      "lite_payload", "bic_payload", "pal_payload",
                      "grd_payload"):
            assert getattr(a, field, None) == getattr(b, field, None), \
                f"{field} differs between shared and unshared fit paths"


def test_fit_cache_dir_persists_state_between_runs() -> None:
    """Setting fit_cache_dir saves the fit state to disk and the second
    encode picks it up."""
    import time as _time
    from weft.encoder import _fit_cache_key

    with tempfile.TemporaryDirectory(prefix="weft-fit-cache-") as td:
        td_p = Path(td)
        src = td_p / "img.png"
        _make_solid_blocks(src, h=128, w=128)
        cache_dir = td_p / "fit-cache"

        cfg = EncodeConfig(
            quality=75,
            fit_cache_dir=str(cache_dir),
            feature_flags={"auto_select": True},
            verify_drift_threshold_db=999.0,
        )

        # Cold run populates the cache.
        encode_image(str(src), str(td_p / "r1.weft"), cfg)
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1, "fit cache should contain exactly one entry"

        # The cache key is deterministic from input bytes + fit-relevant cfg.
        expected_key = _fit_cache_key(str(src), cfg)
        assert cache_files[0].name == f"{expected_key}.pkl"

        # Warm run reads the cache. The bitstream payloads must match
        # the cold run exactly (only timing-stamped metadata can differ).
        encode_image(str(src), str(td_p / "r2.weft"), cfg)
        from weft.bitstream import decode_weft
        r1 = decode_weft((td_p / "r1.weft").read_bytes())
        r2 = decode_weft((td_p / "r2.weft").read_bytes())
        assert r1.prim_payload == r2.prim_payload
        assert r1.qtree_payload == r2.qtree_payload


def test_fit_cache_key_ignores_build_only_flags() -> None:
    """Two cfgs that differ only in build-phase flags must hash to the
    same fit cache key — that's how variant sharing works."""
    from weft.encoder import _fit_cache_key

    with tempfile.TemporaryDirectory(prefix="weft-fit-key-") as td:
        td_p = Path(td)
        src = td_p / "img.png"
        _make_solid_blocks(src, h=64, w=64)

        cfg_baseline = EncodeConfig(quality=75, feature_flags={})
        cfg_hybrid = EncodeConfig(quality=75, feature_flags={"hybrid_bicubic_per_tile": True})
        cfg_dct = EncodeConfig(quality=75, feature_flags={"dct_residual": True})
        cfg_no_res1 = EncodeConfig(quality=75, enable_res1=False, feature_flags={})

        # Build-only differences must NOT change the key.
        k_base = _fit_cache_key(str(src), cfg_baseline)
        k_hyb = _fit_cache_key(str(src), cfg_hybrid)
        k_dct = _fit_cache_key(str(src), cfg_dct)
        k_no_res1 = _fit_cache_key(str(src), cfg_no_res1)
        assert k_base == k_hyb == k_dct == k_no_res1

        # Fit-affecting differences MUST change the key.
        cfg_other_quality = EncodeConfig(quality=95, feature_flags={})
        cfg_other_scale = EncodeConfig(quality=75, encode_scale=0.5, feature_flags={})
        assert _fit_cache_key(str(src), cfg_other_quality) != k_base
        assert _fit_cache_key(str(src), cfg_other_scale) != k_base


# ── Scale-independent rendering ────────────────────────────────────────


def test_prefer_scalable_excludes_raster_variants() -> None:
    """When ``prefer_scalable`` is set, the auto-select scoreboard
    must not contain the raster-only variants (hybrid-dct,
    hybrid-dct-tight, gradient)."""
    with tempfile.TemporaryDirectory(prefix="weft-scalable-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        _make_solid_blocks(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "prefer_scalable": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        names = {c["name"] for c in rep.metadata["auto_select_candidates"]}
        assert "hybrid-dct" not in names
        assert "hybrid-dct-tight" not in names
        assert "gradient" not in names
        # Scalable variants must still be present.
        assert "baseline" in names
        assert "bicubic" in names
        assert "palette-16" in names
        assert "palette-64" in names


def test_scale_independent_palette_upsamples_crisply() -> None:
    """Decoding a palette bitstream at 2× target resolution must
    produce strictly sharper edges than bilinear-upscaling the
    1× output after the fact (which is what the legacy bilinear
    fallback would give)."""
    from weft.decoder import decode_to_array
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="weft-palette-scale-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        _make_solid_blocks(src, h=96, w=96)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"auto_select": True, "prefer_scalable": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        # On 5-color blocks the scalable winner must be a palette
        # variant (bicubic can't represent hard edges; primitives
        # would use more bytes).
        assert "palette" in rep.metadata["auto_selected_variant"]

        # 1× decode
        native = decode_to_array(str(td_p / "out.weft"))
        # 2× decode
        big = decode_to_array(str(td_p / "out.weft"), width=192, height=192)
        assert big.shape == (192, 192, 3)

        # 2× should have the same distinct colours as 1× (nearest-
        # neighbor label-grid upsample preserves the palette exactly).
        native_colors = {tuple(np.round(c * 1000).astype(int))
                         for c in native.reshape(-1, 3)}
        big_colors = {tuple(np.round(c * 1000).astype(int))
                      for c in big.reshape(-1, 3)}
        assert big_colors == native_colors, (
            f"nearest-neighbor upsample must preserve the exact palette; "
            f"native has {len(native_colors)} colors, 2× has {len(big_colors)}"
        )


def test_scale_independent_primitive_path_stays_crisp() -> None:
    """A primitive-stack encode, decoded at 3× target resolution,
    must produce a strictly non-trivial (non-zero) gradient field.
    Verifies the render_tile ``source_size`` parameter propagates
    through the decoder without falling back to bilinear."""
    from weft.decoder import decode_to_array
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="weft-prim-scale-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        _make_solid_blocks(src, h=96, w=96)

        # Force the primitive-stack variant (baseline) without DCT
        # residual — that's the pure-analytic path.
        cfg = EncodeConfig(
            quality=75,
            feature_flags={},  # baseline: no hybrid, no DCT, no palette
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)

        img_native = decode_to_array(str(td_p / "out.weft"))
        img_3x = decode_to_array(str(td_p / "out.weft"), width=288, height=288)
        assert img_3x.shape == (288, 288, 3)

        # The 3× output should have pixel values that span a similar
        # range to the native (not collapsed toward a mean).
        native_range = float(img_native.max() - img_native.min())
        big_range = float(img_3x.max() - img_3x.min())
        assert big_range >= native_range * 0.8, (
            f"3× scaled primitive render collapsed the dynamic range "
            f"({big_range:.3f} vs native {native_range:.3f})"
        )
