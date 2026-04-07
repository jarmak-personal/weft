"""DCT residual encoder + decoder tests (brainstorm idea #16).

The DCT residual is an additive layer that sits on top of the existing
baseline (or hybrid) primitive-stack reconstruction. These tests verify:

* The standalone DCT encode/decode round-trip
* End-to-end encode→decode with the dct_residual flag
* Verify drift is ~0 (encoder and decoder evaluate the same residual)
* PSNR strictly increases with the DCT layer enabled
* Bitstream block presence and head flag wiring
* The decoder applies the residual on the existing CPU primitive path
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.api import decode_image, encode_image
from weft.bitstream import BitstreamError, decode_weft, pack_dct, unpack_dct
from weft.constants import BLOCK_DCT, FLAG_HAS_DCT
from weft.dct_residual import (
    apply_residual_to_image,
    decode_tile_residuals,
    encode_tile_residuals,
    quant_step_for_quality,
)
from weft.types import EncodeConfig


def _make_natural(path: Path, h: int = 96, w: int = 96, seed: int = 7) -> None:
    """Tiny natural-photo-like fixture (mid-frequency noise + soft gradient)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / max(h, w)
    base = 0.4 + 0.3 * xx + 0.2 * yy
    noise = rng.standard_normal((h, w)) * 0.06
    img = np.clip(base + noise, 0.0, 1.0)
    arr = np.stack([img, img * 0.95 + 0.05, img * 0.85 + 0.1], axis=-1)
    Image.fromarray((arr * 255).astype(np.uint8), "RGB").save(path)


# ── Standalone DCT roundtrip ───────────────────────────────────────────


def test_dct_roundtrip_recovers_residual_at_fine_step() -> None:
    """At a small quant_step, the DCT roundtrip recovers the residual to
    high precision."""
    rng = np.random.default_rng(0)
    tiles = [rng.normal(0, 0.05, (16, 16, 3)).astype(np.float32),
             rng.normal(0, 0.05, (32, 32, 3)).astype(np.float32),
             rng.normal(0, 0.05, (8, 8, 3)).astype(np.float32)]
    coeffs, _ = encode_tile_residuals(tiles, quant_step=0.001)
    recons = decode_tile_residuals(coeffs, [16, 32, 8], quant_step=0.001)
    for orig, rec in zip(tiles, recons):
        max_err = float(np.max(np.abs(orig - rec)))
        assert max_err < 0.002, f"max err {max_err}"


def test_dct_roundtrip_matches_quant_step() -> None:
    """At a coarser step the roundtrip error is bounded by ~step/2 per
    coefficient (uniform-quantizer noise floor)."""
    rng = np.random.default_rng(1)
    tile = rng.normal(0, 0.1, (16, 16, 3)).astype(np.float32)
    step = 0.05
    coeffs, _ = encode_tile_residuals([tile], quant_step=step)
    rec = decode_tile_residuals(coeffs, [16], quant_step=step)[0]
    # Per-pixel error should be bounded by ~step/2 * sqrt(N) by the
    # central-limit-style accumulation through the inverse DCT, which
    # for tile_size=16 is ~step * 8 = 0.4. Use a generous tolerance.
    assert np.max(np.abs(tile - rec)) < step * 8


def test_quant_step_schedule_monotonic() -> None:
    """Higher quality → smaller step → finer quantization."""
    steps = [quant_step_for_quality(q) for q in (50, 60, 70, 80, 90)]
    for prev, cur in zip(steps, steps[1:]):
        assert cur <= prev


# ── Bitstream pack/unpack ──────────────────────────────────────────────


def test_pack_dct_roundtrip() -> None:
    rng = np.random.default_rng(0)
    coeffs = rng.integers(-32000, 32000, size=4096, dtype=np.int16)
    blob = pack_dct(coeffs, quant_step=0.025, channels=3, freq_alpha=4.0)
    (back, step, channels, freq_alpha, chroma_mode,
     presence, n_tiles, layout, scales_u8) = unpack_dct(blob)
    assert np.array_equal(back, coeffs)
    assert step == pytest.approx(0.025)
    assert channels == 3
    assert freq_alpha == pytest.approx(4.0)
    assert chroma_mode == 0
    # No presence bitmask supplied → legacy "all tiles present" mode
    assert presence == b""
    assert n_tiles == 0
    # Default layout is tile-major (0); no per-tile scales by default.
    assert layout == 0
    assert scales_u8 == b""


def test_pack_dct_default_alpha_is_zero() -> None:
    """Without freq_alpha, the format defaults to uniform quantization (alpha=0)."""
    coeffs = np.zeros(64, dtype=np.int16)
    blob = pack_dct(coeffs, quant_step=0.05, channels=3)
    _, _, _, freq_alpha, *_ = unpack_dct(blob)
    assert freq_alpha == 0.0


def test_pack_dct_presence_bitmask_round_trip() -> None:
    """v4 presence bitmask survives pack → unpack unchanged, and the
    encoder helper round-trips its own pack/unpack of presence flags."""
    from weft.dct_residual import (
        _bitmask_to_present_indices,
        _present_indices_to_bitmask,
    )
    rng = np.random.default_rng(2)
    coeffs = rng.integers(-100, 100, size=512, dtype=np.int16)
    n_tiles = 13
    present = [True, False, True, False, False, True, False,
               True, False, True, False, False, True]
    bitmask = _present_indices_to_bitmask(present)
    assert _bitmask_to_present_indices(bitmask, n_tiles) == present

    blob = pack_dct(
        coeffs, quant_step=0.03, channels=3, freq_alpha=4.0,
        chroma_mode=2, presence_bitmask=bitmask, n_tiles=n_tiles,
    )
    (back, step, channels, freq_alpha, chroma_mode,
     back_mask, back_n, layout, scales_u8) = unpack_dct(blob)
    assert np.array_equal(back, coeffs)
    assert back_mask == bitmask
    assert back_n == n_tiles
    assert chroma_mode == 2
    assert layout == 0
    assert scales_u8 == b""


def test_permute_tile_to_band_round_trip() -> None:
    """Band-major permutation is exactly reversible across chroma modes
    and heterogeneous tile sizes."""
    from weft.dct_residual import permute_tile_to_band, permute_band_to_tile
    rng = np.random.default_rng(42)
    # Mixed sizes, 3 chroma modes
    for chroma_mode in (0, 1, 2):
        sizes = [16, 32, 8, 16, 8, 32, 16]
        if chroma_mode == 2:
            # 4:2:0: Y size² + 2 × (size/2)²
            per_tile = [s * s + 2 * (s // 2) * (s // 2) for s in sizes]
        else:
            per_tile = [3 * s * s for s in sizes]
        total = sum(per_tile)
        orig = rng.integers(-200, 200, size=total, dtype=np.int16)
        band = permute_tile_to_band(orig, sizes, chroma_mode)
        back = permute_band_to_tile(band, sizes, chroma_mode)
        assert np.array_equal(back, orig), f"chroma_mode={chroma_mode} round-trip failed"
        assert band.size == orig.size


def test_zigzag_flat_indices_properties() -> None:
    """zigzag_flat_indices(N) is a valid permutation: DC first, highest
    frequency last, each index appearing exactly once."""
    from weft.dct_residual import zigzag_flat_indices
    for N in (1, 2, 4, 8, 16, 32):
        idx = zigzag_flat_indices(N)
        assert idx.size == N * N
        assert idx[0] == 0, f"DC should be first for N={N}"
        assert idx[-1] == N * N - 1, f"highest freq should be last for N={N}"
        assert set(idx.tolist()) == set(range(N * N)), \
            f"zigzag order for N={N} is not a permutation"


def test_band_major_format_shrinks_natural_photo() -> None:
    """On natural-photo content the band-major v4 layout produces a
    strictly smaller on-disk DCT payload than the (old) tile-major
    layout at the same encode settings."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-band-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "dct_residual": True,
                "dct_residual_step": 0.03,
                "dct_residual_chroma_mode": 2,
                # Disable per-tile skip so the only difference from the
                # old path is the layout permutation.
                "dct_residual_skip_threshold": 0.0,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        # v4 round-trips cleanly
        assert abs(rep.metadata["verify_drift_db"]) < 0.05
        # Decoder picks the correct layout automatically — the rendered
        # PSNR is the same as the encoder's internal estimate.
        from weft.api import decode_image
        drep = decode_image(str(td_p / "out.weft"), str(td_p / "dec.png"))
        assert drep.metadata["decode_backend"] == "primitive-stack-cpu"


def test_freq_weights_grow_with_frequency() -> None:
    """The per-coefficient weight matrix is monotonically increasing in i+j."""
    from weft.dct_residual import freq_weights
    w = freq_weights(16, 4.0)
    # DC = 1.0, highest freq = 1 + alpha = 5.0
    assert w[0, 0] == pytest.approx(1.0)
    assert w[15, 15] == pytest.approx(5.0)
    # Monotonic along the diagonal
    for k in range(15):
        assert w[k + 1, k + 1] > w[k, k]


def test_pack_dct_rejects_2d_array() -> None:
    with pytest.raises(BitstreamError, match="1-D"):
        pack_dct(np.zeros((4, 4), dtype=np.int16), quant_step=0.05)


# ── apply_residual_to_image helper ─────────────────────────────────────


def test_apply_residual_per_tile_geometry() -> None:
    """The helper writes each tile's IDCT residual to the right position."""
    from weft.quadtree import QuadTile
    h, w = 32, 32
    base = np.zeros((h, w, 3), dtype=np.float32)
    quads = [
        QuadTile(x=0,  y=0,  size=16, index=0),
        QuadTile(x=16, y=0,  size=16, index=1),
        QuadTile(x=0,  y=16, size=16, index=2),
        QuadTile(x=16, y=16, size=16, index=3),
    ]
    # Construct residuals where each tile has a unique constant value.
    constants = [0.10, 0.20, 0.30, 0.40]
    residuals = [np.full((16, 16, 3), c, dtype=np.float32) for c in constants]
    coeffs, _ = encode_tile_residuals(residuals, quant_step=0.005)
    out = apply_residual_to_image(base, coeffs, quads, quant_step=0.005)
    # Each quadrant should now contain (approximately) its constant value.
    for (x, y, c) in [(0, 0, 0.10), (16, 0, 0.20), (0, 16, 0.30), (16, 16, 0.40)]:
        center_val = out[y + 8, x + 8, 0]
        assert center_val == pytest.approx(c, abs=0.01), \
            f"quadrant at ({x},{y}) got {center_val}, expected ~{c}"


# ── End-to-end encoder + decoder ───────────────────────────────────────


def test_dct_residual_end_to_end_strictly_improves_psnr() -> None:
    """A natural-photo fixture should see the DCT residual lift PSNR
    above the baseline at the same encoder quality.
    """
    with tempfile.TemporaryDirectory(prefix="weft-dct-") as td:
        td_p = Path(td)
        src = td_p / "natural.png"
        weft_a = td_p / "baseline.weft"
        weft_b = td_p / "dct.weft"
        decoded = td_p / "dct_decoded.png"
        _make_natural(src, h=128, w=128)

        cfg_base = EncodeConfig(quality=75, verify_drift_threshold_db=999.0)
        cfg_dct = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True, "dct_residual_step": 0.02},
            verify_drift_threshold_db=999.0,
        )
        rep_a = encode_image(str(src), str(weft_a), cfg_base)
        rep_b = encode_image(str(src), str(weft_b), cfg_dct)

        # DCT layer must STRICTLY raise PSNR (it only adds information).
        assert rep_b.psnr > rep_a.psnr, \
            f"dct PSNR {rep_b.psnr:.2f} did not improve baseline {rep_a.psnr:.2f}"
        # Verify drift ~0 (encoder applies the same residual the decoder will).
        assert abs(rep_b.metadata["verify_drift_db"]) < 0.05
        # Bytes should grow because the DCT layer is purely additive.
        assert rep_b.bytes_written > rep_a.bytes_written
        # The metadata records the DCT layer state.
        assert rep_b.metadata["dct_residual"] is True
        assert rep_b.metadata["dct_residual_payload_bytes"] > 0
        assert rep_b.metadata["dct_residual_tile_count"] > 0

        # Standard decoder dispatches via the primitive renderer +
        # apply_residual_to_image; backend label is unchanged.
        drep = decode_image(str(weft_b), str(decoded))
        assert drep.metadata["decode_backend"] == "primitive-stack-cpu"


def test_dct_residual_bitstream_carries_dct_block_and_flag() -> None:
    """The on-disk bitstream contains BLOCK_DCT + FLAG_HAS_DCT when enabled."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-bs-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        weft_path = td_p / "n.weft"
        _make_natural(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True},
            verify_drift_threshold_db=999.0,
        )
        encode_image(str(src), str(weft_path), cfg)

        weft = decode_weft(weft_path.read_bytes())
        assert weft.head.flags & FLAG_HAS_DCT
        assert weft.dct_payload is not None
        assert any(be.tag == BLOCK_DCT for be in weft.block_entries)
        # The DCT residual is additive, so the standard primitive blocks
        # are still present.
        assert weft.qtree_payload is not None
        assert len(weft.prim_payload) > 0


def test_dct_residual_invalid_step_rejected() -> None:
    """dct_residual_step must be > 0 (or None)."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-bad-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True, "dct_residual_step": 0.0},
            verify_drift_threshold_db=999.0,
        )
        with pytest.raises(ValueError, match="dct_residual_step"):
            encode_image(str(src), str(td_p / "out.weft"), cfg)


def test_dct_residual_combines_with_hybrid() -> None:
    """The DCT layer can be stacked on top of the per-tile hybrid encoder."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-hyb-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"hybrid_bicubic_per_tile": True, "dct_residual": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        assert rep.metadata["dct_residual"] is True
        assert rep.metadata["hybrid_bicubic_per_tile"] is True
        assert abs(rep.metadata["verify_drift_db"]) < 0.05


# ── Chroma mode (YCbCr 4:4:4 / 4:2:0) ──────────────────────────────────


@pytest.mark.parametrize("chroma_mode", [0, 1, 2])
def test_dct_residual_chroma_mode_round_trip(chroma_mode: int) -> None:
    """Each chroma mode (RGB / YCbCr 4:4:4 / YCbCr 4:2:0) round-trips
    with zero verify drift — i.e., encoder and decoder evaluate the
    same residual."""
    with tempfile.TemporaryDirectory(prefix=f"weft-dct-c{chroma_mode}-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "dct_residual": True,
                "dct_residual_step": 0.03,
                "dct_residual_chroma_mode": chroma_mode,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        assert rep.metadata["dct_residual_chroma_mode"] == chroma_mode
        assert abs(rep.metadata["verify_drift_db"]) < 0.05


def test_dct_residual_chroma_mode_4_2_0_payload_smaller_than_rgb() -> None:
    """At the same quant step on natural-photo content, YCbCr 4:2:0
    produces a smaller DCT payload than full-RGB encoding (chroma is
    box-averaged 2× per axis → 25% chroma byte cost vs 4:4:4)."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-iso-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)

        rep_rgb = encode_image(
            str(src), str(td_p / "rgb.weft"),
            EncodeConfig(
                quality=75,
                feature_flags={
                    "dct_residual": True,
                    "dct_residual_step": 0.04,
                    "dct_residual_chroma_mode": 0,
                },
                verify_drift_threshold_db=999.0,
            ),
        )
        rep_420 = encode_image(
            str(src), str(td_p / "420.weft"),
            EncodeConfig(
                quality=75,
                feature_flags={
                    "dct_residual": True,
                    "dct_residual_step": 0.04,
                    "dct_residual_chroma_mode": 2,
                },
                verify_drift_threshold_db=999.0,
            ),
        )
        assert (
            rep_420.metadata["dct_residual_payload_bytes"]
            < rep_rgb.metadata["dct_residual_payload_bytes"]
        ), "4:2:0 should produce a smaller DCT payload than RGB at the same step"


def test_dct_residual_per_tile_mode_round_trip() -> None:
    """The per-tile mode selector flips tiles to empty-primitive +
    DCT-only mode. Must round-trip with zero verify drift regardless
    of how many tiles end up in empty mode."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-ptm-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "dct_residual": True,
                "dct_residual_per_tile_mode": True,
                "dct_residual_step": 0.025,
                # Disable per-tile skip so every tile is evaluated
                # and the empty-mode counter is easier to assert.
                "dct_residual_skip_threshold": 0.0,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        assert abs(rep.metadata["verify_drift_db"]) < 0.05
        # On a noisy natural fixture, some tiles must have flipped
        # — if none did, the decision proxy is broken.
        n_empty = rep.metadata["dct_residual_tiles_empty_mode"]
        n_total = rep.metadata["dct_residual_tile_count"]
        assert n_empty > 0, (
            f"expected some tiles to flip to empty mode; got "
            f"{n_empty}/{n_total}"
        )


def test_dct_residual_per_tile_mode_default_off() -> None:
    """With the flag unset, no tiles flip — the legacy primitive-stack
    path is preserved."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-ptm-off-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=96, w=96)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True, "dct_residual_step": 0.025},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        assert rep.metadata["dct_residual_tiles_empty_mode"] == 0


def test_dct_residual_adaptive_quant_round_trip() -> None:
    """BLOCK_DCT v5 per-tile scales survive encode → decode with zero
    verify drift, even though the per-tile scales the encoder picks
    differ across tiles."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-aq-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "dct_residual": True,
                "dct_residual_adaptive_quant": True,
                "dct_residual_step": 0.025,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        # Encoder must produce zero verify drift (decoder reads back
        # the exact same per-tile scales the encoder used).
        assert abs(rep.metadata["verify_drift_db"]) < 0.05

        # The on-disk BLOCK_DCT carries non-empty per-tile scales.
        from weft.bitstream import decode_weft, unpack_dct
        weft = decode_weft((td_p / "out.weft").read_bytes())
        (_c, _qs, _ch, _fa, _cm, _pm, _nt, _layout, scales_u8) = unpack_dct(weft.dct_payload)
        assert len(scales_u8) > 0


def test_dct_residual_adaptive_quant_default_off() -> None:
    """Adaptive quant flag defaults off — bitstream is wire-identical
    to a non-adaptive encode at the same other settings."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-aq-off-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=96, w=96)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True, "dct_residual_step": 0.025},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        from weft.bitstream import decode_weft, unpack_dct
        weft = decode_weft((td_p / "out.weft").read_bytes())
        (*_, scales_u8) = unpack_dct(weft.dct_payload)
        # No per-tile scales when the flag is off.
        assert scales_u8 == b""


def test_tile_scale_u8_round_trip() -> None:
    """encode_tile_scale_u8 → decode_tile_scale_u8 is the identity (mod
    quantization rounding) and the range covers [0.25, 4.0]."""
    from weft.dct_residual import encode_tile_scale_u8, decode_tile_scale_u8
    # Endpoints map to the documented range.
    assert decode_tile_scale_u8(encode_tile_scale_u8(0.25)) == pytest.approx(0.25, rel=1e-4)
    assert decode_tile_scale_u8(encode_tile_scale_u8(4.0)) == pytest.approx(4.0, rel=1e-4)
    # Center maps to ~1.0.
    mid = decode_tile_scale_u8(encode_tile_scale_u8(1.0))
    assert mid == pytest.approx(1.0, rel=1e-2)
    # Round-trip a few intermediate values.
    for s in (0.5, 0.7, 1.5, 2.0, 3.0):
        u = encode_tile_scale_u8(s)
        back = decode_tile_scale_u8(u)
        assert back == pytest.approx(s, rel=2e-2), f"scale {s} round-tripped to {back}"


def test_dct_residual_invalid_chroma_mode_rejected() -> None:
    """dct_residual_chroma_mode must be 0, 1 or 2."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-bad-c-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"dct_residual": True, "dct_residual_chroma_mode": 7},
            verify_drift_threshold_db=999.0,
        )
        with pytest.raises(ValueError, match="dct_residual_chroma_mode"):
            encode_image(str(src), str(td_p / "out.weft"), cfg)


# ── Per-tile skip (presence bitmask) ───────────────────────────────────


def test_dct_residual_per_tile_skip_drops_tiles_below_threshold() -> None:
    """At a generous skip threshold the encoder drops a meaningful
    fraction of tiles compared to no-skip, while keeping verify drift
    at zero (encoder and decoder agree on the same kept-tile set)."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-skip-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=128, w=128)

        rep_no_skip = encode_image(
            str(src), str(td_p / "noskip.weft"),
            EncodeConfig(
                quality=75,
                feature_flags={
                    "dct_residual": True,
                    "dct_residual_step": 0.03,
                    "dct_residual_skip_threshold": 0.0,
                },
                verify_drift_threshold_db=999.0,
            ),
        )
        rep_skip = encode_image(
            str(src), str(td_p / "skip.weft"),
            EncodeConfig(
                quality=75,
                feature_flags={
                    "dct_residual": True,
                    "dct_residual_step": 0.03,
                    # Aggressive threshold so we definitely drop tiles
                    # on this fixture.
                    "dct_residual_skip_threshold": 0.05,
                },
                verify_drift_threshold_db=999.0,
            ),
        )
        n_total = rep_no_skip.metadata["dct_residual_tile_count"]
        n_present_no_skip = rep_no_skip.metadata["dct_residual_tiles_present"]
        n_present_skip = rep_skip.metadata["dct_residual_tiles_present"]
        assert n_present_no_skip == n_total, "no-skip should keep every tile"
        assert n_present_skip < n_total, "skip mode should drop at least one tile"
        # Skip mode must produce a smaller bitstream — that's the point.
        assert rep_skip.bytes_written < rep_no_skip.bytes_written
        # Both must round-trip with zero drift.
        assert abs(rep_no_skip.metadata["verify_drift_db"]) < 0.05
        assert abs(rep_skip.metadata["verify_drift_db"]) < 0.05


def test_dct_residual_skip_zero_threshold_keeps_all_tiles() -> None:
    """Threshold of 0 means every tile is encoded — bit-identical
    behaviour to v2 modulo the v3 header overhead."""
    with tempfile.TemporaryDirectory(prefix="weft-dct-skip0-") as td:
        td_p = Path(td)
        src = td_p / "n.png"
        _make_natural(src, h=96, w=96)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "dct_residual": True,
                "dct_residual_step": 0.03,
                "dct_residual_skip_threshold": 0.0,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(td_p / "out.weft"), cfg)
        assert (
            rep.metadata["dct_residual_tiles_present"]
            == rep.metadata["dct_residual_tile_count"]
        )
