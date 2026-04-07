"""Bicubic-patch encoder/decoder regression tests (brainstorm idea #11).

The bicubic path is a structurally separate encoder: closed-form per-tile
linear-algebra fit, no greedy primitive search, and the decoder skips
OptiX entirely. These tests verify the round-trip works without GPU
dependencies, that the bitstream carries the expected blocks, and that
verify drift is essentially zero (encoder and decoder eval the same
quantized control grids, so they should agree to floating-point noise).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from weft.api import decode_image, encode_image
from weft.bicubic import _bernstein4_basis, eval_tile, fit_tile
from weft.bitstream import decode_weft, pack_bic, unpack_bic
from weft.constants import BLOCK_BIC, FLAG_HAS_BIC
from weft.types import EncodeConfig


def _make_synth(path: Path, size: int = 96) -> None:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = (xx * 255).astype(np.uint8)
    img[..., 1] = (yy * 255).astype(np.uint8)
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4, 2] = 255
    Image.fromarray(img, mode="RGB").save(path)


# ── unit tests on the bicubic primitives ─────────────────────────────

def test_bernstein4_basis_partition_of_unity() -> None:
    """Cubic Bernstein basis should sum to 1 at every sample point."""
    B = _bernstein4_basis(16)
    sums = B.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-10)


def test_fit_tile_recovers_constant_color_exactly() -> None:
    """A constant-colour tile should round-trip with effectively zero error."""
    tile = np.full((16, 16, 3), 0.4, dtype=np.float32)
    cp = fit_tile(tile)
    recon = eval_tile(cp, 16)
    assert np.allclose(recon, tile, atol=1e-5)


def test_fit_tile_recovers_linear_gradient_exactly() -> None:
    """A linear gradient lies in the bicubic span and should be exact."""
    h = w = 16
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / (w - 1)
    tile = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    cp = fit_tile(tile)
    recon = eval_tile(cp, h)
    # Linear functions are in span(B0..B3); error should be tiny.
    assert np.max(np.abs(recon - tile)) < 1e-4


def test_pack_unpack_bic_roundtrip() -> None:
    rng = np.random.default_rng(0)
    grids = rng.random((25, 4, 4, 3), dtype=np.float32)
    blob = pack_bic(grids)
    # 4 bytes count header + 48 bytes per grid.
    assert len(blob) == 4 + 25 * 48
    back = unpack_bic(blob)
    assert back.shape == (25, 4, 4, 3)
    # u8 quantization: max error per channel ≤ 1/255.
    assert np.max(np.abs(back - grids)) <= 1.0 / 255 + 1e-6


def test_pack_bic_rejects_wrong_shape() -> None:
    import pytest
    from weft.bitstream import BitstreamError
    with pytest.raises(BitstreamError, match="must be"):
        pack_bic(np.zeros((10, 4, 4)))  # missing channel dim


# ── end-to-end encoder/decoder ───────────────────────────────────────


def test_bicubic_end_to_end_encode_decode() -> None:
    """encode_image with the bicubic flag round-trips through decode_image."""
    with tempfile.TemporaryDirectory(prefix="weft-bic-") as td:
        td_p = Path(td)
        src = td_p / "tiny.png"
        weft_path = td_p / "tiny.weft"
        decoded_path = td_p / "tiny_decoded.png"
        _make_synth(src, size=128)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"bicubic_patch_tiles": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)

        # The bicubic encoder reports honest results.
        assert rep.bytes_written > 0
        assert rep.tile_count > 0
        assert rep.psnr > 30  # the synth fixture should fit very well
        # Verify drift should be ~0 because both encoder and decoder eval
        # the same dequantized control grids — there is no encode/decode
        # divergence to discover.
        assert abs(rep.metadata["verify_drift_db"]) < 0.01
        assert rep.metadata["bicubic_patch_tiles"] is True
        assert rep.metadata["preset"] == "bicubic-patch"
        assert rep.metadata["bic_payload_bytes"] > 0

        # decode_image must succeed without OptiX in BIC mode.
        drep = decode_image(str(weft_path), str(decoded_path))
        assert drep.metadata["decode_backend"] == "bicubic-cpu"
        assert decoded_path.exists()
        decoded = np.array(Image.open(decoded_path))
        assert decoded.shape == (128, 128, 3)


def test_bicubic_bitstream_has_bic_block_and_flag() -> None:
    """The on-disk bitstream must carry BLOCK_BIC and FLAG_HAS_BIC."""
    with tempfile.TemporaryDirectory(prefix="weft-bic-bs-") as td:
        td_p = Path(td)
        src = td_p / "tiny.png"
        weft_path = td_p / "tiny.weft"
        _make_synth(src, size=96)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"bicubic_patch_tiles": True},
            verify_drift_threshold_db=999.0,
        )
        encode_image(str(src), str(weft_path), cfg)

        weft = decode_weft(weft_path.read_bytes())
        assert weft.head.flags & FLAG_HAS_BIC
        assert weft.bic_payload is not None
        assert any(be.tag == BLOCK_BIC for be in weft.block_entries)
        # Empty PRIM payload sentinel — the bicubic path stores no primitives.
        assert weft.prim_payload == b""
        # QTREE block is required so the decoder knows tile geometry.
        assert weft.qtree_payload is not None


def test_bicubic_handles_non_multiple_of_32_dimensions() -> None:
    """Regression: 1200×630 (cartoon-family-shaped) must not crash on tile padding.

    The quadtree decomposes against an edge-padded image; the bicubic
    encoder must use the same padding when extracting per-tile patches,
    or tiles that straddle the original right/bottom edges become empty
    slices and ``np.pad`` raises ``ValueError: can't extend empty axis``.
    """
    with tempfile.TemporaryDirectory(prefix="weft-bic-pad-") as td:
        td_p = Path(td)
        src = td_p / "ragged.png"
        weft_path = td_p / "ragged.weft"
        # 100×70 → not divisible by 32, both axes need padding to 128×96.
        size_h, size_w = 70, 100
        yy, xx = np.mgrid[0:size_h, 0:size_w].astype(np.float32)
        img = np.zeros((size_h, size_w, 3), dtype=np.uint8)
        img[..., 0] = (xx / size_w * 255).astype(np.uint8)
        img[..., 1] = (yy / size_h * 255).astype(np.uint8)
        Image.fromarray(img, "RGB").save(src)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"bicubic_patch_tiles": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.bytes_written > 0
        # Round-trip through the decoder.
        out_path = td_p / "ragged_decoded.png"
        decode_image(str(weft_path), str(out_path))
        decoded = np.array(Image.open(out_path))
        # The decoder should output at the original (non-padded) source dims.
        assert decoded.shape == (size_h, size_w, 3)


def test_hybrid_per_tile_bicubic_roundtrip() -> None:
    """Hybrid encoder picks bicubic per tile via R-D, decoder dispatches correctly.

    The hybrid mode (``hybrid_bicubic_per_tile=True``) writes a normal
    primitive-stack bitstream where some tiles contain a single
    ``PRIM_BICUBIC`` primitive instead of a primitive stack. The
    standard CPU primitive renderer dispatches PRIM_BICUBIC back to
    ``bicubic.eval_tile`` via ``render_tile``'s fast path.
    """
    with tempfile.TemporaryDirectory(prefix="weft-hybrid-") as td:
        td_p = Path(td)
        src = td_p / "smooth.png"
        weft_path = td_p / "smooth.weft"
        decoded_path = td_p / "smooth_decoded.png"
        # 96px diagonal gradient + bright square — smooth enough that
        # the hybrid encoder should pick bicubic for at least some tiles.
        _make_synth(src, size=96)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"hybrid_bicubic_per_tile": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)

        # Metadata records the per-tile pick count.
        assert rep.metadata["hybrid_bicubic_per_tile"] is True
        assert "hybrid_bicubic_tile_count" in rep.metadata
        # Verify drift must be ~0 — the encoder uses the same renderer
        # the decoder uses, so there's no encode/decode disagreement.
        assert abs(rep.metadata["verify_drift_db"]) < 0.01

        # Standard decoder dispatches the bitstream via the primitive
        # renderer; PRIM_BICUBIC tiles route to bicubic.eval_tile.
        drep = decode_image(str(weft_path), str(decoded_path))
        assert drep.metadata["decode_backend"] == "primitive-stack-cpu"
        decoded = np.array(Image.open(decoded_path))
        assert decoded.shape == (96, 96, 3)


def test_hybrid_psnr_at_least_baseline() -> None:
    """Hybrid encoder must produce PSNR >= baseline on natural content.

    Hybrid only swaps in bicubic when its R-D score is BETTER than the
    primitive stack at the same lambda — so per-tile MSE for hybrid is
    always <= baseline MSE, hence PSNR is always >= baseline. This test
    is a regression catch for the per-tile selection going wrong.
    """
    with tempfile.TemporaryDirectory(prefix="weft-hybrid-vs-") as td:
        td_p = Path(td)
        src = td_p / "smooth.png"
        _make_synth(src, size=128)

        cfg_b = EncodeConfig(quality=75, verify_drift_threshold_db=999.0)
        cfg_h = EncodeConfig(
            quality=75,
            feature_flags={"hybrid_bicubic_per_tile": True},
            verify_drift_threshold_db=999.0,
        )
        rep_b = encode_image(str(src), str(td_p / "b.weft"), cfg_b)
        rep_h = encode_image(str(src), str(td_p / "h.weft"), cfg_h)
        # Allow a small floating-point tolerance.
        assert rep_h.psnr >= rep_b.psnr - 0.05, (
            f"hybrid PSNR {rep_h.psnr:.3f} regressed from baseline {rep_b.psnr:.3f}"
        )


def test_bicubic_encode_scale_half() -> None:
    """encode_scale<1 should still produce a valid bitstream and round-trip."""
    with tempfile.TemporaryDirectory(prefix="weft-bic-scale-") as td:
        td_p = Path(td)
        src = td_p / "tiny.png"
        weft_path = td_p / "tiny.weft"
        decoded_path = td_p / "tiny_decoded.png"
        _make_synth(src, size=128)
        cfg = EncodeConfig(
            quality=75,
            encode_scale=0.5,
            feature_flags={"bicubic_patch_tiles": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.metadata["encode_width"] == 64
        assert rep.metadata["encode_height"] == 64
        decode_image(str(weft_path), str(decoded_path))
        decoded = np.array(Image.open(decoded_path))
        assert decoded.shape == (128, 128, 3)
