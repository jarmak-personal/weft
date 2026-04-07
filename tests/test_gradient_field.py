"""Gradient-field encoder + Poisson-decoder tests (brainstorm idea #1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.api import decode_image, encode_image
from weft.bitstream import BitstreamError, decode_weft, pack_grd, unpack_grd
from weft.constants import BLOCK_GRD, FLAG_HAS_GRD
from weft.gradient_field import decode as grd_decode, encode as grd_encode
from weft.types import EncodeConfig


def _make_shapes(path: Path, h: int = 128, w: int = 128) -> None:
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[h // 8: h // 2, w // 8: w // 2] = (220, 60, 60)
    img[h // 8: h // 2, w // 2: 7 * w // 8] = (60, 200, 80)
    img[h // 2: 7 * h // 8, w // 8: w // 2] = (60, 100, 220)
    Image.fromarray(img, "RGB").save(path)


# ── Standalone math (no bitstream / no encoder) ────────────────────────


def test_gradient_smooth_recovers_at_matched_scale() -> None:
    """The Poisson solve recovers a smooth ramp at a scale matched to the
    per-pixel gradient. Smooth content has a fundamental int8 quantization
    floor — for a ramp of size N, the per-pixel delta is 1/(N-1) and the
    encoder needs ``scale >= N-1`` to represent it without rounding to
    zero. The corresponding ``int8 max representable`` shrinks to
    ``127/scale``, so this only works when the gradient never exceeds
    that range (i.e., for smooth-only content).
    """
    n = 32
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / (n - 1)
    img = np.stack([xx, yy, 1.0 - xx], axis=-1)
    # scale=128: per-pixel delta 1/31 ≈ 0.032 → quantizes to round(4.13) = 4
    # int8 max range 127/128 ≈ 0.99 (no clipping)
    gx, gy, m = grd_encode(img, scale=128, threshold=0.0)
    recon = grd_decode(gx, gy, m, scale=128)
    psnr = 10.0 * np.log10(1.0 / max(float(np.mean((img - recon) ** 2)), 1e-12))
    assert psnr > 30.0, f"smooth ramp PSNR={psnr:.2f} (expected > 30 dB)"


def test_gradient_exact_roundtrip_hard_edges() -> None:
    """With no quantization the Poisson solve recovers hard-edged shapes."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[10:30, 10:30] = (1.0, 0.0, 0.0)
    img[30:50, 30:50] = (0.0, 1.0, 0.0)
    gx, gy, m = grd_encode(img, scale=64, threshold=0.0)
    recon = grd_decode(gx, gy, m, scale=64)
    # Hard edges quantize to integer gradient values exactly at scale=64.
    psnr = 10.0 * np.log10(1.0 / max(float(np.mean((img - recon) ** 2)), 1e-12))
    assert psnr > 50.0, f"hard-edge PSNR={psnr:.2f} (expected > 50 dB)"


def test_gradient_threshold_creates_sparsity() -> None:
    """Threshold zero out small gradients on smooth content."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 63
    img = np.stack([xx, yy, 1.0 - xx], axis=-1)
    gx_low, _, _ = grd_encode(img, scale=64, threshold=0.0)
    gx_high, _, _ = grd_encode(img, scale=64, threshold=0.05)
    # The high threshold should produce strictly more zeros.
    assert (gx_high == 0).sum() >= (gx_low == 0).sum()


# ── Bitstream pack/unpack ──────────────────────────────────────────────


def test_pack_grd_roundtrip() -> None:
    rng = np.random.default_rng(0)
    gx = rng.integers(-127, 127, size=(8, 12, 3), dtype=np.int8)
    gy = rng.integers(-127, 127, size=(8, 12, 3), dtype=np.int8)
    means = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    blob = pack_grd(gx, gy, means, scale=64)
    gx_b, gy_b, means_b, scale_b = unpack_grd(blob)
    assert np.array_equal(gx_b, gx)
    assert np.array_equal(gy_b, gy)
    assert np.allclose(means_b, means)
    assert scale_b == 64


def test_pack_grd_rejects_mismatched_shapes() -> None:
    gx = np.zeros((8, 8, 3), dtype=np.int8)
    gy = np.zeros((8, 9, 3), dtype=np.int8)
    means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(BitstreamError, match="matching"):
        pack_grd(gx, gy, means, scale=64)


# ── End-to-end encoder ────────────────────────────────────────────────


def test_gradient_end_to_end() -> None:
    """Hard-edge image round-trips through encode → decode_image cleanly."""
    with tempfile.TemporaryDirectory(prefix="weft-grd-") as td:
        td_p = Path(td)
        src = td_p / "shapes.png"
        weft_path = td_p / "shapes.weft"
        decoded_path = td_p / "shapes_decoded.png"
        _make_shapes(src, h=128, w=128)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"gradient_field": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.bytes_written > 0
        assert rep.metadata["gradient_field"] is True
        assert "gradient_field_sparsity_pct" in rep.metadata
        # Hard-edge fixture should be highly sparse.
        assert rep.metadata["gradient_field_sparsity_pct"] > 90.0
        # Verify drift ~0 — encoder and decoder eval the same Poisson solve.
        assert abs(rep.metadata["verify_drift_db"]) < 0.01
        # Hard edges quantize cleanly so PSNR should be high.
        assert rep.psnr > 40

        drep = decode_image(str(weft_path), str(decoded_path))
        assert drep.metadata["decode_backend"] == "gradient-cpu"
        decoded = np.array(Image.open(decoded_path))
        assert decoded.shape == (128, 128, 3)


def test_gradient_bitstream_carries_grd_block() -> None:
    """The on-disk bitstream contains a BLOCK_GRD payload + FLAG_HAS_GRD."""
    with tempfile.TemporaryDirectory(prefix="weft-grd-bs-") as td:
        td_p = Path(td)
        src = td_p / "shapes.png"
        weft_path = td_p / "shapes.weft"
        _make_shapes(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"gradient_field": True},
            verify_drift_threshold_db=999.0,
        )
        encode_image(str(src), str(weft_path), cfg)

        weft = decode_weft(weft_path.read_bytes())
        assert weft.head.flags & FLAG_HAS_GRD
        assert weft.grd_payload is not None
        assert any(be.tag == BLOCK_GRD for be in weft.block_entries)
        # Empty PRIM payload sentinel + no QTREE block (gradient is global).
        assert weft.prim_payload == b""
        assert weft.qtree_payload is None


def test_gradient_dispatch_takes_precedence_over_bicubic() -> None:
    """gradient_field flag wins over bicubic_patch_tiles in the dispatcher."""
    with tempfile.TemporaryDirectory(prefix="weft-grd-prec-") as td:
        td_p = Path(td)
        src = td_p / "shapes.png"
        weft_path = td_p / "shapes.weft"
        _make_shapes(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"gradient_field": True, "bicubic_patch_tiles": True},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.metadata.get("gradient_field") is True
        assert rep.metadata.get("preset") == "gradient-field"


def test_gradient_invalid_scale_rejected() -> None:
    """gradient_field_scale must be in [1, 65535]."""
    with tempfile.TemporaryDirectory(prefix="weft-grd-k-") as td:
        td_p = Path(td)
        src = td_p / "shapes.png"
        _make_shapes(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"gradient_field": True, "gradient_field_scale": 0},
            verify_drift_threshold_db=999.0,
        )
        with pytest.raises(ValueError, match="gradient_field_scale"):
            encode_image(str(src), str(td_p / "out.weft"), cfg)
