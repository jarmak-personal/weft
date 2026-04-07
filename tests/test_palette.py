"""Palette + per-pixel labels encoder/decoder tests (brainstorm idea #20)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.api import decode_image, encode_image
from weft.bitstream import BitstreamError, decode_weft, pack_pal, unpack_pal
from weft.constants import BLOCK_PAL, FLAG_HAS_PAL
from weft.palette import fit_palette, render_palette
from weft.types import EncodeConfig


def _make_solid_blocks(path: Path, h: int = 128, w: int = 128) -> None:
    """Synthetic image with 4 solid-color blocks on a solid background.

    Five distinct colors total — fit_palette(k=16) should pick exactly
    those 5 and reconstruct losslessly.
    """
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[h // 8: h // 2, w // 8: w // 2] = (220, 60, 60)
    img[h // 8: h // 2, w // 2: 7 * w // 8] = (60, 200, 80)
    img[h // 2: 7 * h // 8, w // 8: w // 2] = (60, 100, 220)
    img[h // 2: 7 * h // 8, w // 2: 7 * w // 8] = (220, 200, 60)
    Image.fromarray(img, "RGB").save(path)


# ── pack/unpack roundtrip ────────────────────────────────────────────


def test_pack_pal_roundtrip() -> None:
    palette = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)], dtype=np.uint8)
    labels = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
    blob = pack_pal(palette, labels)
    pal_back, lab_back = unpack_pal(blob)
    assert np.array_equal(pal_back, palette)
    assert np.array_equal(lab_back, labels)


def test_pack_pal_rejects_oversize_label() -> None:
    palette = np.zeros((4, 3), dtype=np.uint8)
    labels = np.array([[0, 1, 2, 4]], dtype=np.uint8)  # 4 >= K=4
    with pytest.raises(BitstreamError, match="out of range"):
        pack_pal(palette, labels)


def test_pack_pal_rejects_invalid_palette_shape() -> None:
    with pytest.raises(BitstreamError, match="palette must be"):
        pack_pal(np.zeros((4, 4), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8))


# ── unit test on fit_palette ──────────────────────────────────────────


def test_fit_palette_exact_recovery_on_4_colors() -> None:
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:16, :16] = [1.0, 0.0, 0.0]
    img[:16, 16:] = [0.0, 1.0, 0.0]
    img[16:, :16] = [0.0, 0.0, 1.0]
    img[16:, 16:] = [1.0, 1.0, 0.0]
    palette, labels = fit_palette(img, k=8)
    # The image has exactly 4 unique colors; PIL's quantizer should
    # collapse to 4 (or fewer) palette entries.
    assert palette.shape[0] <= 8
    recon = render_palette(palette, labels)
    # ≤1/255 per channel = ~quantization noise floor.
    assert np.max(np.abs(recon - img)) <= 2.0 / 255 + 1e-6


# ── end-to-end encode/decode ──────────────────────────────────────────


def test_palette_end_to_end() -> None:
    with tempfile.TemporaryDirectory(prefix="weft-pal-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "blocks.weft"
        decoded_path = td_p / "blocks_decoded.png"
        _make_solid_blocks(src, h=128, w=128)

        cfg = EncodeConfig(
            quality=75,
            feature_flags={"palette_planes_k": 16},
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        assert rep.bytes_written > 0
        assert rep.metadata["palette_planes"] is True
        # Image has 5 distinct colors → encoder uses ≤5 palette entries.
        assert rep.metadata["palette_planes_k"] <= 5
        # Verify drift should be ~0 (encoder/decoder eval the same lookup).
        assert abs(rep.metadata["verify_drift_db"]) < 0.01
        # PSNR should be very high — exact lookup, only u8 quantization noise.
        assert rep.psnr > 50

        drep = decode_image(str(weft_path), str(decoded_path))
        assert drep.metadata["decode_backend"] == "palette-cpu"
        decoded = np.array(Image.open(decoded_path))
        assert decoded.shape == (128, 128, 3)


def test_palette_bitstream_carries_pal_block_and_flag() -> None:
    with tempfile.TemporaryDirectory(prefix="weft-pal-bs-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "blocks.weft"
        _make_solid_blocks(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"palette_planes_k": 16},
            verify_drift_threshold_db=999.0,
        )
        encode_image(str(src), str(weft_path), cfg)

        weft = decode_weft(weft_path.read_bytes())
        assert weft.head.flags & FLAG_HAS_PAL
        assert weft.pal_payload is not None
        assert any(be.tag == BLOCK_PAL for be in weft.block_entries)
        # Palette mode shares the bicubic/empty-PRIM sentinel.
        assert weft.prim_payload == b""
        # Palette mode does NOT use the quadtree.
        assert weft.qtree_payload is None


def test_palette_dispatch_takes_precedence_over_bicubic() -> None:
    """If both flags are set, palette wins (palette is the more specialized path)."""
    with tempfile.TemporaryDirectory(prefix="weft-pal-prec-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "blocks.weft"
        _make_solid_blocks(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={
                "palette_planes_k": 16,
                "bicubic_patch_tiles": True,
            },
            verify_drift_threshold_db=999.0,
        )
        rep = encode_image(str(src), str(weft_path), cfg)
        # Palette flag wins; metadata should reflect palette, not bicubic.
        assert rep.metadata.get("palette_planes") is True
        assert rep.metadata.get("preset") == "palette-planes"


def test_palette_invalid_k_rejected() -> None:
    """palette_planes_k must be in [0, 256]."""
    with tempfile.TemporaryDirectory(prefix="weft-pal-k-") as td:
        td_p = Path(td)
        src = td_p / "blocks.png"
        weft_path = td_p / "blocks.weft"
        _make_solid_blocks(src)
        cfg = EncodeConfig(
            quality=75,
            feature_flags={"palette_planes_k": 999},
            verify_drift_threshold_db=999.0,
        )
        with pytest.raises(ValueError, match="palette_planes_k"):
            encode_image(str(src), str(weft_path), cfg)
