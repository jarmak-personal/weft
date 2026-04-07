"""Regression tests for the decode-in-the-loop verification harness (#30).

These exist primarily as a tripwire: future commits must not silently
turn off the verification harness, change its default behavior, or break
the contract that the EncodeReport metadata exposes the verify metrics.

The tests use small synthetic images and force the encoder through the
default adaptive path. The verify pass calls into ``decoder.decode_to_array``
which uses OptiX, so these tests require a working GPU/OptiX stack — they
will be skipped on machines without one.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weft.api import encode_image
from weft.encoder import EncodeError
from weft.types import EncodeConfig


def _make_synth_image(path: Path, w: int = 128, h: int = 128) -> None:
    """Create a small image with some structure (so primitives have work to do)."""
    rng = np.random.default_rng(0xC0DE)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Diagonal gradient
    yy, xx = np.mgrid[0:h, 0:w]
    arr[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    arr[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    # A bright square in the middle so triangles get exercised
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 2] = 255
    # A little noise
    noise = rng.integers(-15, 16, (h, w, 3))
    arr = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)



def test_verify_default_enabled_records_metrics() -> None:
    """Default encode runs the verify pass and surfaces all four metrics."""
    with tempfile.TemporaryDirectory(prefix="weft-verify-") as td:
        root = Path(td)
        src = root / "in.png"
        out = root / "out.weft"
        _make_synth_image(src)

        rep = encode_image(str(src), str(out))
        meta = rep.metadata

        # All four verify metrics must be in metadata.
        assert "verify_psnr" in meta
        assert "verify_ssim" in meta
        assert "verify_decode_hash" in meta
        assert "verify_drift_db" in meta

        # On a happy-path encode, the verify pass must produce a real number.
        assert meta["verify_psnr"] == meta["verify_psnr"]  # not NaN
        assert math.isfinite(meta["verify_psnr"])
        assert meta["verify_drift_db"] == meta["verify_drift_db"]  # not NaN
        assert math.isfinite(meta["verify_drift_db"])
        assert isinstance(meta["verify_decode_hash"], str)
        assert len(meta["verify_decode_hash"]) > 0
        assert meta["verify_failure"] is None

        # report.psnr must reflect the verified end-to-end number, not the
        # encoder's internal estimate.
        assert rep.psnr == pytest.approx(meta["verify_psnr"], abs=1e-9)



def test_verify_can_be_disabled() -> None:
    """``verify_decode=False`` skips the verify pass and report.psnr falls back to the encoder estimate."""
    with tempfile.TemporaryDirectory(prefix="weft-verify-") as td:
        root = Path(td)
        src = root / "in.png"
        out = root / "out.weft"
        _make_synth_image(src)

        rep = encode_image(str(src), str(out), config=EncodeConfig(verify_decode=False))
        meta = rep.metadata

        # When verify is off, the verify metrics are NaN / empty.
        assert meta["verify_psnr"] != meta["verify_psnr"]  # NaN
        assert meta["verify_drift_db"] != meta["verify_drift_db"]  # NaN
        assert meta["verify_decode_hash"] == ""

        # report.psnr falls back to the encoder's software-render estimate.
        assert rep.psnr == pytest.approx(meta["psnr_software"], abs=1e-9)



def test_verify_strict_raises_when_drift_exceeds_threshold() -> None:
    """A drift larger than the threshold must turn into an EncodeError in strict mode.

    Since the encoder and decoder both use the same CPU primitive renderer,
    the natural drift is ~0 dB — there is no organic encode/decode gap to
    exceed the threshold. We inject a deliberately-wrong decoder output
    via monkey-patch so the verify pass sees a large drift, then confirm
    strict mode raises.
    """
    import numpy as np
    from weft import decoder as dec_mod

    with tempfile.TemporaryDirectory(prefix="weft-verify-") as td:
        root = Path(td)
        src = root / "in.png"
        out = root / "out.weft"
        _make_synth_image(src)

        original_decode_to_array = dec_mod.decode_to_array

        def _wrong_decode(*args, **kwargs):
            arr = original_decode_to_array(*args, **kwargs)
            # Shift every pixel by 50% — guarantees a large PSNR drop.
            return np.clip(1.0 - arr, 0.0, 1.0)

        dec_mod.decode_to_array = _wrong_decode
        try:
            cfg = EncodeConfig(
                verify_decode=True,
                verify_strict=True,
                verify_drift_threshold_db=0.5,
            )
            with pytest.raises(EncodeError, match="verify drift"):
                encode_image(str(src), str(out), config=cfg)
        finally:
            dec_mod.decode_to_array = original_decode_to_array



def test_verify_failure_does_not_break_encode() -> None:
    """If the verify pass throws, the encode still completes and records the failure."""
    import weft.encoder as enc

    with tempfile.TemporaryDirectory(prefix="weft-verify-") as td:
        root = Path(td)
        src = root / "in.png"
        out = root / "out.weft"
        _make_synth_image(src)

        # Monkey-patch decode_to_array via the module that the encoder
        # imports it from at call time.
        from weft import decoder as dec_mod

        def _boom(*_args, **_kwargs):
            raise RuntimeError("synthetic verify failure")

        original = dec_mod.decode_to_array
        dec_mod.decode_to_array = _boom
        try:
            rep = encode_image(str(src), str(out))
        finally:
            dec_mod.decode_to_array = original

        meta = rep.metadata
        # The encode itself succeeds (returns a report) and the file
        # exists on disk.
        assert out.exists()
        # The failure is recorded but report.psnr falls back to the
        # encoder's own estimate.
        assert meta["verify_failure"] is not None
        assert "synthetic verify failure" in meta["verify_failure"]
        assert meta["verify_psnr"] != meta["verify_psnr"]  # NaN
        assert rep.psnr == pytest.approx(meta["psnr_software"], abs=1e-9)
