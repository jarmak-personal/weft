from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import weft.encoder as enc
from weft.decoder import DecodeError, decode_image
from weft.encoder import EncodeError, encode_image
from weft.gpu_encoder import GpuEncodeError


def _make_img(path: Path, w: int = 64, h: int = 64) -> None:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    arr[..., 1] = np.tile(np.linspace(255, 0, h, dtype=np.uint8)[:, None], (1, w))
    Image.fromarray(arr, mode="RGB").save(path)


def test_encode_surfaces_gpu_backend_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU failures in the baseline stage-3 fit kernel must propagate as EncodeError.

    Forces the baseline encode path (``_encode_image_gpu_baseline``) because
    that's the one that uses ``fit_tiles_gpu_constant`` as a hard dependency.
    The default encode path (``_encode_image_adaptive``) doesn't call that
    kernel directly and would fall through the monkeypatch unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="weft-encode-policy-") as td:
        root = Path(td)
        src = root / "in.png"
        out = root / "out.weft"
        _make_img(src)

        def _boom(*_args, **_kwargs):
            raise GpuEncodeError("forced failure")

        monkeypatch.setenv("WEFT_BASELINE_ENCODE", "1")
        monkeypatch.setattr(enc, "fit_tiles_gpu_constant", _boom)
        with pytest.raises(EncodeError):
            encode_image(str(src), str(out))


def test_decode_rejects_cpu_fallback_flags() -> None:
    with tempfile.TemporaryDirectory(prefix="weft-decode-policy-") as td:
        root = Path(td)
        fake_in = root / "bad.weft"
        fake_out = root / "out.png"
        fake_in.write_bytes(b"WEFT")

        with pytest.raises(DecodeError):
            decode_image(str(fake_in), str(fake_out), gpu_only=False)
        with pytest.raises(DecodeError):
            decode_image(str(fake_in), str(fake_out), allow_cpu_fallback=True)
        with pytest.raises(DecodeError):
            decode_image(str(fake_in), str(fake_out), require_gpu_entropy=False)
