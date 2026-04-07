from __future__ import annotations

import pytest

from weft.bitstream import BitstreamError, decode_weft


def test_reject_invalid_magic() -> None:
    with pytest.raises(BitstreamError):
        decode_weft(b"NOPE\x01\x00\x00\x00")


def test_reject_truncated_header() -> None:
    with pytest.raises(BitstreamError):
        decode_weft(b"WEFT")


def test_reject_truncated_directory() -> None:
    blob = b"WEFT" + bytes([1, 0]) + (1).to_bytes(2, "little") + (0).to_bytes(4, "little")
    with pytest.raises(BitstreamError):
        decode_weft(blob)
