from __future__ import annotations

import os

from weft.entropy import decode_bytes, encode_bytes


def test_entropy_roundtrip_small_payloads() -> None:
    payloads = [
        b"",
        b"a",
        b"aaaaaa",
        bytes(range(64)),
        bytes([7, 3, 7, 3, 7, 3]) * 20,
        os.urandom(1024),
    ]
    for payload in payloads:
        coded = encode_bytes(payload)
        decoded = decode_bytes(coded)
        assert decoded == payload
