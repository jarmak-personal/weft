from __future__ import annotations

from weft.bitstream import HeadBlock, decode_weft, encode_weft


def test_container_v2_optional_blocks_roundtrip() -> None:
    head = HeadBlock(width=16, height=16, tile_size=16, max_primitives=48, tile_cols=1, tile_rows=1)
    toc = [0, 0]
    blob = encode_weft(
        head=head,
        toc=toc,
        prim_payload=b"",
        residuals=None,
        residual_maps=None,
        res2_payload=b'{"v":1}',
        pstr_payload=b'{"streams":{}}',
        pdel_payload=b'{"delta":[]}',
        meta={"test": True},
        chunk_index=None,
        block_alignment=64,
    )
    rt = decode_weft(blob)
    assert rt.res2_payload == b'{"v":1}'
    assert rt.pstr_payload == b'{"streams":{}}'
    assert rt.pdel_payload == b'{"delta":[]}'


def test_container_v1_compat_still_decodes() -> None:
    head = HeadBlock(width=16, height=16, tile_size=16, max_primitives=48, tile_cols=1, tile_rows=1)
    toc = [0, 0]
    blob = encode_weft(
        head=head,
        toc=toc,
        prim_payload=b"",
        residuals=None,
        residual_maps=None,
        meta={"test": True},
        chunk_index=None,
        block_alignment=64,
    )
    rt = decode_weft(blob)
    assert rt.res2_payload is None
    assert rt.pstr_payload is None
    assert rt.pdel_payload is None
