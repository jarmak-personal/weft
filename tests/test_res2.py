from __future__ import annotations

import numpy as np

from weft.decoder import _apply_res2_sparse
from weft.encoder import _encode_res2_sparse
from weft.primitives import Primitive, TileRecord


def test_encode_res2_sparse_emits_atoms() -> None:
    tile = np.zeros((16, 16, 3), dtype=np.float32)
    tile[8, 8, :] = 1.0
    rec = TileRecord(primitives=[Primitive(kind=0, geom=(), color0=(0.0, 0.0, 0.0), alpha=1.0)])
    res2 = _encode_res2_sparse(tiles=[tile], records=[rec], max_atoms_per_tile=8)
    assert res2["format"] == "sparse_atoms_v1"
    assert len(res2["tiles"]) == 1
    assert len(res2["tiles"][0]) >= 1


def test_apply_res2_sparse_adjusts_pixels() -> None:
    img = np.zeros((16, 16, 3), dtype=np.float32)
    res2 = {
        "format": "sparse_atoms_v1",
        "tile_size": 16,
        "max_atoms_per_tile": 8,
        "tiles": [[[5 * 16 + 7, 25, -10, 5]]],
    }
    out = _apply_res2_sparse(img, res2, width=16, height=16, tile_size=16)
    assert float(out[5, 7, 0]) > 0.0
    assert float(out[5, 7, 1]) == 0.0
