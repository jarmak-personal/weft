from __future__ import annotations

import numpy as np

from weft.encoder import _hierarchical_subtile_candidates


def test_hierarchical_subtile_candidates_mid_nonempty() -> None:
    tile = np.zeros((16, 16, 3), dtype=np.float32)
    tile[:8, :8, :] = 1.0
    cands = _hierarchical_subtile_candidates(tile, "mid")
    assert len(cands) >= 8


def test_hierarchical_subtile_candidates_off_empty() -> None:
    tile = np.zeros((16, 16, 3), dtype=np.float32)
    cands = _hierarchical_subtile_candidates(tile, "off")
    assert cands == []
