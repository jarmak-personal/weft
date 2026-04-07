from __future__ import annotations

from weft.cuda_backend import cccl_compact, cccl_segmented_topk


def test_cccl_compact_indices() -> None:
    assert cccl_compact([False, True, False, True, True]) == [1, 3, 4]


def test_cccl_segmented_topk_smallest_per_segment() -> None:
    scores = [0.8, 0.2, 0.3, 1.1, 0.1, 0.6]
    segs = [0, 0, 1, 1, 1, 2]
    out = cccl_segmented_topk(scores=scores, segment_ids=segs, k=1)
    picked = {segs[i]: i for i in out}
    assert picked[0] == 1
    assert picked[1] == 4
    assert picked[2] == 5


def test_cccl_segmented_topk_k2() -> None:
    scores = [5.0, 1.0, 2.0, 4.0, 3.0, 0.5]
    segs = [0, 0, 0, 1, 1, 1]
    out = cccl_segmented_topk(scores=scores, segment_ids=segs, k=2)
    by_seg = {0: [], 1: []}
    for i in out:
        by_seg[segs[i]].append(i)
    assert set(by_seg[0]) == {1, 2}
    assert set(by_seg[1]) == {5, 4}
