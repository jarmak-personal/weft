"""GPU upload planning helpers for WEFT.

The goal is a single host->device transfer for file bytes, then in-device
addressing for block/chunk decode.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bitstream import BlockEntry, PrimChunkEntry, WeftFile


@dataclass(slots=True)
class DeviceUploadPlan:
    total_bytes: int
    one_shot_upload: bool
    block_views: dict[str, tuple[int, int]]
    prim_chunks: list[PrimChunkEntry]


def build_device_upload_plan(weft: WeftFile, file_size: int) -> DeviceUploadPlan:
    views: dict[str, tuple[int, int]] = {}
    for entry in weft.block_entries:
        key = entry.tag.decode("ascii", errors="ignore").strip() or entry.tag.hex()
        views[key] = (entry.offset, entry.length)

    return DeviceUploadPlan(
        total_bytes=file_size,
        one_shot_upload=True,
        block_views=views,
        prim_chunks=list(weft.chunk_index or []),
    )
