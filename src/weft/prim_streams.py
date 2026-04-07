"""Primitive side-stream builders for container v2 experiments."""

from __future__ import annotations

import base64
import json
from typing import Any

from .primitives import decode_tiles, encode_primitive


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def build_primitive_side_streams(
    *,
    prim_raw: bytes,
    toc: list[int],
    enable_split_streams: bool,
    enable_neighbor_delta: bool,
) -> tuple[bytes | None, bytes | None]:
    """Build optional side streams.

    PSTR groups primitive blobs by type while preserving tile/primitive ordering metadata.
    PDEL stores simple byte deltas to previous primitive blob of the same kind.
    """
    if not enable_split_streams and not enable_neighbor_delta:
        return None, None

    tiles = decode_tiles(prim_raw, toc)
    entries: list[dict[str, Any]] = []
    kind_streams: dict[int, bytearray] = {k: bytearray() for k in range(5)}
    prev_by_kind: dict[int, bytes] = {}
    deltas: list[dict[str, Any]] = []

    for tile_idx, tile in enumerate(tiles):
        for prim_idx, prim in enumerate(tile.primitives):
            blob = encode_primitive(prim)
            kind = int(prim.kind)
            offset = len(kind_streams[kind])
            kind_streams[kind].extend(blob)
            entries.append(
                {
                    "tile": tile_idx,
                    "prim": prim_idx,
                    "kind": kind,
                    "offset": offset,
                    "length": len(blob),
                }
            )

            if enable_neighbor_delta:
                prev = prev_by_kind.get(kind)
                if prev is None:
                    delta = blob
                    prev_len = 0
                else:
                    n = min(len(prev), len(blob))
                    d = bytearray()
                    for i in range(n):
                        d.append((blob[i] - prev[i]) & 0xFF)
                    if len(blob) > n:
                        d.extend(blob[n:])
                    delta = bytes(d)
                    prev_len = len(prev)
                deltas.append(
                    {
                        "tile": tile_idx,
                        "prim": prim_idx,
                        "kind": kind,
                        "prev_len": prev_len,
                        "cur_len": len(blob),
                        "delta_b64": _b64(delta),
                    }
                )
                prev_by_kind[kind] = blob

    pstr_payload = None
    if enable_split_streams:
        pstr_payload = json.dumps(
            {
                "version": 1,
                "layout": "kind_streams_v1",
                "entries": entries,
                "streams": {str(k): _b64(bytes(v)) for k, v in kind_streams.items() if len(v) > 0},
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    pdel_payload = None
    if enable_neighbor_delta:
        pdel_payload = json.dumps(
            {
                "version": 1,
                "layout": "neighbor_delta_v1",
                "entries": deltas,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    return pstr_payload, pdel_payload
