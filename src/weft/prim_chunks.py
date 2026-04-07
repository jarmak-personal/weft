"""Chunked PRIM payload utilities for GPU-friendly decode."""

from __future__ import annotations

from .bitstream import PrimChunkEntry
from .cuda_backend import cccl_exclusive_scan
from .entropy import decode_bytes, encode_bytes


class PrimChunkError(ValueError):
    pass


def _host_exclusive_scan(values: list[int]) -> list[int]:
    out = [0] * len(values)
    running = 0
    for i, v in enumerate(values):
        out[i] = running
        running += int(v)
    return out


def build_prim_chunks(prim_raw: bytes, toc: list[int], chunk_tiles: int) -> tuple[bytes, list[PrimChunkEntry]]:
    if chunk_tiles <= 0:
        raise PrimChunkError("chunk_tiles must be > 0")
    if not toc:
        return b"", []

    tile_count = len(toc) - 1
    if tile_count < 0:
        raise PrimChunkError("invalid TOC")

    starts: list[int] = []
    tile_start = 0
    while tile_start < tile_count:
        starts.append(tile_start)
        tile_start = min(tile_count, tile_start + chunk_tiles)

    raw_spans: list[tuple[int, int, int]] = []
    coded_chunks: list[bytes] = []
    coded_lengths: list[int] = []
    for start in starts:
        end_tile = min(tile_count, start + chunk_tiles)
        raw_start = toc[start]
        raw_end = toc[end_tile]
        if raw_end < raw_start or raw_end > len(prim_raw):
            raise PrimChunkError("invalid TOC bounds")
        raw_spans.append((start, end_tile - start, raw_start))
        coded = encode_bytes(prim_raw[raw_start:raw_end])
        coded_chunks.append(coded)
        coded_lengths.append(len(coded))

    try:
        coded_offsets = cccl_exclusive_scan(coded_lengths)
    except Exception:
        # Container bookkeeping on host; this is not a codec hot path.
        coded_offsets = _host_exclusive_scan(coded_lengths)
    chunks: list[PrimChunkEntry] = []
    coded_payload = bytearray(sum(coded_lengths))
    for i, coded in enumerate(coded_chunks):
        coded_payload[coded_offsets[i] : coded_offsets[i] + len(coded)] = coded

    for i, (start_tile, tile_cnt, raw_start) in enumerate(raw_spans):
        end_tile = start_tile + tile_cnt
        raw_end = toc[end_tile]
        chunks.append(
            PrimChunkEntry(
                start_tile=start_tile,
                tile_count=tile_cnt,
                raw_offset=raw_start,
                raw_length=(raw_end - raw_start),
                coded_offset=coded_offsets[i],
                coded_length=coded_lengths[i],
            )
        )

    return bytes(coded_payload), chunks


def decode_prim_payload(prim_payload: bytes, toc: list[int], chunk_index: list[PrimChunkEntry] | None) -> bytes:
    if not chunk_index:
        raw = decode_bytes(prim_payload)
        if toc and toc[-1] != len(raw):
            raise PrimChunkError("TOC mismatch for non-chunked PRIM")
        return raw

    if not toc:
        return b""

    raw_total = toc[-1]
    out = bytearray(raw_total)
    seen_tiles = 0

    expected_start_tile = 0
    expected_raw_offset = 0
    for c in chunk_index:
        if c.start_tile != expected_start_tile:
            raise PrimChunkError("non-contiguous chunk tile ranges")
        if c.raw_offset != expected_raw_offset:
            raise PrimChunkError("non-contiguous chunk raw ranges")

        coded_end = c.coded_offset + c.coded_length
        if c.coded_offset < 0 or coded_end > len(prim_payload):
            raise PrimChunkError("chunk coded bounds out of range")

        coded = prim_payload[c.coded_offset:coded_end]
        raw = decode_bytes(coded)
        if len(raw) != c.raw_length:
            raise PrimChunkError("chunk raw length mismatch")

        raw_end = c.raw_offset + c.raw_length
        if raw_end > raw_total:
            raise PrimChunkError("chunk raw output out of range")
        out[c.raw_offset:raw_end] = raw

        expected_start_tile += c.tile_count
        expected_raw_offset = raw_end
        seen_tiles += c.tile_count

    if seen_tiles != (len(toc) - 1):
        raise PrimChunkError("chunk tile coverage mismatch")
    if expected_raw_offset != raw_total:
        raise PrimChunkError("chunk raw coverage mismatch")

    return bytes(out)
