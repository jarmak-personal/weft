"""Primitive definitions, quantization and tile-level serialization."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

from .constants import (
    MAX_PRIMITIVES_PER_TILE,
    PRIM_BICUBIC,
    PRIM_CONST_PATCH,
    PRIM_LINEAR_PATCH,
    PRIM_LINE,
    PRIM_POLYGON,
    PRIM_QUAD_CURVE,
    PRIMITIVE_ID_TO_NAME,
    PRIMITIVE_NAME_TO_ID,
)


class PrimitiveError(ValueError):
    pass


@dataclass(slots=True)
class Primitive:
    kind: int
    # Geometry is tile-local in [0, 15] for x/y, thickness in [0, 4].
    geom: tuple[float, ...]
    # Colors are linear RGB in [0, 1].
    color0: tuple[float, float, float]
    color1: tuple[float, float, float] | None = None
    alpha: float = 1.0


@dataclass(slots=True)
class TileRecord:
    primitives: list[Primitive]
    residual_rgb: tuple[int, int, int] = (0, 0, 0)


def _q_u8(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    x = (value - lo) / (hi - lo)
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return int(round(x * 255.0))


def _dq_u8(value: int, lo: float, hi: float) -> float:
    return lo + (value / 255.0) * (hi - lo)


def _q_u16(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    x = (value - lo) / (hi - lo)
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return int(round(x * 65535.0))


def _dq_u16(value: int, lo: float, hi: float) -> float:
    return lo + (value / 65535.0) * (hi - lo)


def _q_i8(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    x = (value - lo) / (hi - lo)
    x = -1.0 if x < -1.0 else 1.0 if x > 1.0 else x
    return int(round(x * 127.0))


def _pack_color(color: tuple[float, float, float]) -> bytes:
    return struct.pack("<BBB", _q_u8(color[0], 0.0, 1.0), _q_u8(color[1], 0.0, 1.0), _q_u8(color[2], 0.0, 1.0))


def _unpack_color(payload: bytes) -> tuple[float, float, float]:
    if len(payload) != 3:
        raise PrimitiveError("invalid color payload")
    r, g, b = struct.unpack("<BBB", payload)
    return (_dq_u8(r, 0.0, 1.0), _dq_u8(g, 0.0, 1.0), _dq_u8(b, 0.0, 1.0))


def primitive_kind(value: str | int) -> int:
    if isinstance(value, int):
        if value not in PRIMITIVE_ID_TO_NAME:
            raise PrimitiveError(f"unknown primitive id: {value}")
        return value
    if value not in PRIMITIVE_NAME_TO_ID:
        raise PrimitiveError(f"unknown primitive name: {value}")
    return PRIMITIVE_NAME_TO_ID[value]


def primitive_name(kind: int) -> str:
    if kind not in PRIMITIVE_ID_TO_NAME:
        raise PrimitiveError(f"unknown primitive id: {kind}")
    return PRIMITIVE_ID_TO_NAME[kind]


def encode_primitive(prim: Primitive) -> bytes:
    kind = primitive_kind(prim.kind)
    alpha = _q_u16(prim.alpha, 0.0, 1.0)

    if kind == PRIM_CONST_PATCH:
        payload = _pack_color(prim.color0) + struct.pack("<H", alpha)
    elif kind == PRIM_LINEAR_PATCH:
        if prim.color1 is None or len(prim.geom) != 4:
            raise PrimitiveError("linear_patch requires 4 geom coords and color1")
        x0, y0, x1, y1 = prim.geom
        payload = struct.pack(
            "<HHHH",
            _q_u16(x0, 0.0, 15.0),
            _q_u16(y0, 0.0, 15.0),
            _q_u16(x1, 0.0, 15.0),
            _q_u16(y1, 0.0, 15.0),
        )
        payload += _pack_color(prim.color0) + _pack_color(prim.color1) + struct.pack("<H", alpha)
    elif kind == PRIM_LINE:
        if len(prim.geom) != 5:
            raise PrimitiveError("line requires 5 geom values")
        x0, y0, x1, y1, thickness = prim.geom
        payload = struct.pack(
            "<HHHHH",
            _q_u16(x0, 0.0, 15.0),
            _q_u16(y0, 0.0, 15.0),
            _q_u16(x1, 0.0, 15.0),
            _q_u16(y1, 0.0, 15.0),
            _q_u16(thickness, 0.0, 4.0),
        )
        payload += _pack_color(prim.color0) + struct.pack("<H", alpha)
    elif kind == PRIM_QUAD_CURVE:
        if len(prim.geom) != 7:
            raise PrimitiveError("quad_curve requires 7 geom values")
        x0, y0, cx, cy, x1, y1, thickness = prim.geom
        payload = struct.pack(
            "<HHHHHHH",
            _q_u16(x0, 0.0, 15.0),
            _q_u16(y0, 0.0, 15.0),
            _q_u16(cx, 0.0, 15.0),
            _q_u16(cy, 0.0, 15.0),
            _q_u16(x1, 0.0, 15.0),
            _q_u16(y1, 0.0, 15.0),
            _q_u16(thickness, 0.0, 4.0),
        )
        payload += _pack_color(prim.color0) + struct.pack("<H", alpha)
    elif kind == PRIM_POLYGON:
        if len(prim.geom) != 6:
            raise PrimitiveError("polygon requires 6 geom values (triangle)")
        x0, y0, x1, y1, x2, y2 = prim.geom
        payload = struct.pack(
            "<HHHHHH",
            _q_u16(x0, 0.0, 15.0),
            _q_u16(y0, 0.0, 15.0),
            _q_u16(x1, 0.0, 15.0),
            _q_u16(y1, 0.0, 15.0),
            _q_u16(x2, 0.0, 15.0),
            _q_u16(y2, 0.0, 15.0),
        )
        payload += _pack_color(prim.color0) + struct.pack("<H", alpha)
    elif kind == PRIM_BICUBIC:
        # geom carries 16 control points × 3 channels = 48 floats in
        # row-major (j, i, ch) order, each in [0, 1] linear RGB.
        # Encoded as 48 u8 bytes; the renderer evaluates Bernstein
        # basis at the hit point. Opaque, replaces whatever's below.
        if len(prim.geom) != 48:
            raise PrimitiveError("bicubic_tile requires 48 geom values (4*4*3 control points)")
        payload = bytes(
            max(0, min(255, int(round(v * 255.0)))) for v in prim.geom
        )
    else:
        raise PrimitiveError(f"unsupported primitive kind: {kind}")

    if len(payload) > 255:
        raise PrimitiveError("primitive payload too large")
    return struct.pack("<BB", kind, len(payload)) + payload


def decode_primitive(blob: bytes, offset: int = 0) -> tuple[Primitive, int]:
    if offset + 2 > len(blob):
        raise PrimitiveError("truncated primitive header")
    kind, payload_len = struct.unpack_from("<BB", blob, offset)
    offset += 2
    end = offset + payload_len
    if end > len(blob):
        raise PrimitiveError("truncated primitive payload")
    payload = blob[offset:end]

    if kind == PRIM_CONST_PATCH:
        if len(payload) != 5:
            raise PrimitiveError("invalid const_patch payload")
        color0 = _unpack_color(payload[:3])
        alpha = _dq_u16(struct.unpack_from("<H", payload, 3)[0], 0.0, 1.0)
        prim = Primitive(kind=kind, geom=(), color0=color0, color1=None, alpha=alpha)
    elif kind == PRIM_LINEAR_PATCH:
        if len(payload) != 16:
            raise PrimitiveError("invalid linear_patch payload")
        x0, y0, x1, y1 = struct.unpack_from("<HHHH", payload, 0)
        color0 = _unpack_color(payload[8:11])
        color1 = _unpack_color(payload[11:14])
        alpha = _dq_u16(struct.unpack_from("<H", payload, 14)[0], 0.0, 1.0)
        prim = Primitive(
            kind=kind,
            geom=(
                _dq_u16(x0, 0.0, 15.0),
                _dq_u16(y0, 0.0, 15.0),
                _dq_u16(x1, 0.0, 15.0),
                _dq_u16(y1, 0.0, 15.0),
            ),
            color0=color0,
            color1=color1,
            alpha=alpha,
        )
    elif kind == PRIM_LINE:
        if len(payload) != 15:
            raise PrimitiveError("invalid line payload")
        x0, y0, x1, y1, t = struct.unpack_from("<HHHHH", payload, 0)
        color0 = _unpack_color(payload[10:13])
        alpha = _dq_u16(struct.unpack_from("<H", payload, 13)[0], 0.0, 1.0)
        prim = Primitive(
            kind=kind,
            geom=(
                _dq_u16(x0, 0.0, 15.0),
                _dq_u16(y0, 0.0, 15.0),
                _dq_u16(x1, 0.0, 15.0),
                _dq_u16(y1, 0.0, 15.0),
                _dq_u16(t, 0.0, 4.0),
            ),
            color0=color0,
            color1=None,
            alpha=alpha,
        )
    elif kind == PRIM_QUAD_CURVE:
        if len(payload) != 19:
            raise PrimitiveError("invalid quad_curve payload")
        x0, y0, cx, cy, x1, y1, t = struct.unpack_from("<HHHHHHH", payload, 0)
        color0 = _unpack_color(payload[14:17])
        alpha = _dq_u16(struct.unpack_from("<H", payload, 17)[0], 0.0, 1.0)
        prim = Primitive(
            kind=kind,
            geom=(
                _dq_u16(x0, 0.0, 15.0),
                _dq_u16(y0, 0.0, 15.0),
                _dq_u16(cx, 0.0, 15.0),
                _dq_u16(cy, 0.0, 15.0),
                _dq_u16(x1, 0.0, 15.0),
                _dq_u16(y1, 0.0, 15.0),
                _dq_u16(t, 0.0, 4.0),
            ),
            color0=color0,
            color1=None,
            alpha=alpha,
        )
    elif kind == PRIM_POLYGON:
        if len(payload) != 17:
            raise PrimitiveError("invalid polygon payload")
        x0, y0, x1, y1, x2, y2 = struct.unpack_from("<HHHHHH", payload, 0)
        color0 = _unpack_color(payload[12:15])
        alpha = _dq_u16(struct.unpack_from("<H", payload, 15)[0], 0.0, 1.0)
        prim = Primitive(
            kind=kind,
            geom=(
                _dq_u16(x0, 0.0, 15.0),
                _dq_u16(y0, 0.0, 15.0),
                _dq_u16(x1, 0.0, 15.0),
                _dq_u16(y1, 0.0, 15.0),
                _dq_u16(x2, 0.0, 15.0),
                _dq_u16(y2, 0.0, 15.0),
            ),
            color0=color0,
            color1=None,
            alpha=alpha,
        )
    elif kind == PRIM_BICUBIC:
        if len(payload) != 48:
            raise PrimitiveError("invalid bicubic_tile payload")
        prim = Primitive(
            kind=kind,
            geom=tuple(b / 255.0 for b in payload),
            color0=(0.0, 0.0, 0.0),
            color1=None,
            alpha=1.0,
        )
    else:
        raise PrimitiveError(f"unsupported primitive kind: {kind}")

    return prim, end


def encode_tile(tile: TileRecord) -> bytes:
    if len(tile.primitives) > MAX_PRIMITIVES_PER_TILE:
        raise PrimitiveError("tile has too many primitives")
    payload = bytearray()
    payload += struct.pack("<B", len(tile.primitives))
    for prim in tile.primitives:
        payload += encode_primitive(prim)
    return bytes(payload)


def decode_tile(blob: bytes, offset: int = 0) -> tuple[TileRecord, int]:
    if offset + 1 > len(blob):
        raise PrimitiveError("truncated tile")
    prim_count = blob[offset]
    if prim_count > MAX_PRIMITIVES_PER_TILE:
        raise PrimitiveError("tile primitive count exceeds max")
    offset += 1
    prims: list[Primitive] = []
    for _ in range(prim_count):
        prim, offset = decode_primitive(blob, offset)
        prims.append(prim)
    return TileRecord(primitives=prims), offset


def encode_tiles(tiles: Iterable[TileRecord]) -> tuple[bytes, list[int]]:
    chunks = []
    toc = [0]
    total = 0
    for tile in tiles:
        raw = encode_tile(tile)
        chunks.append(raw)
        total += len(raw)
        toc.append(total)
    return b"".join(chunks), toc


def decode_tiles(blob: bytes, toc: list[int]) -> list[TileRecord]:
    tiles: list[TileRecord] = []
    if len(toc) < 2:
        return tiles
    for i in range(len(toc) - 1):
        start = toc[i]
        end = toc[i + 1]
        if start < 0 or end < start or end > len(blob):
            raise PrimitiveError("invalid tile toc entry")
        tile, consumed = decode_tile(blob[start:end], 0)
        if consumed != (end - start):
            raise PrimitiveError("tile parse did not consume all bytes")
        tiles.append(tile)
    return tiles
