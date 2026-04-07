"""Byte entropy coding for WEFT payloads.

Implements a compact static-order-0 rANS coder and a raw fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

MODE_RAW = 0
MODE_RANS = 1

PROB_BITS = 12
TOTFREQ = 1 << PROB_BITS
RANS_L = 1 << 23


class EntropyError(ValueError):
    pass


@dataclass(slots=True)
class _Model:
    freqs: list[int]
    cum: list[int]
    lookup: list[int]


def _build_model(data: bytes) -> _Model:
    counts = [0] * 256
    for b in data:
        counts[b] += 1

    total = sum(counts)
    if total <= 0:
        freqs = [0] * 256
        freqs[0] = TOTFREQ
    else:
        raw_freqs = [0] * 256
        for i, c in enumerate(counts):
            if c > 0:
                raw_freqs[i] = max(1, int(round((c / total) * TOTFREQ)))

        cur = sum(raw_freqs)
        if cur == 0:
            raw_freqs[0] = TOTFREQ
            cur = TOTFREQ

        diff = TOTFREQ - cur
        order = sorted(range(256), key=lambda i: counts[i], reverse=True)
        k = 0
        while diff != 0 and k < 100000:
            idx = order[k % len(order)]
            if counts[idx] == 0:
                idx = 0
            if diff > 0:
                raw_freqs[idx] += 1
                diff -= 1
            else:
                if raw_freqs[idx] > 1:
                    raw_freqs[idx] -= 1
                    diff += 1
            k += 1
        if sum(raw_freqs) != TOTFREQ:
            raw_freqs[0] += TOTFREQ - sum(raw_freqs)
        freqs = raw_freqs

    cum = [0] * 257
    running = 0
    for i, f in enumerate(freqs):
        cum[i] = running
        running += f
    cum[256] = running
    if running != TOTFREQ:
        raise EntropyError("invalid normalized frequencies")

    lookup = [0] * TOTFREQ
    for sym in range(256):
        start = cum[sym]
        end = start + freqs[sym]
        if start < end:
            lookup[start:end] = [sym] * (end - start)
    return _Model(freqs=freqs, cum=cum, lookup=lookup)


def _encode_rans(data: bytes) -> bytes:
    if not data:
        return struct.pack("<B", MODE_RAW) + struct.pack("<I", 0)

    model = _build_model(data)
    state = RANS_L
    emitted = bytearray()

    for sym in reversed(data):
        freq = model.freqs[sym]
        start = model.cum[sym]
        if freq <= 0:
            raise EntropyError("symbol frequency is zero")

        x_max = ((RANS_L >> PROB_BITS) << 8) * freq
        while state >= x_max:
            emitted.append(state & 0xFF)
            state >>= 8

        state = ((state // freq) << PROB_BITS) + (state % freq) + start

    pairs = [(sym, model.freqs[sym]) for sym in range(256) if model.freqs[sym] > 0]
    out = bytearray()
    out += struct.pack("<B", MODE_RANS)
    out += struct.pack("<I", len(data))
    out += struct.pack("<H", len(pairs))
    for sym, freq in pairs:
        out += struct.pack("<BH", sym, freq)
    out += struct.pack("<I", state)
    out += struct.pack("<I", len(emitted))
    out += emitted
    return bytes(out)


def _decode_rans(payload: bytes) -> bytes:
    if len(payload) < 1 + 4:
        raise EntropyError("truncated entropy payload")
    mode = payload[0]
    if mode == MODE_RAW:
        if len(payload) < 5:
            raise EntropyError("truncated raw payload")
        raw_len = struct.unpack_from("<I", payload, 1)[0]
        start = 5
        end = start + raw_len
        if end != len(payload):
            raise EntropyError("raw length mismatch")
        return payload[start:end]
    if mode != MODE_RANS:
        raise EntropyError(f"unsupported entropy mode: {mode}")

    if len(payload) < 1 + 4 + 2:
        raise EntropyError("truncated rans header")
    original_size = struct.unpack_from("<I", payload, 1)[0]
    n_syms = struct.unpack_from("<H", payload, 5)[0]
    off = 7
    need = off + n_syms * 3 + 8
    if need > len(payload):
        raise EntropyError("truncated rans model")

    freqs = [0] * 256
    for _ in range(n_syms):
        sym, freq = struct.unpack_from("<BH", payload, off)
        off += 3
        freqs[sym] = freq

    cum = [0] * 257
    running = 0
    for i, f in enumerate(freqs):
        cum[i] = running
        running += f
    cum[256] = running
    if running != TOTFREQ:
        raise EntropyError("invalid rans frequency sum")

    lookup = [0] * TOTFREQ
    for sym in range(256):
        start = cum[sym]
        end = start + freqs[sym]
        if start < end:
            lookup[start:end] = [sym] * (end - start)

    state = struct.unpack_from("<I", payload, off)[0]
    off += 4
    emitted_len = struct.unpack_from("<I", payload, off)[0]
    off += 4
    end = off + emitted_len
    if end != len(payload):
        raise EntropyError("rans emitted length mismatch")
    emitted = payload[off:end]
    ptr = len(emitted) - 1

    out = bytearray(original_size)
    for i in range(original_size):
        slot = state & (TOTFREQ - 1)
        sym = lookup[slot]
        out[i] = sym
        freq = freqs[sym]
        state = freq * (state >> PROB_BITS) + (slot - cum[sym])
        while state < RANS_L and ptr >= 0:
            state = (state << 8) | emitted[ptr]
            ptr -= 1

    return bytes(out)


def encode_bytes(data: bytes) -> bytes:
    """Encode bytes using rANS if smaller than raw, else raw."""
    raw_payload = struct.pack("<B", MODE_RAW) + struct.pack("<I", len(data)) + data
    if not data:
        return raw_payload

    rans_payload = _encode_rans(data)
    return rans_payload if len(rans_payload) < len(raw_payload) else raw_payload


def decode_bytes(payload: bytes) -> bytes:
    if not payload:
        return b""
    mode = payload[0]
    if mode == MODE_RAW:
        if len(payload) < 5:
            raise EntropyError("truncated raw payload")
        raw_len = struct.unpack_from("<I", payload, 1)[0]
        start = 5
        end = start + raw_len
        if end != len(payload):
            raise EntropyError("raw length mismatch")
        return payload[start:end]
    if mode == MODE_RANS:
        return _decode_rans(payload)
    raise EntropyError(f"unknown entropy mode: {mode}")
