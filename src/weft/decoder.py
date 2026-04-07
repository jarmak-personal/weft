"""WEFT decoder implementation."""

from __future__ import annotations

from dataclasses import asdict
import json
import os

import numpy as np

from .bitstream import decode_residual_maps, decode_residuals, decode_weft
from .cuda_backend import detect_gpu_stack, gpu_stack_dict
from .device_plan import build_device_upload_plan
from .gpu_entropy import GpuEntropyError, decode_prim_payload_chunked_gpu
from .image_io import save_image_linear
from .primitives import decode_tiles
from .quadtree import QuadTile, unpack_qtree
from .render import decode_hash, render_scene_adaptive
from .types import DecodeReport


class DecodeError(ValueError):
    pass


def _decode_res1_maps_to_float(blob: bytes | None) -> tuple[int, list[np.ndarray]] | None:
    dec = decode_residual_maps(blob)
    if dec is None:
        return None
    grid_size, maps = dec
    out: list[np.ndarray] = []
    for m in maps:
        arr = np.frombuffer(m, dtype=np.int8).astype(np.float32).reshape((grid_size, grid_size, 3)) / 255.0
        out.append(arr)
    return grid_size, out


def _decode_res2_sparse_payload(payload: bytes | None, meta: dict) -> dict | None:
    if payload is not None:
        try:
            obj = json.loads(payload.decode("utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    res2 = meta.get("res2_sparse")
    if isinstance(res2, dict):
        return res2
    return None


def _apply_res2_sparse(image: np.ndarray, res2: dict | None, *, width: int, height: int, tile_size: int) -> np.ndarray:
    if not isinstance(res2, dict):
        return image
    tiles = res2.get("tiles")
    if not isinstance(tiles, list):
        return image
    cols = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    if len(tiles) != cols * rows:
        return image
    out = image.copy()
    idx = 0
    for ty in range(rows):
        for tx in range(cols):
            atoms = tiles[idx]
            idx += 1
            if not isinstance(atoms, list):
                continue
            x0 = tx * tile_size
            y0 = ty * tile_size
            for atom in atoms:
                if not isinstance(atom, list) or len(atom) != 4:
                    continue
                pix, dr, dg, db = atom
                if not isinstance(pix, int):
                    continue
                lx = pix % tile_size
                ly = pix // tile_size
                x = x0 + lx
                y = y0 + ly
                if x >= width or y >= height:
                    continue
                delta = np.array([dr, dg, db], dtype=np.float32) / 255.0
                out[y, x, :] = np.clip(out[y, x, :] + delta, 0.0, 1.0)
    return out


def _apply_lighting(image: np.ndarray, weft, target_h: int, target_w: int) -> np.ndarray:
    """Phase 2 #17: multiply the decoded albedo by the upsampled lighting grid.

    No-op when the bitstream has no BLOCK_LITE (the typical case for
    most encodes). When BLOCK_LITE is present, the primitive renderer
    produced an albedo image; we unpack the low-res lighting grid,
    bilinearly upsample to the target output dimensions, multiply
    pixel-wise, and clamp to ``[0, 1]``.
    """
    if weft.lite_payload is None:
        return image
    from .bitstream import unpack_lite
    from .intrinsic import upsample_lighting
    grid = unpack_lite(weft.lite_payload)
    lighting = upsample_lighting(grid, target_h, target_w)
    return np.clip(image * lighting, 0.0, 1.0)


def _decode_gradient(weft, target_w: int, target_h: int) -> np.ndarray:
    """Brainstorm #1: render a gradient-field bitstream via DCT Poisson solve.

    The bitstream carries an empty PRIM/TOC plus a BLOCK_GRD payload of
    quantized gradients + per-channel means. Decoder is one
    ``gradient_field.decode`` call (closed-form linear solve, O(N log N)
    per channel via DCT-II), then bilinear-upsamples to the target dims
    if encode_scale<1 was used during encoding.
    """
    if weft.grd_payload is None:
        raise DecodeError("GRD decode requires a BLOCK_GRD payload")
    from .bitstream import unpack_grd
    from .gradient_field import decode as grd_decode
    gx_q, gy_q, means, scale = unpack_grd(weft.grd_payload)
    image = grd_decode(gx_q, gy_q, means, scale=scale)
    encoded_h, encoded_w = gx_q.shape[:2]
    if (target_w, target_h) != (encoded_w, encoded_h):
        from PIL import Image as PILImage
        u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize((target_w, target_h), PILImage.BILINEAR)
        image = np.asarray(upscaled, dtype=np.float32) / 255.0
    return image


def _decode_palette(weft, target_w: int, target_h: int) -> np.ndarray:
    """Render a palette+labels bitstream.

    The bitstream carries an empty PRIM/TOC plus a BLOCK_PAL payload of
    (palette, labels). Decoder is one ``palette[labels]`` lookup at the
    encoded resolution.

    Scale-independent path: when the target resolution differs from
    the encoded label grid, each target pixel nearest-neighbor samples
    the label grid and then looks up the palette color. Palette
    content is typically hard-edged (pixel art, text, screenshots,
    diagrams), so nearest-neighbor is the semantically correct
    upscale: a 2× output has each source label painted onto a 2×2
    block of target pixels, preserving every color boundary crisply
    with zero interpolation blur. This is equivalent to an analytic
    "render the palette quad-tree at any resolution".
    """
    if weft.pal_payload is None:
        raise DecodeError("PAL decode requires a BLOCK_PAL payload")
    from .bitstream import unpack_pal
    from .palette import render_palette
    palette, labels = unpack_pal(weft.pal_payload)
    encoded_h, encoded_w = labels.shape
    if (target_w, target_h) == (encoded_w, encoded_h):
        return render_palette(palette, labels)
    # Nearest-neighbor resample the label grid, then look up the
    # palette once at the target resolution. Avoids materializing the
    # source-resolution image and then resampling it.
    ys = (np.arange(target_h, dtype=np.int64) * encoded_h // max(1, target_h))
    xs = (np.arange(target_w, dtype=np.int64) * encoded_w // max(1, target_w))
    upsampled_labels = labels[np.clip(ys, 0, encoded_h - 1)[:, None],
                              np.clip(xs, 0, encoded_w - 1)[None, :]]
    return render_palette(palette, upsampled_labels)


def _decode_bicubic(weft, target_w: int, target_h: int) -> np.ndarray:
    """Render a bicubic-patch bitstream.

    The bitstream carries an empty PRIM/TOC plus a BLOCK_BIC payload of
    per-tile control grids and a QTREE block of tile layout. Bicubic
    patches are analytic in normalized control-point space, so we
    evaluate them directly at the target resolution — no bilinear
    upscale, crisp at any scale factor.
    """
    if weft.bic_payload is None or weft.qtree_payload is None:
        raise DecodeError("BIC decode requires both BIC and QTREE blocks")
    from .bicubic import render_image as bicubic_render
    from .bitstream import unpack_bic
    encoded_w = int(weft.meta.get("encode_width", weft.head.width))
    encoded_h = int(weft.meta.get("encode_height", weft.head.height))
    quad_tiles = unpack_qtree(weft.qtree_payload)
    control_grids = unpack_bic(weft.bic_payload)
    image = bicubic_render(
        control_grids, quad_tiles,
        width=encoded_w, height=encoded_h,
        target_width=target_w, target_height=target_h,
    )
    return image


def _apply_decode_refinement(image: np.ndarray, meta: dict) -> np.ndarray:
    ff = dict(meta.get("feature_flags") or {})
    level = str(ff.get("decode_refinement_level", "off"))
    if level not in {"mid", "max"}:
        return image
    # Lightweight unsharp mask in linear RGB, deterministic and GPU-friendly to port.
    strength = 0.18 if level == "mid" else 0.28
    out = image.copy()
    # 5-point blur proxy
    center = image
    up = np.vstack([image[:1, :, :], image[:-1, :, :]])
    down = np.vstack([image[1:, :, :], image[-1:, :, :]])
    left = np.hstack([image[:, :1, :], image[:, :-1, :]])
    right = np.hstack([image[:, 1:, :], image[:, -1:, :]])
    blur = (center + up + down + left + right) / 5.0
    out = np.clip(center + (center - blur) * strength, 0.0, 1.0)
    return out


def _decode_prim_payload(
    weft,
) -> tuple[bytes, str]:
    if not weft.chunk_index:
        raise DecodeError("GPU-only decode requires chunked PRIM payload (CIDX)")
    gpu_stack = detect_gpu_stack()
    if not gpu_stack.cuda.available:
        raise DecodeError("GPU-only decode requires CUDA stack for PRIM entropy decode")
    try:
        res = decode_prim_payload_chunked_gpu(
            prim_payload=weft.prim_payload,
            toc=weft.toc,
            chunk_index=weft.chunk_index,
        )
        if weft.toc and weft.toc[-1] != len(res.raw):
            raise DecodeError("TOC final offset mismatch after GPU PRIM decode")
        return res.raw, f"gpu:{res.backend}"
    except GpuEntropyError as exc:
        raise DecodeError(f"GPU chunk entropy decode failed: {exc}") from exc


def _render_primitive_stack_gpu(weft, target_w: int, target_h: int) -> tuple[np.ndarray, str]:
    """GPU primitive-stack renderer (film projector hypothesis Phase 2).

    Mirrors ``_render_primitive_stack_cpu`` but dispatches the per-tile
    primitive walk to the CUDA ``decode_tile_pixels`` kernel via
    ``gpu_render.gpu_render_tiles_to_image``. The kernel runs one CUDA
    block per tile, with threads in the block striding over the tile's
    target pixel rect — so primitive-rich and primitive-sparse tiles
    can coexist in a single launch.

    Falls back to ``_render_primitive_stack_cpu`` (returning its result)
    when the bitstream uses RES1 raster residuals (not yet ported to
    the GPU descriptor) or when the GPU stack is unavailable. The DCT
    residual layer is applied on the CPU on top of the GPU output, the
    same way as the CPU path does, to keep this an isolated swap of
    just the primitive walk.

    Returns ``(image, prim_decode_backend)``.
    """
    from .gpu_render import (
        _PRIM_DTYPE,
        _TILE_DECODE_DTYPE,
        _pack_prims,
        gpu_render_tiles_to_image,
    )

    encoded_w = int(weft.meta.get("encode_width", weft.head.width))
    encoded_h = int(weft.meta.get("encode_height", weft.head.height))

    prim_raw, prim_decode_backend = _decode_prim_payload(weft)

    tiles = decode_tiles(prim_raw, weft.toc)

    residuals = decode_residuals(weft.res0_payload)
    if residuals is not None:
        if len(residuals) != len(tiles):
            raise DecodeError(
                f"RES0 tile count mismatch: {len(residuals)} residuals vs {len(tiles)} tiles"
            )
        for tile, res in zip(tiles, residuals):
            tile.residual_rgb = res

    res1 = _decode_res1_maps_to_float(weft.res1_payload)
    if res1 is not None:
        # RES1 raster residuals aren't in the GPU descriptor yet — fall
        # back to the CPU path so we still produce a correct image.
        return _render_primitive_stack_cpu(weft, target_w, target_h)

    if weft.qtree_payload is not None:
        quad_tiles = unpack_qtree(weft.qtree_payload)
    else:
        ts = weft.head.tile_size
        quad_tiles = []
        idx = 0
        for ty in range(weft.head.tile_rows):
            for tx in range(weft.head.tile_cols):
                quad_tiles.append(QuadTile(x=tx * ts, y=ty * ts, size=ts, index=idx))
                idx += 1

    has_dct_residual = weft.dct_payload is not None
    scaled_path = (
        (target_w, target_h) != (encoded_w, encoded_h)
        and not has_dct_residual
    )

    if scaled_path:
        render_w, render_h = target_w, target_h
        scale_x = target_w / float(encoded_w)
        scale_y = target_h / float(encoded_h)
    else:
        render_w, render_h = encoded_w, encoded_h
        scale_x = scale_y = 1.0

    # Build per-tile descriptors and a flat primitive array. The order
    # of (record, quad_tile) iteration must match the CPU's
    # render_scene_adaptive so RES0/per-tile DCT alignment is preserved.
    n_tiles = len(tiles)
    descs = np.zeros(n_tiles, dtype=_TILE_DECODE_DTYPE)
    packed_lists: list[np.ndarray] = []
    prim_offset = 0
    for i, (rec, qt) in enumerate(zip(tiles, quad_tiles)):
        if scaled_path:
            tx0 = int(round(qt.x * scale_x))
            ty0 = int(round(qt.y * scale_y))
            tx1 = int(round((qt.x + qt.size) * scale_x))
            ty1 = int(round((qt.y + qt.size) * scale_y))
            ts_x = max(1, tx1 - tx0)
            ts_y = max(1, ty1 - ty0)
        else:
            tx0, ty0 = qt.x, qt.y
            ts_x = ts_y = qt.size

        descs[i]["prim_offset"] = prim_offset
        descs[i]["prim_count"] = len(rec.primitives)
        descs[i]["target_x0"] = tx0
        descs[i]["target_y0"] = ty0
        descs[i]["target_w"] = ts_x
        descs[i]["target_h"] = ts_y
        descs[i]["source_size"] = qt.size
        rr, rg, rb = rec.residual_rgb
        descs[i]["res0_r"] = rr / 255.0
        descs[i]["res0_g"] = rg / 255.0
        descs[i]["res0_b"] = rb / 255.0

        if rec.primitives:
            packed_lists.append(_pack_prims(rec.primitives))
            prim_offset += len(rec.primitives)
        # Empty primitive lists contribute nothing to the array; the
        # kernel reads zero primitives so the offset never dereferences.

    if packed_lists:
        all_prims = np.concatenate(packed_lists)
    else:
        all_prims = np.zeros(0, dtype=_PRIM_DTYPE)

    image = gpu_render_tiles_to_image(
        descs, all_prims,
        output_height=render_h, output_width=render_w,
    )
    if image is None:
        # GPU stack unavailable — fall back to CPU.
        return _render_primitive_stack_cpu(weft, target_w, target_h)

    if weft.dct_payload is not None:
        from .bitstream import unpack_dct
        from .dct_residual import (
            apply_residual_to_image,
            permute_band_to_tile,
            _bitmask_to_present_indices,
            decode_tile_scale_u8,
        )
        (
            coeffs, quant_step, channels, freq_alpha, chroma_mode,
            presence_bitmask, n_tiles_hdr, layout, per_tile_scales_u8,
        ) = unpack_dct(weft.dct_payload)
        if layout == 1:
            if n_tiles_hdr > 0 and len(presence_bitmask) > 0:
                present = _bitmask_to_present_indices(presence_bitmask, n_tiles_hdr)
            else:
                present = [True] * len(quad_tiles)
            present_sizes = [int(qt.size) for qt, p in zip(quad_tiles, present) if p]
            coeffs = permute_band_to_tile(coeffs, present_sizes, chroma_mode)
        per_tile_scales: list[float] | None = None
        if per_tile_scales_u8:
            per_tile_scales = [decode_tile_scale_u8(b) for b in per_tile_scales_u8]
        image = apply_residual_to_image(
            image, coeffs, quad_tiles,
            quant_step=quant_step, channels=channels,
            freq_alpha=freq_alpha, chroma_mode=chroma_mode,
            presence_bitmask=presence_bitmask,
            per_tile_scales=per_tile_scales,
        )

    if (target_w, target_h) != (encoded_w, encoded_h) and not scaled_path:
        from PIL import Image as PILImage
        u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize((target_w, target_h), PILImage.BILINEAR)
        image = np.asarray(upscaled, dtype=np.float32) / 255.0

    return image, f"{prim_decode_backend}+gpu-render"


def _render_primitive_stack_cpu(weft, target_w: int, target_h: int) -> tuple[np.ndarray, str]:
    """CPU primitive-stack renderer.

    Decodes the chunked PRIM payload via the GPU entropy kernels,
    reconstructs ``TileRecord`` objects, attaches RES0/RES1 residuals,
    and renders via ``render.render_scene_adaptive`` — the same
    function the encoder uses for its self-consistency PSNR check.
    Bilinear-upsamples to target dims if ``encode_scale < 1`` was used.

    Returns ``(image, prim_decode_backend)``.
    """
    encoded_w = int(weft.meta.get("encode_width", weft.head.width))
    encoded_h = int(weft.meta.get("encode_height", weft.head.height))

    prim_raw, prim_decode_backend = _decode_prim_payload(weft)

    tiles = decode_tiles(prim_raw, weft.toc)

    residuals = decode_residuals(weft.res0_payload)
    if residuals is not None:
        if len(residuals) != len(tiles):
            raise DecodeError(
                f"RES0 tile count mismatch: {len(residuals)} residuals vs {len(tiles)} tiles"
            )
        for tile, res in zip(tiles, residuals):
            tile.residual_rgb = res

    res1 = _decode_res1_maps_to_float(weft.res1_payload)
    res1_grid_size = res1[0] if res1 is not None else 4
    res1_maps = res1[1] if res1 is not None else None

    # Adaptive-quadtree bitstreams carry per-tile (x, y, size) in QTREE.
    # Legacy uniform-grid bitstreams (no QTREE block) get a synthesized
    # uniform tile list from head.tile_cols/tile_rows/tile_size.
    if weft.qtree_payload is not None:
        quad_tiles = unpack_qtree(weft.qtree_payload)
    else:
        ts = weft.head.tile_size
        quad_tiles = []
        idx = 0
        for ty in range(weft.head.tile_rows):
            for tx in range(weft.head.tile_cols):
                quad_tiles.append(QuadTile(x=tx * ts, y=ty * ts, size=ts, index=idx))
                idx += 1

    # Scale-independent rendering: when the caller asked for a target
    # resolution different from the encoded grid, pass it to
    # render_scene_adaptive which samples the analytic primitives at
    # the target resolution directly (no post-hoc bilinear upscale).
    # The DCT residual layer below is raster so it still applies at
    # the encoded resolution; if a DCT residual is present we render
    # at encoded dims and bilinear-upscale at the end. Pure primitive
    # / bicubic bitstreams get crisp scaling for free.
    has_dct_residual = weft.dct_payload is not None
    scaled_path = (
        (target_w, target_h) != (encoded_w, encoded_h)
        and not has_dct_residual
    )
    if scaled_path:
        image = render_scene_adaptive(
            records=tiles,
            quad_tiles=quad_tiles,
            width=encoded_w,
            height=encoded_h,
            residual_maps=res1_maps,
            res1_grid_size=res1_grid_size,
            target_width=target_w,
            target_height=target_h,
        )
    else:
        image = render_scene_adaptive(
            records=tiles,
            quad_tiles=quad_tiles,
            width=encoded_w,
            height=encoded_h,
            residual_maps=res1_maps,
            res1_grid_size=res1_grid_size,
        )

    # DCT residual layer (brainstorm #16). When the bitstream carries a
    # BLOCK_DCT, unpack the per-tile quantized coefficients, IDCT them,
    # and add the result on top of the primitive reconstruction. This
    # is the "JPEG-like residual" path that closes the natural-photo
    # PSNR gap vs DCT-based codecs.
    if weft.dct_payload is not None:
        from .bitstream import unpack_dct
        from .dct_residual import (
            apply_residual_to_image,
            permute_band_to_tile,
            _bitmask_to_present_indices,
            decode_tile_scale_u8,
        )
        (
            coeffs, quant_step, channels, freq_alpha, chroma_mode,
            presence_bitmask, n_tiles_hdr, layout, per_tile_scales_u8,
        ) = unpack_dct(weft.dct_payload)
        if layout == 1:
            # Band-major zigzag: un-permute back to tile-major so
            # apply_residual_to_image can walk tiles linearly.
            if n_tiles_hdr > 0 and len(presence_bitmask) > 0:
                present = _bitmask_to_present_indices(presence_bitmask, n_tiles_hdr)
            else:
                present = [True] * len(quad_tiles)
            present_sizes = [int(qt.size) for qt, p in zip(quad_tiles, present) if p]
            coeffs = permute_band_to_tile(coeffs, present_sizes, chroma_mode)
        per_tile_scales: list[float] | None = None
        if per_tile_scales_u8:
            per_tile_scales = [decode_tile_scale_u8(b) for b in per_tile_scales_u8]
        image = apply_residual_to_image(
            image, coeffs, quad_tiles,
            quant_step=quant_step, channels=channels,
            freq_alpha=freq_alpha, chroma_mode=chroma_mode,
            presence_bitmask=presence_bitmask,
            per_tile_scales=per_tile_scales,
        )

    # Final resize only runs on the raster path (DCT residual present
    # or other raster layers): render_scene_adaptive already rendered
    # at the target resolution on the scaled_path branch, so skipping
    # this avoids an unnecessary bilinear pass that would blur the
    # crisp primitive output.
    if (target_w, target_h) != (encoded_w, encoded_h) and not scaled_path:
        from PIL import Image as PILImage
        u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        upscaled = PILImage.fromarray(u8).resize((target_w, target_h), PILImage.BILINEAR)
        image = np.asarray(upscaled, dtype=np.float32) / 255.0

    return image, prim_decode_backend


def decode_image(
    input_path: str,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    gpu_only: bool = True,
    allow_cpu_fallback: bool = False,
    require_gpu_entropy: bool = True,
) -> DecodeReport:
    if not gpu_only:
        raise DecodeError("CPU decode paths are unsupported; set gpu_only=True")
    if allow_cpu_fallback:
        raise DecodeError("CPU fallback is unsupported in this GPU-only build")
    if not require_gpu_entropy:
        raise DecodeError("GPU-only decode requires GPU entropy decode")

    with open(input_path, "rb") as f:
        blob = f.read()

    weft = decode_weft(blob)
    head = weft.head

    # User-facing dimensions: the HeadBlock stores the ORIGINAL source
    # dimensions (what the user started with). For bitstreams that were
    # encoded with encode_scale<1, the primitive coordinates live in a
    # different (smaller) encoded space; we look that up from meta below.
    source_w = head.width
    source_h = head.height

    # Encoded tile-grid dimensions: where the primitives actually live.
    # New bitstreams record these in meta. Old/baseline bitstreams have
    # source == encoded, so falling back to head.width/height is correct.
    encoded_w = int(weft.meta.get("encode_width", source_w))
    encoded_h = int(weft.meta.get("encode_height", source_h))

    target_w = int(width) if width is not None else source_w
    target_h = int(height) if height is not None else source_h
    if target_w <= 0 or target_h <= 0:
        raise DecodeError("output dimensions must be > 0")

    # Alt-basis bitstreams (brainstorm #1, #11, #20).
    if (
        weft.bic_payload is not None
        or weft.pal_payload is not None
        or weft.grd_payload is not None
    ):
        if weft.bic_payload is not None:
            image = _decode_bicubic(weft, target_w, target_h)
            backend = "bicubic-cpu"
        elif weft.pal_payload is not None:
            image = _decode_palette(weft, target_w, target_h)
            backend = "palette-cpu"
        else:
            image = _decode_gradient(weft, target_w, target_h)
            backend = "gradient-cpu"
        save_image_linear(output_path, image)
        upload_plan = build_device_upload_plan(weft, file_size=len(blob))
        return DecodeReport(
            input_path=input_path,
            output_path=output_path,
            source_width=source_w,
            source_height=source_h,
            output_width=target_w,
            output_height=target_h,
            bytes_read=os.path.getsize(input_path),
            decode_hash=decode_hash(image),
            used_upscaling=(target_w != source_w or target_h != source_h),
            metadata={
                "head": asdict(head),
                "meta": weft.meta,
                "decode_backend": backend,
                "gpu_upload_plan": {
                    "total_bytes": upload_plan.total_bytes,
                    "one_shot_upload": upload_plan.one_shot_upload,
                    "block_count": len(upload_plan.block_views),
                },
                "strict_gpu_only": gpu_only,
                "require_gpu_entropy": require_gpu_entropy,
            },
        )

    image, prim_decode_backend = _render_primitive_stack_cpu(weft, target_w, target_h)

    # RES2 sparse application assumes uniform tiles and no upscaling.
    # Adaptive-quadtree bitstreams skip it.
    if (
        target_w == source_w
        and target_h == source_h
        and weft.qtree_payload is None
    ):
        res2 = _decode_res2_sparse_payload(weft.res2_payload, weft.meta)
        image = _apply_res2_sparse(image, res2, width=source_w, height=source_h, tile_size=head.tile_size)
    image = _apply_lighting(image, weft, target_h, target_w)
    image = _apply_decode_refinement(image, weft.meta)

    save_image_linear(output_path, image)
    upload_plan = build_device_upload_plan(weft, file_size=len(blob))

    return DecodeReport(
        input_path=input_path,
        output_path=output_path,
        source_width=source_w,
        source_height=source_h,
        output_width=target_w,
        output_height=target_h,
        bytes_read=os.path.getsize(input_path),
        decode_hash=decode_hash(image),
        used_upscaling=(target_w != source_w or target_h != source_h),
        metadata={
            "head": asdict(head),
            "meta": weft.meta,
            "decode_backend": "primitive-stack-cpu",
            "gpu_stack": gpu_stack_dict(),
            "prim_decode_backend": prim_decode_backend,
            "gpu_upload_plan": {
                "total_bytes": upload_plan.total_bytes,
                "one_shot_upload": upload_plan.one_shot_upload,
                "block_count": len(upload_plan.block_views),
                "chunk_count": len(upload_plan.prim_chunks),
                "est_host_device_transfers": 1 if upload_plan.one_shot_upload else max(1, len(upload_plan.block_views)),
            },
            "strict_gpu_only": gpu_only,
            "require_gpu_entropy": require_gpu_entropy,
        },
    )


def decode_to_array(input_path: str, width: int | None = None, height: int | None = None):
    with open(input_path, "rb") as f:
        blob = f.read()
    weft = decode_weft(blob)
    head = weft.head

    source_w = head.width
    source_h = head.height
    encoded_w = int(weft.meta.get("encode_width", source_w))
    encoded_h = int(weft.meta.get("encode_height", source_h))

    target_w = int(width) if width is not None else source_w
    target_h = int(height) if height is not None else source_h

    # Alt-basis bitstreams (brainstorm #1/#11/#20) bypass the primitive renderer.
    if weft.bic_payload is not None:
        return _decode_bicubic(weft, target_w, target_h)
    if weft.pal_payload is not None:
        return _decode_palette(weft, target_w, target_h)
    if weft.grd_payload is not None:
        return _decode_gradient(weft, target_w, target_h)

    image, _ = _render_primitive_stack_cpu(weft, target_w, target_h)
    if (
        target_w == source_w
        and target_h == source_h
        and weft.qtree_payload is None
    ):
        res2 = _decode_res2_sparse_payload(weft.res2_payload, weft.meta)
        image = _apply_res2_sparse(image, res2, width=source_w, height=source_h, tile_size=head.tile_size)
    image = _apply_lighting(image, weft, target_h, target_w)
    return _apply_decode_refinement(image, weft.meta)
