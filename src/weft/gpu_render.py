"""GPU-accelerated batch tile rendering and objective evaluation.

Replaces the CPU render_tile + _tile_objective hot loop with a single
CUDA kernel that evaluates multiple (tile, primitive_set) pairs in
parallel. This is the core speedup for the encode refinement loop.

Architecture:
  - One thread block per evaluation (tile + candidate combo)
  - Threads cooperatively render the tile and reduce to MSE
  - Results returned as a flat float32 array
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .primitives import Primitive


_BATCH_RENDER_KERNEL = r"""
#define MAX_PRIMS 96
#define MAX_GEOM 7

struct PrimData {
    int   kind;
    float geom[MAX_GEOM];
    float color0[3];
    float color1[3];
    float alpha;
};

/* Render one tile (stack of primitives) at a pixel and return composited RGB. */
__device__ void render_pixel(
    const PrimData* prims, int n_prims,
    float fx, float fy,
    float& cr, float& cg, float& cb)
{
    cr = 0.0f; cg = 0.0f; cb = 0.0f;
    for (int i = 0; i < n_prims; ++i) {
        const PrimData& P = prims[i];
        float sr, sg, sb, a;
        sr = P.color0[0]; sg = P.color0[1]; sb = P.color0[2]; a = 0.0f;

        if (P.kind == 0) { /* const */
            a = P.alpha;
        } else if (P.kind == 1) { /* linear */
            float vx = P.geom[2]-P.geom[0], vy = P.geom[3]-P.geom[1];
            float den = vx*vx + vy*vy + 1e-8f;
            float t = fminf(fmaxf(((fx-P.geom[0])*vx + (fy-P.geom[1])*vy)/den, 0.0f), 1.0f);
            sr = P.color0[0]*(1.0f-t) + P.color1[0]*t;
            sg = P.color0[1]*(1.0f-t) + P.color1[1]*t;
            sb = P.color0[2]*(1.0f-t) + P.color1[2]*t;
            a = P.alpha;
        } else if (P.kind == 2) { /* line */
            float lx0=P.geom[0],ly0=P.geom[1],lx1=P.geom[2],ly1=P.geom[3];
            float lvx=lx1-lx0, lvy=ly1-ly0;
            float ld = lvx*lvx + lvy*lvy + 1e-8f;
            float lt = fminf(fmaxf(((fx-lx0)*lvx+(fy-ly0)*lvy)/ld, 0.0f), 1.0f);
            float dx=fx-(lx0+lt*lvx), dy=fy-(ly0+lt*lvy);
            float dist = sqrtf(dx*dx+dy*dy);
            float sigma = fmaxf(0.5f, P.geom[4]);
            a = expf(-(dist*dist)/(2.0f*sigma*sigma)) * P.alpha;
        } else if (P.kind == 3) { /* curve */
            /* Matches CPU _eval_curve_distance: 32 sample points (np.linspace(0,1,32))
             * connected by 31 line segments. */
            float bd=1e9f, pcx=P.geom[0], pcy=P.geom[1];
            for(int ci=1; ci<=31; ++ci) {
                float ct=(float)ci/31.0f, cu=1.0f-ct;
                float qx=cu*cu*P.geom[0]+2.0f*cu*ct*P.geom[2]+ct*ct*P.geom[4];
                float qy=cu*cu*P.geom[1]+2.0f*cu*ct*P.geom[3]+ct*ct*P.geom[5];
                float svx=qx-pcx, svy=qy-pcy;
                float sd=svx*svx+svy*svy+1e-8f;
                float st=fminf(fmaxf(((fx-pcx)*svx+(fy-pcy)*svy)/sd,0.0f),1.0f);
                float sdx=fx-(pcx+st*svx), sdy=fy-(pcy+st*svy);
                bd=fminf(bd, sqrtf(sdx*sdx+sdy*sdy));
                pcx=qx; pcy=qy;
            }
            float sigma=fmaxf(0.5f, P.geom[6]);
            a = expf(-(bd*bd)/(2.0f*sigma*sigma)) * P.alpha;
        } else if (P.kind == 4) { /* polygon */
            float x0=P.geom[0],y0=P.geom[1],x1=P.geom[2],y1=P.geom[3],x2=P.geom[4],y2=P.geom[5];
            float td=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);
            float mask=0.0f;
            if(fabsf(td)>1e-6f) {
                float ta=((y1-y2)*(fx-x2)+(x2-x1)*(fy-y2))/td;
                float tb=((y2-y0)*(fx-x2)+(x0-x2)*(fy-y2))/td;
                mask=(ta>=0.0f && tb>=0.0f && 1.0f-ta-tb>=0.0f) ? 1.0f : 0.0f;
            }
            a = mask * P.alpha;
        }
        cr = sr*a + cr*(1.0f-a);
        cg = sg*a + cg*(1.0f-a);
        cb = sb*a + cb*(1.0f-a);
    }
}

/* Batch objective: for each evaluation, render tile and compute MSE.
 *
 * evals[i] = {prim_offset, prim_count, tile_data_offset, tile_size}
 * prims[] = flat array of all PrimData across all evals
 * tiles[] = flat array of all source tile pixels (tile_size^2 * 3 floats each)
 * out_mse[i] = MSE for evaluation i
 */
struct EvalDesc {
    int prim_offset;
    int prim_count;
    int tile_offset;   /* offset into tiles[] in floats */
    int tile_size;
};

extern "C" __global__ void batch_tile_objective(
    const EvalDesc* __restrict__ evals,
    int n_evals,
    const PrimData* __restrict__ prims,
    const float* __restrict__ tiles,
    float* __restrict__ out_mse
) {
    int eid = blockIdx.x;
    if (eid >= n_evals) return;
    int tid = threadIdx.x;

    const EvalDesc& ev = evals[eid];
    const PrimData* my_prims = prims + ev.prim_offset;
    const float* my_tile = tiles + ev.tile_offset;
    int ts = ev.tile_size;
    int npix = ts * ts;

    __shared__ float smem[256];
    smem[tid] = 0.0f;

    for (int px = tid; px < npix; px += blockDim.x) {
        float fx = (float)(px % ts);
        float fy = (float)(px / ts);
        float cr, cg, cb;
        render_pixel(my_prims, ev.prim_count, fx, fy, cr, cg, cb);

        int pidx = px * 3;
        float dr = cr - my_tile[pidx];
        float dg = cg - my_tile[pidx + 1];
        float db = cb - my_tile[pidx + 2];
        smem[tid] += dr*dr + dg*dg + db*db;
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out_mse[eid] = smem[0] / (float)(npix * 3);
}


/* ─────────────────────────────────────────────────────────────────
 * Decode kernel: render primitive-stack tiles into a full-image
 * float32 RGB output buffer at the target resolution.
 *
 * Uses the same render_pixel device function as the encoder R-D
 * scoring kernel above, so the per-pixel composite math is bit-
 * equivalent. Each tile gets one CUDA block; the threads in that
 * block stride-loop over the tile's output pixels.
 *
 * Scale-independent rendering: each tile carries its own
 * source_size (the primitive coordinate space, e.g. 16) and
 * target_w/target_h (the rendered pixel footprint at the user's
 * requested output resolution). When source_size != target_w/h,
 * the kernel samples primitive coords as
 *
 *     fx = (out_x + 0.5) * source_size / target_w
 *     fy = (out_y + 0.5) * source_size / target_h
 *
 * which matches the CPU `_scaled_grid` helper exactly.
 *
 * RES0 bias (per-tile color offset, signed int8 / 255 in [-0.5, 0.5])
 * is added after the primitive composite, mirroring the CPU
 * `render_tile(residual_rgb=...)` path.
 * ──────────────────────────────────────────────────────────────── */
struct TileDecodeDesc {
    int prim_offset;     /* index into the flat PrimData array */
    int prim_count;      /* primitives in this tile */
    int target_x0;       /* top-left of this tile in the output image */
    int target_y0;
    int target_w;        /* rendered pixel footprint of the tile */
    int target_h;
    int source_size;     /* primitive coord space size (encoded tile size) */
    int _pad;            /* keep struct 8-byte aligned */
    float res0_r;        /* RES0 bias added after primitive composite */
    float res0_g;
    float res0_b;
    float _pad2;
};

extern "C" __global__ void decode_tile_pixels(
    const TileDecodeDesc* __restrict__ tiles,
    int n_tiles,
    const PrimData* __restrict__ prims,
    float* __restrict__ output,
    int output_width,
    int output_height
) {
    int tile_id = blockIdx.x;
    if (tile_id >= n_tiles) return;
    int tid = threadIdx.x;

    const TileDecodeDesc& td = tiles[tile_id];
    const PrimData* my_prims = prims + td.prim_offset;
    int npix = td.target_w * td.target_h;
    float inv_w = (float)td.source_size / (float)td.target_w;
    float inv_h = (float)td.source_size / (float)td.target_h;

    for (int px = tid; px < npix; px += blockDim.x) {
        int lx = px % td.target_w;
        int ly = px / td.target_w;
        float fx = ((float)lx + 0.5f) * inv_w;
        float fy = ((float)ly + 0.5f) * inv_h;

        float cr, cg, cb;
        render_pixel(my_prims, td.prim_count, fx, fy, cr, cg, cb);

        cr += td.res0_r;
        cg += td.res0_g;
        cb += td.res0_b;
        cr = fminf(fmaxf(cr, 0.0f), 1.0f);
        cg = fminf(fmaxf(cg, 0.0f), 1.0f);
        cb = fminf(fmaxf(cb, 0.0f), 1.0f);

        int gx = td.target_x0 + lx;
        int gy = td.target_y0 + ly;
        if (gx >= 0 && gx < output_width && gy >= 0 && gy < output_height) {
            int oidx = (gy * output_width + gx) * 3;
            output[oidx + 0] = cr;
            output[oidx + 1] = cg;
            output[oidx + 2] = cb;
        }
    }
}
"""


_PRIM_DTYPE = np.dtype([
    ("kind", np.int32),
    ("geom", np.float32, 7),
    ("color0", np.float32, 3),
    ("color1", np.float32, 3),
    ("alpha", np.float32),
])


def _pack_one(p: Primitive, out: np.ndarray, idx: int) -> None:
    """Pack a single primitive into a pre-allocated structured array at index."""
    out[idx]["kind"] = p.kind
    geom = p.geom
    g = out[idx]["geom"]
    ng = len(geom)
    for j in range(min(ng, 7)):
        g[j] = geom[j]
    for j in range(ng, 7):
        g[j] = 0.0
    out[idx]["color0"][0] = p.color0[0]
    out[idx]["color0"][1] = p.color0[1]
    out[idx]["color0"][2] = p.color0[2]
    if p.color1 is not None:
        out[idx]["color1"][0] = p.color1[0]
        out[idx]["color1"][1] = p.color1[1]
        out[idx]["color1"][2] = p.color1[2]
    else:
        out[idx]["color1"][:] = 0.0
    out[idx]["alpha"] = p.alpha


def _pack_prims(prims: list[Primitive]) -> np.ndarray:
    """Pack primitives into PrimData structured array — vectorized.

    The previous implementation iterated structured-array fields per
    primitive (``arr[i]["color0"][0] = ...``), which is slow because
    each access materializes a view. This version batches all field
    assignments via plain ``np.array(list_comprehension)`` calls, which
    push the per-element work into the numpy C path.
    """
    n_prims = len(prims)
    n = max(1, n_prims)
    arr = np.zeros(n, dtype=_PRIM_DTYPE)
    if not prims:
        return arr

    # kind / alpha are scalar columns — np.fromiter is the fastest
    # iterator-to-1d-array path.
    arr["kind"][:n_prims] = np.fromiter((p.kind for p in prims), dtype=np.int32, count=n_prims)
    arr["alpha"][:n_prims] = np.fromiter((p.alpha for p in prims), dtype=np.float32, count=n_prims)

    # color0 is always present and always (3,). Build via list comp +
    # np.array, then assign to the whole field in one shot.
    arr["color0"][:n_prims] = np.array([p.color0 for p in prims], dtype=np.float32)

    # color1 is optional — substitute zeros for None.
    arr["color1"][:n_prims] = np.array(
        [p.color1 if p.color1 is not None else (0.0, 0.0, 0.0) for p in prims],
        dtype=np.float32,
    )

    # geom is variable length per-kind; right-pad to 7 floats for the
    # PrimGPU layout. The pad-with-tuple trick keeps it in pure Python.
    arr["geom"][:n_prims] = np.array(
        [p.geom + (0.0,) * (7 - len(p.geom)) for p in prims],
        dtype=np.float32,
    )
    return arr


def gpu_batch_objectives_multi_tile(
    tiles: list[np.ndarray],
    tile_indices: list[int],
    prim_sets_per_tile: list[list[list[Primitive]]],
) -> list[np.ndarray] | None:
    """Score candidates across MULTIPLE tiles in ONE kernel launch.

    Args:
        tiles: list of tile pixel arrays (can be different sizes)
        tile_indices: which tiles have work (indices into tiles)
        prim_sets_per_tile: for each tile_index, list of primitive sets to score

    Returns:
        list of MSE arrays (one per tile_index), or None if GPU unavailable.
    """
    # Flatten all evals across all tiles into one batch.
    total_evals = sum(len(prim_sets_per_tile[j]) for j in range(len(tile_indices)))
    if total_evals == 0:
        return [np.empty(0, dtype=np.float32) for _ in tile_indices]

    try:
        state = _get_render_state()
    except Exception:
        return None

    driver = state["driver"]
    stream = state["stream"]
    kernel = state["kernel"]
    from .gpu_encoder import _driver_check

    eval_dtype = np.dtype([
        ("prim_offset", np.int32),
        ("prim_count", np.int32),
        ("tile_offset", np.int32),
        ("tile_size", np.int32),
    ])

    # Pack all tile pixels into one buffer.
    tile_pixel_offsets: dict[int, int] = {}
    pixel_parts: list[np.ndarray] = []
    running_pixels = 0
    for j, ti in enumerate(tile_indices):
        if ti not in tile_pixel_offsets:
            flat = tiles[ti].astype(np.float32, order="C").ravel()
            tile_pixel_offsets[ti] = running_pixels
            pixel_parts.append(flat)
            running_pixels += len(flat)
    all_pixels = np.concatenate(pixel_parts) if pixel_parts else np.zeros(1, dtype=np.float32)

    # Pack all primitive sets and build eval descriptors.
    evals = np.zeros(total_evals, dtype=eval_dtype)
    all_prim_parts: list[np.ndarray] = []
    running_prims = 0
    eval_idx = 0
    # Track which evals belong to which tile for splitting results.
    tile_eval_ranges: list[tuple[int, int]] = []  # (start, count) per tile_index

    for j, ti in enumerate(tile_indices):
        sets = prim_sets_per_tile[j]
        start_eval = eval_idx
        tile_size = tiles[ti].shape[0]
        tile_off = tile_pixel_offsets[ti]

        if sets and len(set(len(s) for s in sets)) == 1 and len(sets[0]) > 0:
            # All same length — use shared-base optimization.
            n_prims = len(sets[0])
            base = _pack_prims(sets[0])
            for k, s in enumerate(sets):
                total_idx = running_prims + k * n_prims
                evals[eval_idx]["prim_offset"] = total_idx
                evals[eval_idx]["prim_count"] = n_prims
                evals[eval_idx]["tile_offset"] = tile_off
                evals[eval_idx]["tile_size"] = tile_size
                eval_idx += 1
            # Build packed array with base copied, patches applied.
            block = np.tile(base, len(sets))
            for k in range(1, len(sets)):
                for p in range(n_prims):
                    if sets[k][p] is not sets[0][p]:
                        _pack_one(sets[k][p], block, k * n_prims + p)
            all_prim_parts.append(block)
            running_prims += len(sets) * n_prims
        else:
            for s in sets:
                packed = _pack_prims(s)
                evals[eval_idx]["prim_offset"] = running_prims
                evals[eval_idx]["prim_count"] = len(s)
                evals[eval_idx]["tile_offset"] = tile_off
                evals[eval_idx]["tile_size"] = tile_size
                eval_idx += 1
                all_prim_parts.append(packed)
                running_prims += len(s) if s else 1

        tile_eval_ranges.append((start_eval, eval_idx - start_eval))

    all_prims_arr = np.concatenate(all_prim_parts) if all_prim_parts else _pack_prims([])
    h_mse = np.zeros(total_evals, dtype=np.float32)

    try:
        def alloc(nbytes):
            ptr = driver.cuMemAlloc(max(4, nbytes))
            return ptr[1] if isinstance(ptr, tuple) and len(ptr) > 1 else ptr

        d_evals = alloc(evals.nbytes)
        d_prims = alloc(all_prims_arr.nbytes)
        d_tile = alloc(all_pixels.nbytes)
        d_mse = alloc(h_mse.nbytes)

        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_evals, evals.ctypes.data, evals.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_prims, all_prims_arr.ctypes.data, all_prims_arr.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_tile, all_pixels.ctypes.data, all_pixels.nbytes, stream))

        arg_evals = np.array([int(d_evals)], dtype=np.uint64)
        arg_n = np.array([total_evals], dtype=np.int32)
        arg_prims = np.array([int(d_prims)], dtype=np.uint64)
        arg_tile = np.array([int(d_tile)], dtype=np.uint64)
        arg_mse = np.array([int(d_mse)], dtype=np.uint64)

        args = np.array([
            arg_evals.ctypes.data, arg_n.ctypes.data,
            arg_prims.ctypes.data, arg_tile.ctypes.data, arg_mse.ctypes.data,
        ], dtype=np.uint64)

        _driver_check(driver, driver.cuLaunchKernel(
            kernel, total_evals, 1, 1, 256, 1, 1, 0, stream, args.ctypes.data, 0))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_mse.ctypes.data, d_mse, h_mse.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        for ptr in (d_evals, d_prims, d_tile, d_mse):
            try:
                driver.cuMemFree(ptr)
            except Exception:
                pass

        # Split results back per tile.
        results: list[np.ndarray] = []
        for start, count in tile_eval_ranges:
            results.append(h_mse[start:start + count].copy())
        return results

    except Exception:
        return None


def gpu_batch_score_prepacked(
    tile_pixels_flat: np.ndarray,
    tile_pixel_offsets: dict[int, int],
    tile_indices: list[int],
    tile_sizes: list[int],
    eval_prim_arrays: list[np.ndarray],
    eval_prim_counts: list[np.ndarray],
) -> list[np.ndarray] | None:
    """Score candidates using pre-packed primitive arrays. Zero per-primitive Python overhead.

    Args:
        tile_pixels_flat: All tile pixels concatenated in one float32 array
        tile_pixel_offsets: {tile_index: offset_in_floats} into tile_pixels_flat
        tile_indices: Which tiles to score
        tile_sizes: tile_size per tile_index
        eval_prim_arrays: Per tile_index, flat packed PrimData array for all evals
        eval_prim_counts: Per tile_index, int array of prim_count per eval
    """
    try:
        state = _get_render_state()
    except Exception:
        return None

    driver = state["driver"]
    stream = state["stream"]
    kernel = state["kernel"]
    from .gpu_encoder import _driver_check

    eval_dtype = np.dtype([
        ("prim_offset", np.int32),
        ("prim_count", np.int32),
        ("tile_offset", np.int32),
        ("tile_size", np.int32),
    ])

    total_evals = sum(len(c) for c in eval_prim_counts)
    if total_evals == 0:
        return [np.empty(0, dtype=np.float32) for _ in tile_indices]

    evals = np.zeros(total_evals, dtype=eval_dtype)
    all_prim_parts: list[np.ndarray] = []
    running_prims = 0
    eval_idx = 0
    tile_eval_ranges: list[tuple[int, int]] = []

    for j, ti in enumerate(tile_indices):
        start_eval = eval_idx
        ts = tile_sizes[j]
        t_off = tile_pixel_offsets[ti]
        prim_arr = eval_prim_arrays[j]
        counts = eval_prim_counts[j]

        off = 0
        for k, nc in enumerate(counts):
            evals[eval_idx]["prim_offset"] = running_prims + off
            evals[eval_idx]["prim_count"] = int(nc)
            evals[eval_idx]["tile_offset"] = t_off
            evals[eval_idx]["tile_size"] = ts
            eval_idx += 1
            off += int(nc)

        all_prim_parts.append(prim_arr)
        running_prims += len(prim_arr)
        tile_eval_ranges.append((start_eval, eval_idx - start_eval))

    all_prims_arr = np.concatenate(all_prim_parts) if all_prim_parts else np.zeros(1, dtype=_PRIM_DTYPE)
    h_mse = np.zeros(total_evals, dtype=np.float32)

    try:
        def alloc(nbytes):
            ptr = driver.cuMemAlloc(max(4, nbytes))
            return ptr[1] if isinstance(ptr, tuple) and len(ptr) > 1 else ptr

        d_evals = alloc(evals.nbytes)
        d_prims = alloc(all_prims_arr.nbytes)
        d_tile = alloc(tile_pixels_flat.nbytes)
        d_mse = alloc(h_mse.nbytes)

        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_evals, evals.ctypes.data, evals.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_prims, all_prims_arr.ctypes.data, all_prims_arr.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_tile, tile_pixels_flat.ctypes.data, tile_pixels_flat.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        arg_evals = np.array([int(d_evals)], dtype=np.uint64)
        arg_n = np.array([total_evals], dtype=np.int32)
        arg_prims = np.array([int(d_prims)], dtype=np.uint64)
        arg_tile = np.array([int(d_tile)], dtype=np.uint64)
        arg_mse = np.array([int(d_mse)], dtype=np.uint64)

        args = np.array([
            arg_evals.ctypes.data, arg_n.ctypes.data,
            arg_prims.ctypes.data, arg_tile.ctypes.data, arg_mse.ctypes.data,
        ], dtype=np.uint64)

        _driver_check(driver, driver.cuLaunchKernel(
            kernel, total_evals, 1, 1, 256, 1, 1, 0, stream, args.ctypes.data, 0))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_mse.ctypes.data, d_mse, h_mse.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        for ptr in (d_evals, d_prims, d_tile, d_mse):
            try:
                driver.cuMemFree(ptr)
            except Exception:
                pass

        results = []
        for start, count in tile_eval_ranges:
            results.append(h_mse[start:start + count].copy())
        return results

    except Exception:
        return None


# ── Decode-side GPU primitive walk ─────────────────────────────────────
#
# The encoder uses ``gpu_batch_score_prepacked`` above to score
# candidate primitive sets via MSE against a target tile (encoder
# R-D path). The decoder needs the OPPOSITE: take the chosen
# primitives and write rendered pixels into an output image. This
# launcher dispatches the ``decode_tile_pixels`` kernel for that —
# one CUDA block per tile, threads stride over output pixels.

_TILE_DECODE_DTYPE = np.dtype([
    ("prim_offset", np.int32),
    ("prim_count", np.int32),
    ("target_x0", np.int32),
    ("target_y0", np.int32),
    ("target_w", np.int32),
    ("target_h", np.int32),
    ("source_size", np.int32),
    ("_pad", np.int32),
    ("res0_r", np.float32),
    ("res0_g", np.float32),
    ("res0_b", np.float32),
    ("_pad2", np.float32),
])


def gpu_render_tiles_to_image(
    tile_descs: np.ndarray,
    all_prims: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray | None:
    """Render a packed-primitive tile list into a full RGB image on the GPU.

    Args:
        tile_descs: structured array of ``_TILE_DECODE_DTYPE`` — one
            entry per tile being rendered. Carries the tile's
            primitive offset/count, target position+size in the
            output image, source coordinate-space size (for
            scale-independent sampling), and RES0 bias.
        all_prims: flat ``_PRIM_DTYPE`` array containing every
            primitive across every tile, indexed by
            ``tile_descs[i]["prim_offset"]``.
        output_height: height of the destination image in pixels.
        output_width: width of the destination image in pixels.

    Returns:
        ``(output_height, output_width, 3)`` float32 RGB image in
        [0, 1], or ``None`` if the GPU stack is unavailable.
    """
    try:
        state = _get_render_state()
    except Exception:
        return None

    driver = state["driver"]
    stream = state["stream"]
    decode_kernel = state["decode_kernel"]
    from .gpu_encoder import _driver_check

    n_tiles = int(tile_descs.shape[0])
    if n_tiles == 0:
        return np.zeros((output_height, output_width, 3), dtype=np.float32)

    # Output is initialized to zero so any pixel that no tile covers
    # stays black. The encoder pads the source up to a multiple of
    # TILE_SIZE_MAX, so the final crop in the caller drops the pad.
    h_output = np.zeros(output_height * output_width * 3, dtype=np.float32)
    descs = np.ascontiguousarray(tile_descs)
    prims = np.ascontiguousarray(all_prims)

    try:
        def alloc(nbytes):
            ptr = driver.cuMemAlloc(max(4, nbytes))
            return ptr[1] if isinstance(ptr, tuple) and len(ptr) > 1 else ptr

        d_descs = alloc(descs.nbytes)
        d_prims = alloc(max(prims.nbytes, 4))
        d_output = alloc(h_output.nbytes)

        _driver_check(driver, driver.cuMemcpyHtoDAsync(
            d_descs, descs.ctypes.data, descs.nbytes, stream))
        if prims.nbytes > 0:
            _driver_check(driver, driver.cuMemcpyHtoDAsync(
                d_prims, prims.ctypes.data, prims.nbytes, stream))
        _driver_check(driver, driver.cuMemsetD8Async(
            d_output, 0, h_output.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        arg_descs = np.array([int(d_descs)], dtype=np.uint64)
        arg_n = np.array([n_tiles], dtype=np.int32)
        arg_prims = np.array([int(d_prims)], dtype=np.uint64)
        arg_output = np.array([int(d_output)], dtype=np.uint64)
        arg_w = np.array([int(output_width)], dtype=np.int32)
        arg_h = np.array([int(output_height)], dtype=np.int32)

        args = np.array([
            arg_descs.ctypes.data, arg_n.ctypes.data,
            arg_prims.ctypes.data, arg_output.ctypes.data,
            arg_w.ctypes.data, arg_h.ctypes.data,
        ], dtype=np.uint64)

        _driver_check(driver, driver.cuLaunchKernel(
            decode_kernel,
            n_tiles, 1, 1,    # grid: one block per tile
            256, 1, 1,        # block: 256 threads stride-loop over pixels
            0, stream,
            args.ctypes.data, 0,
        ))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(
            h_output.ctypes.data, d_output, h_output.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        for ptr in (d_descs, d_prims, d_output):
            try:
                driver.cuMemFree(ptr)
            except Exception:
                pass

        return h_output.reshape(output_height, output_width, 3)

    except Exception:
        return None


_gpu_render_state = None


def _get_render_state():
    """Lazily initialize GPU state for batch rendering."""
    global _gpu_render_state
    if _gpu_render_state is not None:
        return _gpu_render_state

    from .gpu_context import get_cuda_state
    from .gpu_encoder import _driver_check, _nvrtc_check

    cuda = get_cuda_state()
    driver, nvrtc = cuda["driver"], cuda["nvrtc"]

    prog = _nvrtc_check(nvrtc, nvrtc.nvrtcCreateProgram(
        _BATCH_RENDER_KERNEL.encode(), b"batch_render.cu", 0, [], []))
    _nvrtc_check(nvrtc, nvrtc.nvrtcCompileProgram(prog, 1, [b"--std=c++14"]))
    ptx_size = int(_nvrtc_check(nvrtc, nvrtc.nvrtcGetPTXSize(prog)))
    ptx_buf = bytearray(ptx_size)
    try:
        _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog, ptx_buf))
        ptx = bytes(ptx_buf)
    except TypeError:
        ptx = bytes(_nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(prog)))
    _nvrtc_check(nvrtc, nvrtc.nvrtcDestroyProgram(prog))

    try:
        module = _driver_check(driver, driver.cuModuleLoadData(ptx))
    except Exception:
        ptx_arr = np.frombuffer(ptx + b"\x00", dtype=np.uint8).copy()
        module = _driver_check(driver, driver.cuModuleLoadData(ptx_arr.ctypes.data))
    kernel = _driver_check(driver, driver.cuModuleGetFunction(module, b"batch_tile_objective"))
    decode_kernel = _driver_check(driver, driver.cuModuleGetFunction(module, b"decode_tile_pixels"))

    _gpu_render_state = {
        "driver": driver,
        "stream": cuda["stream"],
        "kernel": kernel,
        "decode_kernel": decode_kernel,
    }
    return _gpu_render_state


def gpu_batch_objectives(
    tile: np.ndarray,
    prim_sets: list[list[Primitive]],
) -> np.ndarray | None:
    """Evaluate MSE for multiple primitive sets on the same tile, all on GPU.

    Returns float32 array of shape (len(prim_sets),) or None if GPU unavailable.
    """
    n_evals = len(prim_sets)
    if n_evals == 0:
        return np.empty(0, dtype=np.float32)

    try:
        state = _get_render_state()
    except Exception:
        return None

    driver = state["driver"]
    stream = state["stream"]
    kernel = state["kernel"]
    from .gpu_encoder import _driver_check

    tile_size = tile.shape[0]
    tile_flat = tile.astype(np.float32, order="C").ravel()

    # Optimized packing: detect shared-base pattern (all sets share N-1 prims,
    # only one differs per set). This is the common case in refinement passes.
    eval_dtype = np.dtype([
        ("prim_offset", np.int32),
        ("prim_count", np.int32),
        ("tile_offset", np.int32),
        ("tile_size", np.int32),
    ])
    evals = np.zeros(n_evals, dtype=eval_dtype)

    # Check if all sets have the same length (common in greedy/refine).
    set_lens = [len(s) for s in prim_sets]
    if len(set(set_lens)) == 1 and set_lens[0] > 0:
        n_prims = set_lens[0]
        total = n_evals * n_prims
        all_prims_arr = np.zeros(total, dtype=_PRIM_DTYPE)
        # Pack the first set fully.
        base = _pack_prims(prim_sets[0])
        # Copy base into every slot, then patch the differences.
        for i in range(n_evals):
            start = i * n_prims
            all_prims_arr[start:start + n_prims] = base
            evals[i]["prim_offset"] = start
            evals[i]["prim_count"] = n_prims
            evals[i]["tile_offset"] = 0
            evals[i]["tile_size"] = tile_size
            # Find and patch differing primitives.
            for j in range(n_prims):
                if i > 0 and prim_sets[i][j] is not prim_sets[0][j]:
                    _pack_one(prim_sets[i][j], all_prims_arr, start + j)
    else:
        all_prims_list = []
        prim_offset = 0
        for i, prims in enumerate(prim_sets):
            packed = _pack_prims(prims)
            evals[i]["prim_offset"] = prim_offset
            evals[i]["prim_count"] = len(prims)
            evals[i]["tile_offset"] = 0
            evals[i]["tile_size"] = tile_size
            all_prims_list.append(packed)
            prim_offset += len(prims) if prims else 1
        all_prims_arr = np.concatenate(all_prims_list) if all_prims_list else _pack_prims([])
    h_mse = np.zeros(n_evals, dtype=np.float32)

    try:
        def alloc(nbytes):
            ptr = driver.cuMemAlloc(nbytes)
            return ptr[1] if isinstance(ptr, tuple) and len(ptr) > 1 else ptr

        d_evals = alloc(evals.nbytes)
        d_prims = alloc(all_prims_arr.nbytes)
        d_tile = alloc(tile_flat.nbytes)
        d_mse = alloc(h_mse.nbytes)

        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_evals, evals.ctypes.data, evals.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_prims, all_prims_arr.ctypes.data, all_prims_arr.nbytes, stream))
        _driver_check(driver, driver.cuMemcpyHtoDAsync(d_tile, tile_flat.ctypes.data, tile_flat.nbytes, stream))

        arg_evals = np.array([int(d_evals)], dtype=np.uint64)
        arg_n = np.array([n_evals], dtype=np.int32)
        arg_prims = np.array([int(d_prims)], dtype=np.uint64)
        arg_tile = np.array([int(d_tile)], dtype=np.uint64)
        arg_mse = np.array([int(d_mse)], dtype=np.uint64)

        args = np.array([
            arg_evals.ctypes.data, arg_n.ctypes.data,
            arg_prims.ctypes.data, arg_tile.ctypes.data, arg_mse.ctypes.data,
        ], dtype=np.uint64)

        _driver_check(driver, driver.cuLaunchKernel(
            kernel, n_evals, 1, 1, 256, 1, 1, 0, stream, args.ctypes.data, 0))
        _driver_check(driver, driver.cuMemcpyDtoHAsync(h_mse.ctypes.data, d_mse, h_mse.nbytes, stream))
        _driver_check(driver, driver.cuStreamSynchronize(stream))

        for ptr in (d_evals, d_prims, d_tile, d_mse):
            try:
                driver.cuMemFree(ptr)
            except Exception:
                pass

        return h_mse

    except Exception:
        return None
