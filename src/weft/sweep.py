"""Sweep tool for visualizing WEFT encoder behavior across images × scales × variants.

This is the lightweight, visualization-first counterpart to ``experiments.py``:

* ``experiments.py`` runs systematic profile comparisons with leaderboards,
  pareto fronts, and bootstrap significance tests. Use it when you want to
  rank variants quantitatively across a corpus.
* ``sweep.py`` runs a fixed grid (images × scales × variants) and produces
  a side-by-side HTML contact sheet plus the per-encode JSON. Use it when
  you want to *see* what an encoder change does, fast.

Run via the ``weft sweep`` CLI subcommand or call :func:`run_sweep`
directly from Python.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .api import decode_image, encode_image
from .types import EncodeConfig


# ---------------------------------------------------------------------------
# Built-in variant registry
# ---------------------------------------------------------------------------

# Each variant is a dict of overrides applied on top of the base EncodeConfig.
# Use ``feature_flags`` to set per-variant feature flags; everything else is a
# direct EncodeConfig field.
BUILTIN_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "decompose": {"feature_flags": {"decompose_lighting": True}},
    "decompose-16": {"feature_flags": {"decompose_lighting": True, "lighting_grid_size": 16}},
    "decompose-32": {"feature_flags": {"decompose_lighting": True, "lighting_grid_size": 32}},
    "decompose-64": {"feature_flags": {"decompose_lighting": True, "lighting_grid_size": 64}},
    # Brainstorm #11: closed-form bicubic-patch tile encoder. No greedy
    # primitive search, no OptiX involvement at decode time. Expected to
    # be ~10× faster to encode than baseline; PSNR likely comparable on
    # smooth content and weaker on hard edges (where primitives shine).
    "bicubic": {"feature_flags": {"bicubic_patch_tiles": True}},
    # Brainstorm #20: K-color palette + per-pixel labels. No tiles, no
    # primitives, no smooth basis. Step discontinuities are exact up to
    # the palette quantization. Designed for screenshots / vector art /
    # text / pixel art. Expected to crush hard-edged content and lose
    # on smooth gradients.
    "palette-16": {"feature_flags": {"palette_planes_k": 16}},
    "palette-64": {"feature_flags": {"palette_planes_k": 64}},
    "palette-256": {"feature_flags": {"palette_planes_k": 256}},
    # Brainstorm #1: gradient-field encoder + Poisson decoder. Wins on
    # hard-edge / large-flat-region content where the gradient field is
    # ~99% sparse and zstd crushes the dense int8 maps. Loses on
    # smooth-varying content (gradients quantize below the noise floor).
    # ``gradient`` uses the default scale=128; ``gradient-64`` is the
    # wider-range / coarser-resolution alternative.
    "gradient":    {"feature_flags": {"gradient_field": True}},
    "gradient-64": {"feature_flags": {"gradient_field": True, "gradient_field_scale": 64}},
    # Auto-select: encoder runs bicubic + palette-16 + palette-64
    # internally and writes whichever one wins by R-D score. The
    # bitstream is byte-identical to the winning single-variant encode,
    # so the standard decoder dispatches correctly without auto-aware
    # changes. ``auto-q`` (quality-biased) and ``auto-r`` (rate-biased)
    # vary the lambda for testing the R-D tradeoff curve.
    "auto":   {"feature_flags": {"auto_select": True, "auto_select_lambda": 4.0}},
    "auto-q": {"feature_flags": {"auto_select": True, "auto_select_lambda": 1.0}},
    "auto-r": {"feature_flags": {"auto_select": True, "auto_select_lambda": 12.0}},
    # Per-tile hybrid: baseline encoder + per-tile bicubic R-D pick.
    # Each tile picks bicubic-as-a-single-primitive vs the greedy
    # primitive stack based on the tile's local R-D score. Wins on
    # mixed-content images (smooth photo regions + sharp UI elements).
    "hybrid": {"feature_flags": {"hybrid_bicubic_per_tile": True}},
    # Brainstorm #16: DCT residual layer on top of baseline / hybrid.
    # The frequency-domain residual closes the natural-photo / dense-
    # frequency PSNR ceiling that the primitive bases can't reach.
    # ``dct`` uses the encoder quality knob to derive the quant step;
    # ``dct-coarse`` / ``dct-fine`` are explicit-step variants for
    # tracing the rate-distortion curve.
    "dct":        {"feature_flags": {"dct_residual": True}},
    "hybrid-dct": {"feature_flags": {"hybrid_bicubic_per_tile": True, "dct_residual": True}},
    "dct-fine":   {"feature_flags": {"dct_residual": True, "dct_residual_step": 0.005}},
    "dct-coarse": {"feature_flags": {"dct_residual": True, "dct_residual_step": 0.04}},
}


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SweepResult:
    """One (image × variant × scale) cell of the sweep."""
    image: str            # source filename
    variant: str          # variant name
    scale: float          # encode_scale
    quality: int
    src_bytes: int
    src_w: int
    src_h: int
    raw_bytes: int        # uncompressed RGB size = src_w * src_h * 3
    weft_bytes: int = 0
    bpp: float = 0.0
    psnr: float = float("nan")
    drift_db: float = float("nan")
    psnr_software: float = float("nan")
    encode_w: int = 0
    encode_h: int = 0
    tile_count: int = 0
    encode_s: float = 0.0
    decode_s: float = 0.0
    weft_path: str | None = None
    decoded_path: str | None = None
    error: str | None = None
    # Snapshot of the EncodeConfig.feature_flags overrides used for this
    # cell. Persisted in CSV/JSON so A/B comparisons stay honest — you can
    # always grep "what flags produced this row".
    feature_flags: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio_vs_src(self) -> float:
        """Compression ratio vs the source PNG/JPG file (>1 means WEFT is smaller)."""
        return self.src_bytes / self.weft_bytes if self.weft_bytes else 0.0

    @property
    def ratio_vs_raw(self) -> float:
        """Compression ratio vs uncompressed RGB."""
        return self.raw_bytes / self.weft_bytes if self.weft_bytes else 0.0


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------

def _resolve_variants(
    names: Iterable[str],
    custom: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve variant names against builtin + custom registries.

    Custom variants override builtins of the same name. Unknown names raise.
    """
    registry = dict(BUILTIN_VARIANTS)
    if custom:
        registry.update(custom)
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in registry:
            raise ValueError(
                f"unknown variant: {name!r}. "
                f"Available: {sorted(registry.keys())}"
            )
        out[name] = registry[name]
    return out


def _make_config(quality: int, scale: float, overrides: dict[str, Any]) -> EncodeConfig:
    """Build an EncodeConfig from a (quality, scale) base + variant overrides."""
    kwargs: dict[str, Any] = {
        "quality": quality,
        "encode_scale": scale,
        # Bump verify threshold so the warning doesn't scream on stress fixtures.
        "verify_drift_threshold_db": 999.0,
    }
    for k, v in overrides.items():
        if k == "feature_flags":
            kwargs["feature_flags"] = dict(v)
        else:
            kwargs[k] = v
    return EncodeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Single encode+decode
# ---------------------------------------------------------------------------

def _run_one(
    src: Path,
    *,
    variant_name: str,
    scale: float,
    quality: int,
    overrides: dict[str, Any],
    out_dir: Path,
) -> SweepResult:
    """Encode + decode one (image, variant, scale) cell."""
    pil = Image.open(src)
    pw, ph = pil.size
    src_bytes = src.stat().st_size
    stem = src.stem

    weft_name = f"{stem}__{variant_name}__s{int(scale * 100):03d}.weft"
    decoded_name = weft_name.replace(".weft", "_decoded.png")
    weft_path = out_dir / weft_name
    decoded_path = out_dir / decoded_name

    result = SweepResult(
        image=src.name,
        variant=variant_name,
        scale=scale,
        quality=quality,
        src_bytes=src_bytes,
        src_w=pw,
        src_h=ph,
        raw_bytes=pw * ph * 3,
        feature_flags=dict(overrides.get("feature_flags") or {}),
    )

    cfg = _make_config(quality, scale, overrides)
    try:
        t0 = time.perf_counter()
        rep = encode_image(str(src), str(weft_path), cfg)
        result.encode_s = time.perf_counter() - t0
    except Exception as exc:
        result.error = f"encode: {type(exc).__name__}: {exc}"
        return result

    meta = rep.metadata
    result.weft_bytes = rep.bytes_written
    result.bpp = rep.bits_per_pixel
    result.psnr = float(rep.psnr) if rep.psnr is not None else float("nan")
    result.drift_db = float(meta.get("verify_drift_db", float("nan")))
    result.psnr_software = float(meta.get("psnr_software", float("nan")))
    result.encode_w = int(meta.get("encode_width", 0))
    result.encode_h = int(meta.get("encode_height", 0))
    tsd = meta.get("tile_size_distribution") or {}
    result.tile_count = int(sum(tsd.values()))
    result.weft_path = str(weft_path.relative_to(out_dir))

    try:
        t0 = time.perf_counter()
        decode_image(str(weft_path), str(decoded_path))
        result.decode_s = time.perf_counter() - t0
        result.decoded_path = str(decoded_path.relative_to(out_dir))
    except Exception as exc:
        result.error = f"decode: {type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sweep(
    *,
    images: list[Path | str],
    output_dir: Path | str,
    scales: list[float] | None = None,
    quality: int = 75,
    variants: list[str] | None = None,
    custom_variants: dict[str, dict[str, Any]] | None = None,
    write_html: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Run a sweep and write results to ``output_dir``.

    Parameters
    ----------
    images
        Source image paths (PNG/JPG). Each will be encoded once per
        (variant × scale) cell.
    output_dir
        Where to write the .weft / decoded.png / results.json /
        index.html outputs. Created if missing.
    scales
        List of ``encode_scale`` values to sweep. Default: ``[1.0, 0.5, 0.25]``.
    quality
        Encoder quality value (single, not a sweep axis here — keep
        sweeps to one quality at a time so the contact sheet stays
        readable).
    variants
        Variant names to run. Default: ``["baseline"]``.
    custom_variants
        Optional custom variant definitions, merged on top of the
        built-in registry.
    write_html
        If True (default), generate ``index.html`` contact sheet.
    workers
        Number of process-pool workers for parallel encode+decode. ``1``
        (default) keeps the original sequential behaviour. Each worker
        gets its own OptiX context, so 2-4 is realistic on a single GPU
        before contention starts hurting; the bicubic variant has no
        GPU dependency and scales further.

    Returns
    -------
    dict
        Sweep summary including ``results`` (list of SweepResult dicts),
        ``output_dir``, and ``run_seconds``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_paths = [Path(p) for p in images]
    for p in src_paths:
        if not p.exists():
            raise FileNotFoundError(f"sweep image not found: {p}")
    if scales is None:
        scales = [1.0, 0.5, 0.25]
    if variants is None:
        variants = ["baseline"]
    resolved = _resolve_variants(variants, custom_variants)

    # Build the cell list once — useful for both serial and parallel paths.
    cells: list[tuple[Path, str, dict[str, Any], float]] = [
        (src, var_name, overrides, scale)
        for src in src_paths
        for var_name, overrides in resolved.items()
        for scale in scales
    ]
    n_total = len(cells)

    def _format_ok(r: SweepResult) -> str:
        return (
            f"bytes={r.weft_bytes:>9,}  bpp={r.bpp:.4f}  "
            f"psnr={r.psnr:6.2f}  drift={r.drift_db:5.2f}  "
            f"tiles={r.tile_count:>6,}  enc={r.encode_s:5.1f}s  dec={r.decode_s * 1000:4.0f}ms"
        )

    t0 = time.perf_counter()
    results: list[SweepResult] = []
    if workers <= 1:
        # Sequential path — preserves the original ordered progress output.
        for idx, (src, var_name, overrides, scale) in enumerate(cells, start=1):
            print(
                f"[{idx}/{n_total}] {src.name} variant={var_name} scale={scale:.2f}",
                flush=True,
            )
            r = _run_one(
                src,
                variant_name=var_name,
                scale=scale,
                quality=quality,
                overrides=overrides,
                out_dir=out_dir,
            )
            if r.error:
                print(f"   ERROR: {r.error}")
            else:
                print(f"   {_format_ok(r)}")
            results.append(r)
    else:
        # Parallel path: ProcessPoolExecutor. Each subprocess initializes
        # its own OptiX/CUDA context, so the per-cell launch cost is
        # higher than serial — only worth it when n_total is large
        # relative to workers, or when individual encodes are slow (large
        # images / heavy variants).
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Sweep: {n_total} cells across {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _run_one, src,
                    variant_name=var_name,
                    scale=scale,
                    quality=quality,
                    overrides=overrides,
                    out_dir=out_dir,
                ): (src, var_name, scale)
                for (src, var_name, overrides, scale) in cells
            }
            done = 0
            for fut in as_completed(futures):
                src, var_name, scale = futures[fut]
                done += 1
                try:
                    r = fut.result()
                except Exception as exc:
                    # Defensive: _run_one already catches encode/decode
                    # errors into result.error, so this branch only fires
                    # for genuinely unexpected failures (e.g. process
                    # crash). Synthesize a SweepResult so the run still
                    # completes.
                    pil = Image.open(src)
                    pw, ph = pil.size
                    r = SweepResult(
                        image=src.name, variant=var_name, scale=scale,
                        quality=quality, src_bytes=src.stat().st_size,
                        src_w=pw, src_h=ph, raw_bytes=pw * ph * 3,
                        error=f"worker: {type(exc).__name__}: {exc}",
                    )
                tag = f"[{done}/{n_total}] {src.name} variant={var_name} scale={scale:.2f}"
                if r.error:
                    print(f"{tag}\n   ERROR: {r.error}", flush=True)
                else:
                    print(f"{tag}\n   {_format_ok(r)}", flush=True)
                results.append(r)
        # Sort results back into deterministic (image, variant, scale)
        # order so JSON/CSV output stays comparable across runs.
        order = {(src.name, v, s): i for i, (src, v, _, s) in enumerate(cells)}
        results.sort(key=lambda r: order.get((r.image, r.variant, r.scale), 0))

    elapsed = time.perf_counter() - t0
    summary = {
        "output_dir": str(out_dir),
        "images": [str(p) for p in src_paths],
        "scales": list(scales),
        "quality": quality,
        "variants": list(resolved.keys()),
        "workers": workers,
        "n_results": len(results),
        "run_seconds": elapsed,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "results.csv", results)
    if write_html:
        _write_html(out_dir / "index.html", results, scales=scales, variants=list(resolved.keys()))
    print()
    print(f"Done in {elapsed:.1f}s. Output: {out_dir}")
    if write_html:
        print(f"Open: file://{(out_dir / 'index.html').resolve()}")
    return summary


# ---------------------------------------------------------------------------
# CSV + HTML output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "image", "variant", "scale", "quality",
    "src_bytes", "src_w", "src_h", "raw_bytes",
    "weft_bytes", "bpp", "psnr", "drift_db", "psnr_software",
    "encode_w", "encode_h", "tile_count", "encode_s", "decode_s",
    "ratio_vs_src", "ratio_vs_raw", "feature_flags", "error",
]


def _write_csv(path: Path, results: list[SweepResult]) -> None:
    rows = []
    for r in results:
        d = asdict(r)
        d["ratio_vs_src"] = r.ratio_vs_src
        d["ratio_vs_raw"] = r.ratio_vs_raw
        # Stable JSON dump of the flag snapshot — keys sorted so two rows
        # with the same flags compare equal as strings.
        d["feature_flags"] = json.dumps(r.feature_flags, sort_keys=True) if r.feature_flags else ""
        rows.append({k: d.get(k, "") for k in _CSV_FIELDS})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WEFT sweep · {title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; background: #f7f7f8; color: #1c1d22; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  h2 {{ font-size: 16px; margin-top: 32px; padding-bottom: 4px; border-bottom: 1px solid #ccc; }}
  h3 {{ font-size: 13px; color: #555; margin: 12px 0 4px; font-weight: 600; }}
  .row {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: flex-start; flex-wrap: wrap; }}
  .col {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; max-width: 380px; }}
  .col.source {{ border-color: #4a8; }}
  .col.error {{ border-color: #c44; background: #fee; }}
  .label {{ font-size: 11px; font-weight: 600; color: #555; margin-bottom: 4px; font-family: ui-monospace, monospace; }}
  .meta {{ font-size: 11px; color: #555; font-family: ui-monospace, monospace; line-height: 1.4; margin-top: 4px; }}
  img {{ display: block; max-width: 360px; max-height: 280px; image-rendering: pixelated; background: #222; }}
  .summary {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin-bottom: 24px; font-size: 13px; }}
  .summary table {{ border-collapse: collapse; width: 100%; }}
  .summary th, .summary td {{ padding: 4px 8px; text-align: right; border-bottom: 1px solid #eee; font-family: ui-monospace, monospace; font-size: 12px; }}
  .summary th:first-child, .summary th:nth-child(2), .summary td:first-child, .summary td:nth-child(2) {{ text-align: left; }}
  .warn {{ color: #c44; }}
  .ok {{ color: #4a8; }}
  .nan {{ color: #aaa; }}
</style>
</head>
<body>
<h1>WEFT sweep · {title}</h1>
<div class="subtitle">{subtitle}</div>
"""

_HTML_TAIL = "</body></html>\n"


def _write_html(
    path: Path,
    results: list[SweepResult],
    *,
    scales: list[float],
    variants: list[str],
) -> None:
    """Generate the contact-sheet HTML."""
    title = path.parent.name
    subtitle = (
        f"{len(set(r.image for r in results))} images "
        f"× {len(variants)} variants "
        f"× {len(scales)} scales = {len(results)} encodes"
    )
    out: list[str] = [_HTML_HEAD.format(title=escape(title), subtitle=escape(subtitle))]

    # Summary table
    out.append('<div class="summary"><h2 style="margin-top:0;border:none">Per-encode numbers</h2><table>')
    out.append(
        "<thead><tr>"
        "<th>image</th><th>variant</th><th>scale</th>"
        "<th>bytes</th><th>BPP</th><th>×src</th><th>×raw</th>"
        "<th>PSNR</th><th>drift</th><th>tiles</th><th>enc s</th><th>dec ms</th>"
        "</tr></thead><tbody>"
    )
    for r in results:
        if r.error:
            out.append(
                f'<tr><td>{escape(r.image)}</td><td>{escape(r.variant)}</td>'
                f'<td>{r.scale:.2f}</td><td colspan="9" class="warn">ERROR: {escape(r.error)}</td></tr>'
            )
            continue
        ratio_warn = ' class="warn"' if r.ratio_vs_src < 1.0 else ''
        drift_warn = ' class="warn"' if r.drift_db == r.drift_db and r.drift_db > 10 else ''
        out.append(
            f'<tr><td>{escape(r.image)}</td><td>{escape(r.variant)}</td>'
            f'<td>{r.scale:.2f}</td>'
            f'<td>{r.weft_bytes:,}</td><td>{r.bpp:.4f}</td>'
            f'<td{ratio_warn}>{r.ratio_vs_src:.2f}×</td><td>{r.ratio_vs_raw:.1f}×</td>'
            f'<td>{r.psnr:.2f}</td><td{drift_warn}>{r.drift_db:.2f}</td>'
            f'<td>{r.tile_count:,}</td><td>{r.encode_s:.1f}</td>'
            f'<td>{r.decode_s * 1000:.0f}</td></tr>'
        )
    out.append("</tbody></table></div>")

    # Per-image visual rows, organised by variant.
    images_seen: list[str] = []
    for r in results:
        if r.image not in images_seen:
            images_seen.append(r.image)

    # Resolve relative path back to samples/inputs
    def _src_rel(image_name: str) -> str:
        # Compute relative path from output_dir to samples/inputs/<image_name>
        try:
            inp = (Path("samples") / "inputs" / image_name).resolve()
            return os.path.relpath(inp, start=path.parent)
        except Exception:
            return f"../../inputs/{image_name}"

    for img_name in images_seen:
        rows_for_img = [r for r in results if r.image == img_name]
        if not rows_for_img:
            continue
        sample = rows_for_img[0]
        out.append(
            f'<h2>{escape(img_name)} '
            f'<span style="font-weight:normal;color:#888;font-size:13px">'
            f'{sample.src_w}×{sample.src_h}</span></h2>'
        )
        for var_name in variants:
            var_rows = [r for r in rows_for_img if r.variant == var_name]
            if not var_rows:
                continue
            if len(variants) > 1:
                out.append(f'<h3>variant: {escape(var_name)}</h3>')
            out.append('<div class="row">')
            # Source column (only on the first variant per image, to save vertical space)
            if var_name == variants[0]:
                src_path = _src_rel(img_name)
                out.append('<div class="col source">')
                out.append('<div class="label">SOURCE</div>')
                out.append(f'<a href="{escape(src_path)}"><img src="{escape(src_path)}" alt="source"></a>')
                out.append(
                    f'<div class="meta">{sample.src_w}×{sample.src_h}<br>'
                    f'{sample.src_bytes:,} bytes</div>'
                )
                out.append('</div>')
            # Decoded columns, largest scale first
            for r in sorted(var_rows, key=lambda x: -x.scale):
                if r.error:
                    out.append('<div class="col error">')
                    out.append(f'<div class="label">scale={r.scale:.2f} — ERROR</div>')
                    out.append(f'<div class="meta">{escape(r.error)}</div>')
                    out.append('</div>')
                    continue
                out.append('<div class="col">')
                out.append(f'<div class="label">scale={r.scale:.2f}</div>')
                if r.decoded_path:
                    out.append(
                        f'<a href="{escape(r.decoded_path)}">'
                        f'<img src="{escape(r.decoded_path)}" alt="decoded scale {r.scale}">'
                        f'</a>'
                    )
                drift_warn = ' style="color:#c44"' if r.drift_db == r.drift_db and r.drift_db > 10 else ''
                out.append(
                    f'<div class="meta">'
                    f'{r.weft_bytes:,} bytes ({r.ratio_vs_src:.1f}× src)<br>'
                    f'PSNR {r.psnr:.2f} dB<br>'
                    f'<span{drift_warn}>drift {r.drift_db:.2f} dB</span><br>'
                    f'{r.tile_count:,} tiles · enc {r.encode_s:.1f}s'
                    f'</div>'
                )
                out.append('</div>')
            out.append('</div>')

    out.append(_HTML_TAIL)
    path.write_text("\n".join(out), encoding="utf-8")
