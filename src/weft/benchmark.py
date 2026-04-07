"""Benchmark helpers for WEFT."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

from PIL import Image

from .decoder import decode_image
from .encoder import encode_image
from .image_io import load_image_linear
from .metrics import lpips_score, psnr, ssim
from .types import BenchmarkImageResult, BenchmarkReport, EncodeConfig

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_images(dataset_dir: str) -> list[Path]:
    root = Path(dataset_dir)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files)


def _png_size(path: Path) -> int:
    if path.suffix.lower() == ".png":
        return path.stat().st_size
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        Image.open(path).convert("RGB").save(tmp.name, format="PNG")
        return os.path.getsize(tmp.name)


def benchmark(
    dataset_dir: str,
    quality: int = 75,
    report_path: str | None = None,
    strict_gpu: bool = True,
    require_gpu_entropy: bool = True,
) -> BenchmarkReport:
    images = _collect_images(dataset_dir)
    results: list[BenchmarkImageResult] = []

    with tempfile.TemporaryDirectory(prefix="weft-bench-") as tmpdir:
        for path in images:
            src_linear, w, h = load_image_linear(str(path))
            weft_path = Path(tmpdir) / f"{path.stem}.weft"

            t0 = time.perf_counter()
            enc = encode_image(str(path), str(weft_path), EncodeConfig(quality=quality))
            encode_ms = (time.perf_counter() - t0) * 1000.0

            dec_path = Path(tmpdir) / f"{path.stem}.decoded.png"
            t1 = time.perf_counter()
            drep = decode_image(
                str(weft_path),
                str(dec_path),
                gpu_only=strict_gpu,
                require_gpu_entropy=require_gpu_entropy,
            )
            decode_ms = (time.perf_counter() - t1) * 1000.0
            dec_linear, _, _ = load_image_linear(str(dec_path))

            weft_bytes = weft_path.stat().st_size
            input_bytes = path.stat().st_size
            png_bytes = _png_size(path)
            bpp = weft_bytes * 8.0 / float(w * h)
            p = psnr(src_linear, dec_linear)
            s = ssim(src_linear, dec_linear)
            l = lpips_score(src_linear, dec_linear)

            results.append(
                BenchmarkImageResult(
                    name=str(path.name),
                    width=w,
                    height=h,
                    input_bytes=input_bytes,
                    weft_bytes=weft_bytes,
                    png_bytes=png_bytes,
                    bpp=bpp,
                    psnr=p,
                    ssim=s,
                    lpips=l,
                    encode_ms=encode_ms,
                    decode_ms=decode_ms,
                    decode_backend=str(drep.metadata.get("prim_decode_backend")),
                    strict_gpu=strict_gpu,
                    require_gpu_entropy=require_gpu_entropy,
                    throughput_mpix_s=((w * h) / (decode_ms * 1000.0)) if decode_ms > 0 else None,
                    est_host_device_transfers=int(
                        drep.metadata.get("gpu_upload_plan", {}).get("est_host_device_transfers", 0)
                    )
                    if isinstance(drep.metadata.get("gpu_upload_plan", {}).get("est_host_device_transfers", None), int)
                    else None,
                )
            )

    def _avg(vals: list[float | None]) -> float | None:
        valid = [float(v) for v in vals if v is not None]
        if not valid:
            return None
        return float(sum(valid) / len(valid))

    agg = {
        "avg_bpp": _avg([r.bpp for r in results]),
        "avg_psnr": _avg([r.psnr for r in results]),
        "avg_ssim": _avg([r.ssim for r in results]),
        "avg_lpips": _avg([r.lpips for r in results]),
        "avg_encode_ms": _avg([r.encode_ms for r in results]),
        "avg_decode_ms": _avg([r.decode_ms for r in results]),
        "avg_throughput_mpix_s": _avg([r.throughput_mpix_s for r in results]),
        "weft_vs_png_ratio": (
            float(sum(r.weft_bytes for r in results)) / float(sum(r.png_bytes for r in results))
            if results and sum(r.png_bytes for r in results) > 0
            else None
        ),
    }

    report = BenchmarkReport(
        dataset_dir=dataset_dir,
        quality=quality,
        image_count=len(results),
        results=results,
        aggregate=agg,
    )

    if report_path is not None:
        out = {
            "dataset_dir": report.dataset_dir,
            "quality": report.quality,
            "image_count": report.image_count,
            "aggregate": report.aggregate,
            "results": [asdict(r) for r in report.results],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    return report
