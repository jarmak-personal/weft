"""Public datatypes for WEFT API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EncodeConfig:
    preset: str = "rtx-heavy-v2"
    quality: int = 75
    tile_size: int = 16
    max_primitives: int = 48
    enable_res0: bool = True
    enable_res1: bool = True
    entropy: str = "chunked-rans"
    deterministic: bool = True
    chunk_tiles: int = 64
    block_alignment: int = 64
    encode_scale: float = 1.0  # <1.0 = fit primitives at reduced resolution, decode at original
    feature_flags: dict[str, Any] = field(default_factory=dict)

    # Decode-in-the-loop verification.
    #
    # When ``verify_decode`` is True (default), the encoder runs the
    # production decoder over the freshly-written bitstream and compares
    # the actual decoded output to the source image. With verify on, the
    # top-level ``EncodeReport.psnr`` and ``.ssim`` reflect the actual
    # end-to-end decoded quality, not the encoder's internal estimate.
    #
    # The encoder's internal estimate is preserved in metadata as
    # ``psnr_software``; the difference between it and the verified
    # PSNR is reported as ``verify_drift_db``. Some drift is expected
    # because the encoder uses bilinear upscale internally while the
    # decoder renders primitives analytically — 5–6 dB is normal on
    # dark images. The default threshold is set well above that so the
    # warning fires only on real encoder/decoder divergence.
    #
    # When ``verify_strict`` is True, drift exceeding
    # ``verify_drift_threshold_db`` raises EncodeError instead of just
    # warning. Useful in CI / automated experiments where regressions
    # should fail loudly.
    verify_decode: bool = True
    verify_drift_threshold_db: float = 10.0
    verify_strict: bool = False

    # Optional disk cache for the adaptive encoder's primitive fit
    # phase. When set to a directory path, the encoder serializes the
    # post-greedy ``_AdaptiveFitState`` to ``<dir>/<key>.pkl`` and
    # reuses it on subsequent encodes of the same input + fit-relevant
    # cfg. Designed for benchmark / sweep loops where the same fixture
    # is re-encoded many times — not a realistic end-user setting.
    # ``None`` (the default) disables disk caching entirely; the
    # in-memory fit-share that auto_select uses works regardless.
    fit_cache_dir: str | None = None


@dataclass(slots=True)
class EncodeReport:
    input_path: str
    output_path: str
    width: int
    height: int
    tile_count: int
    bits_per_pixel: float
    bytes_written: int
    psnr: float
    ssim: float | None
    decode_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecodeReport:
    input_path: str
    output_path: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    bytes_read: int
    decode_hash: str
    used_upscaling: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkImageResult:
    name: str
    width: int
    height: int
    input_bytes: int
    weft_bytes: int
    png_bytes: int
    bpp: float
    psnr: float
    ssim: float | None
    lpips: float | None
    encode_ms: float
    decode_ms: float
    decode_backend: str | None
    strict_gpu: bool
    require_gpu_entropy: bool
    throughput_mpix_s: float | None
    est_host_device_transfers: int | None


@dataclass(slots=True)
class BenchmarkReport:
    dataset_dir: str
    quality: int
    image_count: int
    results: list[BenchmarkImageResult]
    aggregate: dict[str, float | None]
