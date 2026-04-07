"""Public API for WEFT."""

from __future__ import annotations

from .benchmark import benchmark as _benchmark
from .decoder import decode_image as _decode_image
from .encoder import encode_image as _encode_image
from .experiments import ExperimentProfile, ExperimentReport, run_experiment_suite as _run_experiment_suite
from .types import BenchmarkReport, DecodeReport, EncodeConfig, EncodeReport


def encode_image(input_path: str, output_path: str, config: EncodeConfig | None = None) -> EncodeReport:
    return _encode_image(input_path, output_path, config=config)


def decode_image(
    input_path: str,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    gpu_only: bool = True,
    allow_cpu_fallback: bool = False,
    require_gpu_entropy: bool = True,
) -> DecodeReport:
    return _decode_image(
        input_path,
        output_path,
        width=width,
        height=height,
        gpu_only=gpu_only,
        allow_cpu_fallback=allow_cpu_fallback,
        require_gpu_entropy=require_gpu_entropy,
    )


def benchmark(
    dataset_path: str,
    config: EncodeConfig | None = None,
    strict_gpu: bool = True,
    require_gpu_entropy: bool = True,
) -> BenchmarkReport:
    cfg = config or EncodeConfig()
    return _benchmark(
        dataset_path,
        quality=cfg.quality,
        strict_gpu=strict_gpu,
        require_gpu_entropy=require_gpu_entropy,
    )


def run_experiment_suite(
    dataset_dir: str,
    output_dir: str,
    profiles: list[ExperimentProfile] | None = None,
    save_decoded: bool = True,
    save_weft: bool = True,
    generate_hybrids: bool = True,
    run_hybrid_pass: bool = True,
) -> ExperimentReport:
    return _run_experiment_suite(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        profiles=profiles,
        save_decoded=save_decoded,
        save_weft=save_weft,
        generate_hybrids=generate_hybrids,
        run_hybrid_pass=run_hybrid_pass,
    )
