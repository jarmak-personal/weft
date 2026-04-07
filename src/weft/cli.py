"""CLI entrypoint for WEFT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from .api import benchmark, decode_image, encode_image
from .experiments import default_profiles, load_profiles_json, load_sweep_json, resolve_dataset_pack, run_experiment_suite
from .types import EncodeConfig


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weft",
        description="Weft: adaptive-basis image codec",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode image to .weft")
    pe.add_argument("input", help="Input image path")
    pe.add_argument("output", help="Output .weft path")
    pe.add_argument("--preset", default="rtx-heavy-v2",
                    help="Encoder preset (internal identifier; rtx-heavy-v2 is the default)")
    pe.add_argument("--single-image-maxcompute", action="store_true",
                    help="Shortcut for preset=rtx-single-maxcompute (experimental)")
    pe.add_argument("--quality", type=int, default=75, help="Quality 0..100")
    pe.add_argument("--no-res0", action="store_true", help="Disable RES0 scalar residual block")
    pe.add_argument("--no-res1", action="store_true", help="Disable RES1 residual map block")
    pe.add_argument("--entropy", default="chunked-rans", choices=["chunked-rans", "rans", "raw"], help="PRIM entropy mode")
    pe.add_argument("--chunk-tiles", type=int, default=64, help="Tiles per PRIM entropy chunk")
    pe.add_argument("--block-align", type=int, default=64, help="Block alignment bytes for GPU-direct reads")
    pe.add_argument("--encode-scale", type=float, default=1.0, help="Encode at reduced resolution (e.g. 0.5 = half-res), decode upscales back")
    pe.add_argument("--feature-flags-json", default=None, help="JSON object for research feature flags")
    pe.add_argument("--multi-rounds", type=int, default=None, help="Feature flag: iterative primitive fitting rounds")
    pe.add_argument("--adaptive-tile-budget", action="store_true", help="Feature flag: complexity-driven per-tile primitive budgets")
    pe.add_argument("--edge-weighted-objective", action="store_true", help="Feature flag: prioritize edge/text reconstruction")
    pe.add_argument("--target-bpp", type=float, default=None, help="Feature flag: global bitrate target for allocator")
    pe.add_argument("--candidate-bank", choices=["linear5", "rich18"], default=None, help="Feature flag: GPU candidate model bank")
    pe.add_argument("--enable-res2", action="store_true", help="Feature flag: emit sparse RES2 high-frequency atoms")
    pe.add_argument("--search-mode", choices=["greedy", "beam", "mcmc"], default=None, help="Feature flag: tile search strategy")
    pe.add_argument("--beam-width", type=int, default=None, help="Feature flag: beam width for beam search")
    pe.add_argument("--mcmc-steps", type=int, default=None, help="Feature flag: MCMC steps for hard tiles")
    pe.add_argument("--stochastic-restarts", type=int, default=None, help="Feature flag: restart count for edge-heavy tiles")
    pe.add_argument("--early-exit-patience", type=int, default=None, help="Feature flag: no-improvement rounds before early exit")
    pe.add_argument("--maxcompute-fit-passes", type=int, default=None, help="Feature flag: repeated GPU fit passes (for single-image saturation)")
    pe.add_argument("--container-v2-blocks", action="store_true", help="Feature flag: write v2 side blocks (PSTR/PDEL/RES2)")
    pe.add_argument("--split-entropy-streams", action="store_true", help="Feature flag: emit primitive split side stream (PSTR)")
    pe.add_argument("--neighbor-delta-coding", action="store_true", help="Feature flag: emit neighbor-delta side stream (PDEL)")
    pe.add_argument("--no-verify", dest="verify_decode", action="store_false", default=True,
                    help="Disable decode-in-the-loop verification (verify is on by default)")
    pe.add_argument("--verify-strict", action="store_true",
                    help="Raise EncodeError when verify drift exceeds threshold instead of warning")
    pe.add_argument("--verify-threshold-db", type=float, default=None,
                    help="Verify drift threshold in dB (default 10.0)")
    pe.add_argument("--decompose-lighting", action="store_true",
                    help="Phase 2 #17: split image into albedo*lighting via Retinex; "
                         "fit primitives against smoother albedo + store low-res lighting grid. "
                         "Best on rendered content; modestly helpful on natural images.")
    pe.add_argument("--lighting-grid-size", type=int, default=None,
                    help="Lighting grid resolution for --decompose-lighting (default 32)")

    pd = sub.add_parser("decode", help="Decode .weft to image")
    pd.add_argument("input", help="Input .weft path")
    pd.add_argument("output", help="Output image path")
    pd.add_argument("--width", type=int, default=None, help="Optional output width")
    pd.add_argument("--height", type=int, default=None, help="Optional output height")
    pd.add_argument(
        "--require-gpu-entropy",
        action="store_true",
        default=True,
        help="Require CUDA kernel PRIM chunk entropy decode; fail if unavailable",
    )

    pb = sub.add_parser("bench", help="Benchmark dataset")
    pb.add_argument("dataset_dir", help="Directory containing images")
    pb.add_argument("report_json", help="Output report path")
    pb.add_argument("--quality", type=int, default=75, help="Quality 0..100")
    pb.add_argument("--require-gpu-entropy", action="store_true", default=True, help="Require GPU entropy decode in benchmark (default)")

    px = sub.add_parser("experiment", help="Run multi-profile experiment sweep")
    px.add_argument("dataset_dir", help="Directory containing images")
    px.add_argument("output_dir", help="Directory for artifacts + reports")
    px.add_argument("--profiles-json", default=None, help="Optional profile list JSON")
    px.add_argument("--sweep-json", default=None, help="Optional sweep grid JSON (Cartesian profiles)")
    px.add_argument("--dataset-pack", default=None, help="Dataset pack key from --dataset-packs-json")
    px.add_argument("--dataset-packs-json", default=None, help="Dataset packs JSON mapping")
    px.add_argument("--no-save-decoded", dest="save_decoded", action="store_false", help="Do not keep decoded PNG artifacts")
    px.add_argument("--no-save-weft", dest="save_weft", action="store_false", help="Do not keep .weft artifacts")
    px.add_argument("--no-generate-hybrids", dest="generate_hybrids", action="store_false", help="Skip hybrid profile suggestion generation")
    px.add_argument("--no-run-hybrid-pass", dest="run_hybrid_pass", action="store_false", help="Do not auto-evaluate generated hybrid profiles")

    psw = sub.add_parser(
        "sweep",
        help="Sweep encoder over images × variants × scales and produce an HTML contact sheet",
    )
    psw.add_argument(
        "images", nargs="+",
        help="Image paths (PNG/JPG). Globs are accepted via shell expansion.",
    )
    psw.add_argument(
        "--output", "-o", required=True,
        help="Output directory; created if missing. Convention: samples/runs/<timestamp>-<label>",
    )
    psw.add_argument(
        "--scales", default="1.0,0.5,0.25",
        help="Comma-separated encode_scale values to sweep (default: 1.0,0.5,0.25)",
    )
    psw.add_argument(
        "--quality", type=int, default=75,
        help="Encoder quality (default 75). Sweeps run at a single quality.",
    )
    psw.add_argument(
        "--variants", default="baseline",
        help=(
            "Comma-separated variant names (default: baseline). "
            "Built-ins: baseline, decompose, decompose-16, decompose-32, decompose-64. "
            "Custom variants can be loaded via --variants-json."
        ),
    )
    psw.add_argument(
        "--variants-json", default=None,
        help="Optional JSON file mapping variant name → encode-config overrides",
    )
    psw.add_argument(
        "--no-html", dest="write_html", action="store_false", default=True,
        help="Skip HTML contact sheet generation (only write results.json + results.csv)",
    )
    psw.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Process-pool workers for parallel encode+decode (default 1). "
            "Each worker spins up its own CUDA context if the GPU encoder "
            "path is enabled, so 2-4 is a reasonable ceiling on a single "
            "GPU. The bicubic variant has no GPU dependency and "
            "parallelizes further."
        ),
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "encode":
        feature_flags = {}
        preset = "rtx-single-maxcompute" if args.single_image_maxcompute else args.preset
        if args.feature_flags_json:
            parsed = json.loads(args.feature_flags_json)
            if not isinstance(parsed, dict):
                raise SystemExit("--feature-flags-json must decode to an object")
            feature_flags = parsed
        if args.multi_rounds is not None:
            feature_flags["multi_rounds"] = int(args.multi_rounds)
        if args.adaptive_tile_budget:
            feature_flags["adaptive_tile_budget"] = True
        if args.edge_weighted_objective:
            feature_flags["edge_weighted_objective"] = True
        if args.target_bpp is not None:
            feature_flags["target_bpp"] = float(args.target_bpp)
        if args.candidate_bank is not None:
            feature_flags["candidate_bank"] = args.candidate_bank
        if args.enable_res2:
            feature_flags["enable_res2"] = True
        if args.search_mode is not None:
            feature_flags["search_mode"] = args.search_mode
        if args.beam_width is not None:
            feature_flags["beam_width"] = int(args.beam_width)
        if args.mcmc_steps is not None:
            feature_flags["mcmc_steps"] = int(args.mcmc_steps)
        if args.stochastic_restarts is not None:
            feature_flags["stochastic_restarts"] = int(args.stochastic_restarts)
        if args.early_exit_patience is not None:
            feature_flags["early_exit_patience"] = int(args.early_exit_patience)
        if args.maxcompute_fit_passes is not None:
            feature_flags["maxcompute_fit_passes"] = int(args.maxcompute_fit_passes)
        if args.container_v2_blocks:
            feature_flags["container_v2_blocks"] = True
        if args.split_entropy_streams:
            feature_flags["split_entropy_streams"] = True
        if args.neighbor_delta_coding:
            feature_flags["neighbor_delta_coding"] = True
        if args.decompose_lighting:
            feature_flags["decompose_lighting"] = True
        if args.lighting_grid_size is not None:
            feature_flags["lighting_grid_size"] = int(args.lighting_grid_size)
        cfg_kwargs: dict[str, Any] = dict(
            preset=preset,
            quality=args.quality,
            enable_res0=not args.no_res0,
            enable_res1=not args.no_res1,
            entropy=args.entropy,
            chunk_tiles=args.chunk_tiles,
            block_alignment=args.block_align,
            encode_scale=args.encode_scale,
            feature_flags=feature_flags,
            verify_decode=bool(args.verify_decode),
            verify_strict=bool(args.verify_strict),
        )
        if args.verify_threshold_db is not None:
            cfg_kwargs["verify_drift_threshold_db"] = float(args.verify_threshold_db)
        cfg = EncodeConfig(**cfg_kwargs)
        rep = encode_image(args.input, args.output, config=cfg)
        _print_json(asdict(rep))
        return 0

    if args.cmd == "decode":
        rep = decode_image(
            args.input,
            args.output,
            width=args.width,
            height=args.height,
            gpu_only=True,
            require_gpu_entropy=args.require_gpu_entropy,
        )
        _print_json(asdict(rep))
        return 0

    if args.cmd == "bench":
        rep = benchmark(
            args.dataset_dir,
            config=EncodeConfig(quality=args.quality),
            strict_gpu=True,
            require_gpu_entropy=args.require_gpu_entropy,
        )
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_dir": rep.dataset_dir,
                    "quality": rep.quality,
                    "image_count": rep.image_count,
                    "aggregate": rep.aggregate,
                    "results": [asdict(r) for r in rep.results],
                },
                f,
                indent=2,
            )
        _print_json({"report_json": args.report_json, "image_count": rep.image_count, "aggregate": rep.aggregate})
        return 0

    if args.cmd == "experiment":
        if args.profiles_json and args.sweep_json:
            raise SystemExit("use only one of --profiles-json or --sweep-json")
        if args.profiles_json:
            profiles = load_profiles_json(args.profiles_json)
        elif args.sweep_json:
            profiles = load_sweep_json(args.sweep_json)
        else:
            profiles = default_profiles()
        resolved_dataset = resolve_dataset_pack(args.dataset_dir, args.dataset_pack, args.dataset_packs_json)
        rep = run_experiment_suite(
            dataset_dir=resolved_dataset,
            output_dir=args.output_dir,
            profiles=profiles,
            save_decoded=args.save_decoded,
            save_weft=args.save_weft,
            generate_hybrids=args.generate_hybrids,
            run_hybrid_pass=args.run_hybrid_pass,
        )
        _print_json(
            {
                "dataset_dir": rep.dataset_dir,
                "output_dir": rep.output_dir,
                "profile_count": len(rep.profiles),
                "image_count": len({r.image for r in rep.results}),
                "leaderboard": rep.leaderboard,
                "hybrid_pass_executed": bool(args.generate_hybrids and args.run_hybrid_pass),
                "report_json": str(Path(args.output_dir) / "experiment_report.json"),
                "leaderboard_csv": str(Path(args.output_dir) / "leaderboard.csv"),
                "pareto_csv": str(Path(args.output_dir) / "pareto_front.csv"),
                "hybrid_profiles_json": str(Path(args.output_dir) / "hybrid_profiles.json"),
            }
        )
        return 0

    if args.cmd == "sweep":
        from .sweep import run_sweep
        scales = [float(s) for s in args.scales.split(",") if s.strip()]
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        custom_variants: dict[str, dict[str, Any]] | None = None
        if args.variants_json:
            with open(args.variants_json, encoding="utf-8") as f:
                custom_variants = json.load(f)
            if not isinstance(custom_variants, dict):
                raise SystemExit("--variants-json must decode to an object")
        summary = run_sweep(
            images=args.images,
            output_dir=args.output,
            scales=scales,
            quality=args.quality,
            variants=variants,
            custom_variants=custom_variants,
            write_html=args.write_html,
            workers=int(args.workers),
        )
        _print_json({
            "output_dir": summary["output_dir"],
            "n_results": summary["n_results"],
            "run_seconds": summary["run_seconds"],
            "results_json": str(Path(summary["output_dir"]) / "results.json"),
            "results_csv": str(Path(summary["output_dir"]) / "results.csv"),
            "index_html": str(Path(summary["output_dir"]) / "index.html") if args.write_html else None,
        })
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
