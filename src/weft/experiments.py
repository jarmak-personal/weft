"""Experiment harness for WEFT model/profile sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .image_io import load_image_linear
from .metrics import lpips_score, psnr, ssim

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(slots=True)
class ObjectiveWeights:
    lpips: float = 1.0
    one_minus_ssim: float = 0.5
    edge_mse: float = 0.5
    bpp: float = 0.25
    encode_seconds: float = 0.02
    decode_seconds: float = 0.02


@dataclass(slots=True)
class ExperimentProfile:
    name: str
    encode_config: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)


@dataclass(slots=True)
class ExperimentImageResult:
    profile: str
    image: str
    width: int
    height: int
    weft_bytes: int
    bpp: float
    psnr: float
    ssim: float | None
    lpips: float | None
    edge_mse: float
    ocr_score: float
    encode_ms: float
    decode_ms: float
    gpu_power_avg_w: float | None
    gpu_util_avg_pct: float | None
    vram_peak_mib: float | None
    objective: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentReport:
    dataset_dir: str
    output_dir: str
    profiles: list[ExperimentProfile]
    results: list[ExperimentImageResult]
    leaderboard: list[dict[str, Any]]


def _avg(vals: list[float | None]) -> float | None:
    vv = [float(v) for v in vals if v is not None]
    if not vv:
        return None
    return float(sum(vv) / len(vv))


def _collect_images(dataset_dir: str) -> list[Path]:
    root = Path(dataset_dir)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files)


def _edge_mse(src: np.ndarray, dec: np.ndarray) -> float:
    y_src = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
    y_dec = 0.2126 * dec[..., 0] + 0.7152 * dec[..., 1] + 0.0722 * dec[..., 2]
    gx_s = np.diff(y_src, axis=1, append=y_src[:, -1:])
    gy_s = np.diff(y_src, axis=0, append=y_src[-1:, :])
    gx_d = np.diff(y_dec, axis=1, append=y_dec[:, -1:])
    gy_d = np.diff(y_dec, axis=0, append=y_dec[-1:, :])
    gm_s = np.sqrt(gx_s * gx_s + gy_s * gy_s)
    gm_d = np.sqrt(gx_d * gx_d + gy_d * gy_d)
    mask = gm_s > np.percentile(gm_s, 85)
    if not np.any(mask):
        return float(np.mean((gm_s - gm_d) ** 2))
    return float(np.mean((gm_s[mask] - gm_d[mask]) ** 2))


def _ocr_proxy_score(src: np.ndarray, dec: np.ndarray) -> float:
    """Edge-structure overlap proxy in [0,1], higher is better."""
    y_src = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
    y_dec = 0.2126 * dec[..., 0] + 0.7152 * dec[..., 1] + 0.0722 * dec[..., 2]
    gx_s = np.diff(y_src, axis=1, append=y_src[:, -1:])
    gy_s = np.diff(y_src, axis=0, append=y_src[-1:, :])
    gx_d = np.diff(y_dec, axis=1, append=y_dec[:, -1:])
    gy_d = np.diff(y_dec, axis=0, append=y_dec[-1:, :])
    gm_s = np.sqrt(gx_s * gx_s + gy_s * gy_s)
    gm_d = np.sqrt(gx_d * gx_d + gy_d * gy_d)
    ts = float(np.percentile(gm_s, 85))
    td = float(np.percentile(gm_d, 85))
    bs = gm_s >= ts
    bd = gm_d >= td
    denom = float(np.sum(bs | bd))
    if denom <= 0.0:
        return 1.0
    inter = float(np.sum(bs & bd))
    return max(0.0, min(1.0, inter / denom))


def _objective(
    *,
    bpp: float,
    ssim_val: float | None,
    lpips_val: float | None,
    edge_mse: float,
    encode_ms: float,
    decode_ms: float,
    w: ObjectiveWeights,
) -> float:
    score = 0.0
    score += w.bpp * float(bpp)
    score += w.edge_mse * float(edge_mse)
    score += w.encode_seconds * (float(encode_ms) / 1000.0)
    score += w.decode_seconds * (float(decode_ms) / 1000.0)
    if ssim_val is not None:
        score += w.one_minus_ssim * (1.0 - float(ssim_val))
    if lpips_val is not None:
        score += w.lpips * float(lpips_val)
    return float(score)


def default_profiles() -> list[ExperimentProfile]:
    return [
        ExperimentProfile(name="q60", encode_config={"quality": 60}),
        ExperimentProfile(name="q75", encode_config={"quality": 75}),
        ExperimentProfile(name="q90", encode_config={"quality": 90}),
        ExperimentProfile(name="q90_chunk32", encode_config={"quality": 90, "chunk_tiles": 32}),
        ExperimentProfile(name="q90_chunk128", encode_config={"quality": 90, "chunk_tiles": 128}),
    ]


def load_profiles_json(path: str) -> list[ExperimentProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("profiles json must be a list")
    out: list[ExperimentProfile] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("profile item must be an object")
        weights = item.get("objective_weights", {})
        out.append(
            ExperimentProfile(
                name=str(item["name"]),
                encode_config=dict(item.get("encode_config", {})),
                env={str(k): str(v) for k, v in dict(item.get("env", {})).items()},
                objective_weights=ObjectiveWeights(
                    lpips=float(weights.get("lpips", 1.0)),
                    one_minus_ssim=float(weights.get("one_minus_ssim", 0.5)),
                    edge_mse=float(weights.get("edge_mse", 0.5)),
                    bpp=float(weights.get("bpp", 0.25)),
                    encode_seconds=float(weights.get("encode_seconds", 0.02)),
                    decode_seconds=float(weights.get("decode_seconds", 0.02)),
                ),
            )
        )
    return out


def load_sweep_json(path: str) -> list[ExperimentProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep json must be an object")
    axes = raw.get("axes", {})
    if not isinstance(axes, dict) or not axes:
        raise ValueError("sweep json requires non-empty 'axes' object")
    name_template = str(raw.get("name_template", "sweep_{i}"))
    base = dict(raw.get("base_encode_config", {}))
    env = {str(k): str(v) for k, v in dict(raw.get("env", {})).items()}
    weights_raw = dict(raw.get("objective_weights", {}))
    weights = ObjectiveWeights(
        lpips=float(weights_raw.get("lpips", 1.0)),
        one_minus_ssim=float(weights_raw.get("one_minus_ssim", 0.5)),
        edge_mse=float(weights_raw.get("edge_mse", 0.5)),
        bpp=float(weights_raw.get("bpp", 0.25)),
        encode_seconds=float(weights_raw.get("encode_seconds", 0.02)),
        decode_seconds=float(weights_raw.get("decode_seconds", 0.02)),
    )

    keys = list(axes.keys())
    vals: list[list[Any]] = []
    for k in keys:
        v = axes[k]
        if not isinstance(v, list) or not v:
            raise ValueError(f"sweep axis '{k}' must be a non-empty list")
        vals.append(v)

    out: list[ExperimentProfile] = []
    for i, combo in enumerate(itertools.product(*vals)):
        cfg = dict(base)
        tags: dict[str, Any] = {"i": i}
        for k, vv in zip(keys, combo):
            cfg[k] = vv
            tags[k] = vv
        name = name_template.format(**tags)
        out.append(
            ExperimentProfile(
                name=name,
                encode_config=cfg,
                env=dict(env),
                objective_weights=weights,
            )
        )
    return out


def resolve_dataset_pack(dataset_dir: str, dataset_pack: str | None, packs_json: str | None) -> str:
    if not dataset_pack:
        return dataset_dir
    if not packs_json:
        raise ValueError("--dataset-pack requires --dataset-packs-json")
    packs = json.loads(Path(packs_json).read_text(encoding="utf-8"))
    if not isinstance(packs, dict) or "packs" not in packs:
        raise ValueError("dataset packs json must contain top-level 'packs' object")
    pack_map = packs.get("packs", {})
    if dataset_pack not in pack_map:
        raise ValueError(f"dataset pack not found: {dataset_pack}")
    rel = str(pack_map[dataset_pack])
    return str((Path(dataset_dir) / rel).resolve())


def run_experiment_suite(
    *,
    dataset_dir: str,
    output_dir: str,
    profiles: list[ExperimentProfile] | None = None,
    save_decoded: bool = True,
    save_weft: bool = True,
    generate_hybrids: bool = True,
    run_hybrid_pass: bool = True,
) -> ExperimentReport:
    imgs = _collect_images(dataset_dir)
    profs = profiles or default_profiles()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentImageResult] = []
    executed_profiles: list[ExperimentProfile] = []

    _evaluate_profiles(
        profiles=profs,
        imgs=imgs,
        out_root=out_root,
        results=results,
        save_decoded=save_decoded,
        save_weft=save_weft,
    )
    executed_profiles.extend(profs)

    base_leaderboard = _build_leaderboard(results)
    hybrids = _suggest_hybrid_profiles(leaderboard=base_leaderboard, profiles=profs) if generate_hybrids else []
    if generate_hybrids and run_hybrid_pass and hybrids:
        existing = {p.name for p in executed_profiles}
        hybrids_to_run = [h for h in hybrids if h.name not in existing]
        if hybrids_to_run:
            _evaluate_profiles(
                profiles=hybrids_to_run,
                imgs=imgs,
                out_root=out_root,
                results=results,
                save_decoded=save_decoded,
                save_weft=save_weft,
            )
            executed_profiles.extend(hybrids_to_run)

    leaderboard = _build_leaderboard(results)
    pareto = _pareto_front(leaderboard)
    bootstrap = _bootstrap_significance(results, leaderboard)

    rep = ExperimentReport(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        profiles=executed_profiles,
        results=results,
        leaderboard=leaderboard,
    )

    (out_root / "experiment_report.json").write_text(
        json.dumps(
            {
                "dataset_dir": rep.dataset_dir,
                "output_dir": rep.output_dir,
                "profiles": [asdict(p) for p in rep.profiles],
                "base_profiles": [asdict(p) for p in profs],
                "leaderboard": rep.leaderboard,
                "pareto_front": pareto,
                "bootstrap_significance": bootstrap,
                "hybrid_suggestions": [asdict(h) for h in hybrids],
                "hybrid_pass_executed": bool(generate_hybrids and run_hybrid_pass and len(hybrids) > 0),
                "hybrid_profiles_evaluated": [p.name for p in executed_profiles if p.name not in {bp.name for bp in profs}],
                "results": [asdict(r) for r in rep.results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_root / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        if leaderboard:
            writer = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
            writer.writeheader()
            writer.writerows(leaderboard)
    with (out_root / "pareto_front.csv").open("w", encoding="utf-8", newline="") as f:
        if pareto:
            writer = csv.DictWriter(f, fieldnames=list(pareto[0].keys()))
            writer.writeheader()
            writer.writerows(pareto)
    with (out_root / "results.csv").open("w", encoding="utf-8", newline="") as f:
        if results:
            rows = [
                {
                    "profile": r.profile,
                    "image": r.image,
                    "width": r.width,
                    "height": r.height,
                    "weft_bytes": r.weft_bytes,
                    "bpp": r.bpp,
                    "psnr": r.psnr,
                    "ssim": r.ssim,
                    "lpips": r.lpips,
                    "edge_mse": r.edge_mse,
                    "ocr_score": r.ocr_score,
                    "encode_ms": r.encode_ms,
                    "decode_ms": r.decode_ms,
                    "gpu_power_avg_w": r.gpu_power_avg_w,
                    "gpu_util_avg_pct": r.gpu_util_avg_pct,
                    "vram_peak_mib": r.vram_peak_mib,
                    "objective": r.objective,
                }
                for r in results
            ]
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_root / "hybrid_profiles.json").write_text(
        json.dumps([asdict(h) for h in hybrids], indent=2),
        encoding="utf-8",
    )
    return rep


def _build_leaderboard(results: list[ExperimentImageResult]) -> list[dict[str, Any]]:
    leaderboard_map: dict[str, list[ExperimentImageResult]] = {}
    for r in results:
        leaderboard_map.setdefault(r.profile, []).append(r)
    leaderboard: list[dict[str, Any]] = []
    for name, rows in leaderboard_map.items():
        leaderboard.append(
            {
                "profile": name,
                "image_count": len(rows),
                "avg_objective": _avg([r.objective for r in rows]),
                "avg_bpp": _avg([r.bpp for r in rows]),
                "avg_psnr": _avg([r.psnr for r in rows]),
                "avg_ssim": _avg([r.ssim for r in rows]),
                "avg_lpips": _avg([r.lpips for r in rows]),
                "avg_edge_mse": _avg([r.edge_mse for r in rows]),
                "avg_ocr_score": _avg([r.ocr_score for r in rows]),
                "avg_encode_ms": _avg([r.encode_ms for r in rows]),
                "avg_decode_ms": _avg([r.decode_ms for r in rows]),
                "avg_gpu_power_w": _avg([r.gpu_power_avg_w for r in rows]),
                "avg_gpu_util_pct": _avg([r.gpu_util_avg_pct for r in rows]),
                "avg_vram_peak_mib": _avg([r.vram_peak_mib for r in rows]),
            }
        )
    leaderboard.sort(key=lambda x: float(x["avg_objective"]) if x["avg_objective"] is not None else float("inf"))
    return leaderboard


def _evaluate_profiles(
    *,
    profiles: list[ExperimentProfile],
    imgs: list[Path],
    out_root: Path,
    results: list[ExperimentImageResult],
    save_decoded: bool,
    save_weft: bool,
) -> None:
    for p in profiles:
        profile_dir = out_root / p.name
        profile_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            src, w, h = load_image_linear(str(img))
            weft_path = profile_dir / f"{img.stem}.weft"
            dec_path = profile_dir / f"{img.stem}.decoded.png"

            old_env: dict[str, str | None] = {}
            try:
                for k, v in p.env.items():
                    old_env[k] = os.environ.get(k)
                    os.environ[k] = v

                t0 = time.perf_counter()
                enc = _run_encode_cli(
                    input_path=str(img),
                    output_path=str(weft_path),
                    encode_cfg=p.encode_config,
                    env=p.env,
                )
                encode_ms = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                dec = _run_decode_cli(
                    input_path=str(weft_path),
                    output_path=str(dec_path),
                    env=p.env,
                )
                decode_ms = (time.perf_counter() - t1) * 1000.0
            finally:
                for k, prev in old_env.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev

            dec_arr, _, _ = load_image_linear(str(dec_path))
            bytes_written = int(enc.get("bytes_written", weft_path.stat().st_size))
            bpp = float(bytes_written * 8.0 / float(w * h))
            p_psnr = float(psnr(src, dec_arr))
            p_ssim = ssim(src, dec_arr)
            try:
                p_lpips = lpips_score(src, dec_arr)
            except Exception:
                p_lpips = None
            e_mse = _edge_mse(src, dec_arr)
            ocr_score = _ocr_proxy_score(src, dec_arr)
            enc_gpu = dict((enc.get("metadata") or {}).get("gpu_sampling") or {})
            dec_gpu = dict((dec.get("metadata") or {}).get("gpu_sampling") or {})
            pwr_vals = [v for v in [enc_gpu.get("avg_power_w"), dec_gpu.get("avg_power_w")] if v is not None]
            util_vals = [v for v in [enc_gpu.get("avg_util_pct"), dec_gpu.get("avg_util_pct")] if v is not None]
            mem_vals = [v for v in [enc_gpu.get("max_mem_mib"), dec_gpu.get("max_mem_mib")] if v is not None]
            gpu_power_avg = float(sum(pwr_vals) / len(pwr_vals)) if pwr_vals else None
            gpu_util_avg = float(sum(util_vals) / len(util_vals)) if util_vals else None
            vram_peak = float(max(mem_vals)) if mem_vals else None

            obj = _objective(
                bpp=bpp,
                ssim_val=p_ssim,
                lpips_val=p_lpips,
                edge_mse=e_mse,
                encode_ms=encode_ms,
                decode_ms=decode_ms,
                w=p.objective_weights,
            )

            results.append(
                ExperimentImageResult(
                    profile=p.name,
                    image=img.name,
                    width=w,
                    height=h,
                    weft_bytes=bytes_written,
                    bpp=bpp,
                    psnr=p_psnr,
                    ssim=p_ssim,
                    lpips=p_lpips,
                    edge_mse=e_mse,
                    ocr_score=ocr_score,
                    encode_ms=encode_ms,
                    decode_ms=decode_ms,
                    gpu_power_avg_w=gpu_power_avg,
                    gpu_util_avg_pct=gpu_util_avg,
                    vram_peak_mib=vram_peak,
                    objective=obj,
                    metadata={
                        "encode": enc,
                        "decode": dec,
                    },
                )
            )

            if not save_weft:
                weft_path.unlink(missing_ok=True)
            if not save_decoded:
                dec_path.unlink(missing_ok=True)


def _pareto_front(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in leaderboard if r.get("avg_bpp") is not None and r.get("avg_psnr") is not None and r.get("avg_decode_ms") is not None]
    front: list[dict[str, Any]] = []
    for a in rows:
        dominated = False
        for b in rows:
            if a["profile"] == b["profile"]:
                continue
            better_or_equal = (
                float(b["avg_bpp"]) <= float(a["avg_bpp"])
                and float(b["avg_decode_ms"]) <= float(a["avg_decode_ms"])
                and float(b["avg_psnr"]) >= float(a["avg_psnr"])
            )
            strictly_better = (
                float(b["avg_bpp"]) < float(a["avg_bpp"])
                or float(b["avg_decode_ms"]) < float(a["avg_decode_ms"])
                or float(b["avg_psnr"]) > float(a["avg_psnr"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(dict(a))
    front.sort(key=lambda x: (float(x["avg_bpp"]), -float(x["avg_psnr"])))
    return front


def _bootstrap_significance(
    results: list[ExperimentImageResult],
    leaderboard: list[dict[str, Any]],
    n_boot: int = 1000,
) -> list[dict[str, Any]]:
    if not leaderboard:
        return []
    top = str(leaderboard[0]["profile"])
    per_profile: dict[str, dict[str, float]] = {}
    for r in results:
        per_profile.setdefault(r.profile, {})[r.image] = float(r.objective)
    if top not in per_profile:
        return []
    top_map = per_profile[top]
    images = sorted(top_map.keys())
    if not images:
        return []
    rng = np.random.default_rng(12345)
    out: list[dict[str, Any]] = []
    for row in leaderboard[1:]:
        other = str(row["profile"])
        other_map = per_profile.get(other, {})
        common = [im for im in images if im in other_map]
        if not common:
            continue
        diffs = np.array([other_map[im] - top_map[im] for im in common], dtype=np.float64)
        obs = float(np.mean(diffs))
        idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        boots = diffs[idx].mean(axis=1)
        p_leq_zero = float(np.mean(boots <= 0.0))
        out.append(
            {
                "top_profile": top,
                "other_profile": other,
                "mean_objective_delta_other_minus_top": obs,
                "p_other_better_or_equal": p_leq_zero,
                "n_images": len(common),
            }
        )
    return out


def _suggest_hybrid_profiles(leaderboard: list[dict[str, Any]], profiles: list[ExperimentProfile]) -> list[ExperimentProfile]:
    if not leaderboard:
        return []
    prof_map = {p.name: p for p in profiles}
    top_names = [str(r["profile"]) for r in leaderboard[: min(3, len(leaderboard))] if str(r["profile"]) in prof_map]
    tops = [prof_map[n] for n in top_names]
    if not tops:
        return []
    # weights: inverse objective
    weights: list[float] = []
    for r in leaderboard[: len(top_names)]:
        obj = float(r.get("avg_objective") or 1.0)
        weights.append(1.0 / max(obj, 1e-9))
    sw = sum(weights) if weights else 1.0
    weights = [w / sw for w in weights] if sw > 0 else [1.0 / len(tops)] * len(tops)

    def pick_str(key: str, default: str) -> str:
        score: dict[str, float] = {}
        for p, wt in zip(tops, weights):
            v = str(p.encode_config.get(key, default))
            score[v] = score.get(v, 0.0) + wt
        return max(score.items(), key=lambda kv: kv[1])[0]

    def pick_bool(key: str, default: bool) -> bool:
        s = 0.0
        for p, wt in zip(tops, weights):
            s += wt * (1.0 if bool(p.encode_config.get(key, default)) else 0.0)
        return s >= 0.5

    def pick_int(key: str, default: int, *, minv: int, maxv: int) -> int:
        v = 0.0
        for p, wt in zip(tops, weights):
            v += wt * float(int(p.encode_config.get(key, default)))
        iv = int(round(v))
        return max(minv, min(maxv, iv))

    weighted = ExperimentProfile(
        name="hybrid_weighted_top3",
        encode_config={
            "preset": pick_str("preset", "rtx-heavy-v2"),
            "quality": pick_int("quality", 75, minv=0, maxv=100),
            "chunk_tiles": pick_int("chunk_tiles", 64, minv=1, maxv=4096),
            "block_alignment": pick_int("block_alignment", 64, minv=1, maxv=4096),
            "entropy": pick_str("entropy", "chunked-rans"),
            "enable_res0": pick_bool("enable_res0", True),
            "enable_res1": pick_bool("enable_res1", True),
        },
    )
    quality_push = ExperimentProfile(
        name="hybrid_quality_push",
        encode_config={
            **weighted.encode_config,
            "quality": min(100, int(weighted.encode_config.get("quality", 75)) + 8),
            "chunk_tiles": max(16, int(weighted.encode_config.get("chunk_tiles", 64))),
        },
    )
    bitrate_push = ExperimentProfile(
        name="hybrid_bitrate_push",
        encode_config={
            **weighted.encode_config,
            "quality": max(0, int(weighted.encode_config.get("quality", 75)) - 6),
            "chunk_tiles": min(256, int(weighted.encode_config.get("chunk_tiles", 64)) * 2),
        },
    )
    return [weighted, quality_push, bitrate_push]


def _run_encode_cli(input_path: str, output_path: str, encode_cfg: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "weft.cli",
        "encode",
        input_path,
        output_path,
        "--preset",
        str(encode_cfg.get("preset", "rtx-heavy-v2")),
        "--quality",
        str(int(encode_cfg.get("quality", 75))),
        "--entropy",
        str(encode_cfg.get("entropy", "chunked-rans")),
        "--chunk-tiles",
        str(int(encode_cfg.get("chunk_tiles", 64))),
        "--block-align",
        str(int(encode_cfg.get("block_alignment", 64))),
    ]
    if not bool(encode_cfg.get("enable_res0", True)):
        cmd.append("--no-res0")
    if not bool(encode_cfg.get("enable_res1", True)):
        cmd.append("--no-res1")
    feature_flags = dict(encode_cfg.get("feature_flags") or {})
    for k in (
        "hierarchical_tiling_level",
        "edge_budget_boost_level",
        "primitive_dictionary_level",
        "stroke_objective_level",
        "subpixel_primitives_level",
        "res2_basis_blocks_level",
        "mixed_action_beam_level",
        "residual_patch_borrow_level",
        "decode_refinement_level",
        "entropy_context_model_level",
    ):
        if k in encode_cfg:
            feature_flags[k] = encode_cfg[k]
    if isinstance(feature_flags, dict) and feature_flags:
        cmd.extend(["--feature-flags-json", json.dumps(feature_flags, separators=(",", ":"))])
    if encode_cfg.get("multi_rounds") is not None:
        cmd.extend(["--multi-rounds", str(int(encode_cfg.get("multi_rounds", 1)))])
    if bool(encode_cfg.get("adaptive_tile_budget", False)):
        cmd.append("--adaptive-tile-budget")
    if bool(encode_cfg.get("edge_weighted_objective", False)):
        cmd.append("--edge-weighted-objective")
    if encode_cfg.get("target_bpp") is not None:
        cmd.extend(["--target-bpp", str(float(encode_cfg["target_bpp"]))])
    if encode_cfg.get("candidate_bank") is not None:
        cmd.extend(["--candidate-bank", str(encode_cfg["candidate_bank"])])
    if bool(encode_cfg.get("enable_res2", False)):
        cmd.append("--enable-res2")
    if encode_cfg.get("search_mode") is not None:
        cmd.extend(["--search-mode", str(encode_cfg["search_mode"])])
    if encode_cfg.get("beam_width") is not None:
        cmd.extend(["--beam-width", str(int(encode_cfg["beam_width"]))])
    if encode_cfg.get("mcmc_steps") is not None:
        cmd.extend(["--mcmc-steps", str(int(encode_cfg["mcmc_steps"]))])
    if encode_cfg.get("stochastic_restarts") is not None:
        cmd.extend(["--stochastic-restarts", str(int(encode_cfg["stochastic_restarts"]))])
    if encode_cfg.get("early_exit_patience") is not None:
        cmd.extend(["--early-exit-patience", str(int(encode_cfg["early_exit_patience"]))])
    if encode_cfg.get("maxcompute_fit_passes") is not None:
        cmd.extend(["--maxcompute-fit-passes", str(int(encode_cfg["maxcompute_fit_passes"]))])
    if bool(encode_cfg.get("container_v2_blocks", False)):
        cmd.append("--container-v2-blocks")
    if bool(encode_cfg.get("split_entropy_streams", False)):
        cmd.append("--split-entropy-streams")
    if bool(encode_cfg.get("neighbor_delta_coding", False)):
        cmd.append("--neighbor-delta-coding")
    out = _run_cli_json(cmd, env=env)
    if not isinstance(out, dict):
        raise RuntimeError("encode CLI did not return JSON object")
    return out


def _run_decode_cli(input_path: str, output_path: str, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "weft.cli",
        "decode",
        input_path,
        output_path,
        "--require-gpu-entropy",
    ]
    out = _run_cli_json(cmd, env=env)
    if not isinstance(out, dict):
        raise RuntimeError("decode CLI did not return JSON object")
    return out


def _run_cli_json(cmd: list[str], env: dict[str, str]) -> Any:
    run_env = os.environ.copy()
    run_env.update(env)
    run_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=run_env)
    gpu_samples: list[tuple[float, float, float]] = []
    while True:
        try:
            out, err = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            try:
                q = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if q.returncode == 0 and q.stdout.strip():
                    line = q.stdout.strip().splitlines()[0]
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 3:
                        gpu_samples.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except Exception:
                pass
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{err}\n{out}")
    txt = out.strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"expected JSON output, got: {txt}")
    payload = json.loads(txt[start : end + 1])
    if isinstance(payload, dict) and gpu_samples:
        md = dict(payload.get("metadata") or {})
        pwr = [s[0] for s in gpu_samples]
        utl = [s[1] for s in gpu_samples]
        mem = [s[2] for s in gpu_samples]
        md["gpu_sampling"] = {
            "samples": len(gpu_samples),
            "avg_power_w": float(sum(pwr) / len(pwr)),
            "max_power_w": float(max(pwr)),
            "avg_util_pct": float(sum(utl) / len(utl)),
            "max_util_pct": float(max(utl)),
            "max_mem_mib": float(max(mem)),
        }
        payload["metadata"] = md
    return payload
