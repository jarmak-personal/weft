#!/usr/bin/env python3
"""Run staged WEFT research campaign (single-technique, pairwise, hybrid)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

LEVELS = ("off", "mid", "max")

TECHNIQUES: dict[str, dict[str, Any]] = {
    "hierarchical_tiling_level": {
        "label": "hierarchical_tiling",
        "mid": "mid",
        "max": "max",
    },
    "edge_budget_boost_level": {
        "label": "edge_budget_boost",
        "mid": "mid",
        "max": "max",
    },
    "primitive_dictionary_level": {
        "label": "primitive_dictionary",
        "mid": "mid",
        "max": "max",
    },
    "stroke_objective_level": {
        "label": "stroke_objective",
        "mid": "mid",
        "max": "max",
    },
    "subpixel_primitives_level": {
        "label": "subpixel_primitives",
        "mid": "mid",
        "max": "max",
    },
    "res2_basis_blocks_level": {
        "label": "res2_basis_blocks",
        "mid": "mid",
        "max": "max",
    },
    "mixed_action_beam_level": {
        "label": "mixed_action_beam",
        "mid": "mid",
        "max": "max",
    },
    "residual_patch_borrow_level": {
        "label": "residual_patch_borrow",
        "mid": "mid",
        "max": "max",
    },
    "decode_refinement_level": {
        "label": "decode_refinement",
        "mid": "mid",
        "max": "max",
    },
    "entropy_context_model_level": {
        "label": "entropy_context_model",
        "mid": "mid",
        "max": "max",
    },
}


def _default_baseline() -> dict[str, Any]:
    return {
        "preset": "rtx-heavy-v2",
        "quality": 82,
        "chunk_tiles": 64,
        "candidate_bank": "rich18",
        "search_mode": "greedy",
        "multi_rounds": 4,
        "adaptive_tile_budget": True,
        "edge_weighted_objective": True,
        "target_bpp": 2.2,
        "enable_res2": True,
        "container_v2_blocks": True,
        "split_entropy_streams": True,
        "neighbor_delta_coding": True,
    }


def _profile(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "encode_config": cfg,
        "objective_weights": {
            "lpips": 1.0,
            "one_minus_ssim": 0.5,
            "edge_mse": 0.5,
            "bpp": 0.25,
            "encode_seconds": 0.02,
            "decode_seconds": 0.02,
        },
    }


def _run_experiment(
    *,
    dataset_dir: str,
    out_dir: Path,
    profiles: list[dict[str, Any]],
    python_exe: str,
    dataset_pack: str | None,
    dataset_packs_json: str | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles_json = out_dir / "profiles.json"
    profiles_json.write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    cmd = [
        python_exe,
        "-m",
        "weft.cli",
        "experiment",
        dataset_dir,
        str(out_dir),
        "--profiles-json",
        str(profiles_json),
    ]
    if dataset_pack:
        cmd.extend(["--dataset-pack", dataset_pack])
    if dataset_packs_json:
        cmd.extend(["--dataset-packs-json", dataset_packs_json])

    env = dict(**__import__("os").environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{py_path}" if py_path else "src"

    subprocess.run(cmd, check=True, env=env)
    rep_path = out_dir / "experiment_report.json"
    return json.loads(rep_path.read_text(encoding="utf-8"))


def _lb_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["profile"]): r for r in report.get("leaderboard", [])}


def _pick_winners(
    *,
    base_row: dict[str, Any],
    report: dict[str, Any],
    bpp_slack: float,
) -> list[str]:
    winners: list[str] = []
    for row in report.get("leaderboard", []):
        name = str(row.get("profile", ""))
        if name == "baseline":
            continue
        d_obj = float(row.get("avg_objective") or 1e99) - float(base_row.get("avg_objective") or 1e99)
        d_bpp = float(row.get("avg_bpp") or 1e99) - float(base_row.get("avg_bpp") or 1e99)
        if d_obj < 0.0 and d_bpp <= bpp_slack:
            winners.append(name)
    if not winners:
        # fallback: take top 5 non-baseline
        winners = [str(r["profile"]) for r in report.get("leaderboard", []) if str(r.get("profile")) != "baseline"][:5]
    return winners


def _decode_profile_name(name: str) -> tuple[str, str] | None:
    # format: t_<label>_<level>
    if not name.startswith("t_"):
        return None
    parts = name.split("_")
    if len(parts) < 3:
        return None
    level = parts[-1]
    label = "_".join(parts[1:-1])
    return label, level


def _write_interaction_matrix(
    out_csv: Path,
    baseline_row: dict[str, Any],
    pair_report: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    b_obj = float(baseline_row.get("avg_objective") or 1e99)
    b_bpp = float(baseline_row.get("avg_bpp") or 1e99)
    b_psnr = float(baseline_row.get("avg_psnr") or -1e99)
    b_dec = float(baseline_row.get("avg_decode_ms") or 1e99)

    for r in pair_report.get("leaderboard", []):
        name = str(r.get("profile", ""))
        if name == "baseline":
            continue
        rows.append(
            {
                "profile": name,
                "delta_objective": float(r.get("avg_objective") or 1e99) - b_obj,
                "delta_bpp": float(r.get("avg_bpp") or 1e99) - b_bpp,
                "delta_psnr": float(r.get("avg_psnr") or -1e99) - b_psnr,
                "delta_decode_ms": float(r.get("avg_decode_ms") or 1e99) - b_dec,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = _default_baseline()
    baseline_profile = _profile("baseline", dict(baseline_cfg))

    # Phase B: single-technique ablation
    phase_b_profiles = [baseline_profile]
    for key, info in TECHNIQUES.items():
        label = str(info["label"])
        for lvl in ("mid", "max"):
            cfg = dict(baseline_cfg)
            cfg[key] = info[lvl]
            phase_b_profiles.append(_profile(f"t_{label}_{lvl}", cfg))

    if args.dry_run:
        (out_root / "phase_b_profiles.json").write_text(json.dumps(phase_b_profiles, indent=2), encoding="utf-8")
        return {
            "mode": "dry_run",
            "phase_b_profile_count": len(phase_b_profiles),
            "phase_c_pair_max": args.max_pairs,
            "phase_d_hybrids": 3,
            "output_dir": str(out_root),
        }

    phase_b_dir = out_root / "phase_b_single"
    rep_b = _run_experiment(
        dataset_dir=args.dataset_dir,
        out_dir=phase_b_dir,
        profiles=phase_b_profiles,
        python_exe=args.python,
        dataset_pack=args.dataset_pack,
        dataset_packs_json=args.dataset_packs_json,
    )
    bm = _lb_map(rep_b)
    if "baseline" not in bm:
        raise RuntimeError("baseline missing from phase B leaderboard")
    winners = _pick_winners(base_row=bm["baseline"], report=rep_b, bpp_slack=float(args.bpp_slack))

    # Phase C: pairwise among winners
    winner_profiles = [p for p in phase_b_profiles if p["name"] in set(winners)]
    combos = list(itertools.combinations(winner_profiles, 2))
    combos = combos[: max(0, int(args.max_pairs))]
    phase_c_profiles = [baseline_profile]
    for a, b in combos:
        cfg = dict(baseline_cfg)
        cfg.update(a["encode_config"])
        cfg.update(b["encode_config"])
        nm = f"pair__{a['name']}__{b['name']}"
        phase_c_profiles.append(_profile(nm, cfg))

    phase_c_dir = out_root / "phase_c_pairs"
    rep_c = _run_experiment(
        dataset_dir=args.dataset_dir,
        out_dir=phase_c_dir,
        profiles=phase_c_profiles,
        python_exe=args.python,
        dataset_pack=args.dataset_pack,
        dataset_packs_json=args.dataset_packs_json,
    )
    _write_interaction_matrix(out_root / "interaction_matrix.csv", _lb_map(rep_c).get("baseline", bm["baseline"]), rep_c)

    # Phase D: hybrid combinations (top singles + top pairs)
    top_b = [str(r["profile"]) for r in rep_b.get("leaderboard", []) if str(r.get("profile")) != "baseline"][:3]
    top_c = [str(r["profile"]) for r in rep_c.get("leaderboard", []) if str(r.get("profile")) != "baseline"][:3]

    profile_map = {p["name"]: p for p in (phase_b_profiles + phase_c_profiles)}

    merged_cfg = dict(baseline_cfg)
    for n in top_b + top_c:
        p = profile_map.get(n)
        if p is not None:
            merged_cfg.update(p["encode_config"])

    bpp_guarded_cfg = dict(merged_cfg)
    bpp_guarded_cfg["target_bpp"] = min(float(merged_cfg.get("target_bpp", 2.2)), float(args.hybrid_bpp_cap))

    maxcompute_cfg = dict(merged_cfg)
    maxcompute_cfg["search_mode"] = "mcmc"
    maxcompute_cfg["mcmc_steps"] = 96
    maxcompute_cfg["stochastic_restarts"] = 3
    maxcompute_cfg["multi_rounds"] = max(int(maxcompute_cfg.get("multi_rounds", 4)), 6)

    phase_d_profiles = [
        baseline_profile,
        _profile("hybrid_readability_push", merged_cfg),
        _profile("hybrid_bpp_guarded", bpp_guarded_cfg),
        _profile("hybrid_maxcompute_text", maxcompute_cfg),
    ]

    phase_d_dir = out_root / "phase_d_hybrids"
    rep_d = _run_experiment(
        dataset_dir=args.dataset_dir,
        out_dir=phase_d_dir,
        profiles=phase_d_profiles,
        python_exe=args.python,
        dataset_pack=args.dataset_pack,
        dataset_packs_json=args.dataset_packs_json,
    )

    base_d = _lb_map(rep_d).get("baseline")
    top_d = (rep_d.get("leaderboard") or [{}])[0]
    gate = {
        "baseline_profile": "baseline",
        "top_profile": top_d.get("profile"),
        "delta_objective_top_minus_baseline": None,
        "delta_decode_ms_top_minus_baseline": None,
        "decode_regression_limit_ms": float(args.decode_regression_limit_ms),
        "requires_two_packs": True,
        "dataset_pack_used": args.dataset_pack,
        "pass": False,
        "reason": "",
    }
    if isinstance(base_d, dict) and isinstance(top_d, dict):
        d_obj = float(top_d.get("avg_objective") or 1e99) - float(base_d.get("avg_objective") or 1e99)
        d_dec = float(top_d.get("avg_decode_ms") or 1e99) - float(base_d.get("avg_decode_ms") or 1e99)
        gate["delta_objective_top_minus_baseline"] = d_obj
        gate["delta_decode_ms_top_minus_baseline"] = d_dec
        enough_packs = args.dataset_pack is None
        improves_obj = d_obj < 0.0
        decode_ok = d_dec <= float(args.decode_regression_limit_ms)
        gate["pass"] = bool(enough_packs and improves_obj and decode_ok)
        if not enough_packs:
            gate["reason"] = "single dataset pack run; requires at least 2 packs for acceptance gate"
        elif not improves_obj:
            gate["reason"] = "objective did not improve vs baseline"
        elif not decode_ok:
            gate["reason"] = "decode regression exceeded limit"
        else:
            gate["reason"] = "passed"

    campaign_summary = {
        "dataset_dir": args.dataset_dir,
        "output_dir": str(out_root),
        "phase_b": {
            "dir": str(phase_b_dir),
            "profile_count": len(phase_b_profiles),
            "winners": winners,
        },
        "phase_c": {
            "dir": str(phase_c_dir),
            "profile_count": len(phase_c_profiles),
            "pairs_run": len(phase_c_profiles) - 1,
            "interaction_matrix_csv": str(out_root / "interaction_matrix.csv"),
        },
        "phase_d": {
            "dir": str(phase_d_dir),
            "profile_count": len(phase_d_profiles),
            "hybrid_candidates": [p["name"] for p in phase_d_profiles if p["name"] != "baseline"],
        },
        "acceptance_gate": gate,
        "top_phase_d": rep_d.get("leaderboard", [])[:5],
    }
    (out_root / "hybrid_candidates.json").write_text(
        json.dumps([p for p in phase_d_profiles if p["name"] != "baseline"], indent=2),
        encoding="utf-8",
    )
    (out_root / "campaign_summary.json").write_text(json.dumps(campaign_summary, indent=2), encoding="utf-8")
    (out_root / "acceptance_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return campaign_summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run WEFT staged campaign for technique testing + hybrid combination")
    ap.add_argument("dataset_dir", help="Dataset root")
    ap.add_argument("output_dir", help="Campaign output directory")
    ap.add_argument("--python", default=sys.executable, help="Python executable for weft.cli invocations")
    ap.add_argument("--dataset-pack", default=None, help="Dataset pack key")
    ap.add_argument("--dataset-packs-json", default=None, help="Dataset packs JSON")
    ap.add_argument("--bpp-slack", type=float, default=0.2, help="Max allowed bpp delta for winner selection")
    ap.add_argument("--max-pairs", type=int, default=15, help="Max pairwise profiles to run in phase C")
    ap.add_argument("--hybrid-bpp-cap", type=float, default=2.4, help="Target bpp cap for hybrid_bpp_guarded")
    ap.add_argument("--decode-regression-limit-ms", type=float, default=25.0, help="Max allowed decode-ms regression for acceptance gate")
    ap.add_argument("--dry-run", action="store_true", help="Only emit phase B profile file and summary")
    args = ap.parse_args()

    summary = run_campaign(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
