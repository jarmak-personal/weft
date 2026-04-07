#!/usr/bin/env python3
"""Compare two WEFT experiment_report.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _leaderboard_map(rep: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rep.get("leaderboard", []):
        name = str(row.get("profile"))
        out[name] = row
    return out


def _f(v):
    if v is None:
        return "n/a"
    return f"{float(v):.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Regression summary between two experiment reports")
    ap.add_argument("baseline_report", help="Path to baseline experiment_report.json")
    ap.add_argument("candidate_report", help="Path to candidate experiment_report.json")
    args = ap.parse_args()

    base = _load(args.baseline_report)
    cand = _load(args.candidate_report)
    bm = _leaderboard_map(base)
    cm = _leaderboard_map(cand)
    common = sorted(set(bm.keys()) & set(cm.keys()))

    print("# WEFT Experiment Regression")
    print()
    print(f"- baseline: `{args.baseline_report}`")
    print(f"- candidate: `{args.candidate_report}`")
    print(f"- common_profiles: `{len(common)}`")
    print()
    print("| profile | delta_objective | delta_bpp | delta_psnr | delta_decode_ms |")
    print("|---|---:|---:|---:|---:|")
    for p in common:
        b = bm[p]
        c = cm[p]
        d_obj = (c.get("avg_objective") or 0.0) - (b.get("avg_objective") or 0.0)
        d_bpp = (c.get("avg_bpp") or 0.0) - (b.get("avg_bpp") or 0.0)
        d_psnr = (c.get("avg_psnr") or 0.0) - (b.get("avg_psnr") or 0.0)
        d_dec = (c.get("avg_decode_ms") or 0.0) - (b.get("avg_decode_ms") or 0.0)
        print(f"| {p} | {_f(d_obj)} | {_f(d_bpp)} | {_f(d_psnr)} | {_f(d_dec)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
