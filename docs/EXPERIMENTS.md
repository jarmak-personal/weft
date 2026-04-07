# Experiments Harness

Run multi-profile sweeps with artifact capture and leaderboard outputs.

## CLI

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./dataset ./exp_out
```

Optional profile file:

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./dataset ./exp_out \
  --profiles-json ./profiles.json
```

Cartesian sweep file:

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./dataset ./exp_out \
  --sweep-json ./sweep.json
```

Dataset packs:

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./datasets ./exp_out \
  --dataset-packs-json ./configs/dataset_packs.json \
  --dataset-pack text_ui
```

Disable hybrid suggestion generation:

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./dataset ./exp_out \
  --no-generate-hybrids
```

Disable automatic second-pass execution of generated hybrids:

```bash
PYTHONPATH=src .venv/bin/python -m weft.cli experiment ./dataset ./exp_out \
  --no-run-hybrid-pass
```

## Default Outputs

Under `./exp_out`:

- `experiment_report.json` — full run report (profiles, per-image results, leaderboard)
- `leaderboard.csv` — per-profile averages sorted by objective
- `pareto_front.csv` — Pareto-optimal profiles on `(bpp, decode_ms, psnr)`
- `results.csv` — per-image rows across all profiles
- `hybrid_profiles.json` — auto-generated profile suggestions from top performers
- `./<profile-name>/*.weft` and `./<profile-name>/*.decoded.png` artifacts

Use `--no-save-weft` and/or `--no-save-decoded` to disable artifact retention.

## Profiles JSON Schema

`profiles.json` must be a list of objects:

```json
[
  {
    "name": "q90_chunk32",
    "encode_config": {
      "quality": 90,
      "chunk_tiles": 32
    },
    "env": {
      "SOME_EXPERIMENT_FLAG": "1"
    },
    "objective_weights": {
      "lpips": 1.0,
      "one_minus_ssim": 0.5,
      "edge_mse": 0.5,
      "bpp": 0.25,
      "encode_seconds": 0.02,
      "decode_seconds": 0.02
    }
  }
]
```

## Sweep JSON Schema

```json
{
  "name_template": "q{quality}_c{chunk_tiles}",
  "base_encode_config": {
    "preset": "rtx-heavy-v2",
    "entropy": "chunked-rans"
  },
  "axes": {
    "quality": [60, 75, 90],
    "chunk_tiles": [32, 64, 128]
  }
}
```

## Objective

Per-image scalar used for profile ranking:

`score = w_lpips*lpips + w_ssim*(1-ssim) + w_edge*edge_mse + w_bpp*bpp + w_enc*encode_s + w_dec*decode_s`

Lower is better.

## Statistical Comparison

`experiment_report.json` includes `bootstrap_significance`, comparing each profile
against the top profile using bootstrap resampling over per-image objective deltas.

- `mean_objective_delta_other_minus_top < 0` suggests the other profile may be better.
- `p_other_better_or_equal` near `0` means low evidence the other profile matches/beats top.
- `p_other_better_or_equal` near `1` means high evidence the other profile matches/beats top.

## Auto Hybrid Second Pass

By default, generated hybrid profiles are immediately evaluated in the same run.
This means leaderboard/pareto/significance include both base profiles and hybrid
profiles unless `--no-run-hybrid-pass` is set.

## Regression Dashboard

Compare two experiment reports:

```bash
python scripts/experiment_regression.py \
  /path/to/baseline/experiment_report.json \
  /path/to/candidate/experiment_report.json
```

## Staged Campaign Runner

Run phased single-technique ablation, pairwise interaction testing, and hybrid
combination search in one command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_campaign.py \
  ./datasets \
  ./campaign_out \
  --dataset-packs-json ./configs/dataset_packs.json \
  --dataset-pack text_ui
```

Key outputs under `./campaign_out`:

- `phase_b_single/` — single-technique ablation run
- `phase_c_pairs/` — pairwise interaction run
- `phase_d_hybrids/` — hybrid candidate run
- `interaction_matrix.csv` — pairwise deltas vs baseline
- `hybrid_candidates.json` — generated hybrid configs
- `campaign_summary.json` — top-level campaign summary
- `acceptance_gate.json` — merge-gate style pass/fail summary

Use `--dry-run` to only emit phase profile definitions without running encodes:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_campaign.py ./datasets ./campaign_out --dry-run
```

## Extended Metrics

Experiment leaderboard/results now include additional operational metrics:

- `ocr_score` (edge-structure readability proxy; higher is better)
- `gpu_power_avg_w`
- `gpu_util_avg_pct`
- `vram_peak_mib`
