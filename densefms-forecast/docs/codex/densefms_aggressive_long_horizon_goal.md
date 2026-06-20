# DenseFMS Aggressive Long-Horizon Refinement Goal

## 0. Purpose

이번 goal의 최우선 목표는 **H=1 성능 개선이 아니라**, `horizon_seconds = 5, 10, 15`에서 validation MAE를 유의미하게 낮추는 것이다.

장시간 실행을 허용하되, 중간에 interrupt되더라도 현재까지의 결과를 확인하고 재개할 수 있도록 모든 진행 상황과 중간 결과를 지속적으로 기록해야 한다.

This is a **validation-only refinement goal**. Do not evaluate on the test set.

---

## 1. Baselines and Targets

Use the previous DenseFMS long target search results as baselines.

### 1.1 Best-score baseline

| Horizon | Baseline validation MAE |
|---:|---:|
| H=5 | ~1.287 |
| H=10 | ~1.626 |
| H=15 | ~1.848 |

### 1.2 Deployment-realistic baseline

| Horizon | Baseline validation MAE |
|---:|---:|
| H=5 | ~1.773 |
| H=10 | ~1.920 |
| H=15 | ~2.038 |

### 1.3 Aggressive targets

#### Best-score aggressive target

| Horizon | Target validation MAE |
|---:|---:|
| H=5 | <= 1.10 |
| H=10 | <= 1.35 |
| H=15 | <= 1.55 |

#### Best-score stretch target

| Horizon | Target validation MAE |
|---:|---:|
| H=5 | <= 1.00 |
| H=10 | <= 1.25 |
| H=15 | <= 1.45 |

#### Deployment-realistic aggressive target

| Horizon | Target validation MAE |
|---:|---:|
| H=5 | <= 1.45 |
| H=10 | <= 1.65 |
| H=15 | <= 1.80 |

#### Deployment-realistic stretch target

| Horizon | Target validation MAE |
|---:|---:|
| H=5 | <= 1.30 |
| H=10 | <= 1.55 |
| H=15 | <= 1.70 |

### 1.4 Success criteria

Primary success:

- Improve validation MAE by at least 0.10 over the corresponding baseline for at least two of H=5/H=10/H=15.

Strong success:

- Improve H=10 or H=15 validation MAE by at least 0.20 over the corresponding baseline.

Excellent success:

- Improve all H=5/H=10/H=15 baselines and reach the aggressive target for either H=10 or H=15.

Stretch success:

- Reach the stretch target for either H=10 or H=15.

---

## 2. Non-negotiable Rules

- Do not use the test set.
- Do not run test evaluation.
- Model selection, pruning, ranking, early stopping, and multi-seed candidate selection must use validation metrics only.
- Do not optimize H=1 as the primary objective.
- H=1 may be used only as auxiliary supervision.
- Do not treat H=1 performance as long-horizon success.
- Do not use head or FMS values after the current index as input.
- Target `FMS[t + horizon_steps]` must never enter the input features.
- Recent window must use only head/motion values at or before current time `t`.
- Sparse anchors must be selected only from FMS values at or before current time `t`.
- Anchor history must include only FMS values at or before current time `t`.
- Cumulative exposure features must use only head/motion values at or before current time `t`.
- `recent_start_observed` is diagnostic/upper-bound only, not deployment-realistic.
- Preserve existing `runs/densefms_long_target_search/` results.
- Do not overwrite the previous H=1 best-score result.
- Do not commit or push.
- Do not add runs, artifacts, checkpoints, or prediction CSV files to git.

---

## 3. Output Directory

All outputs for this goal must be stored under:

```text
runs/densefms_long_horizon_aggressive/
```

Required subdirectories:

```text
diagnostics/
live/
manifests/
summaries/
plots/
selected_candidates/
multiseed/
failed_runs/
logs/
```

---

## 4. Interrupt-safe Logging Requirements

This goal may run for a long time. The run must remain inspectable even if interrupted.

### 4.1 Planned manifest

Create and continuously maintain:

```text
manifests/planned_manifest.json
```

Each planned run must include:

- `run_id`
- `run_name`
- `model`
- `horizon_seconds`
- `anchor_interval`
- `recent_window_seconds`
- `calibration_seconds`
- `variant`
- `seed`
- `command`
- `expected_output_dir`

### 4.2 Run status log

Create append-only JSONL:

```text
manifests/run_status.jsonl
```

Append a record whenever a run changes state.

Fields:

- `timestamp`
- `run_id`
- `run_name`
- `status`: `planned|started|epoch_end|completed|failed|skipped|interrupted`
- `horizon_seconds`
- `model`
- `variant`
- `seed`
- `epoch`
- `best_val_MAE_so_far`
- `best_val_RMSE_so_far`
- `checkpoint_path`
- `metrics_path`
- `error_message` if failed

### 4.3 Progress log

Create append-only JSONL:

```text
live/progress_log.jsonl
```

Log these events:

- `goal_start`
- `sanity_start`
- `sanity_pass`
- `smoke_start`
- `smoke_pass`
- `run_started`
- `epoch_end`
- `validation_evaluated`
- `run_completed`
- `run_failed`
- `leaderboard_updated`
- `pruning_decision`
- `multiseed_started`
- `multiseed_completed`
- `goal_interrupted`
- `goal_completed`

### 4.4 Live leaderboard

Create and update after every completed run:

```text
live/leaderboard_live.csv
```

Required columns:

- `rank_overall`
- `rank_by_horizon`
- `run_id`
- `run_name`
- `model`
- `horizon_seconds`
- `track`: `best_score|deployment_realistic|diagnostic`
- `variant`
- `anchor_interval`
- `recent_window_seconds`
- `calibration_seconds`
- `seed`
- `val_MAE`
- `val_RMSE`
- `val_R2`
- `val_sMAPE`
- `common_val_MAE`
- `high_fms_MAE`
- `high_fms_false_negative_rate`
- `high_change_MAE`
- `trend_macro_f1`
- `true_delta_MAE`
- `pred_delta_saturation_rate`
- `improvement_vs_best_score_baseline`
- `improvement_vs_deployment_baseline`
- `checkpoint_path`
- `metrics_path`
- `predictions_path`
- `plot_dir`
- `status`

### 4.5 Live best summary

Create and update after every completed run:

```text
live/best_by_horizon_live.md
```

Must include:

- H=5 top 10
- H=10 top 10
- H=15 top 10
- deployment-realistic top 10
- best-score top 10
- current aggressive target achievement status
- current stretch target achievement status

### 4.6 Current best JSON

Create and update:

```text
live/current_best.json
```

Must always contain:

- best H=5
- best H=10
- best H=15
- best deployment H=5
- best deployment H=10
- best deployment H=15

### 4.7 Partial summary

Create and update after every completed run:

```text
summaries/aggressive_search_summary_partial.md
```

This file must be readable as an up-to-date partial report if the job is interrupted.

Include:

- completed/failed/skipped run count
- best validation MAE by horizon
- improvement over baselines
- most effective variants so far
- failed variants
- next planned runs
- current interpretation

### 4.8 Per-run logs

Each run must save stdout/stderr:

```text
logs/{run_name}.stdout.log
logs/{run_name}.stderr.log
```

### 4.9 Atomic writes

Write CSV/MD/JSON summaries to a temporary file first, then rename, so interruption does not corrupt files.

### 4.10 Resume support

The runner must support:

```text
--resume
--skip_existing
--retry_failed
```

Rules:

- Completed runs are skipped.
- Failed runs are skipped unless `--retry_failed` is set.
- Interrupted/running runs should inspect output directories and checkpoints, then resume if possible or restart safely.

### 4.11 Signal handling

If possible, catch SIGINT/SIGTERM and:

- mark current run as `interrupted`
- flush `leaderboard_live.csv`
- flush `best_by_horizon_live.md`
- flush `aggressive_search_summary_partial.md`
- exit safely

---

## 5. Initial Diagnostics Before New Training

Before running new variants, analyze previous H=5/H=10/H=15 best-score and deployment validation prediction CSVs.

Output directory:

```text
diagnostics/
```

Required analyses:

- horizon-wise MAE/RMSE/R2/sMAPE
- target FMS bin MAE
  - bins: `0-3`, `3-6`, `6-9`, `9-12`, `12+`
- true delta bin MAE
  - `true_delta = target_fms - anchor_fms`
  - bins: `|delta| < 0.5`, `0.5-1`, `1-2`, `2-3`, `3+`
- predicted delta distribution
  - `pred_delta = pred_fms - anchor_fms`
- true delta vs predicted delta scatter
- delta saturation analysis
- trend-wise MAE: `up|down|stable`
- high-FMS MAE
  - target >= 8
  - target >= 10
- high-FMS false negative rate
  - target >= 8 and pred < 6
  - target >= 10 and pred < 8
- high-change MAE
  - `|target - anchor| >= 2`
  - `|target - anchor| >= 3`
- subject-wise worst MAE
- calibration slope bin MAE
- anchor interval MAE
- recent-window MAE
- horizon-wise predicted-vs-target plot
- horizon-wise residual histogram

Required output files:

```text
diagnostics/diagnostics_summary.csv
diagnostics/diagnostics_summary.md
diagnostics/plots/*.png
```

`diagnostics_summary.md` must answer:

- Does H=10/H=15 mainly underpredict high FMS?
- Does it fail on large true-delta regions?
- Does it miss upward trends?
- Is delta prediction saturated?
- Does it fail on specific subjects or calibration-slope bins?

---

## 6. New Features to Implement

### 6.1 AnchorHistoryEncoder

Rationale: H=10/H=15 may require recent FMS anchor trajectory, not just the latest anchor value.

CLI options:

```text
--use_anchor_history
--anchor_history_k 3|5|7
--anchor_history_encoder mlp|gru|tcn
--anchor_history_dim 32|64
--anchor_history_dropout
```

Each anchor history token must include:

- `anchor_value`
- `delta_from_previous_anchor`
- `time_gap_from_previous_anchor`
- `time_since_current`
- `observed_mask`

Apply to:

- `recent_tcn_summary_calib`
- `lc_sa_tcnformer`
- `anchor_delta_mlp`
- `gated_fusion` if feasible

Fusion:

- RecentTCN+SummaryCalib: concat summary features + recent TCN vector + anchor history embedding.
- LC-SA-TCNFormer: concat anchor history embedding into `z_anchor` or final fusion input.
- AnchorDeltaMLP: concat anchor history embedding to handcrafted feature vector.

Sanity requirements:

- Anchor history must use only FMS values at or before current index.
- It must not use target horizon or future FMS.
- If there are fewer than K anchors, use zero padding and `observed_mask`.

### 6.2 CumulativeExposure features

Rationale: H=10/H=15 may require exposure accumulated so far, not just the recent window.

CLI options:

```text
--use_cumulative_exposure
--exposure_windows 30 60 120
```

Features:

- `session_so_far_motion_mean`
- `session_so_far_motion_std`
- `session_so_far_motion_energy`
- `session_so_far_rotation_energy`
- `session_so_far_acceleration_energy`
- `last_30s_motion_energy`
- `last_60s_motion_energy`
- `last_120s_motion_energy`
- `recent_to_cumulative_energy_ratio`
- `time_since_calibration`
- `time_since_session_start`
- `cumulative_rotation_magnitude`
- `cumulative_acceleration_magnitude`

Apply to:

- `recent_tcn_summary_calib`
- `lc_sa_tcnformer`
- `anchor_delta_mlp`
- `gated_fusion` if feasible

Sanity:

- Exposure features must use only head/motion at or before current index.
- No future motion leakage.

### 6.3 Horizon-dependent delta scale

Rationale: fixed delta scale may limit long-horizon change magnitude.

CLI options:

```text
--delta_scale_mode fixed|horizon_linear|learnable_by_horizon|mlp
--delta_scale_base
--delta_scale_slope
--delta_scale_min
--delta_scale_max
```

Modes:

- `fixed`: preserve existing behavior.
- `horizon_linear`: `delta_scale_h = delta_scale_base + delta_scale_slope * horizon_seconds / 60`.
- `learnable_by_horizon`: learn a separate scale per horizon.
- `mlp`: predict delta scale from horizon embedding and optional anchor/time-since-anchor.

Report:

- true delta distribution by horizon
- predicted delta distribution by horizon
- `pred_delta_saturation_rate`
- high-change MAE change

### 6.4 Long-horizon auxiliary loss

Rationale: previous multi-horizon aggregate was not competitive, but auxiliary trajectory supervision may help long horizons.

CLI options:

```text
--long_horizon_aux_loss
--aux_horizons
--aux_weights
--main_horizon_weight 1.0
```

Examples:

H=5 main:

```text
aux_horizons = 1 2.5
aux_weights = 0.05 0.10
```

H=10 main:

```text
aux_horizons = 1 2.5 5
aux_weights = 0.03 0.07 0.15
```

H=15 main:

```text
aux_horizons = 2.5 5 10
aux_weights = 0.05 0.12 0.25
```

Rules:

- This is not the old aggregate multi-horizon objective.
- Primary selection metric is main-horizon validation MAE.
- Do not select using average auxiliary-horizon MAE.
- If auxiliary loss worsens main-horizon MAE, mark it as failed.

### 6.5 High-change / high-FMS weighted loss

Rationale: long horizons may fail most on high-change or high-FMS regions.

CLI options:

```text
--change_weight_alpha
--high_fms_threshold
--high_fms_weight
--weighting_mode none|change|high_fms|change_high_fms
```

Weight example:

```text
weight = 1 + alpha * abs(target_fms - anchor_fms)
if target_fms >= high_fms_threshold:
    weight *= high_fms_weight
```

Candidate values:

- `change_weight_alpha`: 0.3, 0.5, 0.8
- `high_fms_threshold`: 8
- `high_fms_weight`: 1.25, 1.5, 2.0

Report:

- overall MAE
- high-FMS MAE
- high-FMS false negative rate
- high-change MAE
- trend macro F1

### 6.6 Branch-level gated fusion

Rationale: existing gated fusion may be feature-wise. Add branch-level gating for interpretability and robustness.

CLI options:

```text
--branch_gate_mode none|feature|branch
--branch_dropout
```

Branches:

- calibration summary
- recent TCN
- anchor
- anchor history
- cumulative exposure
- static
- horizon

Report:

- validation MAE impact
- branch gate activation statistics
- whether anchor history or cumulative exposure was actually used

---

## 7. Aggressive Sweep Runner

Create:

```text
scripts/run_densefms_long_horizon_aggressive.py
```

Required options:

```text
--dry_run
--smoke_test
--resume
--skip_existing
--retry_failed
--max_runs
--models
--horizons
--anchor_intervals
--recent_windows
--variants
--seeds
--wall_clock_target_hours
--wall_clock_soft_cap_hours
--wall_clock_hard_cap_hours
--prune_underperforming
--top_k_for_multiseed
--log_every_epoch
```

Default wall-clock budget:

```text
target = 12h
soft_cap = 18h
hard_cap = 24h
```

Default horizons:

```text
5 10 15
```

Default calibration:

```text
120s
```

Best-score track:

```text
anchor intervals: 10s, 30s
models: lc_sa_tcnformer, recent_tcn_summary_calib, anchor_delta_mlp
```

Deployment-realistic track:

```text
anchor intervals: 30s, 60s, 120s
models: recent_tcn_summary_calib, anchor_delta_mlp, lc_sa_tcnformer
```

Recent windows:

```text
10s, 30s, optionally 60s
```

Model priority:

1. `recent_tcn_summary_calib`
2. `lc_sa_tcnformer`
3. `anchor_delta_mlp`
4. `gated_fusion`
5. `anchor_delta_gru` only if time remains

Variant priority:

A. baseline reproduction
B. `+anchor_history`
C. `+cumulative_exposure`
D. `+anchor_history + cumulative_exposure`
E. `+anchor_history + horizon_dependent_delta_scale`
F. `+anchor_history + cumulative_exposure + horizon_dependent_delta_scale`
G. `+long_horizon_aux_loss`
H. `+anchor_history + cumulative_exposure + long_horizon_aux_loss`
I. `+high_change_high_fms_weighting`
J. `+anchor_history + cumulative_exposure + delta_scale + weighted_loss`
K. `+branch_level_gated_fusion`

---

## 8. Search Strategy

### Stage 0. Sanity + smoke

- Run all new sanity tests.
- Run one mini smoke run for each H=5/H=10/H=15.
- Smoke tests must use only a few batches, not full training.

### Stage 1. Baseline reproduction

- Reproduce existing H=5/H=10/H=15 baselines as closely as possible.
- If reproduced results differ, record the difference.
- Compute improvement against both original and reproduced baselines.

### Stage 2. Cheap broad search

Focus:

- `anchor_delta_mlp`
- `recent_tcn_summary_calib`

Variants:

- anchor history
- cumulative exposure
- anchor history + cumulative exposure
- horizon-dependent delta scale

Horizon:

- H=5/H=10/H=15

Anchors:

- 30s/60s

Goal:

- Quickly identify which feature branch helps.

### Stage 3. Neural focused search

Focus:

- `recent_tcn_summary_calib`

Settings:

- H=10/H=15
- anchor=30/60
- recent_window=10/30/60
- expand only promising variants

### Stage 4. Best-score upper search

Focus:

- `lc_sa_tcnformer`

Settings:

- H=5/H=10/H=15
- anchor=10/30
- anchor history + delta scale + auxiliary loss variants

Goal:

- Challenge best-score aggressive targets.

### Stage 5. Deployment-realistic search

Focus:

- `recent_tcn_summary_calib`

Settings:

- H=5/H=10/H=15
- anchor=60/120

Goal:

- Challenge deployment aggressive targets with lower user burden.

### Stage 6. Long-horizon auxiliary experiments

- H=10 main with aux H=1/2.5/5.
- H=15 main with aux H=2.5/5/10.
- Select by main-horizon validation MAE only.

### Stage 7. Multi-seed confirmation

Run seeds 42, 43, 44 for:

- top 2 candidates per horizon
- top 2 deployment-realistic candidates

---

## 9. Pruning / Early Termination

Underperforming runs may be pruned.

Rules:

- After the first 20-30% of epochs, prune if validation MAE is much worse than the corresponding horizon baseline.
- Every new variant must complete at least one full run before being aggressively pruned.
- H=10/H=15 may improve slowly, so pruning should not be too aggressive.

Every pruning decision must be logged in:

```text
live/progress_log.jsonl
manifests/run_status.jsonl
```

Include:

- prune epoch
- best validation MAE before prune
- baseline reference
- prune reason

---

## 10. Multi-seed Confirmation

Run multi-seed confirmation for:

- H=5 best-score best candidate
- H=10 best-score best candidate
- H=15 best-score best candidate
- H=5 deployment best candidate
- H=10 deployment best candidate
- H=15 deployment best candidate

Seeds:

```text
42 43 44
```

Output:

```text
multiseed/multiseed_long_horizon_summary.csv
multiseed/multiseed_long_horizon_summary.md
```

Report:

- mean validation MAE
- std validation MAE
- best seed
- worst seed
- baseline mean improvement
- aggressive target status
- stretch target status

---

## 11. Sanity Tests

Add tests to:

```text
scripts/run_densefms_sanity_tests.py
```

Required tests:

A. Anchor history leakage test

- Confirm no FMS after current index enters anchor history.

B. Anchor history padding/mask test

- Confirm padding and observed mask work when fewer than K anchors exist.

C. Cumulative exposure leakage test

- Confirm exposure features use only head/motion at or before current index.

D. Horizon-dependent delta scale test

- Confirm H=5/H=10/H=15 scale behavior.
- Confirm fixed mode matches previous behavior.

E. Long-horizon auxiliary loss test

- Confirm main and auxiliary losses are separated.
- Confirm selection metric remains main-horizon validation MAE.

F. Weighted loss test

- Confirm high-change and high-FMS weights are applied correctly.

G. Branch gate shape test

- Confirm branch-level gates match branch feature shapes.

H. Backward compatibility test

- When all new options are off, existing baseline forward shape and behavior are preserved.

I. Runner resume test

- Completed runs are skipped.
- Failed runs are skipped unless `--retry_failed` is set.
- Interrupted status is recorded.

J. Live logging test

- Confirm progress log, run status log, and live leaderboard are created before all runs finish.

---

## 12. Required Final Outputs

All outputs under:

```text
runs/densefms_long_horizon_aggressive/
```

Required files:

```text
live/progress_log.jsonl
live/leaderboard_live.csv
live/best_by_horizon_live.md
live/current_best.json
summaries/aggressive_search_summary_partial.md
summaries/aggressive_search_summary_final.md
summaries/best_by_horizon.csv
summaries/best_by_horizon.md
summaries/best_score_track_summary.csv
summaries/deployment_realistic_track_summary.csv
diagnostics/diagnostics_summary.csv
diagnostics/diagnostics_summary.md
multiseed/multiseed_long_horizon_summary.csv
multiseed/multiseed_long_horizon_summary.md
manifests/planned_manifest.json
manifests/run_status.jsonl
```

Required plots:

```text
plots/horizon_mae_comparison.png
plots/baseline_vs_refined_mae.png
plots/best_score_track_h5_h10_h15.png
plots/deployment_track_h5_h10_h15.png
plots/true_delta_vs_pred_delta_h10.png
plots/true_delta_vs_pred_delta_h15.png
plots/high_fms_error.png
plots/high_change_error.png
plots/progress_best_mae_over_time.png
plots/variant_improvement_heatmap.png
plots/anchor_interval_pareto.png
```

---

## 13. Final Summary Requirements

Create:

```text
summaries/aggressive_search_summary_final.md
```

Must include:

1. Goal summary
2. Baseline summary
3. Runtime, completed/failed/skipped run count
4. Best validation MAE by H=5/H=10/H=15
5. Improvement over baselines
6. Best-score track results
7. Deployment-realistic track results
8. Aggressive target achievement status
9. Stretch target achievement status
10. Most effective variants
11. Anchor history effect
12. Cumulative exposure effect
13. Horizon-dependent delta scale effect
14. Long-horizon auxiliary loss effect
15. High-change/high-FMS weighting effect
16. Delta saturation improvement
17. High-FMS/high-change performance change
18. Multi-seed confirmation results
19. Failed experiments and likely reasons
20. Next-step recommendations

Interpretation rules:

- Do not overclaim.
- Report H=5/H=10/H=15 separately.
- Mark single-seed improvements as preliminary.
- Do not mix best-score and deployment-realistic claims.
- Do not claim long-horizon success from H=1.

---

## 14. Final Korean Report Format

Final response/report must be written in Korean and include:

1. 수정/추가한 파일 목록
2. 추가 CLI 옵션
3. sanity/smoke test 결과
4. 실행한 run 수와 주요 설정
5. interrupt-safe logging 구현 여부
6. live leaderboard 위치
7. H=5/H=10/H=15 baseline 대비 validation MAE 개선 여부
8. best-score track 결과
9. deployment-realistic track 결과
10. aggressive/stretch target 달성 여부
11. multi-seed confirmation 결과
12. 실패한 variant와 원인 추정
13. 다음 단계 추천
