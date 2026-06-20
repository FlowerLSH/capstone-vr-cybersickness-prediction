# DenseFMS Long-Horizon Improvement Overnight Goal

## 0. Goal Metadata

- Project: 소융캡디2 / DenseFMS future FMS forecasting
- Date: 2026-05-03 KST
- Goal type: overnight long-running model search
- FULL_TRAINING_ALLOWED: true
- Main run root: `runs/densefms_long_horizon_improvement_20260503/`
- Baseline run root: `runs/densefms_long_target_search/`
- Primary objective: improve validation MAE for horizons longer than H=1
- Secondary objective: keep every intermediate result recoverable even if interrupted
- Git policy:
  - Do not commit.
  - Do not push.
  - Do not add `runs/`, checkpoints, prediction CSVs, datasets, or large artifacts to git.

---

## 1. Current Baseline From Latest Report

Use the previous final report as the baseline.

Do not treat H=1 success as long-horizon success.

### 1.1 Best-score validation baseline

| Horizon | Previous best validation MAE | Previous best validation RMSE | Model | Anchor |
|---:|---:|---:|---|---:|
| H=1 | 0.8839 | 2.1706 | LC-SA-TCNFormer | 10s |
| H=2.5 | 1.0492 | 2.4116 | LC-SA-TCNFormer | 10s |
| H=5 | 1.2870 | 2.7346 | LC-SA-TCNFormer | 10s |
| H=10 | 1.6261 | 3.0926 | LC-SA-TCNFormer | 10s |
| H=15 | 1.8481 | 3.1940 | LC-SA-TCNFormer | 10s |

### 1.2 Deployment-realistic validation baseline

| Horizon | Previous deployment validation MAE | Previous deployment validation RMSE | Model | Anchor |
|---:|---:|---:|---|---:|
| H=5 | 1.7735 | 2.6354 | RecentTCN+SummaryCalib | 60s |
| H=10 | 1.9199 | 2.8597 | RecentTCN+SummaryCalib | 60s |
| H=15 | 2.0376 | 2.9135 | RecentTCN+SummaryCalib | 60s |

### 1.3 Multi-horizon baseline

| Model | Horizon set | Previous aggregate validation MAE |
|---|---|---:|
| RecentTCN+SummaryCalib multi-horizon | 1, 2.5, 5, 10, 15, 30 | 2.0191 |
| LC-SA-TCNFormer multi-horizon | 1, 2.5, 5, 10, 15, 30 | 2.0248 |
| AnchorDeltaMLP multi-horizon | 1, 2.5, 5, 10, 15, 30 | 2.1424 |

---

## 2. Primary Target

Focus on H > 1.

H=1 may be logged as a diagnostic, but it is not the optimization target.

Primary horizons:

```text
H = 5, 10, 15
```

Secondary horizon:

```text
H = 2.5
```

Optional diagnostic horizon:

```text
H = 30
```

### 2.1 Stretch targets

Try to reach at least one of the following validation targets:

| Horizon | Stretch validation MAE target | Previous baseline |
|---:|---:|---:|
| H=2.5 | <= 1.0000 | 1.0492 |
| H=5 | <= 1.1500 | 1.2870 |
| H=10 | <= 1.4500 | 1.6261 |
| H=15 | <= 1.6500 | 1.8481 |

### 2.2 Deployment-realistic stretch targets

| Horizon | Deployment stretch validation MAE target | Previous deployment baseline |
|---:|---:|---:|
| H=5 | <= 1.5500 | 1.7735 |
| H=10 | <= 1.7500 | 1.9199 |
| H=15 | <= 1.9000 | 2.0376 |

If the stretch target is not reached, still report the best relative improvement against the previous baseline.

---

## 3. Non-Negotiable Rules

### 3.1 Validation-only model selection

Model selection must use validation metrics only.

Validation-only decisions include:

- hyperparameter search
- early stopping
- pruning
- ranking
- ensemble member selection
- final role selection

Test metrics must be computed only after validation-selected final roles are fixed.

### 3.2 No leakage

Target:

```text
FMS[t + horizon_steps]
```

No future FMS may enter the input.

Rules:

- Recent motion input must include only samples up to current time `t`.
- Sparse anchor must be at or before current time `t`.
- Calibration input must not include future target FMS.
- `recent_start_observed` must be treated as upper-bound-only, not deployment-realistic.
- Do not use test metrics for search or model selection.

### 3.3 Horizon conversion

Sampling interval is 0.5 seconds.

Use:

| Horizon seconds | Horizon steps |
|---:|---:|
| 1s | 2 |
| 2.5s | 5 |
| 5s | 10 |
| 10s | 20 |
| 15s | 30 |
| 30s | 60 |

---

## 4. Required Interrupt-Safe Logging

Because this run may be interrupted, every important intermediate result must be recoverable.

Create this directory:

```text
runs/densefms_long_horizon_improvement_20260503/
```

Required files:

```text
planned_manifest.json
progress_log.jsonl
progress_log.md
leaderboard_live.csv
leaderboard_live.md
partial_summary.md
interrupt_summary.md
hardware_summary.json
sanity_tests.log
final_report.md
final_selected_models.json
final_test_metrics.csv
```

### 4.1 Before training starts

Write `planned_manifest.json`.

It must include every planned candidate with:

- run name
- command
- model family
- horizon
- horizon steps
- calibration length
- recent window length
- anchor policy
- anchor interval
- static feature flag
- loss type
- delta prediction flag
- seed
- output directory

### 4.2 At every run start

Append a `run_start` event to `progress_log.jsonl`.

Include:

- timestamp
- run name
- command
- horizon
- model
- seed
- output directory

### 4.3 During training

Preserve epoch-level logs if the training script emits them.

If possible, flush best validation metric after every validation evaluation.

Do not rely only on stdout.

### 4.4 At every run end

Append a `run_end` event to `progress_log.jsonl`.

Include:

- best epoch
- best validation MAE
- best validation RMSE
- runtime seconds
- checkpoint path
- metrics path
- prediction path

Immediately update:

```text
leaderboard_live.csv
leaderboard_live.md
partial_summary.md
```

### 4.5 On failure

Append a `run_failed` event.

Include:

- run name
- command
- return code
- exception or error message
- partial output directory

Continue to the next candidate unless the failure indicates a global code/data issue.

### 4.6 On interrupt

Try to catch SIGINT/SIGTERM.

Write `interrupt_summary.md`.

It must include:

- completed runs
- failed runs
- current best leaderboard
- currently running run name if available
- commands not yet executed
- where to resume from

### 4.7 Resume behavior

Support:

```text
--skip_existing
```

A resumed run must not overwrite completed results.

If a run directory exists but metrics are incomplete, mark it as incomplete and either resume or safely rerun into a new directory.

---

## 5. Search Budget

Use an overnight-level budget.

Recommended:

```text
wall_clock_target_hours = 8
wall_clock_soft_cap_hours = 11
wall_clock_hard_cap_hours = 12
```

Budget policy:

- Spend most budget on H=5/H=10/H=15.
- Do not stop early because H=1 is good.
- If H=5/H=10/H=15 improve early, spend remaining budget on:
  1. multi-seed confirmation,
  2. deployment-realistic sparse-anchor candidates,
  3. validation-selected ensembles.
- Stop before hard cap.
- If GPU memory is limited, reduce batch size or candidate count but still prioritize H=5/H=10/H=15.

---

## 6. Required Work Stages

## Stage 0. Environment and baseline inspection

Run first:

```bash
python scripts/run_densefms_sanity_tests.py
```

Inspect whether these exist:

```text
runs/densefms_long_target_search/leaderboard_val.csv
runs/densefms_long_target_search/final_selected_models.json
runs/densefms_long_target_search/progress_log.jsonl
```

If they exist, use them as previous baselines.

Do not rerun all previous baseline jobs unless required.

Record:

- Python version
- PyTorch version
- CUDA availability
- GPU name if available
- git commit hash if available
- git status
- dataset path existence
- previous result artifact existence

Write this to:

```text
hardware_summary.json
progress_log.md
```

---

## Stage 1. Implement or update long-horizon improvement runner

Add a new script if needed:

```text
scripts/run_densefms_long_horizon_improvement.py
```

The runner must support:

```text
--data_dir
--run_root
--baseline_dir
--horizons
--primary_horizons
--include_h1_diagnostic
--wall_clock_target_hours
--wall_clock_soft_cap_hours
--wall_clock_hard_cap_hours
--skip_existing
--dry_run
--smoke_test
--max_runs
--seeds
```

Default horizons:

```text
2.5 5 10 15
```

Primary horizons:

```text
5 10 15
```

The runner should call existing training infrastructure where possible instead of duplicating training logic.

---

## Stage 2. Focused single-horizon refinement

Prioritize horizon-specific models because previous multi-horizon candidates were not competitive.

Primary target horizons:

```text
H = 5, 10, 15
```

Secondary:

```text
H = 2.5
```

### 2.1 LC-SA-TCNFormer refinements

Treat LC-SA-TCNFormer as the previous best-score winner.

Search around:

```text
calibration_seconds: 60, 120, 180 if supported
recent_window_seconds: 10, 30, 60
anchor_interval_seconds: 10, 30, 60
hidden_dim: 128, 256
dropout: 0.05, 0.1, 0.2
loss: l1, smooth_l1
delta_prediction: on
static_features: on first, off only as diagnostic
```

### 2.2 RecentTCN+SummaryCalib refinements

Treat RecentTCN+SummaryCalib as the previous deployment-realistic winner for H=5/H=10/H=15.

Search around:

```text
calibration_seconds: 120, 180 if valid
recent_window_seconds: 10, 30, 60
anchor_interval_seconds: 30, 60, 90
hidden_dim: 128, 256
dropout: 0.1, 0.2, 0.3
loss: smooth_l1, l1
delta_prediction: on
static_features: on
```

### 2.3 Gated Fusion refinements

Use only if the existing code supports it reliably.

Prioritize configurations that combine:

- recent motion dynamics
- calibration summary
- sparse observed anchor
- static features

Do not spend more than 20% of total budget on this family unless it starts winning validation metrics.

### 2.4 AnchorDeltaMLP / AnchorDeltaGRU

Use as baseline or diagnostic only.

Do not over-allocate budget unless they unexpectedly beat LC-SA-TCNFormer or RecentTCN+SummaryCalib on H=5/H=10/H=15.

---

## Stage 3. Long-horizon-specific strategies

Implement only if low-risk and compatible with the current code.

### 3.1 Weighted multi-horizon loss

Previous multi-horizon aggregate was not competitive.

Try a weighted multi-horizon objective excluding H=1 from the main objective.

Main horizon set:

```text
2.5 5 10 15
```

Optional diagnostic:

```text
2.5 5 10 15 30
```

Suggested weights:

```text
H=2.5: 0.5
H=5:   1.0
H=10:  1.5
H=15:  2.0
H=30:  1.0 diagnostic only
```

If H=1 is included as an auxiliary, give it a low weight and do not let it dominate.

### 3.2 Per-horizon heads

Try shared encoder plus separate prediction heads per horizon.

Must log per-horizon validation metrics.

Do not report only aggregate metrics.

### 3.3 Delta/residual target

Try predicting:

```text
future_fms = anchor_fms + predicted_delta
```

or:

```text
future_fms = calibration_end_fms + predicted_delta
```

Validation metric remains absolute FMS MAE/RMSE.

### 3.4 Trend/change auxiliary loss

Add lightweight auxiliary loss for:

```text
target_fms[t+h] - last_available_fms
```

Start with small weights:

```text
0.05
0.1
0.2
```

Do not let trend loss dominate MAE.

### 3.5 Change-aware weighting

Try sample weighting based on:

- absolute future delta magnitude
- high FMS target
- moving vs stationary target trend

Always keep an unweighted validation MAE leaderboard.

### 3.6 Statistical summary branch

Optionally add a lightweight summary branch over recent motion/calibration windows:

- mean
- standard deviation
- min/max
- slope
- simple frequency or jitter descriptors if easy

Fuse this with the sequence encoder output.

Do not rewrite the entire dataset pipeline.

### 3.7 Ordinal/regression auxiliary diagnostic

Optional only if low-risk.

FMS is ordinal-like, but final validation remains MAE/RMSE.

If implemented, report whether ordinal auxiliary improves H=5/H=10/H=15.

---

## Stage 4. Adaptive search policy

Do not blindly run all combinations.

Priority order:

1. H=5 single-horizon LC-SA-TCNFormer and RecentTCN+SummaryCalib
2. H=10 single-horizon LC-SA-TCNFormer and RecentTCN+SummaryCalib
3. H=15 single-horizon LC-SA-TCNFormer and RecentTCN+SummaryCalib
4. Weighted multi-horizon excluding H=1
5. Deployment-realistic sparse-anchor refinements
6. Multi-seed confirmation
7. Validation-selected simple ensembles

Pruning rules:

- If a model family is consistently worse by more than 0.25 MAE across multiple H values, reduce its budget.
- If a configuration beats the previous baseline by at least 5%, spawn nearby refinements.
- If H=10/H=15 improve but H=5 worsens, keep horizon-specific winners instead of forcing one shared model.

---

## Stage 5. Multi-seed and ensemble validation

For each of H=5/H=10/H=15:

1. Take the best 1-3 validation candidates.
2. Run additional seeds if budget permits:

```text
42
43
44
```

3. Compute mean/std validation metrics.
4. Try simple validation-only ensemble:
   - average predictions from top 2 or top 3 models of the same horizon
   - ensemble member selection must use validation only
5. Save ensemble prediction CSV and metrics.

Do not use test metrics to choose ensemble members.

---

## Stage 6. Final test evaluation

Only after validation-selected final roles are chosen.

Required final roles:

```text
best_score_h2p5
best_score_h5
best_score_h10
best_score_h15
deployment_h5
deployment_h10
deployment_h15
multi_horizon_diagnostic
ensemble_diagnostic
```

`multi_horizon_diagnostic` and `ensemble_diagnostic` are required only if relevant candidates were actually run.

For each final role, evaluate test exactly once and save:

```text
final_test_metrics.csv
final_selected_models.json
```

Include:

- validation MAE/RMSE
- test MAE/RMSE/R2
- common test MAE if supported
- checkpoint path
- prediction CSV path
- role name
- selection rationale

---

## 7. Required Metrics

Every run should log at least:

```text
run_name
model
horizon
horizon_steps
calibration_seconds
recent_window_seconds
anchor_policy
anchor_interval
use_static
loss_type
delta_prediction
seed
best_epoch
val_MAE
val_RMSE
val_R2
val_sMAPE if available
checkpoint_path
metrics_path
val_predictions_path
runtime_seconds
status
```

Test metrics are allowed only for final selected roles.

If already implemented, also keep:

```text
common_val_MAE
common_val_RMSE
derivative_mae_all
trend_macro_f1_2s_eps0.5
trend_macro_f1_5s_eps0.5
high_fms_false_positive_rate
```

---

## 8. Required Plots

Generate plots under:

```text
runs/densefms_long_horizon_improvement_20260503/plots/
```

Required plots:

```text
best_by_horizon_improvement.png
horizon_mae_curve_previous_vs_new.png
deployment_horizon_mae_curve_previous_vs_new.png
progress_best_mae_over_time.png
model_family_comparison_h5_h10_h15.png
```

For selected final models:

```text
val_predicted_vs_target_h2p5.png
val_predicted_vs_target_h5.png
val_predicted_vs_target_h10.png
val_predicted_vs_target_h15.png
residual_histogram_h5.png
residual_histogram_h10.png
residual_histogram_h15.png
```

---

## 9. Smoke Test / Dry Run Requirements

Before full training:

1. Run sanity tests.
2. Run dry-run manifest generation.
3. Run at least two smoke tests if feasible:
   - H=5 short run
   - H=10 short run
4. Verify:
   - target shift uses correct horizon steps
   - no future FMS enters input
   - log files are created before full training
   - live leaderboard updates after smoke test
   - `--skip_existing` works

---

## 10. Suggested Commands

### 10.1 Dry run

```bash
python scripts/run_densefms_long_horizon_improvement.py \
  --data_dir ./DenseFMS/Dataset \
  --baseline_dir runs/densefms_long_target_search \
  --run_root runs/densefms_long_horizon_improvement_20260503 \
  --horizons 2.5 5 10 15 \
  --primary_horizons 5 10 15 \
  --wall_clock_target_hours 8 \
  --wall_clock_soft_cap_hours 11 \
  --wall_clock_hard_cap_hours 12 \
  --skip_existing \
  --dry_run
```

### 10.2 Smoke test

```bash
python scripts/run_densefms_long_horizon_improvement.py \
  --data_dir ./DenseFMS/Dataset \
  --baseline_dir runs/densefms_long_target_search \
  --run_root runs/densefms_long_horizon_improvement_20260503 \
  --horizons 5 10 \
  --primary_horizons 5 10 \
  --max_runs 2 \
  --smoke_test \
  --skip_existing
```

### 10.3 Full overnight run

```bash
python scripts/run_densefms_long_horizon_improvement.py \
  --data_dir ./DenseFMS/Dataset \
  --baseline_dir runs/densefms_long_target_search \
  --run_root runs/densefms_long_horizon_improvement_20260503 \
  --horizons 2.5 5 10 15 \
  --primary_horizons 5 10 15 \
  --wall_clock_target_hours 8 \
  --wall_clock_soft_cap_hours 11 \
  --wall_clock_hard_cap_hours 12 \
  --seeds 42 43 44 \
  --skip_existing
```

### 10.4 Optional H=30 diagnostic

Run this only after H=5/H=10/H=15 have been sufficiently explored.

```bash
python scripts/run_densefms_long_horizon_improvement.py \
  --data_dir ./DenseFMS/Dataset \
  --baseline_dir runs/densefms_long_target_search \
  --run_root runs/densefms_long_horizon_improvement_20260503 \
  --horizons 30 \
  --primary_horizons 30 \
  --max_runs 6 \
  --skip_existing
```

---

## 11. Final Report Requirements

Write final report in Korean:

```text
runs/densefms_long_horizon_improvement_20260503/final_report.md
```

Must include:

1. 작업 요약
2. 사용한 baseline
3. 변경/추가한 파일
4. 새 CLI/config 옵션
5. sanity test 결과
6. search budget 실제 사용량
7. 완료 run 수 / 실패 run 수 / interrupt-safe log 위치
8. validation leaderboard
9. H=2.5/H=5/H=10/H=15별 이전 대비 개선량
10. deployment-realistic leaderboard
11. multi-horizon 결과
12. ensemble 결과, 있으면
13. final selected model 목록
14. final test metrics
15. plots 목록
16. 해석
17. 남은 이슈
18. `git status --short`

Interpretation rules:

- H=1 성능을 장기 예측 성공으로 주장하지 말 것.
- H=5/H=10/H=15를 분리해서 해석할 것.
- validation 기준 개선과 test 결과를 구분할 것.
- test는 final-report-only라고 명시할 것.
- deployment-realistic과 best-score track을 섞지 말 것.
- `recent_start_observed`는 upper-bound-only라고 명시할 것.
- stretch target 실패 시에도 가장 좋은 상대 개선량을 정확히 보고할 것.

---

## 12. Acceptance Criteria

The task is successful if:

1. Full training search runs with continuous logs.
2. Partial results are readable even after interruption.
3. At least H=5/H=10/H=15 each have new attempted candidates.
4. Validation leaderboard compares previous vs new results.
5. Best-score and deployment-realistic tracks are separated.
6. Final test is only run for validation-selected final roles.
7. Final Korean report is produced.
8. No commit/push is performed.

The task is highly successful if at least one of these is achieved:

1. H=2.5 validation MAE <= 1.0000
2. H=5 validation MAE <= 1.1500
3. H=10 validation MAE <= 1.4500
4. H=15 validation MAE <= 1.6500
5. Deployment-realistic H=5/H=10/H=15 improves by >=10%
6. A validation-selected ensemble improves a single model for H=5/H=10/H=15