# Calibration-Only No-Post-FMS Ablation Goal

FULL_TRAINING_ALLOWED = true

## Objective

Evaluate whether DenseFMS future FMS forecasting can work without any
post-calibration user FMS input. FMS may be used only during the calibration
phase.

## Input Policy

- `max_session_points = 420`
- `target_index <= 419`
- `calibration_seconds = 90`
- `recent_window_seconds = 10` or `30`
- `fms_context_mode = calibration_history`
- No `start_only` recent-window FMS anchor.
- No sparse observed FMS.
- No `recent_start_observed`.
- No current/target/future FMS input.
- `anchor_mode = calibration_end` is allowed only as the last calibration FMS
  delta base.
- `anchor_mode = none` is allowed for pure level baselines.
- Static features are disabled for this first ablation.

## Test Policy

- Training runs must use validation only.
- Every run must include `--no_test_eval`.
- Do not create or inspect test metrics/predictions/plots in this ablation.
- Final test is not part of this task unless the user explicitly asks after
  validation results are reviewed.

## Required Output

- `runs/calibration_only_no_post_fms_0507_full/leaderboard.csv`
- `runs/calibration_only_no_post_fms_0507_full/calibration_only_ablation_report.md`

## Budget

- Run the predefined calibration-only candidate list once.
- Early stopping is allowed.
- Do not rerun completed full runs unless their artifacts are missing or invalid.
