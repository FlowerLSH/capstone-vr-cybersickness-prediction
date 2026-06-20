# Session-Level Trajectory Forecaster Goal, 2026-05-07

FULL_TRAINING_ALLOWED = true

## Objective

Build and evaluate a leakage-safe DenseFMS forecaster that receives FMS only during the initial calibration period, then predicts future FMS continuously from post-calibration head motion.

Success criterion:

- primary validation MAE <= 1.75, where primary MAE is the mean validation MAE over h=5s, h=10s, and h=15s.
- validation plots should follow the broad actual FMS trajectory, not merely improve MAE by smoothing or exploiting labels.

## Required Input Policy

Allowed inputs:

- head motion from 0 through the current time t
- FMS from 0 through calibration time T only
- horizon value, for h=5s, h=10s, and h=15s
- optional static Age/Gender/MSSQ features, using train-split-only normalization and binary2 gender one-hot encoding

Forbidden inputs:

- any real FMS after calibration time T
- current FMS at t
- target FMS at t+h
- future FMS
- recent-window dense FMS sequence
- start-FMS anchor, recent_start_observed, sparse_observed, or other post-calibration anchor-assisted FMS
- participant/session/trial/file-derived identity as model input
- validation/test statistics for train preprocessing

Calibration-end FMS may be used only as a delta base because it is inside the allowed calibration window. It must be recorded as calibration-end use, not as a post-calibration start-FMS anchor.

## Evaluation Policy

- Use validation metrics for all model selection.
- General search runs must use `--no_test_eval`.
- Do not read or create test metrics, test predictions, or test plots during search.
- Test evaluation is allowed only once after selecting a final candidate by validation.
- Do not change metric/evaluation code to improve reported MAE.
- Do not optimize against generated test results.

## Data Window Policy

- sampling_interval = 0.5
- calibration_seconds = 90
- max_session_points = 420
- target_index must be <= 419 for every validation/test prediction
- if cap420 changes sample counts, record it explicitly

## Model Direction

Main structure:

1. Encode calibration head motion plus calibration FMS.
2. Initialize a latent sickness state from the calibration encoder.
3. Update that latent state causally using post-calibration head motion only.
4. Predict FMS at t+h for h=5/10/15 from latent state, calibration summary, and horizon context.

Recommended first-pass branches:

- level prediction, no calibration-end delta
- calibration-end delta prediction
- calibration-end delta prediction with larger `delta_scale`
- current-FMS auxiliary head trained from labels, without feeding post-calibration FMS to the model
- motion-derived features from head motion only: norm, norm_delta, norm_delta_energy
- optional static branch only after the no-static model is working

## Shape/Flow Review

Besides MAE, inspect validation plots and sequence metrics:

- trend/change F1
- trend sign accuracy
- derivative MAE
- dynamic range and smoothing behavior from plots
- high-change and zero-tail failure patterns where available

Plots should be stored for validation candidates. Each plot must remain validation-only during the search phase.

## Required Artifacts

Create or update:

- `runs/session_level_trajectory_0507/run_state_forecaster_search.py`
- `runs/session_level_trajectory_0507/*/leaderboard.csv`
- `runs/session_level_trajectory_0507/*/state_forecaster_report.md`
- validation prediction CSVs and validation plots for completed candidates

The report must clearly state:

- no post-calibration FMS was used as model input
- whether calibration-end delta was used
- whether static features were used
- validation primary MAE and horizon MAE
- plot/trajectory observations
- leakage audit result
- whether the success criterion was met
