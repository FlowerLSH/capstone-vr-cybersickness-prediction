# Online FMS Risk Tracking Goal - 2026-05-07

FULL_TRAINING_ALLOWED = true

## Objective

Implement and smoke/full-train a deployment-realistic online tracker that uses calibration head motion + calibration FMS, then post-calibration head motion only, to estimate current FMS and rapid-rise risk.

## Task

- Task mode: `online_current_risk`
- Calibration: 90.0 seconds
- Recent motion window: 10.0 seconds
- Sampling interval: 0.5 seconds
- No post-calibration observed FMS may be used as input.
- Model selection must use validation metrics only.
- Test set is final-report-only after selecting a configuration from validation results.

## Targets

- Current FMS: `FMS[t]`
- Rapid-rise 5s: `max(FMS[t+1:t+10]) - FMS[t] >= 2`
- Rapid-rise 10s: `max(FMS[t+1:t+20]) - FMS[t] >= 3`
- High FMS caution threshold: 8
- High FMS warning threshold: 12
- Ordinal FMS bins: `[0, 2, 4, 6, 8, 10, 12, 15, 20]`
- User-facing final warning mode: rapid-rise only
- High-FMS outputs may remain diagnostic, but must not drive `final_warning`.
- Model selection metric for rapid-rise-only experiments: validation `rapid_rise.10s.auprc` maximized.

## Architecture Update

- Current-FMS head may use `dual_delta_gate`:
  - direct level prediction
  - calibration-end-FMS + session drift prior + dynamic delta prediction
  - learned gate between direct level and delta-from-calibration value
- Risk head should consume fused latent state plus predicted current-FMS/drift diagnostics.
- Current FMS remains a latent/state supervision target, not the primary user-facing alarm.

## Experiment Budget

- Risk representative runs already completed: 3
- Current-FMS focused representative runs: 5
- Max epochs per run: 80
- Early stopping patience: 10
- Validation search only: run representative configs with `--no_test_eval`
- Final test evaluation: only once for the validation-selected configuration
- No commit or push

## Current-FMS Focus Update

- Primary objective: estimate current FMS as accurately as possible after calibration.
- Model selection metric: validation `mae` minimized.
- Rapid-rise outputs may remain present for diagnostics, but `risk_loss_weight` should be `0.0` in current-FMS-focused runs.
- Compare at least:
  - basic current head without motion stats
  - dual delta-gated current head without motion stats
  - heavier dual delta-gated encoder with motion stats and causal TCN motion encoder stem
- Follow-up if validation MAE does not beat the previous online reference:
  - basic head with previous current/ordinal/smoothness weights and no risk loss
  - dual head with previous current/ordinal/smoothness weights and auxiliary risk loss, selected only by validation MAE

## Representative Configurations

1. `online_risk_no_stats`
   - model: `online_fms_risk_tracker`
   - motion_feature_mode: `norm`
   - motion_stats_branch: false
   - state_feedback_mode: `predicted_current`

2. `online_risk_stats`
   - model: `online_fms_risk_tracker`
   - motion_feature_mode: `norm`
   - motion_stats_branch: true
   - state_feedback_mode: `predicted_current`

## Required Verification

- import check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- anchor policy check
- online model forward shape check
- online rapid-rise label check
- checkpoint saving
- metrics JSON generation
- prediction CSV generation where enabled
- validation leaderboard generation
- final selected model evaluation on test only after validation selection
