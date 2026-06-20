# Start-FMS Teacher Distillation Goal, 2026-05-07

FULL_TRAINING_ALLOWED = true

## Objective

Optimize a deployment-realistic DenseFMS student that does not receive any
post-calibration FMS at inference time, while using a fixed start-FMS teacher
only as a train-split distillation signal.

Primary success criterion:

- Minimize validation primary MAE, defined as the mean validation MAE over
  h=5s, h=10s, and h=15s.
- Compare against the current strict no-post-FMS best:
  `runs/session_level_trajectory_0507/full/state_selfdelta_static_time_multiscale_norm_scale10_aux10_curdelta05_beta02_e120_s7`
  with validation primary MAE 2.171292.

## Input And Leakage Policy

Student allowed inputs:

- Head motion from session start through current time t.
- FMS only inside the initial calibration interval.
- Horizon value for h=5s, h=10s, and h=15s.
- Optional static Age/Gender/MSSQ with train-split-only normalization.

Student forbidden inputs:

- Any real FMS after calibration.
- Current FMS, target FMS, future FMS, recent-window dense FMS.
- `start_only`, `recent_start_observed`, `sparse_observed`, or any
  post-calibration FMS anchor as student input.
- Participant/session/file identity.
- Validation/test statistics for training preprocessing.

Privileged teacher policy:

- A fixed teacher checkpoint with `fms_context_mode=start_only` may be loaded
  during training only.
- The teacher may receive full train-split FMS inside the distillation loss.
- Teacher outputs are used only as auxiliary soft targets for the student.
- The teacher must not run on validation/test during model selection.
- The deployed/evaluated model is the student alone.
- Reports must label this as privileged train-time distillation, not as a
  strict label-free training signal.

## Evaluation Policy

- All search runs must include `--no_test_eval`.
- Model selection must use validation metrics only.
- Do not inspect or create test metrics, test predictions, or test plots during
  search.
- Final test is allowed only once after freezing a validation-selected
  candidate and writing a validation selection lock.

## Data Window Policy

- sampling_interval = 0.5
- calibration_seconds = 90
- recent_window_seconds = 10
- max_session_points = 420
- target_index must be <= 419 for validation/test prediction rows.
- split file:
  `runs/goal_mae_under1_search_0506/under1_delta_norm_mae_changew3_trend002_e120_s7_splitlock/split.json`

## Budget

- Run smoke/dry-run checks before full training.
- First pass: up to 4 full validation-only distillation candidates.
- Adaptive pass: up to 4 additional validation-only candidates if the first
  pass improves or reveals a clear weight/loss trend.
- Boundary extension: if the best adaptive candidate is at the tested
  distillation-weight boundary, run up to 2 additional validation-only
  candidates adjacent to that boundary.
- Representation extension: after output distillation, run up to 5 additional
  validation-only candidates that align the student's final per-time/horizon
  latent representation to the fixed start-FMS teacher representation. Include
  at least one representation-only control and small-weight combinations with
  the current best output-distillation setting.
- Delta/current-state extension: after representation distillation, run up to
  6 additional validation-only candidates that keep the student deployment
  policy strict but distill the teacher's future trajectory as a delta relative
  to the student's predicted current FMS state. Include predicted-current
  feedback and current-auxiliary weight variants if the first delta candidates
  do not reach primary validation MAE <= 2.0.
- Architecture/loss extension: if delta/current-state distillation does not
  reach primary validation MAE <= 2.0, run up to 6 additional validation-only
  candidates that keep the best output-distillation setting but change student
  motion/state capacity or supervised weighting. Prioritize richer motion
  features, GRU+TCN stream context, calibration summary features, larger hidden
  dimensions, stronger current-state supervision, and high-target weighting.
- Calibration-length extension: if the 90s calibration setting still does not
  reach primary validation MAE <= 2.0, run up to 4 validation-only candidates
  with longer calibration intervals. These candidates must be reported as a
  separate calibration-length ablation because they change the deployment
  requirement and the evaluated current-time range. If teacher distillation is
  disabled for compatibility, label the candidate as no-teacher.
- Session-latent extension: if the calibration-length extension still does not
  reach primary validation MAE <= 2.0, run up to 4 validation-only candidates
  that keep the 90s deployment input policy and predict a session-level FMS
  summary from calibration-only inputs. The predicted summary may be injected
  into the student latent state at inference time, but the true post-calibration
  FMS summary may be used only as a train-split auxiliary target, never as a
  validation/test input.
- Seed-ensemble extension: if the session-latent extension still does not
  reach primary validation MAE <= 2.0, run up to 4 validation-only candidates
  that repeat the current best single-model configuration with different random
  seeds. Use these only to evaluate whether a deployable validation-selected
  ensemble can close the remaining gap. Do not use test metrics for ensemble
  selection.
- Synthetic-anchor extension: if the strict seed-ensemble extension still does
  not reach primary validation MAE <= 2.0, run up to 3 validation-only
  candidates that do not receive post-calibration FMS, but predict the
  recent-window start FMS internally from calibration FMS and causal head
  motion. Prefer the roll-in variant: with h=5s and recent_window=10s, predict
  the FMS tape at time u from the window ending at u-h, then reuse that lagged
  prediction as the start-FMS delta base for later forecasts. True
  post-calibration FMS may supervise the internal current, level-tape, or
  synthetic-anchor auxiliary targets on the train split only; it must never be
  fed to the model during validation/test forward passes.
- Each full run may use up to 120 epochs with early stopping patience 12.
- Do not rerun completed valid runs.

## Required Outputs

- `runs/start_fms_distillation_0507/full/leaderboard.csv`
- `runs/start_fms_distillation_0507/full/distillation_report.md`
- `command.json` and `command.txt` for each run.
- validation prediction CSVs and validation plots for completed candidates.
- Clear audit fields showing no post-calibration FMS anchor in student
  validation predictions.
- For synthetic-anchor candidates, report the predicted-anchor mode separately
  and verify `requires_full_fms=False`, `anchor_mode=none`, and no non-null
  real `anchor_fms` or `start_fms_value` in validation predictions.
