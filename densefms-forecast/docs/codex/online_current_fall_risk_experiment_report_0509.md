# Online Current Fall-Risk Auxiliary Experiment 0509

## Goal

Test whether adding a rapid-drop auxiliary head improves recovery/drop trajectory tracking over the current DeepTCN calibration + risk035 baseline.

## Implementation

- Added optional `fall_risk_head_enabled` to `OnlineFMSRiskTracker`.
- Added leakage-safe fall-risk labels:
  - 5s: future-window minimum FMS drops at least 2 points below current FMS.
  - 10s: future-window minimum FMS drops at least 3 points below current FMS.
- Added `fall_loss_weight` BCE auxiliary loss and `fall_risk_pos_weight`.
- Added validation/test logging fields:
  - `p_rapid_drop_{horizon}s`
  - `rapid_drop_label_{horizon}s`
  - `rapid_drop_valid_{horizon}s`
  - `rapid_drop`, `rapid_drop_any`
- Final warning remains rise-only/high-FMS based; fall-risk is not used as sickness-warning alarm.

## Validation Search

All runs used the same split as `deeptcn_imp_risk035_seed42`. Model selection used validation MAE only. Test was not evaluated for fall candidates.

| Run | fall_loss_weight | Best epoch | Val MAE | Val RMSE | Val R2 | Rapid-rise any F1 | Rapid-drop any F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk035 baseline | 0.00 | 70 | 1.740192 | 2.512162 | 0.687399 | 0.395817 | n/a |
| fall005 | 0.05 | 49 | 1.839278 | 2.678713 | 0.644575 | 0.391587 | 0.438399 |
| fall010 | 0.10 | 76 | 1.786435 | 2.636467 | 0.655698 | 0.362322 | 0.430643 |
| fall020 | 0.20 | 49 | 1.836794 | 2.619141 | 0.660208 | 0.392604 | 0.449519 |

## Trajectory/Drop Analysis

Validation prediction CSV analysis:

| Run | Centered MAE | Delta corr 5s | Direction acc 5s | Drop dir acc 5s | Drop F1 5s | Drop F1 10s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---:|---|
| risk035 | 1.344220 | 0.395389 | 0.671277 | 0.611727 | 0.343733 | 0.367559 | 6 / 1 / 11 |
| fall005 | 1.370723 | 0.370119 | 0.676574 | 0.591918 | 0.317425 | 0.342276 | 6 / 1 / 11 |
| fall010 | 1.385190 | 0.387986 | 0.657740 | 0.595087 | 0.376248 | 0.369565 | 6 / 0 / 12 |
| fall020 | 1.366211 | 0.396171 | 0.693938 | 0.574485 | 0.362334 | 0.357143 | 5 / 1 / 12 |

## Decision

Do not promote fall-risk auxiliary as the new baseline.

The fall head learns a real rapid-drop classification signal, especially at `fall_loss_weight=0.20`, but it does not improve the primary current-FMS validation objective or plot proxy. The best fall candidate, `fall010`, improves validation drop F1 at 5s but worsens MAE, RMSE, centered MAE, rapid-rise F1, and plot bad count.

Final selected configuration remains:

- `configs/online_current/selected_deeptcn_risk035_static4.yaml`
- checkpoint: `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/best.pt`

## Final Test

Final-only test was rerun for the selected risk035 checkpoint after validation selection.

| Metric | Value |
|---|---:|
| Test MAE | 2.082228 |
| Test RMSE | 2.874417 |
| Test R2 | 0.601676 |
| Acc <= 1 | 0.377244 |
| Warning high-FMS F1 | 0.672291 |
| Rapid-rise any F1 | 0.299726 |
| Final warning F1 | 0.299726 |

## Artifacts

- Fall config: `configs/online_current/selected_deeptcn_risk035_fall010_static4.yaml`
- Fall runs: `runs/online_fms_current_tracking_0509_fall_risk/`
- Validation analysis: `runs/online_fms_current_tracking_0509_fall_risk/analysis/`
- Selected final test: `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/eval_test/`
