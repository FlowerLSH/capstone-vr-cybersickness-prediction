# Online Current FMS Model Registry - 2026-05-09

## Selected path

Use `configs/online_current/selected_deeptcn_risk035_static4.yaml` as the canonical online-current configuration.

Reference run:

`runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42`

## Why this is selected

The previous selected path was `configs/online_current/selected_fds_static4.yaml` with reference run
`runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42`.

The selected DeepTCN calibration path first improved validation MAE/RMSE/R2 and final test MAE/RMSE/R2 on the same
participant split. A follow-up risk-loss sweep then improved the validation and final test regression metrics again
with `risk_loss_weight=0.35`, while keeping test final-report-only.

| model | MAE | RMSE | centered MAE | delta corr 5s | direction acc 5s | flat rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous_best | 1.9228 | 2.7988 | 1.3627 | 0.4634 | 0.7231 | 0.0508 |
| selected_deeptcn_static4 | 1.7537 | 2.6311 | 1.3395 | 0.4096 | 0.6795 | 0.0000 |
| selected_deeptcn_risk035_static4 | 1.7402 | 2.5122 | 1.3442 | 0.3954 | 0.6713 | 0.0000 |

Final test comparison:

| model | MAE | RMSE | R2 | warning F1 | rapid-rise-any F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_best | 2.2295 | 3.0056 | 0.5645 | 0.6893 | 0.2831 |
| selected_deeptcn_static4 | 2.1543 | 2.8862 | 0.5984 | 0.6841 | 0.2854 |
| selected_deeptcn_risk035_static4 | 2.0822 | 2.8744 | 0.6017 | 0.6723 | 0.2997 |

The selected model improves level/regime fit and reduces flat amplitude underfit. Short 5s direction metrics are still not the strongest point, and warning F1 trades down slightly, so follow-up work should target transition/delta supervision without over-regularizing the level head.

## Selected architecture

- `task.mode`: `online_current_risk`
- Calibration encoder: `deep_tcn`
- Calibration dilation stages: `[1, 2, 4, 8, 16]`
- Stream encoder: `deep_tcn_latent_gru`
- Calibration length: 120 seconds / 240 steps
- Session cap: first 420 sampled steps
- Static features: 4D `age`, `mssq`, `gender_male`, `gender_female`
- Decoder context: `state`
- Current head: `basic`
- Ordinal head: `cumulative`
- Motion features: `causal_dynamics_v1`
- Ordinal/regression blend: 0.15 ordinal, 0.85 regression
- Risk head: enabled as an auxiliary training target
- Loss: MAE + current-reg aux + ordinal aux + rapid-rise risk aux (`risk_loss_weight=0.35`)
- Imbalance handling: LDS + FDS (`fds_blend=0.75`)

The risk head is not used as the model-selection metric; validation MAE remains the selection target. The risk auxiliary loss is kept because same-seed ablations degraded when the risk loss was removed.

## Deprecated for current work

Keep these paths for reproducibility, but do not use them as the default online-current path:

- `dual_delta_gate`: MAE is slightly lower, but predictions are too flat and calibration-anchor-biased.
- `transition_weighting`: custom transition-weighted MAE, not DenseLoss; validation MAE regressed.
- `DILATE-lite`: trajectory surrogate, not full DILATE; improved some direction metrics but hurt MAE/RMSE.
- `teacher_distill_w015`: current-FMS distillation from shape teacher; validation MAE regressed.
- `risk_head_removed` / `risk_head_loss0`: same-seed ablations degraded validation MAE and trajectory metrics, so they are not the current default.
- `selected_fds_static4`: superseded by `selected_deeptcn_static4`; keep for reproducibility.
- `selected_deeptcn_static4`: superseded by `selected_deeptcn_risk035_static4`; keep as the clean DeepTCN calibration baseline.

## Refactor rule

New online-current work should start from `selected_deeptcn_risk035_static4.yaml`. Legacy heads and experimental losses may remain available for old checkpoints and ablations, but they should stay behind explicit config flags rather than leaking into the selected path.
