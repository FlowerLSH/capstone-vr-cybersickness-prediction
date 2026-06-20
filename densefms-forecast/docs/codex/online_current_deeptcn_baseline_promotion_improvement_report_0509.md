# Online Current DeepTCN Baseline Promotion + Follow-up Improvement - 2026-05-09

## 1. Modified or Added Files

- `configs/online_current/selected_deeptcn_static4.yaml`
  - Clean promoted DeepTCN calibration baseline.
- `configs/online_current/selected_deeptcn_risk035_static4.yaml`
  - Final canonical config after follow-up improvement.
- `docs/codex/online_current_model_registry_0509.md`
  - Canonical online-current path updated to `selected_deeptcn_risk035_static4.yaml`.
- `docs/codex/online_current_deeptcn_baseline_promotion_improvement_report_0509.md`
  - This report.
- Previously modified for DeepTCN support:
  - `src/densefms_forecast/model.py`
  - `src/densefms_forecast/train.py`
  - `scripts/run_densefms_sanity_tests.py`

## 2. New CLI / Config Options

- New config files:
  - `configs/online_current/selected_deeptcn_static4.yaml`
  - `configs/online_current/selected_deeptcn_risk035_static4.yaml`
- Existing new CLI support used:
  - `--calibration_encoder_mode deep_tcn`
  - `--risk_loss_weight 0.35`
- Final canonical config uses:
  - `calibration_encoder_mode: deep_tcn`
  - `motion_feature_mode: causal_dynamics_v1`
  - `fds_blend: 0.75`
  - `fms_combine_weight_ordinal: 0.15`
  - `risk_loss_weight: 0.35`

## 3. Dataset / Windowing Changes

- No dataset/windowing logic changes.
- Dataset: `DenseFMS/Dataset`
- Split file: `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42/split.json`
- Split: train 316 / val 60 / test 52 sessions.
- Sampling interval: 0.5s
- Max session points: 420
- Calibration: 120s = 240 steps
- Recent window: 10s
- Test set was not used during search. It was evaluated only after validation selection.

## 4. Model Changes

Final selected architecture:

- Calibration encoder: DeepTCN only.
- Calibration dilation stages: `[1, 2, 4, 8, 16]`.
- Each stage is a residual `TCNBlock` with two causal conv layers, so calibration branch has 5 dilation blocks / 10 causal conv layers.
- No Transformer layer is used in the final calibration branch.
- Stream encoder: `deep_tcn_latent_gru`
- Decoder context: `state`
- Current head: `basic`
- Ordinal head: `cumulative`
- Static features: `age`, `mssq`, `gender`

## 5. Anchor / Static / Multi-Horizon Support Status

- Anchor/FMS policy unchanged.
- Calibration input uses only the first `calibration_steps`.
- post-calibration FMS does not enter calibration input.
- Recent motion remains causal and ends at current time `t`.
- Static support remains enabled.
- Existing rapid-rise auxiliary horizons remain 5s and 10s.
- Multi-horizon/future auxiliary code paths remain available, but the final selected config does not enable extra future/delta/event auxiliary losses.

## 6. Sanity Test Results

Commands run:

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m py_compile src/densefms_forecast/model.py src/densefms_forecast/train.py scripts/run_densefms_sanity_tests.py
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/run_densefms_sanity_tests.py
```

Results:

- `py_compile`: pass
- Full sanity suite: pass
- Covered checks include:
  - import check
  - seconds-to-steps conversion
  - target shift correctness
  - calibration leakage check
  - recent-window leakage check
  - anchor policy check
  - model forward shape check
  - dry-run sweep command generation
  - DeepTCN calibration mode no-post-calib-FMS leakage check
- Config load check:
  - `selected_deeptcn_static4.yaml`: `deep_tcn`, `[1,2,4,8,16]`, `causal_dynamics_v1`, FDS blend 0.75, risk 0.25.
  - `selected_deeptcn_risk035_static4.yaml`: same model path, risk 0.35.

## 7. Full-Training Search Budget Actually Used

Baseline promotion run already completed:

| Run | Purpose | Epochs | Best epoch |
|---|---|---:|---:|
| `calib_deep_tcn_mean_seed42` | clean DeepTCN calibration baseline | 80 | 75 |

Follow-up improvement runs:

| Run | Change | Epochs | Best epoch |
|---|---|---:|---:|
| `deeptcn_imp_delta_aux_w005_seed42` | future delta auxiliary 0.05 | 27 | 17 |
| `deeptcn_imp_trajectory_shape_w002_seed42` | trajectory shape loss 0.02 | 80 | 72 |
| `deeptcn_imp_risk015_seed42` | risk loss 0.15 | 30 | 20 |
| `deeptcn_imp_risk035_seed42` | risk loss 0.35 | 80 | 70 |

Additional follow-up training budget: 217 epochs.

Only `deeptcn_imp_risk035_seed42`, selected by validation, was evaluated on test.

## 8. Validation Leaderboard

| Rank | Model | Val MAE | Val RMSE | Val R2 | Acc <= 1.0 | Warning F1 | Rapid-any F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `risk035` | 1.740192 | 2.512162 | 0.687399 | 0.442685 | 0.732161 | 0.395817 |
| 2 | `selected_deeptcn` | 1.753715 | 2.631096 | 0.657099 | 0.472037 | 0.747837 | 0.387816 |
| 3 | `trajectory_shape_w002` | 1.784108 | 2.577908 | 0.670823 | 0.453056 | 0.723468 | 0.398527 |
| 4 | `risk015` | 2.043652 | 2.977196 | 0.560954 | 0.378519 | 0.746433 | 0.398988 |
| 5 | `delta_aux_w005` | 2.085781 | 2.948916 | 0.569255 | 0.352870 | 0.756955 | 0.364462 |

Interpretation:

- `risk_loss_weight=0.35` is the only follow-up that beats the promoted DeepTCN baseline on validation MAE/RMSE/R2.
- Delta auxiliary and risk 0.15 over-regularized the model.
- Trajectory shape loss improved RMSE/R2 relative to the promoted baseline but did not beat MAE.

## 9. Plot / Trajectory Result

Validation trend metrics:

| Model | Pearson session mean | Centered MAE | Delta corr 5s | Direction acc 5s | Flat rate |
|---|---:|---:|---:|---:|---:|
| `risk035` | 0.477009 | 1.344220 | 0.395389 | 0.671277 | 0.000000 |
| `selected_deeptcn` | 0.488245 | 1.339468 | 0.409595 | 0.679517 | 0.000000 |
| `trajectory_shape_w002` | 0.478910 | 1.361328 | 0.407358 | 0.679223 | 0.000000 |
| `risk015` | 0.448223 | 1.408319 | 0.382488 | 0.723661 | 0.118644 |
| `delta_aux_w005` | 0.445177 | 1.411139 | 0.372877 | 0.718069 | 0.067797 |

Validation plot proxy sample:

| Model | Good | Medium | Bad | Total |
|---|---:|---:|---:|---:|
| `selected_deeptcn` | 4 | 0 | 8 | 12 |
| `delta_aux_w005` | 4 | 0 | 8 | 12 |
| `trajectory_shape_w002` | 3 | 0 | 9 | 12 |
| `risk015` | 4 | 0 | 8 | 12 |
| `risk035` | 4 | 0 | 8 | 12 |

Final test trend metrics:

| Model | Test MAE | Test RMSE | Pearson session mean | Centered MAE | Delta corr 5s | Direction acc 5s | Flat rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `risk035` | 2.082228 | 2.874417 | 0.369421 | 1.271289 | 0.297259 | 0.664301 | 0.000000 |
| `selected_deeptcn` | 2.154288 | 2.886184 | 0.381830 | 1.260353 | 0.307483 | 0.666430 | 0.000000 |

Test plot proxy sample:

| Model | Good | Medium | Bad | Total |
|---|---:|---:|---:|---:|
| `selected_deeptcn` | 4 | 3 | 5 | 12 |
| `risk035` | 4 | 3 | 5 | 12 |

Plot interpretation:

- `risk035` improves pointwise level metrics but does not improve short 5s direction metrics over `selected_deeptcn`.
- Plot bad count is unchanged against the promoted DeepTCN baseline on the fixed test sample.
- Compared with the older pre-DeepTCN baseline, the DeepTCN family still reduces plot bad cases.

## 10. Final Selected Configuration

Final selected config:

- `configs/online_current/selected_deeptcn_risk035_static4.yaml`

Final selected run:

- `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42`

Selection basis:

- Best validation MAE among the promoted DeepTCN baseline and follow-up candidates.
- Test evaluated only after this validation selection.

## 11. Final Test-Set Metrics

| Model | Test MAE | Test RMSE | Test R2 | Acc <= 0.5 | Acc <= 1.0 | Acc <= 2.0 | Warning F1 | Rapid-any F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old `previous_best` | 2.229539 | 3.005608 | 0.564486 | 0.185470 | 0.348825 | 0.591987 | 0.689286 | 0.283066 |
| `selected_deeptcn` | 2.154288 | 2.886184 | 0.598408 | 0.193376 | 0.332158 | 0.583226 | 0.684145 | 0.285373 |
| final `risk035` | 2.082228 | 2.874417 | 0.601676 | 0.225748 | 0.377244 | 0.620833 | 0.672291 | 0.299726 |

Final interpretation:

- `risk035` improves MAE, RMSE, R2, acc<=0.5/1/2, and rapid-any F1 over both earlier baselines.
- Warning F1 drops, because warning precision/recall tradeoff changed under the stronger risk auxiliary.

## 12. Generated Plots / Tables

- Validation analysis:
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_val/online_current_validation_leaderboard.csv`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_val/plot_judgment_summary.csv`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_val/plot_judgment_sessions.csv`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_val/trend_metric_summary.png`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_val/trajectory_*.png`
- Final test analysis:
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_test/online_current_validation_leaderboard.csv`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_test/plot_judgment_summary.csv`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_test/trend_metric_summary.png`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/analysis_test/trajectory_*.png`
- Final selected test outputs:
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/eval_test/metrics.json`
  - `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/eval_test/test_predictions.csv`

## 13. Git Status Summary

Tracked modified files include:

- `scripts/run_densefms_sanity_tests.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`
- Existing tracked files modified by prior work: `AGENTS.md`, `src/densefms_forecast/data.py`, `src/densefms_forecast/evaluate.py`, `src/densefms_forecast/losses.py`, `src/densefms_forecast/utils.py`

Untracked additions include:

- `configs/online_current/selected_deeptcn_static4.yaml`
- `configs/online_current/selected_deeptcn_risk035_static4.yaml`
- `docs/codex/online_current_deeptcn_baseline_promotion_improvement_report_0509.md`
- Other pre-existing untracked configs/scripts/docs/artifacts.

No commit or push was performed.

## 14. Remaining Issues or Warnings

- The best current config still does not improve short 5s direction metrics over the clean DeepTCN baseline.
- Stronger risk loss improves regression and rapid-any F1 but lowers warning F1 on test.
- Direct delta auxiliary and trajectory shape loss did not beat the final validation MAE; they should not be promoted as-is.
- Next useful work should target transition/delta behavior more surgically, for example with weaker or scheduled shape loss, or with a validation objective that explicitly balances MAE and direction metrics.
