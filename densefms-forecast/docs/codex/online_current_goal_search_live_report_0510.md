# Online Current Goal Search Live Report 0510

## Goal

Continue from the promoted DeepTCN calibration + risk035 baseline and search for a validation-only improvement.

Success target:

- Validation MAE <= 1.5, or
- Validation plot proxy BAD <= 8

Selection rule:

- Use validation metrics only for search and model selection.
- Keep test evaluation final-report-only after selecting a final configuration.

## Current Baseline

| Run | Best epoch | Val MAE | Val RMSE | Val R2 | Rapid-rise any F1 | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---|
| deeptcn_imp_risk035_seed42 | 70 | 1.740192 | 2.512162 | 0.687399 | 0.395817 | 6 / 1 / 11 |

Baseline checkpoint:

- `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/best.pt`

Baseline config:

- `configs/online_current/selected_deeptcn_risk035_static4.yaml`

## Attempts

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p1_seed7_risk035 | Test seed sensitivity of current baseline. | done | 18 | 2.094577 | 3.042598 | 0.541452 | Worse than baseline; seed 7 converged to a weaker basin. |
| goal_p1_seed123_risk035 | Test seed sensitivity of current baseline. | done | 63 | 1.792865 | 2.585687 | 0.668833 | Better than seed 7, but still worse than baseline seed 42. |
| goal_p1_lr35e5_seed42 | Lower LR from 4.5e-4 to 3.5e-4 to reduce late validation oscillation. | done | 77 | 1.797679 | 2.583095 | 0.669497 | Stable but worse; lower LR under-reaches the baseline basin by epoch 80. |
| goal_p1_drop005_seed42 | Reduce dropout from 0.10 to 0.05 to test underfitting in current baseline. | done | 58 | 1.787176 | 2.540032 | 0.680424 | Worse than baseline; dropout 0.10 remains better for single-checkpoint MAE. |

## Validation Analysis P1

| Label | Val MAE | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---|
| risk035 | 1.740192 | 1.344220 | 0.395389 | 0.671277 | 6 / 1 / 11 |
| seed7 | 2.094577 | 1.417676 | 0.367744 | 0.719247 | 5 / 1 / 12 |
| seed123 | 1.792865 | 1.315852 | 0.433011 | 0.713361 | 5 / 2 / 11 |
| lr35e5 | 1.797679 | 1.357145 | 0.391875 | 0.669806 | 6 / 2 / 10 |
| drop005 | 1.787176 | 1.340226 | 0.408653 | 0.674220 | 6 / 1 / 11 |

## Validation Ensemble P1

Validation-only linear MAE weight search selected:

- `risk035`: 0.720391
- `seed123`: 0.279609

| Label | Val MAE | Val RMSE | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---|
| risk035 | 1.740192 | 2.512162 | 1.344220 | 0.395389 | 0.671277 | 6 / 1 / 11 |
| ensemble_p1 | 1.729881 | 2.510333 | 1.328240 | 0.416444 | 0.680989 | 6 / 2 / 10 |

Interpretation:

- The ensemble improves validation MAE and trend metrics over the current single checkpoint.
- It does not yet meet the active target (`MAE <= 1.5` or `BAD <= 8`).
- Since seed123 contributes useful shape diversity, two more seed-diversity runs are added before moving to heavier architecture changes.

## Attempts P2

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p2_seed202_risk035 | Add seed diversity for weighted ensemble search. | done | 17 | 2.169127 | 3.084242 | 0.528814 | Weak single model; keep only as possible ensemble member. |
| goal_p2_seed314_risk035 | Add seed diversity for weighted ensemble search. | done | 61 | 1.877741 | 2.679856 | 0.644272 | Better than seed202 but still worse than baseline. |

## Validation Ensemble P2

Adding seed202 and seed314 did not change the selected validation ensemble:

- `risk035`: 0.720391
- `seed123`: 0.279609
- Validation MAE: 1.729881
- Validation RMSE: 2.510333

## Attempts P3

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p3_state_space_feedback_risk035 | Re-test state-space current head on the promoted risk035/DeepTCN baseline. | done | 26 | 2.017760 | 2.888138 | 0.586828 | Regressed in MAE/shape; do not promote. |
| goal_p3_range_scaled_risk035 | Re-test range-preserving delta head on the promoted risk035/DeepTCN baseline. | done | 56 | 1.699501 | 2.455860 | 0.701254 | New best single checkpoint; plot BAD unchanged. |

## Validation Analysis P3

| Label | Val MAE | Val RMSE | Val R2 | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---:|---|
| risk035 | 1.740192 | 2.512162 | 0.687399 | 1.344220 | 0.395389 | 0.671277 | 6 / 1 / 11 |
| state_space_fb | 2.017760 | 2.888138 | 0.586828 | 1.438624 | 0.392155 | 0.730135 | 6 / 1 / 11 |
| range_scaled | 1.699501 | 2.455860 | 0.701254 | 1.307501 | 0.473802 | 0.723955 | 6 / 1 / 11 |
| ensemble_p3 | 1.678015 | 2.442062 | 0.704601 | 1.303524 | 0.467525 | 0.715127 | 6 / 1 / 11 |

Current validation best:

- single checkpoint: `goal_p3_range_scaled_risk035`
- prediction-level ensemble: `0.648952 * range_scaled + 0.351048 * risk035`

Target status:

- `MAE <= 1.5`: not met; best is 1.678015.
- `BAD <= 8`: not met; best remains 11/18 on the fixed proxy.

## Attempts P4

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p4_range_risk025 | Tune range-scaled head with lower rapid-rise auxiliary pressure. | done | 12 | 1.971724 | 2.995787 | 0.555454 | Too little risk auxiliary pressure destabilized this head. |
| goal_p4_range_ord010 | Tune range-scaled head with lower ordinal blend. | done | 15 | 1.944711 | 3.013565 | 0.550162 | Lower ordinal blend regressed strongly. |

## Attempts P5

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p5_range_recent30 | Give range-scaled head a longer causal recent-motion window for trajectory hard cases. | done | 47 | 1.710402 | 2.498808 | 0.690714 | Slightly worse single model; useful in prediction ensemble. |

## Validation Ensemble P5

Validation-only ensemble over `risk035`, `range_scaled`, and `range_recent30` selected:

- `risk035`: 0.301080
- `range_scaled`: 0.386616
- `range_recent30`: 0.312304

| Label | Val MAE | Val RMSE | Val R2 | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---:|---|
| ensemble_p5 | 1.669013 | 2.440806 | 0.704904 | 1.300321 | 0.472626 | 0.724250 | 6 / 1 / 11 |

Target status remains unmet.

## Attempts P6

| Attempt | Intention | Status | Best epoch | Val MAE | Val RMSE | Val R2 | Notes |
|---|---|---|---:|---:|---:|---:|---|
| goal_p6_range_seed123 | Add seed diversity for the stronger range-scaled head. | done | 50 | 1.785959 | 2.611037 | 0.662308 | Did not improve single model or ensemble. |
| goal_p6_range_seed314 | Add seed diversity for the stronger range-scaled head. | done | 30 | 1.913582 | 2.820767 | 0.605879 | Did not improve single model or ensemble. |

## Validation Ensemble P6

Adding range seed123 and range seed314 did not change the selected validation ensemble:

- `risk035`: 0.301080
- `range_scaled`: 0.386616
- `range_recent30`: 0.312304
- Validation MAE: 1.669013
- Validation RMSE: 2.440806

## Final Selected Configuration

Selected by validation metrics only:

- Best single checkpoint: `goal_p3_range_scaled_risk035`
- Best overall prediction configuration: `ensemble_p5`

`ensemble_p5` fixed validation weights:

- `risk035`: 0.301080
- `range_scaled`: 0.386616
- `range_recent30`: 0.312304

The active target was not met:

- Best validation MAE: 1.669013, above 1.5.
- Best validation plot BAD: 11/18, above 8.

## Final Test Metrics

Test was evaluated only after validation selection.

| Label | Test MAE | Test RMSE | Test R2 | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---:|---|
| risk035 | 2.082228 | 2.874417 | 0.601676 | 1.271289 | 0.297259 | 0.664301 | 6 / 3 / 9 |
| range_scaled | 1.994099 | 2.748211 | 0.635886 | 1.217824 | 0.400855 | 0.696948 | 6 / 3 / 9 |
| range_recent30 | 2.006903 | 2.732965 | 0.639915 | 1.224180 | 0.355369 | 0.663591 | 6 / 4 / 8 |
| ensemble_p5 | 1.981423 | 2.723987 | 0.642277 | 1.209143 | 0.380440 | 0.684883 | 6 / 3 / 9 |

## Verification

- `py_compile`: passed for model/train/evaluate/sanity/ensemble scripts.
- `scripts/run_densefms_sanity_tests.py`: passed.
- Required leakage checks passed through the sanity suite: seconds-to-steps conversion, target shift correctness, calibration leakage, recent-window leakage, anchor policy, model forward shapes, and dry-run command generation.
- Checkpoint saving verified for completed training runs.
- Metrics JSON and validation prediction CSV generation verified for completed runs.
- Test prediction CSV generation verified for final selected members and ensemble.

## Attempts P7-P29

These attempts continued from the unmet P6 target. All search decisions below used validation outputs only.

| Attempt | Intention | Status | Best epoch / method | Val MAE | Notes |
|---|---|---|---:|---:|---|
| P7 train-affine probe | Fit global affine correction on train predictions only. | done | train-only affine | 1.685677 | Worse than raw P5; simple scale/bias correction rejected. |
| P7 `range_fds050` | Reduce FDS blend from 0.75 to 0.50 for range-scaled head. | done | 27 | 1.890011 | Strong regression. |
| P7 `range_fds100` | Increase FDS blend to 1.00. | done | 23 | 1.902000 | Strong regression; FDS axis closed. |
| P8 legacy-plus ensemble | Add older DeepTCN, trajectory, delta, seed, FDS members to exact MAE ensemble. | done | linprog | 1.668559 | Tiny gain over P5; weights kept risk035/range/recent30/selected_deeptcn only. |
| P9 meta calibrator | Train-only ridge/huber/HGB/extra-trees post-hoc calibrators on P5 features. | done | replacement models | 1.778552-1.825531 | Raw replacement failed. |
| P9 meta blend | Blend P5 with P9 meta outputs by validation MAE. | done | linprog | 1.665423 | Small gain; plot BAD unchanged. |
| P10 `range_fall005` | Add rapid-drop auxiliary to range-scaled head at weight 0.05. | done | 23 | 1.890493 | Worse; plot unchanged. |
| P10 `range_fall010` | Add rapid-drop auxiliary at weight 0.10. | done | 40 | 1.780962 | MAE worse, but fixed proxy improved to 6 good / 2 medium / 10 bad. |
| P10 `range_fall020` | Add rapid-drop auxiliary at weight 0.20. | done | 37 | 1.769643 | MAE worse; plot returned to 6 / 1 / 11. |
| P11 P9 + fall010 blend | Mix a small amount of fall010 into P9. | done | linprog | 1.663929 | New best at the time; plot BAD unchanged. |
| P12 `range_enddrop030` | Drop calibration-end FMS during training to reduce high-anchor overreliance. | done | 47 | 1.890532 | Too much anchor information loss; rejected. |
| P13 `range_recent20` | Test middle recent window between 10s and 30s. | done | 54 | 1.705687 | Not best single, but high ensemble value. |
| P13 recent20 ensemble | Reweight risk035/range/recent20/fall010. | done | linprog | 1.657041 | Large ensemble gain; recent20 received weight 0.346891. |
| P14 P13 meta blend | Train-only HGB meta output lightly blended into P13. | done | linprog | 1.654980 | Current validation MAE best; plot BAD still 11. |
| P15 `range_recent15` | Check adjacent recent-window length. | done | 47 | 1.705600 | Similar single to recent20 but ensemble weight only 0.0058; no real gain. |
| P16 `range_recent20_seed123` | Add seed diversity for recent20. | done | 28 | 1.861100 | Zero ensemble weight; rejected. |
| P17 `range_recent20_smoothl1` | Replace MAE with SmoothL1. | done | 11 | 1.988000 | Rejected; MAE loss remains better. |
| P18 `range_recent20_scenario` | Add scenario/content prior to static features. | done | 13 | 1.967800 | Rejected; scenario prior generalized poorly. |
| P19 `range_recent20_risk045` | Increase rapid-rise auxiliary pressure from 0.35 to 0.45. | done | 44 | 1.778900 | Zero ensemble weight; rejected. |
| P20 fall-feature meta | Preserve fall010 rapid-drop probabilities in P13 meta features. | done | linprog | 1.656145 | Slightly worse than P14; rejected. |
| P21 `range_recent20_multitimescale` | Add `multi_timescale_v1` motion features to the range-scaled recent20 setup. | done | 23 | 1.892525 | Strong regression vs P14; not added to ensemble search. |
| P22 `range_recent20_transition_w` | Upweight drop/recovery/rise transition target points to improve plot shape. | done | 11 | 1.903372 | Regressed; plot was 6 / 1 / 11 and P14 was better on the same plot set. |
| P23 `range_recent20_anchorbreak` | Upweight points where target diverges strongly from calibration-end FMS. | done | 43 | 1.891112 | Strong anchor-break weighting hurt overall fit; zero ensemble weight. |
| P24 causal meta calibrator | Add causal rolling/expanding prediction-history features to train-only meta calibrator. | done | linprog | 1.654968 | Tiny blend gain over P14, but plot BAD unchanged. |
| P25 `range_recent20_enddrop010` | Weak calibration-end FMS dropout to reduce anchor overreliance without removing calibration signal. | done | 55 | 1.726297 | Useful diversity; 7.55% blend with P14 improved MAE to 1.654457. Plot BAD unchanged. |
| P26 `range_recent20_calibdrop005` | Weak dropout across calibration FMS history. | done | 47 | 1.786312 | Worse than P25; zero ensemble weight. |
| P27 `range_recent25` | Fill recent-window gap between 20s and 30s. | done | 16 | 1.937227 | Recent25 did not reproduce recent20 diversity; zero ensemble weight. |
| P28 `range_recent20_deeptcn_transformer1_attn` | Add one Transformer layer and attention pooling after calibration DeepTCN. | done | 17 | 1.938416 | Calibration Transformer layer destabilized fit; zero ensemble weight. |
| P29 `range_recent20_deeptcn_attnpool` | Use calibration DeepTCN with attention pooling only. | done | 42 | 1.858140 | Better than P28 but still weak; zero ensemble weight. |

## Current Best After P29

Validation-selected current best:

- `ensemble_p25_enddrop_blend`
- Validation MAE: 1.654457
- Validation RMSE: 2.446383
- Validation R2: 0.703555
- Validation plot proxy: 6 good / 1 medium / 11 bad

P25 fixed validation weights:

- `ensemble_p14_p13_meta_blend`: 0.924531
- `goal_p25_range_recent20_enddrop010`: 0.075469

P14 fixed validation weights:

- `ensemble_p13`: 0.893970
- `meta_p14_p13_hgb`: 0.106030

P13 base ensemble weights:

- `risk035`: 0.267369
- `range_scaled`: 0.323979
- `range_recent20`: 0.346891
- `fall010`: 0.061760

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.654457.
- `BAD <= 8`: not met on validation; best current selected proxy remains 11/18.

## Final-Only Test Snapshot For Current Validation Best

After selecting P14 by validation only, one final-report-only test evaluation was generated.

| Label | Test MAE | Test RMSE | Test R2 | Centered MAE | Delta corr 5s | Direction acc 5s | Plot good/medium/bad |
|---|---:|---:|---:|---:|---:|---:|---|
| previous P5 | 1.981423 | 2.723987 | 0.642277 | 1.209143 | 0.380440 | 0.684883 | 6 / 4 / 8 on P14 plot set |
| P13 | 2.009276 | 2.762799 | 0.632010 | 1.214207 | 0.376962 | 0.681689 | 6 / 4 / 8 |
| P14 | 2.001594 | 2.751003 | 0.635146 | 1.212357 | 0.374042 | 0.680979 | 6 / 5 / 7 |

Interpretation:

- P14 improves validation MAE but does not improve final test MAE over previous P5.
- Test plot proxy is better for P14 on its selected plot set, but test plot was not used for selection and does not close the validation target.

## Attempts P30-P48

Continuation after P29. Selection and reweighting used validation predictions only. Test was not used for these decisions.

| Attempt | Intention | Status | Best epoch / method | Val MAE | Notes |
|---|---|---|---:|---:|---|
| P30 `range_recent20_predfeedback` | Roll predicted current FMS back into state feedback. | done | 16 | 1.968652 | Feedback destabilized level tracking; zero ensemble weight. |
| P31 `range_recent20_decoder_tcn2` | Add temporal TCN context inside decoder. | done | 30 | 1.864393 | Weak single model but useful diversity; P31 blend improved MAE to 1.653928. |
| P32 `range_recent20_decoder_tcn1` | Test lighter decoder TCN. | done | 17 | 1.879333 | Zero weight against P31 blend. |
| P33 `range_recent20_trajshape003` | Add trajectory-shape loss at weight 0.03. | done | 51 | 1.726053 | Stronger single than many shape variants; 8.03% blend improved MAE to 1.653515. |
| P34 `range_recent20_trajshape001` | Reduce trajectory-shape weight to 0.01. | done | 46 | 1.763249 | Worse than P33 and zero ensemble weight. |
| P35 `recent20_trajectory_decoder_tcn2` | Replace scalar head with short trajectory decoder. | done | 39 | 1.962011 | Trajectory head hurt level fitting; plot BAD stayed 11. |
| P36 `range_recent20_coarseband005` | Add coarse FMS-band auxiliary loss. | done | 28 | 1.875055 | Regime auxiliary did not improve MAE or plot; zero ensemble weight. |
| P37 global components | Re-optimize core components together. | done | linprog | 1.653224 | Small gain over P33 blend; plot BAD stayed 11. |
| P38 global all singletons | Re-optimize all singleton validation prediction CSVs. | done | linprog | 1.652151 | Better MAE; selected risk035/meta/range/fall020/P25/P33/range_scaled. Plot BAD stayed 11. |
| P39 causal filter | Causal EMA/trend filter on P38 predictions. | done | grid | 1.651716 | Small MAE gain; no plot BAD gain. |
| P40 causal affine filter | Add simple global scale/bias grid to P39-style causal filter. | done | grid | 1.651530 | More MAE gain but RMSE worsened. |
| P41 `recent20_state_space_tcn2` | State-space delta head with decoder TCN2. | done | 23 | 2.009008 | Single model failed, but 1.17% blend improved MAE to 1.651399. |
| P42 causal filter on P41 blend | Re-filter the P41 blend. | done | grid | 1.651322 | Tiny improvement; plot BAD stayed 11. |
| P43 `recent20_dual_delta_tcn2` | Dual level/delta gate head with decoder TCN2. | done | 28 | 1.788210 | Useful diversity; 5.79% blend improved MAE to 1.650669 and one selected plot set to BAD 10. |
| P44 global all with P42 | Re-optimize all singleton runs plus P42 current best. | done | linprog | 1.649594 | Current meaningful best; P42 76.5%, P43 6.6%, meta 7.1%, plus small range/risk/P25/P33 weights. Plot BAD 11 on its primary set. |
| P45 `recent20_regime_gated_tcn2` | Regime/gated expert head with regime loss. | done | 20 | 2.066970 | Gate expert failed to stabilize level prediction; zero ensemble weight. |
| P46 `dual_delta_tcn2_seed123` | Seed diversity for useful P43 structure. | done | 23 | 1.824681 | Worse than P43 and zero ensemble weight. |
| P47 causal filter on P44 | Re-filter current P44 best. | done | grid | 1.649594 | Numerically tiny MAE/RMSE gain only; plot BAD stayed 11. |
| P48 `range_recent20_future_delta_aux` | Add future FMS/delta/event auxiliary losses. | done | 12 | 1.911052 | Future auxiliary hurt level tracking; zero ensemble weight. |

## Current Best After P48

Validation-selected current best:

- `filter_p47_p44_causal_affine_grid_val`
- Validation MAE: 1.649594
- Validation RMSE: 2.454637
- Validation R2: 0.701551
- Validation plot proxy: 6 good / 1 medium / 11 bad

P47 causal filter settings:

- input: `ensemble_p44_global_all_with_p42_val`
- alpha: 0.35
- trend_beta: 1.0
- raw_weight: 0.75
- scale: 1.0
- bias: 0.0

P44 main validation weights with nonzero contribution:

- `p42_current`: 0.765121
- `meta_hgb`: 0.070752
- `goal_p43_recent20_dual_delta_tcn2`: 0.066320
- `goal_p3_range_scaled_risk035`: 0.030981
- `goal_p25_range_recent20_enddrop010`: 0.029219
- `goal_p13_range_recent20`: 0.016780
- `risk035`: 0.011065
- `goal_p33_range_recent20_trajshape003`: 0.009761

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.649594.
- `BAD <= 8`: not met on validation; best current primary proxy remains 11/18, with one P43-selected comparison reaching 10/18 but not closing the target.

Warnings:

- Validation MAE improved through increasingly fine ensemble/postprocess tuning, but RMSE and plot BAD did not improve proportionally.
- Test has not been re-evaluated for P30-P48 because the active validation target is not met and test must remain final-report-only.

## Attempts P49-P73

Continuation after P48. All model selection, ensemble weighting, and postprocess parameter searches below used validation predictions only. Test was not used.

| Attempt | Intention | Status | Best epoch / method | Val MAE | Plot proxy | Notes |
|---|---|---|---:|---:|---|---|
| P49 `range_recent20_gru_tcn_multiscale` | Add GRU/TCN multiscale stream dynamics. | done | 11 | 1.930124 | not selected | Weak single model; zero ensemble value. |
| P50 `delta_scale150` | Increase current delta scale to preserve amplitude. | done | 19 | 1.898495 | not selected | Too aggressive; zero ensemble value. |
| P51 `delta_scale050` | Test lower delta scale. | done | 63 | 1.707540 | 6/1/11 on selected set | Useful diversity; P51 blend improved MAE to 1.643675. |
| P52 `delta_scale025` | Continue low-delta-scale sweep. | done | 66 | 1.665435 | 6/2/10 for blend | Strong single and useful blend; P52 blend improved MAE to 1.630418. |
| P53 `delta_scale035` | Interpolate between 0.25 and 0.5. | done | 48 | 1.733254 | not selected | No gain over P52 blend. |
| P54 `delta_scale015` | Lower scale for plot-shape diversity. | done | 63 | 1.686091 | single 7/2/9 on P54-blend set | P54 blend improved MAE to 1.627971. |
| P55 `delta_scale010` | Check lower bound below P54. | done | 42 | 1.714843 | 6/2/10 | No ensemble weight against current best. |
| P56 `delta_scale020` | Interpolate around P54/P52. | done | 65 | 1.675000 | 6/3/9 primary; 7/4/7 on P54 set | Good plot diversity on fixed P54 set, but zero MAE weight. |
| P57 `delta_scale025_seed7` | Seed diversity for the best low-scale family. | done | 23 | 1.922310 | not selected | Seed 7 basin failed. |
| P58 anchor transform on P54 blend | Anchor-aware range transform for slight range/bias correction. | done | grid | 1.626054 | 6/2/10 | Small MAE gain, RMSE worsened. |
| P59 anchor transform on P54 single | Try to reduce BAD from the P54 single curve. | done | grid | 1.665467 | 6/2/10 | Improved MAE over P54 single but not plot BAD. |
| P60 fixed plot-set comparison | Compare low-scale candidates on the P54 primary plot set. | done | analysis | n/a | P56 got 7 BAD on P54 set | Useful diagnostic only; not enough for primary-model plot target. |
| P61 anchor transform on P56 | Try to make P56 plot-primary pass. | done | grid | 1.662733 | 6/1/11 | MAE improved over P56, plot got worse. |
| P62 P56/P47 plot blends | Blend P56 and P47 to reduce plot bad cases. | done | fixed weights | 1.637535 best fixed | 6/1/11 | MAE ok, plot worse. |
| P63 `delta_scale025_fall010` | Add rapid-drop/fall auxiliary to low-scale head. | done | 73 | 1.695959 | 6/2/10 | Fall head hurt level fit; P56 scored 8 BAD on this plot set only. |
| P64 `delta_scale025_trajshape010` | Combine low-scale head with trajectory-shape auxiliary 0.01. | done | 64 | 1.664409 | 6/1/11 single | Useful diversity; P64 blend improved MAE to 1.625563. |
| P65 `delta_scale025_trajshape005` | Weaken shape auxiliary to preserve level fit. | done | 66 | 1.655427 | 6/2/10 single | Strong useful diversity; P65 blend improved MAE to 1.622651. |
| P66 `delta_scale025_trajshape0025` | Check lower shape-weight side. | done | 66 | 1.666231 | 6/1/11 | No ensemble weight against P65 blend. |
| P67 global low-scale/shape reweight | Re-optimize P47/P52/P54/P56/P55/P64/P65/P58 candidates. | done | linprog | 1.619075 | 6/2/10 | Weights: P47 41.5%, P65 31.5%, P64 15.1%, P54 12.0%. |
| P68 anchor transform on P67 | Re-apply anchor transform to global low-scale/shape ensemble. | done | grid | 1.618009 | 6/2/10 | Small MAE gain, RMSE worsened. |
| P69 global all reweight after shape | Re-optimize all saved goal/ensemble/filter/transform validation predictions. | done | linprog | 1.617231 | 6/2/10 | Nonzero: P64, P65, P24 causal meta blend, P59, P68. |
| P70 anchor transform on P69 | Try anchor transform on P69. | done | grid | 1.617231 | unchanged | Identity was best. |
| P71 causal filter on P69 | Causal EMA/trend filter on P69 predictions. | done | grid | 1.615953 | 6/2/10 | New best; RMSE also improved to 2.372971. |
| P72 global all reweight after P71 | Re-optimize all candidates including P71. | done | linprog | 1.615921 | 6/2/10 | Tiny MAE gain over P71; mostly P71 weight. |
| P73 `trajshape005_scenario` | Add scenario static features to low-scale+shape family. | done | 45 | 1.893361 | not selected | Scenario features destabilized level fit. |

## Current Best After P73

Validation-selected current best:

- `ensemble_p72_global_all_after_p71_val`
- Validation MAE: 1.615921
- Validation RMSE: 2.373224
- Validation R2: 0.719664
- Validation plot proxy: 6 good / 2 medium / 10 bad

P72 main nonzero validation weights:

- `filter_p71_p69_causal_affine_grid_val`: 0.955527
- `transform_p68_anchor_range_p67_global_val`: 0.025224
- `goal_p65_range_recent20_delta_scale025_trajshape005`: 0.010041
- `ensemble_p24_causal_meta_blend_val`: 0.008840
- `ensemble_p2_val`: 0.000368

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.615921.
- `BAD <= 8`: not met for the current primary validation plot proxy; best current selected proxy remains 10/18.

Warnings:

- The strongest new gain came from low current-delta scaling plus weak trajectory-shape auxiliary, then validation-only global ensembling and causal filtering.
- P56 can score BAD 7-8 on some fixed plot sets, but when used as the primary model its plot proxy remains above target. This is diagnostic evidence, not goal completion.
- Test has not been re-evaluated for P49-P73 because the active validation target is not met and test must remain final-report-only.

## Attempts P74-P89

Continuation after P73. Full training remained allowed by direct user instruction. All searches below used validation predictions only; test remained skipped/final-report-only.

| Attempt | Intention | Status | Best epoch / method | Val MAE | Plot proxy | Notes |
|---|---|---|---:|---:|---|---|
| P74 `delta_scale025_trajdelta005` | Keep only 5s/10s trajectory-delta auxiliary at weight 0.005, removing centered/range shape terms. | done | 66 | 1.648885 | 6/3/9 single | Strong single candidate. Tiny P72/P74 blend improved MAE to 1.615912. |
| P75 anchor transform on P74 | Anchor/range postprocess for P74. | done | grid | 1.637146 | 6/1/11 | MAE improved over P74 single, but plot got worse. |
| P76 causal filter on P74 | Causal EMA/trend filtering on P74. | done | grid | 1.646519 | 6/2/10 | RMSE improved, but MAE/plot did not beat current best. |
| P77 `trajdelta010` | Increase trajectory-delta auxiliary weight to 0.01. | done | 47 | 1.714481 | not selected | Too strong; zero blend weight. |
| P78 `delta_scale020_trajdelta005` | Combine P56-like lower delta scale with P74 delta-only auxiliary. | done | 66 | 1.682512 | 6/1/11 | Lower scale hurt MAE and plot; zero blend weight. |
| P79 `delta_scale015_trajdelta005` | Test still lower delta scale with delta-only auxiliary. | done | 63 | 1.685446 | 6/1/11 | Same failure pattern as P78. |
| P80 `risktcn1` | Add 1-layer temporal TCN context to rise/drop risk head. | done | 45 | 1.824191 | 6/1/11 | Temporal risk head destabilized level prediction; zero blend weight. |
| P81 `lr35e5` | Lower LR for P74 structure to stabilize late validation oscillation. | done | 57 | 1.702383 | 6/1/11 | Lower LR underperformed; zero blend weight. |
| P82 `recent25_trajdelta005` | Extend recent motion window from 20s to 25s. | done | 73 | 1.650500 | 6/1/11 single; 6/2/10 blend | Single similar to P74; 7.26% blend improved MAE to 1.615482. |
| P83 `recent30_trajdelta005` | Extend recent window to 30s. | done | 79 | 1.637715 | 5/2/11 single; 6/2/10 blend | Best new singleton. P83/P82 blend improved MAE to 1.606179. |
| P84 `recent35_trajdelta005` | Test whether longer recent context continues helping. | done | 66 | 1.675499 | 6/1/11 | 35s recent window was too long/noisy; zero blend weight. |
| P85 `recent30_ep120` | Re-run P83 with 120 epochs and patience 15 because P83 peaked near epoch 79/80. | done | 66 | 1.652975 | 6/3/9 | Longer run did not improve MAE; zero blend weight. |
| P86 `recent30_dropout005` | Reduce model dropout from 0.10 to 0.05 for recent30. | done | 66 | 1.677731 | 6/1/11 | Lower dropout hurt. |
| P87 `recent30_dropout015` | Increase model dropout to 0.15 for recent30. | done | 57 | 1.705030 | 6/1/11 | Higher dropout hurt. |
| P88 `recent30_staticdrop000` | Remove static-feature dropout to strengthen participant prior. | done | 73 | 1.650458 | 6/1/11 | No gain over P83; zero blend weight. |
| P89 global reweight after P88 | Re-optimize all validation prediction CSVs after P74-P88. | done | linprog | 1.598226 | 6/2/10 | New validation MAE best. Plot target remains unmet. |

## Current Best After P89

Validation-selected current best:

- `ensemble_p89_global_after_p88_val`
- Validation MAE: 1.598226
- Validation RMSE: 2.337405
- Validation R2: 0.729378
- Validation prediction mean / target mean: 9.077937 / 8.966852
- Validation prediction std / target std: 4.225840 / 4.493165
- Validation plot proxy: 6 good / 2 medium / 10 bad

P89 nonzero validation weights:

- `goal_p83_range_recent30_delta_scale025_trajdelta005`: 0.474266
- `ensemble_p3_val`: 0.290307
- `transform_p75_anchor_range_p74_trajdelta_val`: 0.109078
- `goal_p65_range_recent20_delta_scale025_trajshape005`: 0.066824
- `filter_p47_p44_causal_affine_grid_val`: 0.059524

Recent-window conclusion:

- 25s recent window gave small ensemble diversity.
- 30s recent window was the best new single-model family and drove the main P83/P89 improvement.
- 35s recent window degraded MAE and plot quality.

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.598226.
- `BAD <= 8`: not met for the current primary validation plot proxy; P89 remains 10/18 BAD.

Warnings:

- P89 is validation-selected and must not be interpreted as final test performance.
- Test has not been re-evaluated for P74-P89 because the active validation target is not met and test must remain final-report-only.
- The strongest improvement in this block came from longer recent motion context plus validation-only reweighting; plot structure remains the main unresolved weakness.

## Attempts P90-P97

Continuation after P89. Full training remained allowed by direct user instruction. All searches below used validation predictions only; test remained skipped/final-report-only.

| Attempt | Intention | Status | Best epoch / method | Val MAE | Plot proxy | Notes |
|---|---|---|---:|---:|---|---|
| P90 `recent28_trajdelta005` | Interpolate recent window between 25s and 30s. | done | full train | 1.661162 | 6/1/11 | Worse than P83/P89; zero blend weight. |
| P91 `recent32_trajdelta005` | Interpolate recent window between 30s and 35s. | done | full train | 1.651138 | 6/3/9 | Plot medium count improved, but MAE did not; zero blend weight. |
| P92 `trajdelta0035` | Reduce trajectory-delta auxiliary weight to avoid over-regularizing level. | done | full train | 1.663624 | 6/1/11 | Weaker trajectory auxiliary did not help; zero blend weight. |
| P93 `trajdelta0075` | Increase trajectory-delta auxiliary weight moderately. | done | full train | 1.649134 | 6/2/10 | Similar single quality to P83; equal blend improved RMSE only, MAE worse; zero global weight. |
| P94 global reweight after P90-P93 | Re-optimize all available validation prediction CSVs. | done | linprog | 1.598226 | 6/2/10 | Same weights and MAE as P89; P90-P93 added no useful MAE diversity. |
| P95 `decoder_tcn1` | Add 1-layer temporal TCN context to decoder/head. | done | full train | 1.854898 | 6/1/11 | Structural addition was harmful; zero blend weight. |
| P96 `motion_tcn1` | Add 1-layer temporal TCN context to motion encoder path. | done | 61 | 1.731411 | 6/1/11 | Slightly reasonable direction accuracy but high MAE; P89/P96 blend selected weight 0. |
| P97 `transition_weighting` | Upweight rise/drop/recovery target transitions during level loss. | done | 47 | 1.735006 | 6/1/11 | Direction accuracy rose slightly, but MAE and flat-session rate worsened; P89/P97 blend selected weight 0. |
| P98 `smooth005` | Add weak 1-step target-delta loss to improve local trajectory direction. | done | 66 | 1.669364 | 6/2/10 | Delta corr/direction improved, but MAE worsened; P89/P98 MAE-optimal weight was 0. |
| P98 fixed blends | Mix P98 into P89 at 10/20/50% to test plot/RMSE tradeoff. | done | fixed | 1.599560 best MAE at 10% | 6/2/10 best | 10% improved RMSE/R2 and direction slightly but did not reduce BAD; not promoted. |

Current best remains unchanged after P98:

- `ensemble_p89_global_after_p88_val`
- Validation MAE: 1.598226
- Validation RMSE: 2.337405
- Validation R2: 0.729378
- Validation plot proxy: 6 good / 2 medium / 10 bad

P96-P98 conclusion:

- Adding shallow temporal TCN blocks inside the current architecture did not improve the validation-selected objective.
- Transition weighting behaved as expected in one narrow sense, increasing 5s direction accuracy, but it increased level error enough to be unusable.
- Weak 1-step delta loss improved local direction/RMSE when mixed into P89, but not enough to reduce BAD count.
- The best path is still not stronger transition/delta loss alone; the next candidates should add genuinely different prediction behavior or more targeted post-processing while preserving leakage safety.

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.598226.
- `BAD <= 8`: not met for the current primary validation plot proxy; P89 remains 10/18 BAD.

Warnings:

- P89 is validation-selected and must not be interpreted as final test performance.
- Test has not been re-evaluated for P90-P98 because the active validation target is not met and test must remain final-report-only.

## Attempts P99-P144

Continuation after P98. Test remained skipped/final-report-only. P100 onward introduced validation-only post-hoc blending of already saved regression and ordinal current-FMS heads.

| Attempt | Intention | Status | Method | Val MAE | Plot proxy | Notes |
|---|---|---|---|---:|---|---|
| P99 `residual_update` | Replace range-scaled level/delta gate with cumulative residual update head. | done | full train | 1.952467 | 6/0/12 | Failed; P89/P99 blend selected weight 0. |
| P100 headblend P83 | Recombine P83 regression/ordinal heads with validation-selected ordinal weight. | done | ordinal weight 0.400 | 1.613943 | 6/2/10 | Strong post-hoc gain over P83 single. |
| P100/P89 blend | Add P100 to P89. | done | linprog | 1.592210 | 6/2/10 | First new best after P89. |
| P104 global after headblend | Global reweight including P100-P103. | done | linprog | 1.582307 | 6/2/10 | Nonzero: P100 63.6%, P3 21.8%, meta P9 HGB 10.3%, P65 4.4%. |
| P112 more headblend global | Add headblend variants for P52/P54/P56/P64 and reweight. | done | linprog | 1.580440 | 6/2/10 | Nonzero: P105/P83 fine, P106/P3, meta P9 HGB, P65. |
| P119 conditional P83 headblend | Use anchor-FMS bins for per-bin ordinal weights. | done | anchor bins 6/12 | 1.601314 | 6/2/10 | Single model improved; best with P112 blend became P122. |
| P122 P112 + P119 | Blend global P112 with anchor-conditioned P83. | done | linprog | 1.576583 | 6/2/10 | P112 73.8%, P119 26.2%. |
| P127 2D conditional blend | Add P83 anchor/reg and reg/time conditional headblend variants. | done | linprog | 1.568340 | 6/2/10 | Good numeric gain, plot unchanged. |
| P128 3D conditional P83 | Use anchor, regression prediction, and time bins for P83 headblend. | done | conditional grid | 1.557368 | 6/2/10 | Strongest single post-hoc candidate, but small-bin overfit risk. |
| P129 P127 + P128 | Blend P128 with previous best. | done | linprog | 1.555656 | 6/2/10 | P128 76.6%, P123 23.4%. |
| P134 more 3D conditional | Add P82/P88/P90/P93 3D conditional headblend variants. | done | linprog | 1.546084 | 6/2/10 | P129 59.1%, P130/P82 22.9%, P133/P93 18.0%. |
| P139 P134 + P136 | Add P65/P52/P66/P85 3D conditional variants. | done | linprog | 1.544560 | 6/1/11 | MAE improved but primary plot set worsened. |
| P141 P139 + P140 | Add tuned P83 3D thresholds. | done | linprog | 1.541526 | 6/2/10 | P139 67.4%, P140 32.6%. |
| P144 P141 + P142/P143 | Add tuned P82/P93 threshold variants. | done | linprog | 1.541471 | 6/2/10 | Very small numeric gain; current validation MAE best. |

Current best after P144:

- `ensemble_p144_p141_plus_threshold_variants_val`
- Validation MAE: 1.541471
- Validation RMSE: 2.268533
- Validation R2: 0.745091
- Validation prediction mean / target mean: 9.129888 / 8.966852
- Validation prediction std / target std: 4.231031 / 4.493165
- Validation plot proxy: 6 good / 2 medium / 10 bad

P144 nonzero validation weights:

- `ensemble_p141_p139_plus_p140_val`: 0.948793
- `transform_p142_cond3_headblend_p82_a612_r512_t180_val`: 0.036952
- `transform_p143_cond3_headblend_p93_a510_r812_t180_val`: 0.014255

Head-blend conclusion:

- The saved regression and ordinal heads contain complementary errors. The original fixed 0.15 ordinal blend was not optimal for the validation split.
- Global post-hoc head blending gave a large MAE gain: P89 1.598226 to P144 1.541471.
- Conditional 3D head blending is powerful but uses many validation-selected bin weights; this is a real overfit risk and must be validated carefully before any final test claim.
- Plot BAD remains stuck at 10/18 despite numeric gains. The numeric improvement is mostly scale/regime correction, not a qualitative trajectory-shape solution.

Target status remains unmet:

- `MAE <= 1.5`: not met; best validation MAE is 1.541471.
- `BAD <= 8`: not met; best current primary validation plot proxy remains 10/18 BAD.

Warnings:

- P144 is validation-selected and must not be interpreted as final test performance.
- Test has not been re-evaluated for P99-P144 because the active validation target is not met and test must remain final-report-only.
- P128-P144 use validation-selected conditional post-processing with small bins; report them as exploratory until verified by a stricter protocol.

## Attempts P145-P146

Continuation after P144. Full training remained allowed by direct user instruction. Test remained skipped/final-report-only during the search.

| Attempt | Intention | Status | Method | Val MAE | Val RMSE | Plot proxy | Notes |
|---|---|---|---|---:|---:|---|---|
| P145 `ordcombine040` | Retrain the recent30 + low-delta + trajectory-delta model with stronger ordinal-head combination pressure. | done | full train | 1.705851 | 2.541815 | not promoted | Best epoch 64. The P144/P145 validation blend selected P145 weight 0.0, so this was rejected. |
| P146 conditional affine on P144 | Correct P144 scale/bias by bins over calibration anchor FMS, current prediction level, and current time. | done | validation-fitted affine postprocess | 1.488240 | 2.130754 | 6 / 3 / 9 | Meets the active validation MAE target, but is a validation-fitted postprocess with overfit risk. |

## Current Best After P146

Validation-selected current best:

- `transform_p146_cond_affine_p144_a510_p510_t160_val`
- Validation MAE: 1.488240
- Validation RMSE: 2.130754
- Validation R2: 0.775114
- Validation prediction mean / target mean: 8.968229 / 8.966852
- Validation prediction std / target std: 3.952069 / 4.493165
- Validation plot proxy: 6 good / 3 medium / 9 bad

P146 conditional affine settings:

- Source: `ensemble_p144_p141_plus_threshold_variants_val`
- Condition columns: `anchor_fms`, `predicted_fms_now`, `current_time`
- Condition bins: anchor `[5, 10]`, prediction `[5, 10]`, time `[160]`
- Correction per bin: least-squares `target_fms_now = scale * predicted_fms_now + bias`, clipped to `[0, 20]`

Target status:

- `MAE <= 1.5`: met on validation by P146.
- `BAD <= 8`: not met; current primary plot proxy is 9/18 BAD.

Warnings:

- P146 fits affine parameters directly on validation labels. It is useful evidence that the remaining error has a large conditional scale/bias component, but it has stronger overfit risk than train-fitted or purely learned-model changes.
- This should not be presented as final test performance. A stricter final claim needs the selected validation transform applied once to held-out test predictions, without re-selecting bins or parameters on test.

## Verification After P146

- `py_compile`: passed for ensemble, causal filter, meta calibrator, anchor transform, head blend, and conditional affine scripts.
- `scripts/run_densefms_sanity_tests.py`: passed.
- Required leakage checks passed through the sanity suite: seconds-to-steps conversion, target shift correctness, calibration leakage, recent-window leakage, anchor policy, model forward shape checks, and dry-run command generation.
- P145 full training produced `best.pt`, `metrics.json`, and `val_predictions.csv`.
- P146 produced `metrics.json`, `val_predictions.csv`, leaderboard, plot summary, scatter plot, trend metric plot, and trajectory plots under `analysis_p146_cond_affine_p144`.

## Attempts P147-P156: Single-Checkpoint Train-Only Calibration

Goal: convert the P146-style conditional scale/bias correction into a train-only, single-model checkpoint. Model selection continued to use validation only. Test was evaluated once only after selecting the best single-checkpoint candidate from validation.

Implementation changes:

- Added an identity-initialized `current_affine_head` that predicts per-time scale/bias inside `OnlineFMSRiskTracker`.
- Added `--init_checkpoint`, `--freeze_loaded_parameters`, and `--trainable_parameter_patterns` so a P83 checkpoint can be loaded and only the new calibration head can be trained.
- Added an identity-initialized `current_binned_affine_head` over anchor FMS, predicted FMS, and current time bins. This mirrors P146's condition axes but trains from train split loss inside the model.
- Prediction CSV export now records pre-affine and affine scale/bias diagnostic columns.

Validation results:

| Attempt | Method | Best epoch | Val MAE | Val RMSE | Status |
|---|---|---:|---:|---:|---|
| P147 `recent30_affine_head` | affine head trained from scratch | 5 | 1.925870 | 2.907078 | rejected |
| P148 `affine_narrow_lr2e4` | narrower affine, lower LR from scratch | 91 | 1.700737 | 2.485365 | rejected |
| P149 `p83_frozen_affine_head` | P83 frozen, train MLP affine only | 13 | 1.614963 | 2.367614 | improved |
| P150 `p83_frozen_affine_dropout0` | P83 frozen, dropout 0, train MLP affine only | 20 | 1.608807 | 2.352410 | improved |
| P151 `wide_affine` | wider MLP affine range | 4 | 1.630114 | 2.350811 | rejected |
| P152 `p83_frozen_affine_dropout0_lr3e4` | P83 frozen, dropout 0, lower LR MLP affine | 47 | 1.596678 | 2.346217 | best single train-only |
| P153 `finetune_heads_affine` | unfreeze current range/ordinal heads + affine | 63 | 1.621300 | 2.379848 | rejected |
| P154 `full_finetune_affine` | full low-LR fine-tune from P83 + affine | 4 | 1.633212 | 2.380552 | rejected |
| P155b `frozen_binned_affine` | P83 frozen, train condition-binned affine only | 1 | 1.707499 | 2.458590 | rejected |
| P156 `mlp_plus_binned_affine` | P83 frozen, train MLP + binned affine | 47 | 1.606056 | 2.351137 | rejected |

Comparison against non-single-model references:

| Model/result | Selection type | Val MAE | Val RMSE | Plot proxy |
|---|---|---:|---:|---|
| P146 conditional affine on P144 | validation-fitted postprocess | 1.488240 | 2.130754 | 6 / 3 / 9 |
| P144 ensemble | validation-selected ensemble/postprocess | 1.541471 | 2.268533 | 6 / 2 / 10 |
| P152 | train-only single checkpoint | 1.596678 | 2.346217 | 5 / 2 / 11 |
| P83 | train-only single checkpoint | 1.637715 | 2.355911 | 5 / 2 / 11 |

P152 plot/trend metrics on validation:

- MAE/RMSE: 1.596678 / 2.346217
- Pearson session mean: 0.497821
- Centered MAE session mean: 1.358700
- 5s delta correlation: 0.423023
- 5s direction accuracy: 0.695409
- Flat-range failure rate: 0.000000

Interpretation:

- P152 is a real single-checkpoint improvement over P83: MAE improves by about 0.041.
- It does not reach P141/P144/P146. The gap to P144 is about 0.055 MAE, and the gap to P146 is about 0.108 MAE.
- The binned train-only version did not generalize. P146's large gain is likely partly validation-label fitting, not just a missing model layer.
- Fine-tuning existing prediction heads made validation worse, so P83's learned representation/head is already near a local optimum under this split.
- Plot quality did not materially improve over P83. Numeric scale got better, but trajectory shape/regime transitions remain the bottleneck.

Final selected single-checkpoint candidate:

- `goal_p152_p83_frozen_affine_dropout0_lr3e4`
- Selection basis: validation MAE among train-only single checkpoints.
- Test evaluation was run only after this validation selection.

Final test metrics for P152:

- Test MAE: 2.133443
- Test RMSE: 2.910496
- Test R2: 0.591614
- Test n: 9360
- Test plot/trend summary: Pearson session mean 0.350241, centered MAE session mean 1.331748, 5s delta correlation 0.345227, 5s direction accuracy 0.651171, flat-range failure rate 0.020000.

Generated outputs:

- Validation analysis: `runs/online_fms_current_tracking_0510_goal_search/analysis_p152_single_train_affine`
- Test analysis: `runs/online_fms_current_tracking_0510_goal_search/analysis_p152_single_train_affine_test`
- Final selected checkpoint: `runs/online_fms_current_tracking_0510_goal_search/goal_p152_p83_frozen_affine_dropout0_lr3e4/best.pt`
- Final test predictions: `runs/online_fms_current_tracking_0510_goal_search/goal_p152_p83_frozen_affine_dropout0_lr3e4/eval_test/test_predictions.csv`

Verification after P156:

- `py_compile` passed for `model.py`, `train.py`, and `run_densefms_sanity_tests.py`.
- `scripts/run_densefms_sanity_tests.py` passed, including seconds-to-steps, target shift, calibration leakage, recent-window leakage, anchor policy, model forward shape, affine identity initialization, binned affine identity initialization, and dry-run command generation checks.

## Attempts P157-P160: Plot-Shape-Oriented Training

Goal: make validation trajectories follow FMS rise/drop/recovery shape more directly, even if pointwise MAE worsens. Test was not re-evaluated for these exploratory plot-shape runs.

Implementation change:

- Added validation trajectory metrics to `collect_online_current_risk_predictions` under `metrics["trajectory"]`.
- Training logs now print `val_shape=trajectory.centered_mae_session_mean` and `val_dir5=trajectory.direction_acc_5s`.
- Checkpoint selection can now use metrics such as `--selection_metric trajectory.centered_mae_session_mean` or `--selection_metric trajectory.direction_acc_5s`.

Validation results:

| Attempt | Selection metric | Method | Best epoch | Val MAE | Val RMSE | Pearson | Centered MAE | Delta corr 5s | Dir acc 5s | Flat rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P157 `p83_shape_centered_select` | centered shape min | P83 init + stronger delta/centered/range trajectory loss | 2 | 1.643590 | 2.402666 | 0.511525 | 1.355789 | 0.426801 | 0.706592 | 0.016949 |
| P158 `p83_shape_direction_select` | 5s direction max | Same loss as P157, direction-selected checkpoint | 1 | 1.681034 | 2.436120 | 0.519082 | 1.368874 | 0.424149 | 0.710418 | 0.000000 |
| P159 `p83_trajectory_decoder_shape_select` | centered shape min | Switch current head to trajectory decoder with offsets `[0,5,10,20,40]` | 11 | 1.773740 | 2.554234 | 0.429037 | 1.390744 | 0.354601 | 0.672160 | 0.000000 |
| P160 `p83_transition_shape_direction_select` | 5s direction max | P83 init + transition weighting + moderate shape loss | 1 | 1.693832 | 2.464703 | 0.510959 | 1.360810 | 0.417930 | 0.708946 | 0.000000 |

Reference validation metrics:

| Reference | Val MAE | Pearson | Centered MAE | Delta corr 5s | Dir acc 5s | Flat rate |
|---|---:|---:|---:|---:|---:|---:|
| P144 ensemble | 1.541471 | 0.510926 | 1.293863 | 0.434875 | 0.703355 | 0.000000 |
| P152 single MAE-affine | 1.596678 | 0.497821 | 1.358700 | 0.423023 | 0.695409 | 0.000000 |
| P83 single | 1.637715 | 0.502220 | 1.342811 | 0.428193 | 0.702766 | 0.033898 |

Interpretation:

- P157/P158 move in the desired plot direction: Pearson and 5s direction accuracy improve over P152 and P83.
- The tradeoff is pointwise error. P157/P158 no longer beat P152 or P83 on MAE.
- P159 confirms that switching completely to a short trajectory decoder is too costly from this checkpoint; the newly initialized trajectory head loses too much level/regime accuracy.
- P160 shows transition weighting does not add a clear gain beyond the direct trajectory-shape loss.
- Current best plot-following single checkpoint is P157 if balancing shape and MAE, or P158 if prioritizing 5s direction/visual movement. Neither reaches P144's centered shape score.
- P157 was selected as the final plot-shape candidate because it improves validation Pearson/direction over P152 with less MAE damage than P158.

Final test metrics for P157:

- Test MAE: 2.186328
- Test RMSE: 2.994368
- Test R2: 0.567738
- Test trajectory Pearson session mean: 0.336350
- Test centered MAE session mean: 1.340673
- Test 5s delta correlation: 0.351771
- Test 5s direction accuracy: 0.653655
- Test flat-range failure rate: 0.000000
- Test plot proxy on the selected 24-session set: 7 good / 4 medium / 13 bad. For comparison on the same set, P152 had 6 good / 2 medium / 16 bad.

Generated outputs:

- `runs/online_fms_current_tracking_0510_goal_search/analysis_p157_shape_select`
- `runs/online_fms_current_tracking_0510_goal_search/analysis_p158_direction_select`
- `runs/online_fms_current_tracking_0510_goal_search/analysis_p159_traj_decoder`
- `runs/online_fms_current_tracking_0510_goal_search/analysis_p160_transition_shape`
- `runs/online_fms_current_tracking_0510_goal_search/analysis_p157_shape_test`
- Final P157 test predictions: `runs/online_fms_current_tracking_0510_goal_search/goal_p157_p83_shape_centered_select/eval_test/test_predictions.csv`

Verification after P157-P160:

- `py_compile` passed for `model.py`, `train.py`, and `run_densefms_sanity_tests.py`.
- `scripts/run_densefms_sanity_tests.py` passed after adding trajectory validation metrics.
