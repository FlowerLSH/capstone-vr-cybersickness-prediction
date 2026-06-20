# Online Current FMS Remaining + Param Search Final Report 0509

## 1. Modified or added files

- `docs/codex/online_current_integrated_improvement_plan_0509.md`: `FULL_TRAINING_ALLOWED = true` confirmed at line 3.
- `src/densefms_forecast/data.py`: scenario static feature parsing/encoding and static report support.
- `src/densefms_forecast/online_current/heads.py`: `residual_update`, `person_prior` current-head modes.
- `src/densefms_forecast/model.py`: multi-timescale motion features, coarse/regime/uncertainty outputs.
- `src/densefms_forecast/train.py`: new aux losses, uncertainty loss stabilization, motion-pretrain loading, prediction sigma export, CLI wiring.
- `scripts/pretrain_online_current_motion_encoder.py`: motion-only pretraining script.
- `scripts/run_online_current_remaining_experiments.py`: remaining document-item experiment runner.
- `scripts/run_online_current_promising_param_search.py`: narrow promising-candidate parameter search runner.
- `scripts/run_densefms_sanity_tests.py`: new leakage/static/head/aux/pretrain/runner checks.
- `docs/codex/online_current_remaining_param_search_final_report_0509.md`: this report.

## 2. New CLI/config options

- Model/data: `--motion_feature_mode multi_timescale_v1`, `--current_head_mode residual_update|person_prior`, `--static_features ... scenario`.
- Aux heads/losses: `--coarse_band_bins`, `--coarse_band_loss_weight`, `--regime_head_enabled`, `--regime_class_count`, `--regime_loss_weight`, `--regime_delta_slow_threshold`, `--regime_delta_rapid_threshold`, `--regime_high_threshold`.
- Uncertainty: `--uncertainty_head_enabled`, `--uncertainty_loss_weight`.
- Pretraining: `--motion_pretrain_checkpoint`.
- Existing integrated search options were reused: `--motion_feature_mode causal_dynamics_v1`, FDS blend, ordinal blend, risk/future/delta/event/trajectory aux options.

## 3. Dataset/windowing changes

- Scenario is inferred from DenseFMS filenames and can be one-hot encoded as a static feature.
- Static feature pipeline now supports `age`, `gender`, `mssq`, and `scenario`.
- All experiments used `sampling_interval=0.5` and `max_session_points=420`; selected final config used `calibration_seconds=120.0`, `recent_window_seconds=10.0`, `horizon_seconds=10.0`.
- Leakage-safe rules are preserved: calibration FMS only from calibration range, recent motion only up to current time, target is shifted to `t + horizon`, and no target FMS is used as input.

## 4. Model changes

- Added multi-timescale motion feature bank.
- Added residual update and person-prior current prediction heads.
- Added optional coarse-band, regime, and uncertainty heads.
- Added motion-only pretraining path for the deep TCN stream.
- Final selected model remains the online risk tracker with `motion_feature_mode=causal_dynamics_v1`, FDS smoothing, ordinal blend, and static age/MSSQ/gender features.

## 5. Anchor/static/multi-horizon support status

- Anchor policy checks pass for calibration-end, sparse anchor, start-only, and related online-current paths.
- Static support is complete for age, gender, MSSQ, and scenario. Final selected config uses age/MSSQ/gender only.
- Multi-horizon support is active for rapid-rise horizons 5s and 10s; future/delta/event aux horizon paths were implemented and tested, but they were not the final validation winner.

## 6. Sanity test results

- Import/compile check passed for modified Python files.
- `scripts/run_densefms_sanity_tests.py` passed all checks after final runs.
- Covered checks include seconds-to-steps conversion, target shift correctness, calibration leakage, recent-window leakage, anchor policy, model forward shapes, scenario static encoding, remaining-runner dry-run generation, and parameter-search dry-run generation.

## 7. Full-training search budget actually used

- Full training was allowed by the goal file: `FULL_TRAINING_ALLOWED = true`.
- Integrated stage already completed before the remaining sweep: 9 validation-only candidates, 382 total trained epochs.
- Remaining document-item stage: 11 validation-only candidates, 345 total trained epochs.
- Motion pretraining: 1 run, best validation loss at epoch 28.
- Promising parameter search: 10 validation-only candidates, 413 total trained epochs.
- Final test evaluation: exactly 1 final selected checkpoint evaluated on the test split after validation-based selection.
- No commit or push was performed. No parallel training was used.

## 8. Validation leaderboard

Validation-only selection leaderboard from `runs/online_fms_current_tracking_0509_param_search/analysis/online_current_validation_leaderboard.csv`:

| Rank | Label | Val MAE | Val RMSE | Best epoch |
| --- | --- | ---: | ---: | ---: |
| 1 | `psearch_causal_dyn_fds075_ord015_seed42` | 1.922834 | 2.798767 | 38 |
| 2 | `psearch_causal_dyn_risk020_ord015_seed42` | 1.924195 | 2.820190 | 48 |
| 3 | `psearch_person_prior_risk020_seed42` | 1.924522 | 2.829794 | 7 |
| 4 | `integrated_p4_causal_dynamics_v1_seed42` | 1.930335 | 2.837943 | 38 |
| 5 | `psearch_causal_dyn_trajectory_w002_seed42` | 1.935011 | 2.814309 | 38 |
| 6 | `fds_static4` | 1.945173 | 2.771334 | 50 |
| 7 | `psearch_person_prior_ord010_seed42` | 1.948968 | 2.850596 | 7 |
| 8 | `remaining_p5_person_prior_seed42` | 1.961731 | 2.889938 | 11 |
| 9 | `psearch_causal_dyn_risk030_ord015_seed42` | 1.965917 | 2.914693 | 32 |
| 10 | `psearch_causal_dyn_dropout015_seed42` | 1.995081 | 2.891530 | 37 |
| 11 | `psearch_causal_dyn_event_delta_light_seed42` | 2.013798 | 2.869545 | 37 |
| 12 | `psearch_explicit_state_aux_light_seed42` | 2.032062 | 2.860967 | 43 |
| 13 | `remaining_p8_explicit_state_shared_aux_seed42` | 2.050808 | 2.831572 | 33 |
| 14 | `psearch_causal_dyn_lr035_seed42` | 2.097510 | 3.095688 | 26 |

## 9. Final selected configuration

- Selected by validation MAE only: `psearch_causal_dyn_fds075_ord015_seed42`.
- Checkpoint: `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42/best.pt`.
- Base config: `configs/online_current/selected_fds_static4.yaml`.
- Key settings: `motion_feature_mode=causal_dynamics_v1`, `fds_blend=0.75`, `fms_combine_weight_ordinal=0.15`, `risk_loss_weight=0.25`, `learning_rate=0.00045`, `batch_size=48`, `epochs=80`, `patience=10`.
- Data settings: max 420 points, calibration 120s, recent 10s, horizon 10s, static age/MSSQ/gender.

## 10. Final test-set metrics for the selected configuration

Final test evaluation was run only after selecting the configuration from validation results:

- Test MAE: 2.229539
- Test RMSE: 3.005608
- Test R2: 0.564486
- Test within 1 FMS: 0.348825
- Test within 2 FMS: 0.591987
- Rapid-rise 5s: AUROC 0.749780, AUPRC 0.201460, F1 0.237394
- Rapid-rise 10s: AUROC 0.726772, AUPRC 0.280043, F1 0.251852
- Final-warning mode: `rapid_rise_only`; final warning AUROC 0.672451, AUPRC 0.158239, F1 0.283066.

## 11. Generated plots/tables

- Validation leaderboard CSV/JSON: `runs/online_fms_current_tracking_0509_param_search/analysis/online_current_validation_leaderboard.csv`, `.json`.
- Parameter search manifest and dry-run commands: `candidate_manifest.json`, `dry_run_commands.txt`.
- Analysis plots/tables: `prediction_scatter_all.png`, `trend_metric_summary.png`, `plot_judgment_summary.csv`, `plot_judgment_sessions.csv`, trajectory best/dynamic/worst PNGs.
- Final selected test outputs: `eval_test/metrics.json`, `eval_test/test_predictions.csv` with 9360 prediction rows, and 12 test trajectory plots.
- Checkpoints and per-run validation predictions were generated for completed candidates but are under `runs/` and not added to git.

## 12. Git status summary

- Current worktree is dirty.
- Modified tracked files include `AGENTS.md`, `scripts/run_densefms_sanity_tests.py`, `src/densefms_forecast/data.py`, `src/densefms_forecast/evaluate.py`, `src/densefms_forecast/losses.py`, `src/densefms_forecast/model.py`, `src/densefms_forecast/train.py`, and `src/densefms_forecast/utils.py`.
- Many untracked docs/configs/scripts are present, including the online-current configs, scripts, reports, and PDFs already in the workspace.
- No commit and no push were performed.

## 13. Remaining issues or warnings

- The uncertainty-head run initially produced NaN loss. The uncertainty loss path was stabilized, rerun, and completed with validation MAE 2.076671.
- The trainer accepts `--resume` for CLI compatibility but does not perform partial checkpoint resume; practical resume behavior is through `--skip_existing` on completed run directories.
- `nvidia-smi` path configured in the environment was unavailable, but the training logs show CUDA execution.
- Do not treat previous intermediate test evaluations as search signals. The final selected model above was chosen from validation metrics, and only then evaluated on the test split.
