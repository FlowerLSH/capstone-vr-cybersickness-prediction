# LC-SA-TCNFormer full-training search goal

FULL_TRAINING_ALLOWED = true

## Goal

Implement, train, compare, and report LC-SA-TCNFormer variants for DenseFMS future FMS forecasting.

The goal is not only implementation. The goal is to run a bounded full-training search and produce:

1. a validation leaderboard,
2. a selected best configuration based on validation metrics,
3. final test-set metrics for the selected configuration,
4. prediction CSVs,
5. plots,
6. a markdown report.

Do not commit or push.

Do not use test metrics for model selection.

## Core research question

DenseFMS is head/motion-only compared with multimodal DeepTCN-style settings.

This project asks:

Can long calibration, recent motion encoding, static user features, and FMS state anchors compensate for missing multimodal eye/physio input and enable stable future FMS forecasting?

## Required model

Implement and train:

LC-SA-TCNFormer
Long-Calibrated State-Anchored TCN-Transformer

Required branches:

1. Calibration Encoder
   - input: first C seconds of motion + FMS
   - shape: [B, C_steps, 7]
   - structure: Linear -> Dilated TCN -> Transformer Encoder -> Pooling

2. Recent Motion Encoder
   - input: current-time recent W seconds of motion only
   - shape: [B, W_steps, 6]
   - structure: Linear -> Dilated TCN -> Pooling

3. Anchor Encoder
   - input: anchor_fms + time_since_anchor
   - shape: [B, 2]
   - optional depending on anchor_mode

4. Static Encoder
   - input: age, MSSQ, gender one-hot
   - shape: [B, 4]
   - optional depending on use_static

5. Horizon Encoder
   - input: horizon_seconds or normalized horizon
   - shape: [B, 1]
   - used for single-horizon models

6. Fusion Head
   - concatenate branch embeddings
   - MLP
   - output future FMS

## Phase 0. Inspect repository

Before editing, inspect the repo and identify:

- dataset class
- preprocessing code
- train script
- evaluation script
- model registry or model factory
- config files
- metrics code
- plotting code
- split-file handling
- static feature handling, if any
- existing hard-coded calibration/horizon/recent-window values
- current checkpoint and metrics format

Do not assume file names.

## Phase 1. Implementation requirements

Add config/CLI support for:

Data/window options:

- sampling_interval, default 0.5
- calibration_seconds
- recent_window_seconds
- horizon_seconds
- multi_horizon, default false
- horizon_set, default null

Common-window evaluation options:

- common_eval_current_start
- common_eval_current_end
- common_eval_target_start
- common_eval_target_end

Model options:

- model, including lc_sa_tcnformer
- d_model, default 64
- kernel_size, default 3
- dropout, default 0.1
- calib_dilations, default [1, 2, 4, 8, 16]
- recent_dilations, default auto
- transformer_layers, default 1
- transformer_heads, default 4
- transformer_ff_dim, default 128
- pooling, choices [mean, last, attention], default mean

Static options:

- use_static
- no_static

Anchor options:

- anchor_mode, choices [none, calibration_end, recent_start_observed, sparse_observed]
- anchor_interval_seconds, default 60.0
- predict_delta_from_anchor, default false

Loss options:

- loss_type, choices [mse, smooth_l1], default smooth_l1
- loss_mode, choices [level_only, level_plus_trend], default level_only

Training options:

- epochs
- batch_size
- learning_rate
- weight_decay
- patience
- seed
- max_train_batches
- max_eval_batches
- resume
- skip_existing
- save_predictions
- save_plots

## Phase 2. Windowing and leakage rules

Seconds-to-steps conversion:

- calibration_steps = round(calibration_seconds / sampling_interval)
- recent_window_steps = round(recent_window_seconds / sampling_interval)
- horizon_steps = round(horizon_seconds / sampling_interval)

Sanity expectations:

- 30s -> 60 steps
- 60s -> 120 steps
- 90s -> 180 steps
- 120s -> 240 steps
- 10s -> 20 steps
- 30s -> 60 steps
- 2.5s -> 5 steps
- 5s -> 10 steps
- 10s -> 20 steps
- 15s -> 30 steps
- 30s -> 60 steps

For prediction current time t:

Calibration input:

- motion[:C_steps]
- fms[:C_steps]
- concat -> calib_seq [C_steps, 7]

Recent input:

- motion[t - W_steps + 1 : t + 1]
- recent_seq [W_steps, 6]
- no FMS in recent_seq
- no motion after t

Target:

- target = FMS[t + horizon_steps]

Valid time:

- valid_t_start = max(C_steps, W_steps - 1)
- valid_t_end = T - horizon_steps - 1

For multi-horizon:

- valid_t_end = T - max_horizon_steps - 1

## Phase 3. Anchor policies

Implement:

### none

No post-calibration FMS input.

### calibration_end

anchor_index = C_steps - 1

### sparse_observed

anchor_interval_steps = round(anchor_interval_seconds / sampling_interval)

anchor_index = floor(t / anchor_interval_steps) * anchor_interval_steps

If anchor_index < C_steps:

anchor_index = C_steps - 1

Rules:

- anchor_index <= t
- anchor_index < target_index

### recent_start_observed

anchor_index = t - W_steps + 1

Mark this as upper_bound = true.

This setting is allowed in search, but must be reported as an upper-bound condition.

## Phase 4. Dynamic recent dilation

If recent_dilations = auto:

- W <= 10s: [1, 2, 4]
- 10s < W <= 20s: [1, 2, 4]
- 20s < W <= 45s: [1, 2, 4, 8]
- W > 45s: [1, 2, 4, 8, 16]

Approximate receptive field:

RF = 1 + 2 * (kernel_size - 1) * sum(dilations)

For kernel_size=3:

RF = 1 + 4 * sum(dilations)

Log RF in steps and seconds for every run.

## Phase 5. Mandatory sanity tests before full training

Before any full training, run or create tests for:

1. seconds-to-steps conversion
2. target shift correctness
3. calibration leakage
4. recent window leakage
5. anchor policy correctness
6. valid prediction count
7. dynamic dilation schedule
8. TCN forward shape
9. model forward shape
10. metrics/evaluation import check

If any sanity test fails, fix it before training.

## Phase 6. Hardware and budget inspection

Before running full training, inspect available resources.

Run if available:

- python --version
- nvidia-smi
- free -h
- df -h
- git status

Report:

- GPU model, if available
- CUDA availability from PyTorch
- estimated number of runs in the search
- chosen budget

If no GPU is available, reduce the search budget and train only a minimal subset.

## Phase 7. Experiment budget

Use bounded search. Do not run an unbounded AutoML search.

Default full-search budget:

Stage 1: coarse search
Maximum runs: 24
Seeds: [42]
Max epochs per run: 40
Early stopping patience: 6

Stage 2: refined search
Maximum runs: 8
Seeds: [42, 43, 44] if budget allows
Max epochs per run: 80
Early stopping patience: 10

Stage 3: final selected model
Train best validation configuration with seeds [42, 43, 44] if budget allows
Max epochs: 100
Early stopping patience: 12

If runtime is too high:

- reduce Stage 1 to 12 runs
- reduce Stage 2 to 4 runs
- use seed [42] only
- keep final test evaluation on selected model

Never silently expand the budget beyond this file.

## Phase 8. Stage 1 coarse search space

Run a validation-based coarse search over the following core settings.

Base model:

- model = lc_sa_tcnformer
- d_model = 64
- kernel_size = 3
- loss_type = smooth_l1
- loss_mode = level_only
- pooling = mean
- transformer_layers = 1
- transformer_heads = 4

Search dimensions:

Calibration seconds:

- 60
- 90
- 120

Recent window seconds:

- 10
- 30
- 60

Horizon seconds:

- 5
- 10
- 15

Anchor mode:

- none
- calibration_end
- sparse_observed

Static:

- no_static
- static

Because the full Cartesian product is too large, do not run all combinations.

Use a structured subset:

1. Default core:
   - C=90, W=30, H=5, anchor=calibration_end, no_static
   - C=90, W=30, H=10, anchor=calibration_end, no_static
   - C=90, W=30, H=15, anchor=calibration_end, no_static

2. Calibration comparison:
   - C=60, W=30, H=10, anchor=calibration_end, no_static
   - C=90, W=30, H=10, anchor=calibration_end, no_static
   - C=120, W=30, H=10, anchor=calibration_end, no_static

3. Recent window comparison:
   - C=90, W=10, H=10, anchor=calibration_end, no_static
   - C=90, W=30, H=10, anchor=calibration_end, no_static
   - C=90, W=60, H=10, anchor=calibration_end, no_static

4. Anchor comparison:
   - C=90, W=30, H=10, anchor=none, no_static
   - C=90, W=30, H=10, anchor=calibration_end, no_static
   - C=90, W=30, H=10, anchor=sparse_observed, anchor_interval=60, no_static

5. Static comparison:
   - C=90, W=30, H=10, anchor=calibration_end, no_static
   - C=90, W=30, H=10, anchor=calibration_end, static
   - C=90, W=30, H=10, anchor=sparse_observed, anchor_interval=60, static

6. Longer-horizon stress:
   - C=90, W=30, H=20, anchor=calibration_end, no_static
   - C=90, W=30, H=30, anchor=calibration_end, no_static

7. Upper-bound anchor, clearly marked:
   - C=90, W=30, H=10, anchor=recent_start_observed, no_static

This should be around 17-20 runs.

If budget allows, add:

- C=120, W=60, H=10, anchor=calibration_end, static
- C=120, W=30, H=15, anchor=sparse_observed, static
- C=90, W=60, H=15, anchor=sparse_observed, static

## Phase 9. Stage 2 refined search

After Stage 1, select the top 3 configurations by validation MAE.

Do not use test metrics.

For each top config, run small architecture refinements:

1. pooling:
   - mean
   - attention

2. transformer_layers:
   - 1
   - 2

3. d_model:
   - 64
   - 128 only if GPU memory allows

4. predict_delta_from_anchor:
   - false
   - true only for anchor modes other than none

Limit to maximum 8 refined configurations.

Selection metric:

Primary:
- validation MAE

Tie-breakers:
1. validation RMSE
2. common-window validation MAE, if available
3. horizon trend stability if multi-horizon is used
4. lower model complexity

## Phase 10. Optional multi-horizon experiment

If Stage 1/2 completes successfully and budget remains, run a small multi-horizon comparison.

Horizon set:

- [1, 2.5, 5, 10, 15, 30]

Configuration:

- use best C, W, anchor, static from validation search
- output_dim = len(horizon_set)
- max epochs = 60
- patience = 8
- seed = 42

Compare against single-horizon models only descriptively.

Do not use multi-horizon test metrics to replace the final selected single-horizon result unless it wins by validation metrics and was selected before test evaluation.

## Phase 11. Final selection and test evaluation

After Stage 2, select one final configuration by validation MAE.

Then evaluate on test set exactly once for that selected configuration.

If multiple seeds were run for the same final configuration:

- report mean ± std on validation
- report mean ± std on test if each seed was independently trained and selected config was fixed before test

Do not change the selected configuration after looking at test metrics.

## Phase 12. Metrics

Compute and save:

Validation:

- val_MAE
- val_RMSE
- val_R2
- val_sMAPE
- common_val_MAE if common-window is available
- common_val_RMSE if common-window is available

Test for selected model only:

- test_MAE
- test_RMSE
- test_R2
- test_sMAPE
- common_test_MAE
- common_test_RMSE

High-FMS metrics:

- high_fms_precision
- high_fms_recall
- high_fms_false_positive_rate

Use threshold:

- high_fms_threshold = 10.0

Also save:

- best_epoch
- train_loss
- val_loss
- training_time
- parameter_count
- recent RF steps/seconds

## Phase 13. Prediction CSVs

For each evaluated run, save prediction CSV with:

- run_name
- model_name
- split
- session_id
- participant_id
- current_index
- target_index
- current_time
- target_time
- session_length_steps
- session_duration
- calibration_seconds
- recent_window_seconds
- horizon_seconds
- calibration_steps
- recent_window_steps
- horizon_steps
- anchor_mode
- anchor_index
- anchor_time
- anchor_fms
- time_since_anchor
- is_upper_bound_anchor
- use_static
- predicted_fms
- target_fms
- absolute_error
- squared_error

For Stage 1, save validation prediction CSVs if not too large.

For final selected model, save both validation and test prediction CSVs.

## Phase 14. Plots

Generate plots for the final selected model:

1. predicted vs target scatter
2. residual histogram
3. FMS trajectory examples for several sessions
4. horizon degradation curve if horizon sweep was run
5. validation leaderboard bar plot
6. calibration/recent-window comparison plot if available

Save plots under:

runs/<run_name>/plots/

or a consistent existing plot directory.

## Phase 15. Leaderboard and report

Create:

- runs/lc_sa_tcnformer_full_search/leaderboard_val.csv
- runs/lc_sa_tcnformer_full_search/leaderboard_val.md
- runs/lc_sa_tcnformer_full_search/final_test_metrics.csv
- runs/lc_sa_tcnformer_full_search/final_report.md

Leaderboard columns:

- rank
- run_name
- calibration_seconds
- recent_window_seconds
- horizon_seconds
- anchor_mode
- anchor_interval_seconds
- use_static
- predict_delta_from_anchor
- d_model
- transformer_layers
- pooling
- recent_dilations
- recent_rf_seconds
- val_MAE
- val_RMSE
- val_R2
- val_sMAPE
- common_val_MAE
- common_val_RMSE
- best_epoch
- parameter_count
- checkpoint_path
- metrics_path
- prediction_csv_path

Final report must include:

1. What was implemented
2. Search space
3. Actual budget used
4. Validation leaderboard
5. Selected best configuration
6. Why it was selected
7. Final test metrics
8. Plots and paths
9. Interpretation:
   - Did long calibration help?
   - Did recent window length matter?
   - Did static features help?
   - Did anchor help?
   - How far into the future was prediction reliable?
10. Limitations:
   - head/motion-only
   - subjectivity of FMS
   - upper-bound anchor caveat
   - validation search size
   - no external dataset transfer yet
11. Reproducibility:
   - exact commands
   - seed
   - git status
   - hardware summary

## Phase 16. Sweep runner

If not already present, implement or update:

scripts/run_lc_sa_tcnformer_full_search.py

It should:

1. run sanity tests,
2. generate Stage 1 commands,
3. run Stage 1 unless dry_run is true,
4. aggregate validation metrics,
5. select top configurations,
6. generate Stage 2 commands,
7. run Stage 2,
8. select final configuration by validation MAE,
9. evaluate test set for final configuration,
10. generate plots and final report.

CLI options:

- data_dir
- split_file
- output_dir
- seed
- dry_run
- skip_existing
- max_stage1_runs
- max_stage2_runs
- max_epochs_stage1
- max_epochs_stage2
- max_epochs_final
- patience_stage1
- patience_stage2
- patience_final
- allow_test_eval
- no_test_eval
- device
- num_workers
- batch_size
- learning_rate
- weight_decay

Default:

- output_dir = runs/lc_sa_tcnformer_full_search
- allow_test_eval = true only after validation selection
- skip_existing = true

## Phase 17. Recommended command

After implementation and sanity tests, run:

python scripts/run_lc_sa_tcnformer_full_search.py \
  --data_dir ./DenseFMS/Dataset \
  --split_file ./artifacts/densefms_split_seed42.json \
  --output_dir ./runs/lc_sa_tcnformer_full_search \
  --seed 42 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --max_epochs_stage1 40 \
  --patience_stage1 6 \
  --max_epochs_stage2 80 \
  --patience_stage2 10 \
  --max_epochs_final 100 \
  --patience_final 12 \
  --skip_existing

If GPU memory is insufficient, reduce batch_size to 32.

If runtime is too long, use:

--max_stage1_runs 12
--max_stage2_runs 4
--max_epochs_stage1 25
--max_epochs_stage2 50

## Phase 18. Git hygiene

Before starting:

git status

After finishing:

git status

Do not commit.
Do not push.

Do not add to git:

- DenseFMS/Dataset
- runs/
- artifacts/
- checkpoints/
- *.pt
- *.pth
- *.ckpt
- large prediction CSV files

Update .gitignore if necessary.

## Final response

Report in Korean.

Include:

1. Modified/added files
2. Sanity test results
3. Hardware summary
4. Full-training budget actually used
5. Validation leaderboard summary
6. Selected best configuration
7. Final test metrics
8. Plot/report paths
9. Generated commands
10. git status summary
11. Remaining issues

Do not claim that validation-selected metrics are test metrics.
Do not hide failed runs.
If a run failed, report why and whether it was retried or skipped.

## Adaptive continuation rule

If Stage 1 and Stage 2 results are mediocre, follow:

docs/codex/ADAPTIVE_NEXT_STEP_POLICY.md

The adaptive stage is allowed to run bounded additional full-training experiments.

Rules:

- Do not use test metrics during adaptive diagnosis or adaptive experiment selection.
- Use validation metrics only.
- Maximum adaptive runs: 8 unless user explicitly approves more.
- If a data/windowing bug is suspected, stop model search and fix the bug first.
- If results remain mediocre after adaptive stage, stop experiments and write a limitation-focused report.
- Do not keep expanding the search indefinitely.