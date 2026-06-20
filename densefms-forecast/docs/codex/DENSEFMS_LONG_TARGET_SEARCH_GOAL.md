# DenseFMS long target search goal

FULL_TRAINING_ALLOWED = true

WALL_CLOCK_TARGET_HOURS = 8
WALL_CLOCK_SOFT_CAP_HOURS = 10
WALL_CLOCK_HARD_CAP_HOURS = 12

## Goal

Run a long, target-driven DenseFMS model search.

The stretch goal is:

- validation MAE <= 1.0 for short-horizon forecasting if achievable

If MAE <= 1.0 is not achievable, find the best validation-selected model in the low-1 MAE range.

The search must not optimize only H=1 and claim general forecasting success. Report results separately for:

1. H = 1s
2. H = 2.5s
3. H = 5s
4. H = 10s
5. H = 15s
6. H = 20s
7. H = 30s

Use validation metrics for all model selection.
Evaluate test only after validation-based final selection.

Do not commit or push.

## Context from previous report

Previous best final selected model:

- model: lc_sa_tcnformer
- C = 120s
- W = 30s
- H = 1s
- anchor_mode = sparse_observed
- anchor_interval = 60s
- static = true
- predict_delta_from_anchor = true
- validation MAE = 1.7570
- test MAE = 1.7076

Important caveat:

This was a 1-second short-horizon model. It must not be presented as long-horizon forecasting performance.

Previous 15s deployment-style candidate:

- validation MAE around 2.10

Previous diagnosis:

- sparse FMS anchor was critical
- no-anchor models were much weaker
- delta prediction from anchor was useful
- static features may help
- recent/calibration branch contributions were not fully isolated
- multi-horizon was not fully exploited
- model family search was still limited

## Primary objective

Search for models that can reduce DenseFMS future FMS forecasting MAE toward 1.0 or low-1 range by exploring:

1. stronger anchor-delta formulations
2. simpler feature/statistical models
3. recurrent recent-motion models
4. gated fusion models
5. multi-horizon models
6. classical ML baselines
7. longer adaptive refinement

## Non-negotiable rules

1. Do not use test metrics for model selection.
2. Do not evaluate test until final validation-selected models are fixed.
3. Do not select recent_start_observed as a deployment model.
4. recent_start_observed is upper-bound only.
5. Do not claim H=1 performance as long-horizon performance.
6. Keep best-score and deployment-realistic tracks separate.
7. If leakage/windowing bug is suspected, stop training and fix it first.
8. Do not run unlimited search beyond WALL_CLOCK_HARD_CAP_HOURS.
9. Save progress after every run.
10. Write periodic progress summaries.

## Tracks

### Track A. Best-score track

Goal:

Find the lowest validation MAE possible.

Allowed:

- sparse_observed anchor
- anchor intervals 10s, 20s, 30s, 60s
- static features
- delta prediction
- multi-horizon training
- stronger model families

Must report user burden if frequent anchors are used.

### Track B. Deployment-realistic track

Goal:

Find best practical model.

Preferred:

- H = 5s, 10s, or 15s
- anchor_mode = sparse_observed
- anchor_interval_seconds >= 60
- static allowed
- delta prediction allowed

Do not use recent_start_observed as deployment model.

### Track C. Upper-bound track

Goal:

Estimate performance ceiling when dense FMS state is available.

Allowed:

- recent_start_observed

Must label upper-bound clearly.

## Target tiers

### H = 1s

Stretch:

- validation MAE <= 1.0

Strong:

- validation MAE <= 1.25

Acceptable:

- validation MAE <= 1.50

### H = 2.5s and H = 5s

Stretch:

- validation MAE <= 1.25

Strong:

- validation MAE <= 1.50

Acceptable:

- validation MAE <= 1.75

### H = 10s and H = 15s

Stretch:

- validation MAE <= 1.50

Strong:

- validation MAE <= 1.75

Acceptable:

- validation MAE <= 2.00

### H = 20s and H = 30s

Treat as stress tests.

Do not expect MAE <= 1 unless validation evidence supports it.

## Required files to read first

Before doing anything, read:

- AGENTS.md
- docs/codex/LC_SA_TCNFORMER_WORK_REPORT_KO.md
- runs/lc_sa_tcnformer_full_search/leaderboard_val.csv, if exists
- runs/lc_sa_tcnformer_full_search/final_report.md, if exists
- runs/lc_sa_tcnformer_full_search/final_test_metrics.csv, if exists
- src/densefms_forecast/data.py
- src/densefms_forecast/model.py
- src/densefms_forecast/train.py
- src/densefms_forecast/evaluate.py
- src/densefms_forecast/losses.py
- scripts/run_lc_sa_tcnformer_full_search.py

Summarize previous results before starting.

## Stage 0. Sanity and previous result reconstruction

Run existing sanity tests before training.

Required checks:

- seconds-to-steps conversion
- target shift
- calibration leakage
- recent motion future leakage
- anchor policy
- sparse anchor fallback
- model forward shape
- prediction_start correctness

Also reconstruct or compute validation baselines:

1. global train mean
2. calibration_end anchor
3. sparse_observed anchor
4. recent_start_observed upper bound
5. ridge/linear feature baseline if available

If sanity tests fail, fix them before training.

## Stage 1. Controlled comparable sweep

Purpose:

Establish controlled comparable curves around previous best.

Base config:

- C = 120
- W = 30
- anchor_mode = sparse_observed
- static = true
- predict_delta_from_anchor = true
- loss_type = smooth_l1
- loss_mode = level_only

Sweep:

Horizon:

- 1
- 2.5
- 5
- 10
- 15
- 20
- 30

Anchor interval:

- 10
- 30
- 60
- 90

Model:

- lc_sa_tcnformer

This is up to 28 runs.

If runtime is high, prioritize:

1. H = 1, 2.5, 5, 10, 15
2. anchor intervals = 30, 60
3. then add 10 and 90

Required output:

- horizon vs validation MAE
- anchor interval vs validation MAE
- best-score track summary
- deployment-realistic track summary

## Stage 2. New model family implementation and search

Implement and search the following model families.

### Family A. AnchorDeltaMLP

Purpose:

Test whether a simple state-transition model beats sequence encoders.

Inputs:

- anchor_fms
- time_since_anchor
- horizon_seconds
- static features
- calibration summary features
- recent motion statistics

Calibration summary features:

- calibration_start_fms
- calibration_end_fms
- calibration_mean_fms
- calibration_std_fms
- calibration_max_fms
- calibration_min_fms
- calibration_delta_fms
- calibration_slope_fms

Recent motion statistics:

- per-channel mean
- per-channel std
- per-channel min
- per-channel max
- acceleration magnitude mean/std/max
- angular velocity magnitude mean/std/max
- jerk magnitude mean/std if easy

Architecture candidates:

1. MLP 128 -> 128 -> 64 -> 1
2. MLP 256 -> 128 -> 64 -> 1
3. residual MLP if easy

Default output:

- delta from anchor

Search:

- H = 1, 2.5, 5, 10, 15
- anchor_interval = 10, 30, 60, 90
- static = true
- max 20 runs

### Family B. AnchorDeltaGRU

Purpose:

Model recent motion as a recurrent state-transition signal.

Inputs:

- recent motion sequence
- anchor input
- static features
- horizon embedding
- calibration summary features

Architecture:

- GRU recent encoder
- calibration summary MLP
- anchor MLP
- static MLP
- fusion MLP
- output delta from anchor

Search:

- hidden_size = 64, 128
- num_layers = 1, 2
- H = 1, 5, 10, 15
- anchor_interval = 30, 60
- max 16 runs

### Family C. RecentTCN + SummaryCalib

Purpose:

Keep recent TCN but replace complex calibration TCN-Transformer with summary features.

Inputs:

- recent motion TCN
- calibration summary MLP
- anchor
- static
- horizon

Search:

- H = 1, 2.5, 5, 10, 15
- anchor_interval = 30, 60
- recent_window = 10, 30, 60
- max 18 runs

### Family D. Gated Fusion model

Purpose:

Learn branch reliance and reduce over-dependence on anchor.

Inputs:

- z_calib
- z_recent
- z_anchor
- z_static
- z_horizon

Add learned gates:

- gate_calib
- gate_recent
- gate_anchor
- gate_static

Also allow branch dropout.

Search:

- H = 1, 5, 10, 15
- anchor_interval = 30, 60
- branch_dropout = 0.0, 0.1, 0.2
- max 16 runs

### Family E. MultiHorizon model

Purpose:

Use dense FMS labels more fully.

Horizon set:

- [1, 2.5, 5, 10, 15, 30]

Candidates:

1. LC-SA-TCNFormer multi-horizon
2. RecentTCN + SummaryCalib multi-horizon
3. AnchorDeltaMLP multi-horizon

Loss:

- mean SmoothL1 over horizons
- optional horizon weights:
  - equal
  - short-horizon emphasis
  - practical-horizon emphasis

Max 8 runs.

### Family F. Classical feature baselines

If sklearn is available, run:

- Ridge
- ElasticNet
- RandomForestRegressor
- HistGradientBoostingRegressor
- GradientBoostingRegressor

Inputs:

- same engineered features as AnchorDeltaMLP

Use train/val only for selection.

If these beat neural models, report honestly.

Do not install heavy packages unless already available.

## Stage 2 budget

This is a long search. The target wall-clock is at least 7 hours if runs are still making progress.

Maximum Stage 2 neural runs:

- 70

Maximum classical baseline runs:

- 20, if cheap

If wall-clock exceeds 7 hours:

- continue only if the latest 10 runs produced at least one meaningful validation improvement
- otherwise move to Stage 3 refinement

Meaningful improvement:

- validation MAE improves by >= 0.03 for H=1 or H=2.5
- validation MAE improves by >= 0.05 for H=5, H=10, H=15
- a new deployment-realistic best is found

## Stage 3. Progressive refinement

After Stage 2, select top candidates by validation MAE.

Keep separate pools:

1. best H=1
2. best H=2.5
3. best H=5
4. best H=10
5. best H=15
6. best deployment-realistic model
7. best multi-horizon model
8. best classical baseline

Refine top 6 to 10 candidates.

Allowed refinements:

- learning_rate: 3e-4, 1e-3, 3e-3
- weight_decay: 1e-5, 1e-4, 1e-3
- dropout: 0.0, 0.1, 0.2, 0.3
- hidden size: 64, 128, 256 depending on model
- loss_type: smooth_l1, mae/l1 if implemented, mse
- predict_delta_from_anchor: true/false
- anchor interval: 10, 30, 60, 90
- branch_dropout: 0.0, 0.1, 0.2
- anchor_dropout: 0.0, 0.1, 0.2
- recent_window: 10, 30, 60

Maximum Stage 3 runs:

- 40

Use validation only.

Stop refinement for a horizon if:

- H=1 reaches validation MAE <= 1.0
- H=2.5/H=5 reaches validation MAE <= 1.25
- H=10/H=15 reaches validation MAE <= 1.50

If target is reached, run stability confirmation instead of more broad search.

## Stage 4. Multi-seed confirmation

Run multiple seeds for selected candidates.

Candidates:

1. best H=1 model
2. best H=2.5 model
3. best H=5 deployment candidate
4. best H=10 or H=15 deployment candidate
5. best multi-horizon model
6. best classical baseline if applicable

Seeds:

- 42
- 43
- 44

If runtime is high:

- use seeds 42 and 43 only

Report:

- validation MAE mean
- validation MAE std
- validation RMSE mean/std
- training stability
- best epoch distribution

Do not evaluate test during this stage.

## Stage 5. Final validation-based selection

Select final models using validation only.

Evaluate on test only these final models:

1. best short-horizon model
   - H=1 or H=2.5

2. best deployment-realistic model
   - prefer H=5, H=10, or H=15
   - prefer anchor_interval >= 60 if performance is close

3. best multi-horizon model, if competitive

4. optional upper-bound model
   - recent_start_observed
   - clearly labeled upper-bound

Do not change model choices after seeing test metrics.

## Search scheduling rules

This task may run longer than 7 hours.

Use this scheduling policy:

### First 1 hour

- implement missing model families
- run sanity tests
- run short smoke tests
- reconstruct previous leaderboard
- start Stage 1

### Hours 1 to 4

- complete Stage 1
- run broad Stage 2 model family search
- prioritize AnchorDeltaMLP and classical baselines first
- then GRU, RecentTCN+Summary, Gated Fusion, MultiHorizon

### Hours 4 to 7

- run Stage 3 refinement
- focus on best-performing model families
- focus on H=1, H=2.5, H=5, H=10, H=15
- avoid spending time on bad families

### After 7 hours

Continue only if progress is meaningful.

Continue if:

- best MAE is still improving
- low-1 target is within reach
- deployment-realistic model improved recently
- multi-horizon result is promising but incomplete

Stop if:

- no meaningful validation improvement in last 10 runs
- repeated training failures
- GPU/CPU limits make progress impractical
- likely data/windowing bug appears

### Hard cap

Stop by WALL_CLOCK_HARD_CAP_HOURS unless final evaluation/report is already in progress.

## Checkpointing and progress reporting

Create progress logs:

- runs/densefms_long_target_search/progress_log.md
- runs/densefms_long_target_search/progress_log.jsonl

After each run, log:

- timestamp
- elapsed time
- run name
- model family
- horizon
- anchor mode
- anchor interval
- validation MAE
- validation RMSE
- validation R2
- best so far
- whether target tier was reached
- next planned action

Every 60 minutes, write a checkpoint summary:

- current best H=1
- current best H=2.5
- current best H=5
- current best H=10
- current best H=15
- current best deployment-realistic
- current best multi-horizon
- remaining budget
- next step decision

## Training defaults

Default for broad search:

- max_epochs = 60
- patience = 8
- batch_size = 64
- learning_rate = 1e-3
- weight_decay = 1e-4

Refinement:

- max_epochs = 90
- patience = 12

Final multi-seed:

- max_epochs = 120
- patience = 15

If GPU memory is insufficient:

- batch_size = 32

If training is too slow:

- reduce max_epochs for bad families
- keep early stopping
- prioritize cheap baselines and MLP models first

## Required runner

Create or update:

scripts/run_densefms_long_target_search.py

The runner should:

1. run sanity tests
2. reconstruct previous results
3. run Stage 1 controlled sweep
4. run Stage 2 model family search
5. run Stage 3 progressive refinement
6. run Stage 4 multi-seed confirmation
7. select final models by validation
8. evaluate test only for selected final models
9. generate plots
10. write final report

Required CLI:

- --data_dir
- --split_file
- --output_dir
- --seed
- --device
- --batch_size
- --learning_rate
- --weight_decay
- --max_epochs
- --patience
- --dry_run
- --skip_existing
- --wall_clock_target_hours
- --wall_clock_soft_cap_hours
- --wall_clock_hard_cap_hours
- --reduced_budget
- --aggressive_budget
- --allow_final_test_eval
- --no_test_eval

Default output_dir:

runs/densefms_long_target_search

## Required output files

Create:

- runs/densefms_long_target_search/progress_log.md
- runs/densefms_long_target_search/progress_log.jsonl
- runs/densefms_long_target_search/leaderboard_val.csv
- runs/densefms_long_target_search/leaderboard_val.md
- runs/densefms_long_target_search/horizon_sweep.csv
- runs/densefms_long_target_search/anchor_interval_sweep.csv
- runs/densefms_long_target_search/model_family_comparison.csv
- runs/densefms_long_target_search/refinement_results.csv
- runs/densefms_long_target_search/multiseed_results.csv
- runs/densefms_long_target_search/final_selected_models.json
- runs/densefms_long_target_search/final_test_metrics.csv
- runs/densefms_long_target_search/final_long_target_search_report.md

Plots:

- plots/horizon_mae_curve.png
- plots/anchor_interval_curve.png
- plots/model_family_comparison.png
- plots/best_by_horizon.png
- plots/progress_best_mae_over_time.png
- plots/val_predicted_vs_target_best_h1.png
- plots/val_predicted_vs_target_best_deployment.png
- plots/test_predicted_vs_target_selected.png
- plots/residual_histogram_selected.png
- plots/multi_horizon_curve.png, if available

## Final report structure

Write:

runs/densefms_long_target_search/final_long_target_search_report.md

The report must include:

1. Previous result recap
2. Target definition
3. Search budget actually used
4. Hardware summary
5. Sanity test results
6. Baseline results
7. Controlled horizon/anchor sweep
8. Model family comparison
9. Progressive refinement results
10. Multi-seed confirmation
11. Best validation models by horizon
12. Whether MAE <= 1.0 was reached
13. Whether low-1 MAE was reached
14. Best short-horizon model
15. Best deployment-realistic model
16. Best multi-horizon model
17. Final test metrics
18. Interpretation:
    - Did simple AnchorDeltaMLP beat complex models?
    - Did GRU help?
    - Did recent motion help?
    - Did calibration help?
    - Did static help?
    - Did multi-horizon help?
    - Is performance anchor-dominated?
19. Limitations:
    - head/motion-only
    - FMS subjectivity
    - anchor frequency burden
    - H=1 vs long-horizon distinction
    - single-dataset validation
20. Next recommended research step

## Recommended commands

Dry run:

```bash
python scripts/run_densefms_long_target_search.py \
  --data_dir ./DenseFMS/Dataset \
  --split_file ./artifacts/densefms_split_seed42.json \
  --output_dir ./runs/densefms_long_target_search \
  --seed 42 \
  --wall_clock_target_hours 7 \
  --wall_clock_soft_cap_hours 10 \
  --wall_clock_hard_cap_hours 12 \
  --dry_run