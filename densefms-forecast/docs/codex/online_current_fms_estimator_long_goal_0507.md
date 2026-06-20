# Online Current FMS Estimator Long Goal - MAE + Trend Tracking - Autonomous Revision - 2026-05-07

FULL_TRAINING_ALLOWED = true

## Mission Summary

Build and improve a deployment-realistic online system that estimates the user's current FMS from head-motion data after an initial calibration phase.

The product goal has two acceptable validation-success modes.

### Success Mode A: Level-Accurate Current FMS

The model predicts current FMS accurately on the original 0-20 FMS scale.

Targets:

- First success: beat validation MAE `2.2955` under the same fixed split.
- Strong success: validation MAE `<= 2.20`.
- Ambitious success: validation MAE `<= 2.00`.

### Success Mode B: Trend-Tracking Current FMS

Even if the absolute scale, offset, or amplitude is somewhat imperfect, the predicted trajectory should visibly follow the user's current-FMS rises and drops.

This means:

- when true FMS rises, prediction should tend to rise;
- when true FMS drops, prediction should tend to drop;
- prediction should not collapse to a flat average line;
- prediction should not react with an unacceptable delay;
- prediction should not look good only because of cherry-picked plots.

A trend-first candidate may be selected even if its raw MAE is slightly worse than the best MAE model, but only if validation-only movement metrics and fixed validation trajectory plots show clearly superior rise/drop tracking.

Default guardrail for trend-first selection:

- raw validation MAE should generally be `<= 2.45`;
- trend metrics must improve over the comparable baseline;
- fixed plot review must show less flatness/lag failure than the baseline;
- the reason for selection must be written in `validation_selection_lock.md` before test.

The final selected candidate may be chosen by either:

1. best validation MAE, or
2. clearly superior validation trend-following evidence with acceptable MAE.

Test results must never be used to choose between these routes.

## Product Requirement

During deployment:

- During calibration, the system may receive both head motion and FMS.
- After calibration, the system must receive no additional user-entered FMS.
- At online time `t`, the system may use head motion up to and including current index `t`.
- In Python slice language, if `t` is a zero-based current index, the allowed head-motion prefix is `head[:t+1]`, not `head[:t]`.
- The primary output is current-time FMS.
- The primary product is not future FMS forecasting.
- The primary product is not a warning label or rapid-rise alarm.

Risk, rapid-rise, high-FMS, or trend labels may exist only as auxiliary diagnostics or auxiliary training heads. They must not replace current-FMS prediction as the main output.

## Fresh-Context Bootstrap

This document is intended to be sufficient after context reset. Start from this file, then inspect the repository before editing.

Workspace:

- repo root: `/mnt/c/users/rio/documents/github/codex`
- dataset path: `DenseFMS/Dataset`
- main package: `src/densefms_forecast`
- current date when this goal was written: `2026-05-07`
- user/reporting language: Korean

Python/CUDA environment:

- preferred Python executable: `/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe`
- this environment has PyTorch with CUDA.
- observed GPU: `NVIDIA GeForce RTX 4070`
- observed free VRAM before current-FMS runs: about 11.6 GB free out of 12.9 GB.
- Miniforge env did not have torch; avoid it unless rechecked.

Important local rules:

- Do not commit.
- Do not push.
- Do not revert user or pre-existing changes.
- Do not add datasets, checkpoints, prediction CSVs, runs, or artifacts to git.
- Full training is allowed for this goal because this file contains `FULL_TRAINING_ALLOWED = true`.
- Model selection must use validation evidence only.
- Test is final-report-only and may be run once after validation selection is locked.
- If no candidate is worth locking, stop as validation-only exploration and do not run test.

Repository files to inspect before implementation:

- dataset/windowing: `src/densefms_forecast/data.py`
- models: `src/densefms_forecast/model.py`
- training: `src/densefms_forecast/train.py`
- evaluation: `src/densefms_forecast/evaluate.py`
- losses: `src/densefms_forecast/losses.py`
- metrics/utilities: `src/densefms_forecast/utils.py`
- plotting: `src/densefms_forecast/plot_compare.py`, `scripts/plot_online_current_comparison.py`
- sanity checks: `scripts/run_densefms_sanity_tests.py`
- existing current-FMS configs: `configs/online_fms_current_tracker_*.yaml`
- previous risk/current goal file: `docs/codex/online_fms_risk_tracking_goal_0507.md`

Useful commands:

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m py_compile src/densefms_forecast/model.py src/densefms_forecast/train.py src/densefms_forecast/evaluate.py scripts/run_densefms_sanity_tests.py
```

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/run_densefms_sanity_tests.py
```

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config <CONFIG> --model online_fms_risk_tracker --run_name <RUN_NAME> --no_test_eval --split_file <FIXED_SPLIT_JSON_IF_AVAILABLE>
```

Plot the first 3 current-FMS validation runs:

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/plot_online_current_comparison.py --run_dirs runs/online_fms_current_tracking_0507/current_fms_base_basic runs/online_fms_current_tracking_0507/current_fms_dual_medium runs/online_fms_current_tracking_0507/current_fms_encoder_heavy --labels base_basic dual_medium encoder_heavy --out_dir runs/online_fms_current_tracking_0507/comparison_3runs --split val --primary_label dual_medium --trajectory_count 6
```

## Split Reproducibility Requirement

All new validation/test comparisons must use the same split as the reported baselines whenever that split is available.

Rules:

- Before running new experiments, locate the split definition used by the existing baseline runs.
- Prefer an existing `split.json`, `splits.json`, run-specific split artifact, or explicit split file recorded in baseline configs/logs.
- If the training CLI supports `--split_file`, use it for all new runs.
- If no explicit split file exists, inspect baseline configs and logs to identify the exact split seed, split mode, fold index, and subject/sample split policy.
- Write the resolved split policy into the leaderboard and checkpoint notes.
- Do not directly compare a new run against validation MAE `2.2955` unless it uses the same validation/test split or the difference is explicitly labeled as not directly comparable.
- If a new split is unavoidable, create a new baseline under that exact split before comparing candidates.
- Final `validation_selection_lock.md` must state the split file or exact split-generation parameters used.

The validation MAE reference `2.2955` is meaningful only under a comparable split and leakage-safe online input contract.

## Execution Guardrails

Use staged execution, but do not stop after a small initial batch. This goal is intended to run autonomously while the user is away.

### Phase A: before any new full training

Complete these first:

1. Inspect repo structure.
2. Run `git status --short`.
3. Do not revert unrelated changes.
4. Resolve the fixed split policy or split file.
5. Run py_compile/import checks.
6. Run sanity tests.
7. Rebuild or parse the existing validation leaderboard from current metrics files.
8. Inspect existing validation trajectory plots from `comparison_3runs`.
9. Write a short progress note in Korean summarizing:
   - baseline state,
   - split policy,
   - whether sanity checks passed,
   - first concrete modeling hypothesis.

### First exploration stage

Start conservatively, then continue autonomously.

- First run one smoke-training candidate to verify the new scaffold.
- After the smoke run passes, launch a small diverse initial batch.
- The initial batch should cover multiple distinct hypotheses, not tiny variants of the same config.
- 3-5 candidates is a reasonable initial diversity target, but it is not a hard limit.
- After the initial batch, continue adaptive validation-only search while the user is away.
- Do not run test during this stage.

After every serious batch:

- update the validation leaderboard;
- compute trend-following metrics;
- generate or update fixed validation plots;
- write a checkpoint note in Korean;
- identify the next concrete hypothesis.

### Autonomous long-run mode

This goal is intended to continue while the user is away.

Codex should keep exploring validation-only candidates until one of these stop conditions occurs:

- a candidate achieves validation MAE `<= 2.00` with no leakage concern;
- a candidate clearly satisfies the trend-first route with acceptable MAE and strong plots;
- repeated batches show no meaningful improvement and a failure-mode report has been written;
- compute becomes unstable due to OOM, repeated crashes, or corrupted outputs;
- sanity/leakage checks fail;
- the run reaches a practical local-resource limit.

Do not stop only because a small number of runs has completed.

If interrupted, the user should be able to inspect:

- latest leaderboard;
- latest checkpoint note;
- list of completed/running/failed candidates;
- best MAE-first candidate;
- best trend-first candidate;
- plots for the best candidates.

### Expansion rule

Continue adaptive search only when the next batch has a concrete hypothesis, such as:

- current models are too flat, so add delta/slope loss;
- current models lag, so reduce smoothing or improve recent-context encoding;
- current models have correct shape but wrong scale, so test train-only affine calibration or output decomposition;
- current models overreact, so adjust loss, smoothing, or context window;
- current models miss long trends, so add multi-scale causal context.

Do not run a blind grid.

## Existing Implementation Snapshot

As of this document, the codebase already has an online current/risk path. It may be reused as a baseline or replaced.

Implemented task mode:

- `task.mode = online_current_risk`
- current target: `FMS[t]`
- optional rapid-rise labels remain available but are not the main objective for this goal.

Implemented model:

- class: `OnlineFMSRiskTracker` in `src/densefms_forecast/model.py`
- builder names: `online_fms_risk_tracker`, `online_risk_tracker`, `online_current_risk`
- uses calibration head motion + calibration FMS.
- after calibration, it uses post-calibration head motion only.
- supports `current_head_mode = basic`.
- supports `current_head_mode = dual_delta_gate`.
- supports `motion_feature_mode = none | norm | norm_delta | norm_delta_energy`.
- supports `motion_stats_branch`.
- supports `stream_context_mode = gru | gru_multiscale | gru_tcn | gru_tcn_multiscale`.
- supports `state_feedback_mode = none | predicted_current`.
- supports optional causal motion encoder stem:
  - `motion_encoder_context = linear | tcn`
  - `motion_encoder_layers = int`
- supports optional temporal risk context:
  - `risk_temporal_context = none | tcn`
  - `risk_temporal_layers = int`

Implemented training/eval support:

- `compute_online_current_risk_targets`
- `compute_online_current_risk_loss`
- `collect_online_current_risk_predictions`
- validation selection via:
  - `training.selection_metric`
  - `training.selection_mode`
- test skipping via:
  - CLI `--no_test_eval`
  - config `evaluation.no_test_eval`
- current-FMS-focused runs should set:
  - `selection_metric: mae`
  - `selection_mode: min`

Important CLI options already added:

- `--task_mode online_current_risk`
- `--rise_horizon_seconds`
- `--rise_thresholds`
- `--ordinal_bins`
- `--fms_combine_weight_ordinal`
- `--current_head_mode`
- `--current_delta_scale`
- `--motion_encoder_context`
- `--motion_encoder_layers`
- `--risk_temporal_context`
- `--risk_temporal_layers`
- `--selection_metric`
- `--selection_mode`
- `--current_reg_aux_weight`
- `--ordinal_loss_weight`
- `--risk_loss_weight`
- `--smoothness_weight`
- `--motion_stats_branch`
- `--rollout_mode`

Existing sanity coverage includes online-current checks for:

- target shift and rapid-rise label alignment
- calibration-only FMS input
- recent/future motion leakage
- online model forward shapes
- loss smoke check
- rapid-rise-only final warning behavior
- causal motion encoder stem reporting

The existing implementation is not sacred. If a cleaner current-FMS-only system requires a new class, new dataset, or new training script, create it. Keep the old baseline path runnable.

## I/O and Deployment Contract

Default timing:

- `sampling_interval = 0.5`
- `calibration_seconds = 90.0`
- `recent_window_seconds = 10.0` as a starting point, but this may be searched.
- Prediction begins after calibration unless a candidate explicitly defines a later warm-up period.

Required inputs:

- `head`: dense head-motion sequence with 6 raw channels, shape `[B, T, 6]`.
- `fms_calibration`: FMS values for the first `calibration_steps` only, shape `[B, C]`.
- `lengths`: valid sequence length per session, shape `[B]`.
- optional `static`: user/session static features such as age, gender, MSSQ.

Static features are allowed only if train-only preprocessing and ablation reporting are maintained.

Allowed derived inputs:

- causal motion statistics computed from head motion at or before the current index;
- causal deltas;
- velocity/energy-like features;
- rolling statistics;
- causal frequency summaries;
- learned motion embeddings;
- calibration summaries derived only from calibration head motion and calibration FMS.

Forbidden inputs:

- FMS after `calibration_steps`;
- target FMS at the current prediction index as model input;
- head motion after current index `t`;
- validation/test labels or statistics in train preprocessing;
- any feature computed using future labels, future motion, or test-set model selection feedback.

Required outputs:

- `current_fms`: predicted current FMS for each valid online index, denormalized to the 0-20 FMS scale for metrics and CSVs.
- `current_fms_norm`: normalized prediction on the training scale if used internally.
- `mask`: valid prediction mask.
- `current_index` and `current_time` in prediction CSVs.

Optional diagnostics:

- uncertainty;
- ordinal probabilities;
- latent norm;
- predicted delta;
- calibration-end baseline;
- dynamic residual;
- smoothing state;
- rapid-rise risk;
- high-FMS diagnostic flags;
- trend labels.

Optional diagnostics must not replace the primary current-FMS objective.

## Target Definition

For each valid current index `t`:

- target = `FMS[t]`

Model input may use:

- `head[:t+1]`, meaning motion up to and including current index `t`;
- `head` windows ending at `t`, inclusive;
- `FMS[0:calibration_steps]` only;
- optional static features.

Model input must not use:

- `FMS[t]`;
- `FMS[calibration_steps:]`;
- `head[t+1:]`.

Primary metric:

- validation MAE on current FMS, 0-20 scale.

Primary selection remains validation-only. The default numerical selection metric is validation `mae` minimized, but final candidate selection may choose either the best-MAE route or the trend-first route if the validation evidence justifies it.

## Required Metrics

Every serious validation candidate should report both level metrics and trend-following metrics.

### Level metrics

Report at least:

- validation MAE;
- validation RMSE;
- R2 / explained variance if implemented;
- error by FMS range:
  - low,
  - mid,
  - high;
- error by session time bucket;
- error during rise/drop/plateau segments;
- prediction variance;
- calibration bias.

### Trend-following metrics

Add validation-only movement metrics. These are important for this goal.

Recommended metrics:

1. Per-session correlation

   - Pearson correlation between predicted FMS and true FMS.
   - Spearman correlation if easy to add.
   - Report mean, median, and valid session count.
   - Handle constant predictions or constant targets safely.

2. Bias-removed MAE

   For each session:

   - subtract the session mean from the prediction;
   - subtract the session mean from the target;
   - compute MAE on the centered trajectories.

   This measures whether the shape is right even when absolute baseline is shifted.

3. Scale/bias-normalized shape error

   Optional but useful:

   - compare z-scored prediction and z-scored target per session;
   - report shape MAE or RMSE.
   - skip sessions with near-zero target variance or handle them separately.

4. Train-only affine-calibrated MAE

   Fit a simple global correction on the training split only:

   - `y_cal = a * y_pred + b`

   Then evaluate the corrected predictions on validation.

   Implementation requirement:

   - Implement or reuse a train prediction collection path.
   - Collect training-split predictions and training labels using the selected checkpoint.
   - Fit `a` and `b` using training predictions and training labels only.
   - Apply the frozen correction to validation predictions.

   Rules:

   - Do not fit `a` or `b` on validation labels.
   - Do not fit `a` or `b` on test labels.
   - Report both raw validation MAE and train-affine-corrected validation MAE.
   - If the affine correction is part of the final deployment candidate, include it in the validation lock.

   This helps identify candidates that follow movement but have wrong scale or offset.

5. Multi-horizon delta MAE

   For horizons:

   - 5 seconds,
   - 10 seconds,
   - 20 seconds,
   - 30 seconds.

   Compare:

   - `pred[t] - pred[t-h]`
   - against `target[t] - target[t-h]`.

   Report delta MAE for each horizon.

6. Multi-horizon delta correlation

   For the same horizons, compute correlation between predicted delta and true delta.

7. Direction accuracy

   For each horizon, compute sign agreement between predicted and true delta.

   Ignore near-flat true deltas by default:

   - ignore if `abs(delta_true) < 0.3`.

   Report direction accuracy for:

   - all non-flat deltas;
   - rises;
   - drops.

8. Rise/drop F1

   For each horizon, classify movement:

   - rise if true delta `>= +1.0` FMS;
   - drop if true delta `<= -1.0` FMS;
   - flat otherwise.

   Report F1 or balanced accuracy for rise/drop detection from predicted deltas.

9. Lag diagnostic

   Compare predicted slope and true slope within a small lag window.

   Suggested lag window:

   - ±5 seconds,
   - optionally ±10 seconds.

   Report whether the best alignment requires positive lag. A model that follows the target but only after a long delay is less useful.

10. Flatness / dynamic-range diagnostic

   Report:

   - prediction standard deviation vs target standard deviation;
   - per-session predicted range vs true range;
   - percentage of sessions where prediction range is less than 25% of true range.

A candidate that has slightly worse MAE but clearly better trend metrics and plots should not be discarded automatically.

## Selection Logic

Use validation evidence only.

### MAE-first route

A candidate can be selected through the MAE-first route if:

- it has the best validation MAE among serious candidates, or
- it beats the `2.2955` reference under the same fixed split and has stable trajectory plots.

Priority levels:

- ambitious: validation MAE `<= 2.00`;
- strong: validation MAE `<= 2.20`;
- useful improvement: validation MAE `< 2.2955` under the same split.

### Trend-first route

A candidate can be selected through the trend-first route only if it satisfies at least 3 of the following validation-only conditions:

1. raw validation MAE `<= 2.45`;
2. bias-removed MAE improves over the comparable baseline;
3. delta correlation improves over the comparable baseline for at least one major horizon, preferably 10s or 20s;
4. direction accuracy improves over the comparable baseline for non-flat segments;
5. rise/drop F1 or balanced rise/drop accuracy improves over the comparable baseline;
6. flatness diagnostic improves, such as predicted range or standard deviation being closer to target dynamics;
7. fixed plot set shows fewer flatness or lag failures than the comparable baseline.

Additionally:

- it must visibly follow rises and drops better than the MAE baseline;
- it must avoid degenerate flat prediction;
- if raw MAE is worse than `2.45`, the candidate should not be selected unless the validation evidence is exceptional and the reason is clearly scale/bias mismatch rather than wrong movement.

### Tie-breakers

If multiple candidates are plausible, use these tie-breakers:

1. validation MAE;
2. validation RMSE;
3. bias-removed MAE;
4. delta correlation and direction accuracy;
5. high-FMS-range MAE;
6. fixed trajectory plot review for lag/smoothing;
7. simpler, more stable deployment model.

Test metrics must not be used as tie-breakers.

## Validation Trajectory Plot Protocol

Do not cherry-pick only visually good sessions.

For every serious candidate, generate a fixed validation plot set.

The fixed plot set should include:

- the same fixed validation sessions across all candidates, if session IDs are stable;
- sessions with largest true target variance;
- sessions with largest true rise;
- sessions with largest true drop;
- worst-error sessions by MAE;
- representative low/mid/high FMS sessions;
- at least a few random or predeclared validation sessions.

Each plot should show:

- true current FMS;
- raw predicted current FMS;
- optionally train-only affine-corrected prediction;
- calibration boundary;
- post-calibration online region;
- optionally prediction error;
- optionally true and predicted delta traces.

Plot review should explicitly label failure modes:

- flat prediction;
- delayed response;
- correct direction but wrong amplitude;
- correct amplitude but wrong baseline;
- overreaction/noisy prediction;
- underreaction/oversmoothing;
- good local motion but bad long-term level;
- good level but bad movement;
- high-FMS underprediction;
- low-FMS overprediction.

The plot protocol must be stable enough that candidates can be compared fairly.

## Current Validation Baselines

Use these only as validation-only references. Do not treat them as final test performance.

| Run | Selection | Best Epoch | Val MAE | Val RMSE | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `online_risk_no_stats` | legacy validation MAE | 23 | 2.2955 | 3.0576 | Previous online reference, risk auxiliary was active. |
| `current_fms_dual_auxrisk` | validation MAE | 23 | 2.3058 | 3.0884 | Current-FMS selected, small auxiliary risk loss. |
| `current_fms_dual_medium` | validation MAE | 14 | 2.3183 | 3.1703 | Best among the first 3 plotted current-FMS runs. |
| `current_fms_encoder_heavy` | validation MAE | 15 | 2.3276 | 3.2229 | Added causal TCN motion encoder stem. |
| `current_fms_base_basic` | validation MAE | 14 | 2.3808 | 3.1949 | Basic current head baseline. |

First success threshold:

- Beat validation MAE `2.2955` under the same fixed split and leakage-safe online input contract.

Strong success threshold:

- Validation MAE `<= 2.20`.

Ambitious threshold:

- Validation MAE `<= 2.00` with stable validation plots and no leakage concerns.

Trend-first threshold:

- raw validation MAE may be slightly worse than the best MAE candidate, but should generally remain `<= 2.45`;
- trend metrics and fixed validation plots must show clear improvement in rise/drop tracking;
- at least 3 trend-first conditions must be met before final lock.

Existing baseline artifacts:

- `runs/online_fms_risk_tracking_0507/online_risk_no_stats/metrics.json`
- `runs/online_fms_current_tracking_0507/current_fms_base_basic/metrics.json`
- `runs/online_fms_current_tracking_0507/current_fms_dual_medium/metrics.json`
- `runs/online_fms_current_tracking_0507/current_fms_encoder_heavy/metrics.json`
- `runs/online_fms_current_tracking_0507/current_fms_base_v1weights/metrics.json`
- `runs/online_fms_current_tracking_0507/current_fms_dual_auxrisk/metrics.json`

Existing 3-run comparison plots:

- `runs/online_fms_current_tracking_0507/comparison_3runs/validation_mae_curves.png`
- `runs/online_fms_current_tracking_0507/comparison_3runs/prediction_scatter.png`
- `runs/online_fms_current_tracking_0507/comparison_3runs/error_distribution.png`
- `runs/online_fms_current_tracking_0507/comparison_3runs/current_fms_3run_leaderboard.csv`
- `runs/online_fms_current_tracking_0507/comparison_3runs/trajectory_*.png`

Existing current-FMS configs:

- `configs/online_fms_current_tracker_base_basic.yaml`
- `configs/online_fms_current_tracker_dual_medium.yaml`
- `configs/online_fms_current_tracker_heavy.yaml`
- `configs/online_fms_current_tracker_base_v1weights.yaml`
- `configs/online_fms_current_tracker_dual_auxrisk.yaml`

Existing risk/current configs that may be useful as references:

- `configs/online_fms_risk_tracker.yaml`
- `configs/online_fms_risk_tracker_heavy.yaml`
- `configs/online_fms_risk_tracker_heavy_balanced.yaml`

## Current Working Tree Warning

At the time this document was written, the working tree already contained uncommitted changes and untracked files. Do not revert them unless the user explicitly asks.

Known modified files:

- `scripts/run_densefms_sanity_tests.py`
- `src/densefms_forecast/data.py`
- `src/densefms_forecast/evaluate.py`
- `src/densefms_forecast/losses.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`

Known relevant untracked files from this work:

- `configs/online_fms_current_tracker_base_basic.yaml`
- `configs/online_fms_current_tracker_base_v1weights.yaml`
- `configs/online_fms_current_tracker_dual_auxrisk.yaml`
- `configs/online_fms_current_tracker_dual_medium.yaml`
- `configs/online_fms_current_tracker_heavy.yaml`
- `configs/online_fms_risk_tracker.yaml`
- `configs/online_fms_risk_tracker_heavy.yaml`
- `configs/online_fms_risk_tracker_heavy_balanced.yaml`
- `docs/codex/online_current_fms_estimator_long_goal_0507.md`
- `docs/codex/online_fms_risk_tracking_goal_0507.md`
- `scripts/plot_online_current_comparison.py`

Some modified files, especially `data.py` and `losses.py`, may include pre-existing user/session changes from before the current-FMS work. Inspect diffs before touching them.

## Model Freedom

The implementation may add a new task mode, new dataset windowing class, new model class, new loss, new evaluator, or new training script if that is cleaner than extending the current path.

Allowed architecture families include, but are not limited to:

- calibration-conditioned recurrent state trackers;
- causal TCN/ConvNeXt-style motion encoders;
- causal Transformer or attention pooling over recent and long motion context;
- multi-scale motion encoders with short/mid/long windows;
- learned latent-state update models;
- direct current-FMS regressors;
- delta-from-calibration or delta-from-latent baselines;
- baseline-plus-dynamic-residual models;
- ordinal-regression plus continuous-regression multi-head models;
- quantile or uncertainty-aware current-FMS heads;
- mixture-of-experts or session-cluster specialists;
- self-supervised or teacher-assisted pretraining, if deployment inputs remain valid;
- validation-selected ensembles, if every member uses only allowed inputs.

The first new family should be current-FMS-first. Keep rapid-rise or risk heads disabled unless a validation-only ablation shows they improve current-FMS MAE, trend metrics, or plots.

Recommended output decomposition:

```text
predicted_current_fms = calibration_level_baseline + dynamic_motion_residual
```

Useful heads:

- direct absolute current-FMS head;
- delta-from-calibration-end FMS head;
- multi-horizon delta heads;
- short-term slope head;
- rise/drop/flat auxiliary head;
- uncertainty or quantile head.

Useful losses:

- current FMS regression loss;
- SmoothL1 / L1 / Huber variants;
- multi-horizon delta loss;
- slope loss;
- direction classification loss for rise/drop/flat;
- high-FMS weighting only if it improves validation evidence;
- smoothness regularization only if it does not make predictions flat or lagged.

Avoid excessive smoothness regularization if it causes flat or delayed predictions.

Recommended exploration axes:

- calibration length: 30s, 60s, 90s, 120s, 180s;
- recent context: 5s, 10s, 20s, 30s, 60s, full post-calibration causal stream;
- static features: off, raw, encoded, FiLM/gating;
- motion features: raw only, norm, deltas, energy, causal statistics, learned feature stem;
- state feedback: none, predicted-current feedback, latent-only feedback;
- output heads: regression, ordinal, hybrid, quantile, delta, gated direct/delta;
- losses: SmoothL1, L1, Huber variants, ordinal CE, high-FMS weighting, delta loss, slope loss, lag penalty, smoothness regularization;
- model scale: small baseline, medium, heavy, over-parameterized diagnostic;
- training: seeds, LR, weight decay, dropout, batch size, gradient clipping, scheduler, EMA/SWA if supported.

Do not keep any added complexity unless validation metrics or validation plots justify it.

## Search Policy

This is an autonomous adaptive validation search. Experiments should be selected based on observed validation behavior, not by completing a fixed grid.

### Initial phase

- Run or reuse sanity checks before full training.
- Establish a clean validation leaderboard with the current baselines.
- Add trend-following evaluation before expanding the search too much.
- Start with representative validation-only candidates across clearly different architecture/loss families.
- Continue beyond the initial candidates if the next run is hypothesis-driven.

### Adaptive phase

There is no fixed hard limit on validation-only runs, and there is no fixed wall-clock time limit in this goal.

Allowed:

- full training;
- validation-only adaptive search;
- parallel training when VRAM/RAM permit it;
- increasing epochs for promising candidates;
- replacing the model family if the current one is structurally wrong.

Required after each serious batch:

- update leaderboard;
- report validation MAE/RMSE;
- report trend-following metrics;
- inspect trajectory plots for representative sessions;
- inspect worst-session errors;
- identify whether the model is:
  - flat,
  - lagged,
  - overshooting,
  - under-reacting,
  - scale-shifted,
  - too noisy,
  - high-FMS biased,
  - low-FMS biased.

Continue only if the next batch has a concrete hypothesis.

If many runs are executed, periodically write a checkpoint note summarizing what has and has not worked.

### Epoch and early stopping defaults

- no hard max epoch limit is imposed by this goal;
- candidate configs should still define explicit `epochs` and `patience`;
- max epochs may be increased for promising heavy models;
- patience: 10 by default;
- larger patience is allowed if validation curves are still improving;
- prune clearly bad runs early if validation MAE remains far worse than baselines after a fair warm-up.

### Compute/model-size constraints

- Model size is free to increase if it does not cause OOM.
- Training time is acceptable as long as a normal full-training epoch is under about 60 seconds on the available machine.
- If epoch time exceeds 60 seconds, reduce model size, sequence length, batch size, or parallelism unless the user explicitly approves that specific run.
- OOM is not acceptable as a planned operating point.
- If OOM occurs, reduce batch size/model size or run fewer jobs in parallel.
- Prefer using available VRAM/RAM efficiently, but do not sacrifice validation/test separation or leakage safety for speed.

## Warm-Up Handling

Prediction begins after calibration by default.

If a candidate uses a post-calibration warm-up period:

- the warm-up duration must be recorded;
- the prediction mask must be saved in metrics and prediction CSVs;
- compare it against baselines evaluated with the same mask, or report it in a separate leaderboard section;
- do not hide hard early-online regions to make a model look better.

## Static Feature Handling

Static features such as age, gender, or MSSQ are allowed only if:

- preprocessing is fit on the training split only;
- validation/test statistics are not used for normalization or imputation;
- ablation compares static on vs static off;
- missingness is handled explicitly;
- final report states whether static features helped.

## Affine Calibration Handling

Train-only affine calibration may be used as a diagnostic or as part of the final deployment model.

Implementation requirement:

- Implement or reuse train-split prediction collection.
- Use the selected checkpoint to collect training predictions and training labels.
- Fit affine correction only on training predictions and training labels.
- Apply the frozen correction to validation predictions.
- If the affine correction is part of the final candidate, freeze it before test and apply the same correction to test predictions.

Rules:

- Never fit correction on validation labels.
- Never fit correction on test labels.
- Always report raw metrics and affine-corrected metrics separately.
- If final candidate uses affine correction, include correction coefficients and fitting split in `validation_selection_lock.md`.

Purpose:

- identify models that learn movement shape but have wrong scale or offset.

## Final Selection and Test Policy

Model selection must use validation evidence only.

Before final test, write:

- `validation_selection_lock.md`

This lock file must include:

- selected route:
  - MAE-first, or
  - trend-first;
- selected run name;
- selected config path;
- selected checkpoint path;
- selected best epoch;
- split file or exact split-generation parameters;
- raw validation metrics;
- trend-following validation metrics;
- validation plots used for review;
- reason for selection;
- trend-first checklist results if trend-first route is used;
- confirmation that test was not used for selection;
- whether train-only affine correction is part of the selected deployment candidate;
- affine coefficients if used;
- whether any warm-up mask is used.

Run test exactly once only after `validation_selection_lock.md` is written.

Test may be run only if one of these validation-only conditions is met.

### Route A: MAE-first lock

- selected candidate is the best validation-MAE candidate;
- ideally validation MAE `<= 2.00`, or at least beats the `2.2955` reference under the same split.

### Route B: trend-first lock

- selected candidate may have slightly worse MAE than the best-MAE model;
- it must satisfy at least 3 trend-first conditions listed in `Selection Logic`;
- it must show clearly better validation trend-following evidence;
- raw MAE should generally be no worse than about `2.45`.

Test metrics must not be used to decide architecture, hyperparameters, seed, checkpoint, threshold, ensemble members, affine correction, warm-up duration, or route selection.

If no candidate beats the validation baseline and no trend-first candidate is convincing:

- do not run test;
- update the validation leaderboard;
- write failure-mode notes;
- propose the next concrete hypotheses.

## Required Verification

Always run or create lightweight checks for:

- import check;
- seconds-to-steps conversion;
- fixed split loading or exact split reproduction;
- target shift correctness for current FMS;
- calibration leakage check;
- recent-window leakage check;
- causal stream no-future-motion check;
- inclusive current-index head usage check, i.e. allowed motion up to `head[:t+1]`;
- post-calibration FMS exclusion check;
- optional static train-only preprocessing check;
- model forward shape check;
- trend metric computation sanity check;
- affine calibration train-only check, if implemented;
- warm-up mask check, if implemented;
- dry-run sweep command generation if sweep scripts are modified.

If full training is run, also verify:

- checkpoint saving;
- resume behavior if supported;
- metrics CSV/JSON generation;
- validation prediction CSV generation;
- validation plot generation;
- validation leaderboard generation;
- fixed plot session protocol;
- final selected model evaluation on test only after validation lock.

## Required Artifacts

During the search:

- config files for each meaningful candidate;
- model/training code changes, if any;
- validation-only metrics JSON/CSV;
- validation prediction CSVs for serious candidates;
- train prediction CSVs if affine calibration is implemented;
- validation trajectory plots;
- fixed validation plot set;
- trend-following metric table;
- validation leaderboard;
- autonomous checkpoint notes in Korean;
- failure-mode notes for non-winning families.

Before final test:

- `validation_selection_lock.md`

Final report:

- modified or added files;
- new CLI/config options;
- dataset/windowing changes;
- split file or split-generation policy;
- model changes;
- loss changes;
- trend metric implementation;
- plotting/audit implementation;
- train-only affine implementation if used;
- anchor/static/multi-horizon support status;
- sanity test results;
- full-training budget actually used;
- validation leaderboard;
- MAE-first vs trend-first selection discussion;
- final selected configuration;
- final test-set metrics for the selected configuration, if test was run;
- raw vs affine-corrected metrics, if affine correction was used;
- generated plots/tables;
- git status summary;
- remaining issues or warnings.

Final report language should be Korean.

## Non-Goals

- Do not optimize future FMS forecasting in this goal.
- Do not optimize rapid-rise alarm as the primary objective.
- Do not use absolute high-FMS warning as the main product decision.
- Do not present validation search results as final test performance.
- Do not use test results to choose architecture, hyperparameters, thresholds, seeds, checkpoints, plot sessions, affine correction, or ensembles.
- Do not keep a model just because it is complex.
- Do not accept a flat predictor as successful just because MAE is passable.

## Current Recommended Next Step

Create a clean current-FMS search scaffold separate from the risk-oriented tracker.

Recommended immediate plan:

1. Keep the existing online tracker as a baseline.
2. Resolve and lock the comparable split policy.
3. Add validation trend-following metrics.
4. Add fixed validation trajectory plot protocol.
5. Add one new current-FMS-first model family focused on motion-to-latent dynamics.
6. Prefer output decomposition:

   ```text
   predicted_current_fms = calibration_level_baseline + dynamic_motion_residual
   ```

7. Add optional multi-horizon delta/slope auxiliary losses.
8. Run smoke training first.
9. Run an initial diverse validation-only batch.
10. Continue autonomous validation-only adaptive search while compute is stable.
11. Compare both:
   - raw MAE/RMSE;
   - trend-following metrics and plots.
12. Decide whether to expand search, select a candidate, or write failure-mode notes.

## Fresh-Context Execution Plan

When starting from a reset context, proceed in this order:

1. Read this document.
2. Inspect repository structure and the files listed in `Fresh-Context Bootstrap`.
3. Run `git status --short`.
4. Do not revert unrelated changes.
5. Resolve fixed split file or exact split-generation parameters.
6. Run import/py_compile check.
7. Run `scripts/run_densefms_sanity_tests.py`.
8. Parse the existing baseline metrics listed above and recreate a validation leaderboard.
9. Inspect validation plots from `comparison_3runs`.
10. Implement or verify trend-following metric computation.
11. Implement or verify fixed validation trajectory plot protocol.
12. Decide the first new architecture family.
13. Implement the smallest clean scaffold needed for that family.
14. Add leakage/shape/trend-metric/split tests for the new family.
15. Run smoke training first.
16. Run validation-only full training candidates with `--no_test_eval`.
17. Update leaderboard and plot review.
18. Continue autonomous adaptive search while the user is away, only with concrete hypotheses.
19. Before final test, write `validation_selection_lock.md`.
20. Evaluate test exactly once for the locked validation-selected candidate, if a candidate is worth locking.
21. Write final Korean report.

## Codex Operating Instruction

Execute this goal end-to-end within the repository.

Important priorities:

1. Preserve deployment realism and leakage safety.
2. Use the same fixed validation/test split as the baseline whenever available.
3. Optimize for current-time FMS, not future FMS.
4. Try to reach validation MAE `<= 2.00`.
5. Also treat trend-following as a first-class success mode:
   - scale may be imperfect,
   - but rises and drops should be followed.
6. Continue autonomous validation-only adaptive search while the user is away.
7. Do not stop merely because a small initial batch has completed.
8. Use validation only for all selection decisions.
9. Do not run test until validation selection is locked.
10. Do not commit, push, or revert unrelated changes.
11. Report progress and final results in Korean.
