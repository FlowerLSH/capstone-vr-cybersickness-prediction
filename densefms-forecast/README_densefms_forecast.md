# DenseFMS COFFLSTM Forecasting Prototype

This prototype trains real-time-compatible models for calibration-conditioned online future FMS forecasting on `./DenseFMS/Dataset`.

## Task

- Sampling interval defaults to 0.5 seconds.
- Calibration: first 30 seconds, 60 steps.
- Recent online window: 10 seconds, 20 steps.
- Forecast horizon: 5 seconds, 10 steps.
- Main long-horizon experiments use `fms_context_mode=start_only`: models receive calibration head/FMS history, recent head motion, and exactly one scalar FMS from the start of each recent online motion window.
- In no-anchor settings, ground-truth FMS after the allowed calibration/window-start context is used only as a loss target.
- Sparse-anchor, calibration-history-only, recent-start-anchor, or static-feature settings are diagnostics rather than the main window-start-FMS track. They require different user/context inputs such as prompted FMS anchors, age/gender, or MSSQ.

## FMS Context Modes

`fms_context_mode` controls which FMS values are allowed as model input:

- `none`: no FMS context; motion-only baseline.
- `start_only`: main track; uses calibration-phase FMS history plus one rolling scalar, `FMS[t - recent_window_steps + 1]`, at each prediction current time `t`.
- `calibration_history`: diagnostic only; allows the full calibration FMS sequence but no rolling window-start FMS scalar.
- `sparse_anchor`: anchor-assisted diagnostic only; uses sparse observed FMS anchors such as 60s prompts and must not be interpreted as window-start-FMS performance.

The main comparison tracks are `motion_only`, `start_fms_only` (window-start FMS), `calibration_history` diagnostic, and `sparse_anchor_60s` diagnostic. Reports and leaderboards include `fms_context_mode` so these tracks remain separated.

## Deployment wording

Use `sparse-anchor-assisted diagnostic` for tracks that require sparse observed FMS anchors. Do not describe those tracks as window-start-FMS, passive real-time, or head/motion-only deployment.

Report the user burden with each deployment-style result:

- whether the rolling recent-window-start FMS scalar is used, or whether only calibration FMS history is enabled as a diagnostic
- sparse FMS prompt interval, for example 60 seconds when `anchor_interval_seconds=60`
- whether MSSQ is required
- whether age/gender are required
- whether the model can run with head/motion only after calibration

`recent_start_observed` is not part of the main experiment track.

## CLIs

Inspect dataset:

```bash
python scripts/inspect_densefms.py --data_dir ./DenseFMS/Dataset
```

Train the 2x2 COFFLSTM experiments with a shared split:

```bash
python -m src.densefms_forecast.train \
  --data_dir ./DenseFMS/Dataset \
  --config configs/coff_lstm_no_static_level.yaml \
  --model coff_lstm \
  --loss_mode level_only \
  --split_file ./artifacts/densefms_split_seed42.json
```

```bash
python -m src.densefms_forecast.train \
  --data_dir ./DenseFMS/Dataset \
  --config configs/coff_lstm_no_static_trend.yaml \
  --model coff_lstm \
  --loss_mode level_trend_raw \
  --trend_weight 0.1 \
  --split_file ./artifacts/densefms_split_seed42.json
```

```bash
python -m src.densefms_forecast.train \
  --data_dir ./DenseFMS/Dataset \
  --config configs/coff_lstm_static_level.yaml \
  --model coff_lstm \
  --loss_mode level_only \
  --use_static \
  --static_features age gender \
  --split_file ./artifacts/densefms_split_seed42.json
```

```bash
python -m src.densefms_forecast.train \
  --data_dir ./DenseFMS/Dataset \
  --config configs/coff_lstm_static_trend.yaml \
  --model coff_lstm \
  --loss_mode level_trend_raw \
  --trend_weight 0.1 \
  --use_static \
  --static_features age gender \
  --split_file ./artifacts/densefms_split_seed42.json
```

Train a baseline:

```bash
python -m src.densefms_forecast.train \
  --data_dir ./DenseFMS/Dataset \
  --config configs/coff_lstm.yaml \
  --model recent10_tcn
```

Evaluate a checkpoint:

```bash
python -m src.densefms_forecast.evaluate \
  --checkpoint ./runs/<run_name>/best.pt \
  --data_dir ./DenseFMS/Dataset
```

Real-time simulation:

```bash
python -m src.densefms_forecast.realtime \
  --checkpoint ./runs/<run_name>/best.pt \
  --csv_path ./DenseFMS/Dataset/<one_session_csv>
```

## Models

- `coff_lstm`: calibration encoder + FiLM-conditioned online LSTM + recent-window TCN + direct future-level forecast head.
- `recent10_tcn`: recent 10-second head-window TCN baseline.
- `calib_only`: calibration encoder plus learned time-index embedding baseline.

Default training uses only future sequence prediction:

```text
level_only:
  SmoothL1(S_pred, S_true)

level_trend_raw:
  SmoothL1(S_pred, S_true)
  + trend_weight * SmoothL1(diff(S_pred), diff(S_true))
```

The legacy now/delta/gate fusion path is disabled by default with:

```yaml
model:
  use_legacy_multihead: false
```

Optional static age/gender personalization is controlled by:

```yaml
data:
  use_static: true
  static_features: ["age", "gender"]
  allow_missing_static: false

model:
  use_static: true
  static_dim: 4
  static_hidden_dim: 64
  static_dropout: 0.1
```

The static vector is `[age_z, gender_male, gender_female, gender_unknown]`.
Age scaling is fit on the train split only; gender is case-insensitive and maps unknown or missing values to `unknown` only when `--allow_missing_static` is used.
FMS labels are normalized with the fixed DenseFMS measurement range `0-20`, not with split-specific observed min/max values.
Sequence analysis reports both the legacy raw exact sign metric (`trend_sign_accuracy_raw_exact`) and thresholded trend metrics with default `eps_fms=0.5` on the original FMS scale. Because DenseFMS labels are plateau-heavy and predictions are continuous, use the thresholded trend metrics for interpretation.

### Age/Gender/MSSQ full static feature

DenseFMS-style full static personalization can use Age, Gender, and MSSQ together:

```yaml
data:
  use_static: true
  static_features: ["age", "gender", "mssq"]

model:
  use_static: true
  static_dim: 5
```

The full static vector is `[age_z, mssq_z, gender_male, gender_female, gender_unknown]`.
MSSQ is standardized with train-split mean/std and acts as a user-level motion sickness susceptibility covariate; it is fused with `z_calib` through the static encoder/context fusion path, not fed to the online LSTM at every step.
Compare `no_static`, `age_gender`, and `age_gender_mssq` with the same split file and eligible session subset.

Useful ablations:

```bash
--no_film
--no_recent_encoder
--no_aux_now
--loss_mode level_only
--loss_mode level_trend_raw
--trend_weight 0.1
--use_static
--static_features age gender
--allow_missing_static
--fms_context_mode start_only
--calibration_seconds 15
--horizon_seconds 10
--recent_window_seconds 10
```

## Calibration length and forecast horizon ablation

The online forecasting task can vary the initial calibration length, future horizon, and recent head-motion window without changing the model backbone or loss formula.

Canonical config fields:

```yaml
data:
  sampling_interval: 0.5
  calibration_seconds: 30.0
  horizon_seconds: 5.0
  recent_window_seconds: 10.0
```

Legacy aliases remain supported: `default_sampling_interval` and `recent_seconds`.

Seconds are converted to steps with `round(seconds / sampling_interval)`, so `30s -> 60`, `5s -> 10`, and `2.5s -> 5` at 0.5-second sampling. Prediction targets are always shifted by the resulting horizon steps. Recent windows are causal and left-padded when the current history is shorter than the requested window, so no sample after current time `t` is used.

Natural metrics use every valid prediction in each run. Common-window metrics filter predictions to a shared time range so calibration/horizon sweeps are comparable despite different valid prediction intervals.

Dry-run a calibration sweep:

```bash
python scripts/run_calibration_horizon_sweep.py \
  --data_dir ./DenseFMS/Dataset \
  --split_file ./artifacts/densefms_split_seed42.json \
  --calibration_seconds 30 60 90 \
  --horizon_seconds 5 \
  --recent_window_seconds 10 \
  --loss_mode level_only \
  --run_prefix calib_sweep \
  --dry_run
```

Dry-run a horizon sweep:

```bash
python scripts/run_calibration_horizon_sweep.py \
  --data_dir ./DenseFMS/Dataset \
  --split_file ./artifacts/densefms_split_seed42.json \
  --calibration_seconds 30 \
  --horizon_seconds 1 2.5 5 10 15 \
  --recent_window_seconds 10 \
  --loss_mode level_only \
  --run_prefix horizon_sweep \
  --dry_run
```

Interpret calibration improvements against both natural and common-window metrics. Longer calibration changes the first valid prediction time, so common-window metrics are the safer comparison. Horizon results should be read descriptively: larger horizons are expected to be harder, but the sweep does not establish causal explanations.

## Performance optimization runner

Use the adaptive optimization runner when experiments need durable logs and resumable summaries:

```bash
python scripts/run_densefms_optimization.py \
  --data_dir ./DenseFMS/Dataset \
  --split_file ./artifacts/densefms_split_seed42.json \
  --stages stage1 stage2 \
  --conditions no_static full_static \
  --run_prefix opt \
  --dry_run
```

Every run directory stores `command.txt`, `run_config.json`, `git_commit.txt`, `status.json`, `events.jsonl`, `stdout.log`, `stderr.log`, training curves, checkpoints, and metrics. Partial or interrupted runs can still be summarized with:

```bash
python scripts/run_densefms_optimization.py --run_prefix opt --summary_only
```

The recent-window encoder can be switched with:

```bash
--recent_encoder tcn
--recent_encoder transformer
```

The transformer recent encoder uses only the causal recent window ending at current time `t`; it never receives future head motion.

## Outputs

- `artifacts/data_report.json`
- `artifacts/column_mapping.json`
- `runs/<run_name>/best.pt`
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/plots/*.png`

The checkpoint stores the model config, inferred columns, participant/session split, train-set head/static scalers, the fixed FMS `0-20` scaler, and normalized-scale `delta_max`.

Comparison plots can be generated after both runs finish:

```bash
python -m src.densefms_forecast.plot_compare \
  --level_only_checkpoint ./runs/<level_only_run>/best.pt \
  --level_trend_checkpoint ./runs/<level_trend_run>/best.pt \
  --data_dir ./DenseFMS/Dataset \
  --split test
```
