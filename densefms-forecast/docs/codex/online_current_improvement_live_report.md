## Integrated Improvement Plan Update - 2026-05-09 06:30:32

- mode: dry-run only
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 0
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | pending-dry-run |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | pending-dry-run |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | pending-dry-run |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | pending-dry-run |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-dry-run |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-dry-run |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-dry-run |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-dry-run |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-dry-run |

### Commands

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Motion Dynamics Diagnostic - 2026-05-09 06:31

- mode: offline diagnostic only, no training
- output: `runs/online_fms_current_tracking_0509_integrated/motion_dynamics_diagnostic/`
- rows: 72,700 causal current-time samples after `calibration_seconds=120.0`, `horizon_seconds=5.0`, `max_session_points=420`
- key table: `motion_dynamics_correlations.csv`
- plot: `complexity_vs_fms.png`
- note: future FMS and rapid-rise labels were used only as diagnostic labels, not as model inputs.
- strongest absolute FMS-rank signals were still weak: `motion_delta_energy_long` Spearman -0.0665, `motion_jerk_energy` -0.0446, `gyro_jerk_norm` -0.0440.
- strongest 5s future-delta/rapid-rise signal among this bank was from magnitude/gyro features, e.g. `motion_norm` delta Pearson 0.1027 and rapid-rise Pearson 0.1086.
## Integrated Improvement Plan Update - 2026-05-09 07:36:01

- mode: dry-run only
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 5
- completed in this invocation: 0
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | pending-dry-run |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | pending-dry-run |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | pending-dry-run |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | pending-dry-run |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-dry-run |

### Commands

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```

## Completed Run - integrated_p1_risk015_seed42 - 2026-05-09 07:39:44

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--risk_loss_weight 0.15`
- purpose: weaker rapid-rise auxiliary regularization around the selected baseline
- changed factor family: `risk_loss_weight`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=36, best_epoch=26, elapsed_seconds=187.7
- analysis exit_code: 0
- validation MAE/RMSE: 2.086854871853634 / 3.008457244674399
- validation session Pearson: 0.4180240200721151
- validation centered MAE: 1.4180499303924694
- validation delta corr 5s: 0.4049580746346776
- validation direction acc 5s: 0.7160094173042967
- validation flat rate: 0.06779661016949153
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.141681
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk015_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk015_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk015_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk015_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:39:44

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 1
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | pending-or-skipped |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | pending-or-skipped |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | pending-or-skipped |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-or-skipped |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-or-skipped |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p1_risk035_seed42 - 2026-05-09 07:45:05

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--risk_loss_weight 0.35`
- purpose: stronger rapid-rise auxiliary regularization around the selected baseline
- changed factor family: `risk_loss_weight`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=60, best_epoch=50, elapsed_seconds=308.5
- analysis exit_code: 0
- validation MAE/RMSE: 1.9435687677948563 / 2.786586605627599
- validation session Pearson: 0.46347239797711365
- validation centered MAE: 1.372416453329256
- validation delta corr 5s: 0.43205487206859905
- validation direction acc 5s: 0.7004120070629782
- validation flat rate: 0.03389830508474576
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: -0.001605
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk035_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk035_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk035_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_risk035_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:45:05

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 2
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | pending-or-skipped |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | pending-or-skipped |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-or-skipped |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-or-skipped |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p1_ordblend015_seed42 - 2026-05-09 07:47:17

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--fms_combine_weight_ordinal 0.15`
- purpose: lower ordinal blend to preserve continuous amplitude
- changed factor family: `fms_combine_weight_ordinal`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=23, best_epoch=13, elapsed_seconds=120.3
- analysis exit_code: 0
- validation MAE/RMSE: 2.1820900001349273 / 3.065466039733883
- validation session Pearson: 0.43755422864063137
- validation centered MAE: 1.4234276015493605
- validation delta corr 5s: 0.3604593778688744
- validation direction acc 5s: 0.6986462625073573
- validation flat rate: 0.0847457627118644
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.236917
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend015_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend015_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend015_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend015_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:47:17

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 3
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | pending-or-skipped |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-or-skipped |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-or-skipped |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p1_ordblend025_seed42 - 2026-05-09 07:52:43

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--fms_combine_weight_ordinal 0.25`
- purpose: higher ordinal blend to stabilize severity ordering
- changed factor family: `fms_combine_weight_ordinal`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=60, best_epoch=50, elapsed_seconds=313.1
- analysis exit_code: 0
- validation MAE/RMSE: 1.9734655660997937 / 2.8456425328358237
- validation session Pearson: 0.45459758780233944
- validation centered MAE: 1.3926934824476151
- validation delta corr 5s: 0.4472484794076036
- validation direction acc 5s: 0.7242495585638611
- validation flat rate: 0.01694915254237288
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.028292
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend025_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend025_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend025_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_ordblend025_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:52:43

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 4
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | pending-or-skipped |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-or-skipped |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p1_fdsblend075_seed42 - 2026-05-09 07:56:09

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--fds_blend 0.75`
- purpose: weaken FDS pull toward average trajectories
- changed factor family: `fds_blend`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=37, best_epoch=27, elapsed_seconds=193.0
- analysis exit_code: 0
- validation MAE/RMSE: 2.1192465585911715 / 3.1127367479928987
- validation session Pearson: 0.41647145210402187
- validation centered MAE: 1.4195112949720134
- validation delta corr 5s: 0.3941317035976776
- validation direction acc 5s: 0.7154208357857563
- validation flat rate: 0.1016949152542373
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.174073
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_fdsblend075_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_fdsblend075_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_fdsblend075_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p1_fdsblend075_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:56:09

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 5
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | pending-or-skipped |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p2_future_delta_event_light_seed42 - 2026-05-09 07:59:49

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0`
- purpose: near-future FMS, delta, and rise/drop/plateau auxiliary supervision
- changed factor family: `future_delta_event_aux`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=39, best_epoch=29, elapsed_seconds=206.2
- analysis exit_code: 0
- validation MAE/RMSE: 2.0803287381375277 / 2.9181237234314064
- validation session Pearson: 0.42653019154377964
- validation centered MAE: 1.416806937458216
- validation delta corr 5s: 0.41057937427968255
- validation direction acc 5s: 0.723955267804591
- validation flat rate: 0.06779661016949153
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.135155
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_future_delta_event_light_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_future_delta_event_light_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_future_delta_event_light_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_future_delta_event_light_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 07:59:49

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 6
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | completed |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | pending-or-skipped |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p2_delta_only_light_seed42 - 2026-05-09 08:03:31

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0`
- purpose: isolate future-delta supervision without future-level or event losses
- changed factor family: `delta_aux`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=39, best_epoch=29, elapsed_seconds=207.8
- analysis exit_code: 0
- validation MAE/RMSE: 2.0796475115308057 / 2.9586582157035863
- validation session Pearson: 0.43138366175647846
- validation centered MAE: 1.4122040446171298
- validation delta corr 5s: 0.39642996195251357
- validation direction acc 5s: 0.7289582107121836
- validation flat rate: 0.06779661016949153
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.134474
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_delta_only_light_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_delta_only_light_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_delta_only_light_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_delta_only_light_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 08:03:31

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 7
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | completed |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | completed |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | pending-or-skipped |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p2_trajectory_w003_d5_seed42 - 2026-05-09 08:07:13

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae`
- purpose: weak trajectory-shape auxiliary with 5s deltas
- changed factor family: `trajectory_loss`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=40, best_epoch=30, elapsed_seconds=207.7
- analysis exit_code: 0
- validation MAE/RMSE: 2.1210882392415296 / 3.0319323700407916
- validation session Pearson: 0.4299742464616306
- validation centered MAE: 1.411927869085299
- validation delta corr 5s: 0.40406874775482693
- validation direction acc 5s: 0.7271924661565626
- validation flat rate: 0.0847457627118644
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: 0.175915
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_trajectory_w003_d5_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_trajectory_w003_d5_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_trajectory_w003_d5_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p2_trajectory_w003_d5_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 08:07:13

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 8
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | completed |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | completed |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | completed |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | pending-or-skipped |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Completed Run - integrated_p4_causal_dynamics_v1_seed42 - 2026-05-09 08:11:38

- config: `configs/online_current/selected_fds_static4.yaml`
- CLI override: `--motion_feature_mode causal_dynamics_v1`
- purpose: append causal derivative, energy, sign-change, and complexity proxies
- changed factor family: `motion_feature_mode`
- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`
- training budget: epochs_completed=48, best_epoch=38, elapsed_seconds=250.3
- analysis exit_code: 0
- validation MAE/RMSE: 1.9303348613613183 / 2.8379432181837916
- validation session Pearson: 0.4571691502182401
- validation centered MAE: 1.3722154387992964
- validation delta corr 5s: 0.44416640429545
- validation direction acc 5s: 0.7257210123602119
- validation flat rate: 0.06779661016949153
- PLOT proxy judgment: good=4, medium=0, bad=8 on fixed baseline-selected validation set
- baseline MAE delta: -0.014839
- decision: `reject`
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_integrated\integrated_p4_causal_dynamics_v1_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_integrated\integrated_p4_causal_dynamics_v1_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_integrated\integrated_p4_causal_dynamics_v1_seed42\val_predictions.csv`, plots=`runs\online_fms_current_tracking_0509_integrated\integrated_p4_causal_dynamics_v1_seed42\plots`
- leaderboard: `runs\online_fms_current_tracking_0509_integrated\analysis\online_current_validation_leaderboard.csv`
- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.
## Integrated Improvement Plan Update - 2026-05-09 08:11:38

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 9
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | completed |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | completed |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | completed |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | completed |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

## Integrated Improvement Plan Update - 2026-05-09 08:11:44

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_integrated`
- test evaluation: skipped by `--no_test_eval`
- selected candidates: 9
- completed in this invocation: 9
- failed in this invocation: 0

| phase | run | isolated factor | purpose | status |
| --- | --- | --- | --- | --- |
| phase1 | `integrated_p1_risk015_seed42` | `risk_loss_weight` | weaker rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_risk035_seed42` | `risk_loss_weight` | stronger rapid-rise auxiliary regularization around the selected baseline | completed |
| phase1 | `integrated_p1_ordblend015_seed42` | `fms_combine_weight_ordinal` | lower ordinal blend to preserve continuous amplitude | completed |
| phase1 | `integrated_p1_ordblend025_seed42` | `fms_combine_weight_ordinal` | higher ordinal blend to stabilize severity ordering | completed |
| phase1 | `integrated_p1_fdsblend075_seed42` | `fds_blend` | weaken FDS pull toward average trajectories | completed |
| phase2 | `integrated_p2_future_delta_event_light_seed42` | `future_delta_event_aux` | near-future FMS, delta, and rise/drop/plateau auxiliary supervision | completed |
| phase2 | `integrated_p2_delta_only_light_seed42` | `delta_aux` | isolate future-delta supervision without future-level or event losses | completed |
| phase2 | `integrated_p2_trajectory_w003_d5_seed42` | `trajectory_loss` | weak trajectory-shape auxiliary with 5s deltas | completed |
| phase4 | `integrated_p4_causal_dynamics_v1_seed42` | `motion_feature_mode` | append causal derivative, energy, sign-change, and complexity proxies | completed |

### Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_risk035_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --risk_loss_weight 0.35
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend015_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.15
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_ordblend025_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fms_combine_weight_ordinal 0.25
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p1_fdsblend075_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --fds_blend 0.75
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_future_delta_event_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.05 --delta_aux_loss_weight 0.10 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_delta_only_light_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p2_trajectory_w003_d5_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --trajectory_loss_weight 0.03 --trajectory_delta_seconds 5.0 --trajectory_delta_weight 1.0 --trajectory_centered_weight 0.3 --trajectory_range_weight 0.1 --trajectory_loss_type mae
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name integrated_p4_causal_dynamics_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_integrated --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1
```

