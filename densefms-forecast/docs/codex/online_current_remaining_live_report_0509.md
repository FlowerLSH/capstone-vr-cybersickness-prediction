## Remaining Experiment Update - 2026-05-09 08:40:50

- mode: dry-run only
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | pending-dry-run |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | pending-dry-run |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | pending-dry-run |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | pending-dry-run |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-dry-run |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-dry-run |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-dry-run |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-dry-run |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-dry-run |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-dry-run |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-dry-run |

### Pretrain Command

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Remaining Experiment Update - 2026-05-09 08:41:25

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | pending-or-skipped |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | pending-or-skipped |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | pending-or-skipped |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | pending-or-skipped |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-or-skipped |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p5_multitimescale_v1_seed42 - 2026-05-09 08:45:35

- phase: `phase5`
- changed factor: `motion_feature_mode`
- purpose: 5/15/30/60s causal motion-response summaries and complexity-drop proxies
- CLI override: `--motion_feature_mode multi_timescale_v1`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.140894448183201 / 3.0695606659222006
- validation session Pearson: 0.4352819609385288
- validation centered MAE: 1.407364847124359
- validation delta corr 5s: 0.34912030798678034
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_multitimescale_v1_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_multitimescale_v1_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_multitimescale_v1_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 08:45:35

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | pending-or-skipped |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | pending-or-skipped |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | pending-or-skipped |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-or-skipped |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p5_stream_multiscale_seed42 - 2026-05-09 08:47:39

- phase: `phase5`
- changed factor: `stream_context_mode`
- purpose: GRU+TCN multiscale state response path as a structural lag-response check
- CLI override: `--stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.1617609459603275 / 3.1707914387015057
- validation session Pearson: 0.43984481137993103
- validation centered MAE: 1.4204124541004255
- validation delta corr 5s: 0.33399403966425767
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_stream_multiscale_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_stream_multiscale_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_stream_multiscale_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 08:47:39

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | pending-or-skipped |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | pending-or-skipped |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-or-skipped |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p5_event_only_seed42 - 2026-05-09 08:50:45

- phase: `phase5`
- changed factor: `event_aux_loss`
- purpose: rise/fall/plateau event auxiliary without future-level or delta losses
- CLI override: `--future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.147632080614567 / 3.014343637376292
- validation session Pearson: 0.4303744409301791
- validation centered MAE: 1.4071797928680116
- validation delta corr 5s: 0.40671583926893
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_event_only_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_event_only_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_event_only_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 08:50:45

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | pending-or-skipped |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-or-skipped |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p5_person_prior_seed42 - 2026-05-09 08:52:48

- phase: `phase5`
- changed factor: `current_head_mode`
- purpose: calibration/static-conditioned bias, scale, and response-speed prior
- CLI override: `--current_head_mode person_prior`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 1.961731317219911 / 2.889937940701366
- validation session Pearson: 0.3903222272654459
- validation centered MAE: 1.4540136310845742
- validation delta corr 5s: 0.2692573603600577
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_person_prior_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_person_prior_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p5_person_prior_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 08:52:48

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | pending-or-skipped |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p7_residual_update_seed42 - 2026-05-09 08:55:26

- phase: `phase7`
- changed factor: `current_head_mode`
- purpose: predicted-state residual update head with bounded per-step deltas
- CLI override: `--current_head_mode residual_update --current_delta_scale 1.0`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.026138138130859 / 3.007980541770685
- validation session Pearson: 0.3080313070699754
- validation centered MAE: 1.4765611720887226
- validation delta corr 5s: 0.20611908078410013
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p7_residual_update_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p7_residual_update_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p7_residual_update_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 08:55:26

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | pending-or-skipped |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p8_explicit_state_shared_aux_seed42 - 2026-05-09 09:00:32

- phase: `phase8`
- changed factor: `shared_latent_state_aux`
- purpose: latent-GRU state with predicted-current feedback plus future/event/regime supervision
- CLI override: `--state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.0508079855144024 / 2.8315716960870922
- validation session Pearson: 0.42507134036571376
- validation centered MAE: 1.416245209482349
- validation delta corr 5s: 0.41229906254253895
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p8_explicit_state_shared_aux_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p8_explicit_state_shared_aux_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p8_explicit_state_shared_aux_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:00:32

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | pending-or-skipped |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p9_coarse_band_aux_seed42 - 2026-05-09 09:03:34

- phase: `phase9`
- changed factor: `coarse_band_aux`
- purpose: low/mid/high/very-high severity band auxiliary head
- CLI override: `--coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.125099767965299 / 3.0346551108871544
- validation session Pearson: 0.4309109552832196
- validation centered MAE: 1.4175388056797746
- validation delta corr 5s: 0.38850052176721717
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p9_coarse_band_aux_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p9_coarse_band_aux_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p9_coarse_band_aux_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:03:34

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | pending-or-skipped |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p10_motion_pretrained_seed42 - 2026-05-09 09:05:55

- phase: `phase10`
- changed factor: `motion_pretraining`
- purpose: initialize causal-dynamics DeepTCN stream from motion-only future-energy pretraining
- CLI override: `--motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.183343600332737 / 3.1141765635103815
- validation session Pearson: 0.4645927543126068
- validation centered MAE: 1.3973076874444517
- validation delta corr 5s: 0.3920981098980265
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p10_motion_pretrained_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p10_motion_pretrained_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p10_motion_pretrained_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:05:55

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | completed |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | pending-or-skipped |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p11_scenario_prior_seed42 - 2026-05-09 09:09:25

- phase: `phase11`
- changed factor: `static_features`
- purpose: deployment-visible scenario/content one-hot prior parsed from session filename
- CLI override: `--static_features age mssq gender scenario`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.146909729540348 / 3.074271131383615
- validation session Pearson: 0.4329439446067862
- validation centered MAE: 1.4188077700851873
- validation delta corr 5s: 0.41536176611783127
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p11_scenario_prior_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p11_scenario_prior_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p11_scenario_prior_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:09:25

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | completed |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | completed |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | pending-or-skipped |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Completed Remaining Run - remaining_p12_regime_aux_seed42 - 2026-05-09 09:12:32

- phase: `phase12`
- changed factor: `regime_head`
- purpose: stable/slow-rise/rapid-rise/high-plateau/recovery regime classifier auxiliary
- CLI override: `--regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.1182917276466333 / 3.0483159200282532
- validation session Pearson: 0.43072871858892026
- validation centered MAE: 1.4144408923504037
- validation delta corr 5s: 0.4126254172954141
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_regime_aux_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_regime_aux_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_regime_aux_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:12:32

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | completed |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | completed |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | pending-or-skipped |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Remaining Experiment Update - 2026-05-09 09:13:36

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | completed |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | completed |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | failed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.02
```
## Remaining Experiment Update - 2026-05-09 09:15:27

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | failed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.005
```
## Remaining Experiment Update - 2026-05-09 09:17:05

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | failed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.005
```
## Completed Remaining Run - remaining_p12_uncertainty_head_seed42 - 2026-05-09 09:21:52

- phase: `phase12`
- changed factor: `uncertainty_head`
- purpose: heteroscedastic current-FMS uncertainty head with small NLL auxiliary weight
- CLI override: `--uncertainty_head_enabled --uncertainty_loss_weight 0.005`
- test evaluation: skipped by `--no_test_eval`
- analysis exit_code: 0
- validation MAE/RMSE: 2.076670883010935 / 2.964941156217728
- validation session Pearson: 0.4129882597926929
- validation centered MAE: 1.4183540644401633
- validation delta corr 5s: 0.4236087162352384
- outputs: checkpoint=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_uncertainty_head_seed42\best.pt`, metrics=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_uncertainty_head_seed42\metrics.json`, predictions=`runs\online_fms_current_tracking_0509_remaining\remaining_p12_uncertainty_head_seed42\val_predictions.csv`
- leaderboard: `runs\online_fms_current_tracking_0509_remaining\analysis\online_current_validation_leaderboard.csv`
## Remaining Experiment Update - 2026-05-09 09:21:52

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | completed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.005
```
## Remaining Experiment Update - 2026-05-09 09:21:55

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | completed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.005
```
## Remaining Experiment Update - 2026-05-09 09:22:13

- mode: execute validation-only training
- base config: `configs/online_current/selected_fds_static4.yaml`
- fixed split: `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_remaining`
- motion pretrain checkpoint: `runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt`
- test evaluation: skipped by `--no_test_eval`

| phase | run | isolated factor | status |
| --- | --- | --- | --- |
| phase5 | `remaining_p5_multitimescale_v1_seed42` | `motion_feature_mode` | completed |
| phase5 | `remaining_p5_stream_multiscale_seed42` | `stream_context_mode` | completed |
| phase5 | `remaining_p5_event_only_seed42` | `event_aux_loss` | completed |
| phase5 | `remaining_p5_person_prior_seed42` | `current_head_mode` | completed |
| phase7 | `remaining_p7_residual_update_seed42` | `current_head_mode` | completed |
| phase8 | `remaining_p8_explicit_state_shared_aux_seed42` | `shared_latent_state_aux` | completed |
| phase9 | `remaining_p9_coarse_band_aux_seed42` | `coarse_band_aux` | completed |
| phase10 | `remaining_p10_motion_pretrained_seed42` | `motion_pretraining` | completed |
| phase11 | `remaining_p11_scenario_prior_seed42` | `static_features` | completed |
| phase12 | `remaining_p12_regime_aux_seed42` | `regime_head` | completed |
| phase12 | `remaining_p12_uncertainty_head_seed42` | `uncertainty_head` | completed |

### Pretrain Command

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe scripts/pretrain_online_current_motion_encoder.py --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --out_dir runs/online_fms_current_tracking_0509_remaining/motion_pretrain --run_name motion_energy_causal_dynamics_v1_seed42 --motion_feature_mode causal_dynamics_v1 --hidden_dim 192 --deep_tcn_dilations 1 2 4 8 16 --kernel_size 3 --dropout 0.10 --batch_size 48 --epochs 30 --patience 5
```

### Training Commands

```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_multitimescale_v1_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode multi_timescale_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_stream_multiscale_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --stream_context_mode gru_tcn_multiscale --motion_feature_mode causal_dynamics_v1
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_event_only_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.0 --delta_aux_loss_weight 0.0 --event_aux_loss_weight 0.03 --event_delta_threshold 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p5_person_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode person_prior
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p7_residual_update_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --current_head_mode residual_update --current_delta_scale 1.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p8_explicit_state_shared_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --state_feedback_mode predicted_current --future_aux_horizon_seconds 5.0 10.0 15.0 --future_aux_loss_weight 0.03 --delta_aux_loss_weight 0.05 --event_aux_loss_weight 0.02 --regime_head_enabled --regime_loss_weight 0.02
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p9_coarse_band_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --coarse_band_bins 5.0 10.0 15.0 --coarse_band_loss_weight 0.05
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p10_motion_pretrained_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --motion_feature_mode causal_dynamics_v1 --motion_pretrain_checkpoint runs\online_fms_current_tracking_0509_remaining\motion_pretrain\motion_energy_causal_dynamics_v1_seed42\best_motion_encoder.pt
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p11_scenario_prior_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --static_features age mssq gender scenario
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_regime_aux_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --regime_head_enabled --regime_loss_weight 0.03 --regime_delta_slow_threshold 0.5 --regime_delta_rapid_threshold 2.0 --regime_high_threshold 12.0
```
```bash
C:\Users\rio\AppData\Local\Programs\Python\Python310\python.exe -m src.densefms_forecast.train --data_dir DenseFMS/Dataset --config configs/online_current/selected_fds_static4.yaml --model online_fms_risk_tracker --run_name remaining_p12_uncertainty_head_seed42 --runs_dir runs/online_fms_current_tracking_0509_remaining --split_file runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json --no_test_eval --skip_existing --save_predictions --save_plots --uncertainty_head_enabled --uncertainty_loss_weight 0.005
```
