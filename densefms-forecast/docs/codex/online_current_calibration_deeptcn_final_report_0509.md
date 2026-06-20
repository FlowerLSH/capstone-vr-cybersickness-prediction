# Online Current Calibration DeepTCN 실험 보고서 - 2026-05-09

## 1. Modified or Added Files

- `src/densefms_forecast/model.py`
  - `calibration_encoder_mode`에 `deep_tcn`, `deep_tcn_transformer`를 추가했다.
  - `deep_tcn`: calibration sequence를 `DeepTCNEncoder`로 인코딩한 뒤 바로 pooling한다. Transformer는 `nn.Identity()`로 우회한다.
  - `deep_tcn_transformer`: calibration sequence를 `DeepTCNEncoder`로 먼저 인코딩한 뒤 기존 Transformer encoder stack을 적용한다.
- `src/densefms_forecast/train.py`
  - CLI 선택지에 `--calibration_encoder_mode deep_tcn deep_tcn_transformer`를 추가했다.
- `scripts/run_densefms_sanity_tests.py`
  - calibration encoder mode sanity에 `deep_tcn`, `deep_tcn_transformer`를 추가했다.
- `docs/codex/online_current_calibration_deeptcn_final_report_0509.md`
  - 본 보고서.

## 2. New CLI / Config Options

- 확장된 옵션:
  - `--calibration_encoder_mode deep_tcn`
  - `--calibration_encoder_mode deep_tcn_transformer`
- 기존 옵션과의 관계:
  - `deep_tcn`은 `--transformer_layers` 값이 config에 남아 있어도 calibration branch에서는 Transformer를 쓰지 않는다.
  - `deep_tcn_transformer`는 DeepTCN 뒤에 `--transformer_layers`만큼 Transformer layer를 쌓는다.
  - `--pooling mean|attention`은 DeepTCN output sequence summary에 그대로 적용된다.

## 3. Dataset / Windowing Changes

- 데이터/windowing 로직은 변경하지 않았다.
- 사용 데이터: `DenseFMS/Dataset`
- split: `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42/split.json`
- split count: train 316 / val 60 / test 52 sessions
- participant group split 유지.
- sampling interval: 0.5s
- max session points: 420
- calibration: 120s = 240 steps
- recent window: 10s
- horizon/rise auxiliary: current FMS tracking + rapid-rise 5s/10s
- static features: `age`, `mssq`, `gender`
- validation search 중 test는 `--no_test_eval`로 비활성화했다.

## 4. Model Changes

이번 실험에서 실제 사용된 calibration DeepTCN 구조:

- input: `[head_dim 6 + calibration FMS 1] = 7`
- projection: `Linear(7, d_model)` + GELU + LayerNorm + Dropout
- dilation stages: `[1, 2, 4, 8, 16]`
- 각 dilation stage는 `TCNBlock` 1개이고, `TCNBlock` 내부에는 causal conv 2개가 있다.
- 따라서 이번 run의 calibration DeepTCN은 5 residual dilation blocks, 실제 causal conv depth 10개다.
- `deep_tcn_mean`: 위 DeepTCN output을 mean pooling.
- `deep_tcn_attention`: 위 DeepTCN output을 attention pooling.
- `deep_tcn_transformer2_mean`: 위 DeepTCN output 뒤에 Transformer encoder 2층을 추가하고 mean pooling.

중요 정정:

- 코드 기본 경로는 `self.deep_tcn_dilations`를 calibration DeepTCN에 연결했다.
- 이번 config snapshot의 실제 값은 `[1, 2, 4, 8, 16]`이었다. 즉 이번 학습 결과는 32 dilation stage를 포함하지 않는다.

## 5. Anchor / Static / Multi-Horizon Support Status

- Anchor/FMS policy: 변경 없음.
- Calibration FMS input은 여전히 first `calibration_steps` 내부 값만 사용한다.
- post-calibration FMS는 calibration branch에 들어가지 않는다.
- Recent motion은 current time `t`까지만 사용한다.
- Static support: 유지. 이번 run은 `age`, `mssq`, `gender` 사용.
- Multi-horizon/future auxiliary 경로: 기존 지원 유지. 이번 변경은 calibration encoder ablation이다.

## 6. Sanity Test Results

실행:

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe -m py_compile src/densefms_forecast/model.py src/densefms_forecast/train.py scripts/run_densefms_sanity_tests.py
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/run_densefms_sanity_tests.py
```

결과:

- `py_compile`: pass
- 전체 sanity suite: pass
- 포함 확인:
  - import check
  - seconds-to-steps conversion
  - target shift correctness
  - calibration leakage check
  - recent-window leakage check
  - anchor policy check
  - model forward shape check
  - dry-run sweep command generation
  - `deep_tcn` / `deep_tcn_transformer` calibration mode forward + no post-calib FMS leakage
- 경고:
  - PyTorch Transformer nested tensor warning은 `deep_tcn_transformer`와 기존 Transformer 경로에서 출력됐지만 실패는 아니었다.

Full-training artifact 확인:

- `best.pt`, `final.pt`, `metrics.json`, `training_curves.csv`, `val_predictions.csv`, plots 생성 확인.
- 최종 선택 모델의 final-only test에서 `eval_test/metrics.json`, `eval_test/test_predictions.csv`, `eval_test/plots` 생성 확인.
- trainer의 `--resume`는 CLI compatibility 수준이며 실제 이어학습은 지원하지 않는다고 코드가 명시한다.

## 7. Full-Training Search Budget Actually Used

공통 설정:

- config: `configs/online_current/selected_fds_static4.yaml`
- runs dir: `runs/online_fms_current_tracking_0509_calib_deeptcn`
- GPU: CUDA
- max epochs: 80
- patience: 10
- batch size: 48
- seed: 42
- validation search 중 test evaluation 비활성화: `--no_test_eval`

실제 사용 budget:

| Run | Calibration encoder | Pooling | Transformer layers after DeepTCN | 실제 epochs | Best epoch | Trainable params |
|---|---|---|---:|---:|---:|---:|
| `calib_deep_tcn_mean_seed42` | DeepTCN | mean | 0 | 80 | 75 | 2,017,689 |
| `calib_deep_tcn_attention_seed42` | DeepTCN | attention | 0 | 80 | 75 | 2,017,786 |
| `calib_deep_tcn_transformer2_mean_seed42` | DeepTCN + Transformer | mean | 2 | 58 | 48 | 2,204,313 |

총 training epoch 사용량: 218 epochs.

추가로 validation 기준 최종 선택 모델 `calib_deep_tcn_mean_seed42`에 대해서만 final-only test evaluation을 1회 실행했다.

## 8. Validation Leaderboard

Validation selection metric은 MAE min이다. test는 search/selection에 사용하지 않았다.

| Rank | Label | Val MAE | Val RMSE | Val R2 | Acc <= 1.0 | Warning F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `deep_tcn_mean` | 1.753715 | 2.631096 | 0.657099 | 0.472037 | 0.747837 |
| 2 | `previous_best` | 1.922834 | 2.798767 | 0.612003 | 0.415648 | 0.704575 |
| 3 | `deep_tcn_attention` | 1.931219 | 2.720600 | 0.633373 | 0.406481 | 0.695896 |
| 4 | `deep_tcn_transformer2_mean` | 2.065792 | 2.921548 | 0.577213 | 0.379167 | 0.676176 |

해석:

- `deep_tcn_mean`이 기존 best를 validation MAE 기준으로 `1.922834 -> 1.753715`까지 개선했다.
- attention pooling은 mean보다 나빴다.
- DeepTCN 뒤에 Transformer 2층을 얹으면 오히려 크게 악화됐다.

## 9. Plot / Trajectory Result

Validation trend metric:

| Label | Pearson session mean | Centered MAE session mean | Delta corr 5s | Direction acc 5s | Flat range <25% session rate |
|---|---:|---:|---:|---:|---:|
| `deep_tcn_mean` | 0.488245 | 1.339468 | 0.409595 | 0.679517 | 0.000000 |
| `previous_best` | 0.468224 | 1.362749 | 0.463442 | 0.723072 | 0.050847 |
| `deep_tcn_attention` | 0.460767 | 1.365513 | 0.384116 | 0.683049 | 0.000000 |
| `deep_tcn_transformer2_mean` | 0.407636 | 1.420065 | 0.401070 | 0.705121 | 0.050847 |

Validation plot judgment sample:

| Label | Good | Medium | Bad | Total |
|---|---:|---:|---:|---:|
| `previous_best` | 4 | 1 | 7 | 12 |
| `deep_tcn_mean` | 4 | 0 | 8 | 12 |
| `deep_tcn_attention` | 4 | 0 | 8 | 12 |
| `deep_tcn_transformer2_mean` | 4 | 0 | 8 | 12 |

Test trend metric for final selected vs previous:

| Label | Test MAE | Test RMSE | Pearson session mean | Centered MAE session mean | Delta corr 5s | Direction acc 5s | Flat range <25% session rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `deep_tcn_mean` | 2.154288 | 2.886184 | 0.381830 | 1.260353 | 0.307483 | 0.666430 | 0.000000 |
| `previous_best` | 2.229539 | 3.005608 | 0.367449 | 1.267480 | 0.311418 | 0.690916 | 0.040000 |

Test plot judgment sample:

| Label | Good | Medium | Bad | Total |
|---|---:|---:|---:|---:|
| `previous_best` | 3 | 2 | 7 | 12 |
| `deep_tcn_mean` | 3 | 4 | 5 | 12 |

Plot 관점 결론:

- `deep_tcn_mean`은 level/regime, centered error, flat-range underfit 측면에서 좋아졌다.
- test plot sample에서는 bad가 7개에서 5개로 줄고 medium이 늘었다.
- 단, 5초 변화 방향성(`delta_corr_5s`, `direction_acc_5s`)은 기존 best가 더 낫거나 비슷하다.
- 따라서 DeepTCN calibration은 전반적인 FMS level/regime와 amplitude 보존에는 도움이 되지만, 아주 짧은 상승/하락 방향성까지 완전히 개선한 것은 아니다.

## 10. Final Selected Configuration

최종 선택 모델:

- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_mean_seed42`

핵심 설정:

- model: `online_fms_risk_tracker`
- calibration encoder: `deep_tcn`
- calibration dilation stages: `[1, 2, 4, 8, 16]`
- calibration pooling: `mean`
- Transformer after calibration DeepTCN: none
- stream context: `deep_tcn_latent_gru`
- state feedback: `none`
- motion feature mode: `causal_dynamics_v1`
- FDS blend: 0.75
- ordinal combine weight: 0.15
- static features: `age`, `mssq`, `gender`
- selection metric: validation MAE min

선택 이유:

- validation MAE/RMSE/R2, Acc<=1.0, Warning F1이 기존 best보다 모두 좋아졌다.
- final-only test에서도 MAE/RMSE/R2가 기존 best보다 좋아졌다.
- plot에서는 short-term direction metric 일부는 손해지만, test plot sample의 bad count가 줄었다.

## 11. Final Test-Set Metrics

최종 선택 모델 `deep_tcn_mean`:

| Metric | Value |
|---|---:|
| Test MAE | 2.154288 |
| Test RMSE | 2.886184 |
| Test R2 | 0.598408 |
| Acc <= 1.0 | 0.332158 |
| Warning F1 | 0.684145 |
| Rapid-rise-any F1 | 0.285373 |

기존 best와 비교:

| Model | Test MAE | Test RMSE | Test R2 | Acc <= 1.0 | Warning F1 | Rapid-rise-any F1 |
|---|---:|---:|---:|---:|---:|---:|
| `previous_best` | 2.229539 | 3.005608 | 0.564486 | 0.348825 | 0.689286 | 0.283066 |
| `deep_tcn_mean` | 2.154288 | 2.886184 | 0.598408 | 0.332158 | 0.684145 | 0.285373 |

해석:

- 핵심 회귀 지표는 개선됐다.
- `Acc <= 1.0`과 warning F1은 소폭 하락했다.
- rapid-rise-any F1은 소폭 상승했다.

## 12. Generated Plots / Tables

주요 생성물:

- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_mean_seed42/`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_attention_seed42/`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_transformer2_mean_seed42/`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_mean_seed42/eval_test/metrics.json`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/calib_deep_tcn_mean_seed42/eval_test/test_predictions.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_val/online_current_validation_leaderboard.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_val/plot_judgment_summary.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_val/plot_judgment_sessions.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_val/trend_metric_summary.png`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_val/trajectory_*.png`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_test/online_current_validation_leaderboard.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_test/plot_judgment_summary.csv`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_test/trend_metric_summary.png`
- `runs/online_fms_current_tracking_0509_calib_deeptcn/analysis_test/trajectory_*.png`

## 13. Git Status Summary

현재 tracked modified 파일:

- `AGENTS.md`
- `scripts/run_densefms_sanity_tests.py`
- `src/densefms_forecast/data.py`
- `src/densefms_forecast/evaluate.py`
- `src/densefms_forecast/losses.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`
- `src/densefms_forecast/utils.py`

현재 untracked 파일/디렉터리가 다수 있다:

- `configs/online_current/`
- `src/densefms_forecast/online_current/`
- `scripts/analyze_online_current_tracking.py`
- `scripts/run_online_current_*.py`
- `docs/codex/*.md`
- 논문 PDF/기존 report/기타 산출물

이번 작업에서 commit/push는 하지 않았다.

## 14. Remaining Issues or Warnings

- DeepTCN calibration은 명확히 유효한 개선이다. 특히 mean pooling이 가장 좋았다.
- DeepTCN 뒤에 Transformer를 얹는 것은 이번 조건에서는 손해였다.
- short-term direction metrics는 아직 약하다. 다음 개선은 calibration encoder가 아니라 decoder/state transition 쪽에서 5s/10s delta supervision 또는 transition-aware loss를 직접 강화하는 방향이 더 맞다.
- `Acc <= 1.0`, warning F1은 test에서 약간 하락했다. MAE/RMSE/R2를 우선할지 warning metric을 우선할지에 따라 후속 selection metric을 조정할 수 있다.
- PyTorch Transformer nested tensor warning은 남아 있지만 학습/평가 실패는 아니다.
