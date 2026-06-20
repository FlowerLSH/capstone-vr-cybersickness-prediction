# Online Current Architecture Shift Final Report 0509

## 요약

`state_t` 업데이트는 기존 `OnlineFMSRiskTracker`의 `deep_tcn_latent_gru` 경로에 이미 있었다. 이번 실험은 state를 새로 추가한 것이 아니라, 같은 causal state 뒤의 current-FMS decoder를 네 가지 구조로 바꿔 full training validation search를 수행했다.

테스트는 validation으로 고른 뒤 final-report-only로만 실행했다. 새 구조 4개 중 validation 1등은 `state_space_delta + predicted_current feedback`이었지만, 전체 비교에서는 기존 best `psearch_causal_dyn_fds075_ord015_seed42`가 아직 validation MAE 1등이다.

## 1. Modified or Added Files

- `src/densefms_forecast/online_current/heads.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`
- `scripts/run_densefms_sanity_tests.py`
- `docs/codex/online_current_architecture_shift_final_report_0509.md`

## 2. New CLI or Config Options

- `--current_head_mode trajectory_decoder`
- `--current_head_mode regime_gated`
- `--current_head_mode state_space_delta`
- `--current_head_mode range_scaled_delta`
- `--current_trajectory_offsets ...`
- `--trajectory_decoder_loss_weight ...`

## 3. Dataset and Windowing Changes

- Dataset split은 기존 selected run split을 그대로 사용했다.
- `max_session_points=420`, `sampling_interval=0.5`, calibration 120s, recent window 10s 유지.
- calibration input, recent motion input, target shift, anchor policy는 변경하지 않았다.
- test set은 search에 사용하지 않았다.

## 4. Model Changes

- `trajectory_decoder`: decoder가 `t, t+5s, t+10s, t+20s` FMS를 동시에 예측하고 `t` offset을 current prediction으로 사용한다.
- `regime_gated`: regime gate logits와 expert values를 만들고 softmax gate로 expert 출력을 혼합한다. regime loss는 gate logits에 직접 걸 수 있게 했다.
- `state_space_delta`: `base_fms`에서 시작해 drive, leak, equilibrium으로 current trajectory를 recurrent하게 누적한다.
- `range_scaled_delta`: person/session range scale을 calibration embedding에서 예측하고, 누적 signed delta와 direct level을 gate로 섞는다.

## 5. Anchor, Static, Multi-Horizon Support

- Anchor policy: 변경 없음. online-current task는 calibration FMS만 입력으로 사용.
- Static features: 기존 `age, mssq, gender` 4D binary2 static 그대로 사용.
- Multi-horizon: rapid-rise auxiliary horizon 5s/10s 유지. current head 자체는 단일 current 출력이고, trajectory decoder만 보조 offset을 추가 예측한다.

## 6. Sanity Test Results

통과:

- `py_compile`: `heads.py`, `model.py`, `train.py`, `run_densefms_sanity_tests.py`
- `scripts/run_densefms_sanity_tests.py`
- 포함 검증: seconds-to-steps, target shift, calibration leakage, recent-window leakage, anchor policy, model forward shape, trajectory decoder loss path, dry-run command generation.

경고:

- PyTorch transformer nested tensor warning만 발생. 실패 아님.

## 7. Full-Training Search Budget Used

공통 조건:

- GPU: CUDA available, RTX 4070 확인.
- 후보당 최대 80 epochs, patience 10, batch size 48, seed 42.
- validation-only training, test skipped during search.

실제 사용:

| run | best epoch | epochs run |
|---|---:|---:|
| trajectory_decoder | 35 | 45 |
| regime_gated | 38 | 48 |
| state_space_feedback | 50 | 60 |
| range_scaled_delta | 25 | 35 |

## 8. Validation Leaderboard

| rank | run | val MAE | val RMSE | val R2 | centered MAE | 5s delta corr | 5s direction acc | plot |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | previous_best | 1.922834 | 2.798767 | 0.612003 | 1.362749 | 0.463442 | 0.723072 | 4 good / 0 medium / 8 bad |
| 2 | state_space_feedback | 1.940143 | 2.741209 | 0.627797 | 1.395249 | 0.391259 | 0.702766 | 4 / 0 / 8 |
| 3 | range_scaled_delta | 1.985598 | 2.900706 | 0.583224 | 1.456786 | 0.396291 | 0.709829 | 4 / 0 / 8 |
| 4 | regime_gated | 2.007997 | 2.793799 | 0.613379 | 1.382132 | 0.416592 | 0.705415 | 4 / 0 / 8 |
| 5 | trajectory_decoder | 2.072357 | 2.944740 | 0.570474 | 1.400208 | 0.439278 | 0.733667 | 4 / 0 / 8 |

## 9. Final Selected Configuration

전체 선택 기준은 validation MAE이므로 기존 best 유지:

- `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42`

새 구조 4개 안에서의 대표는:

- `runs/online_fms_current_tracking_0509_arch_shift/arch_state_space_delta_feedback_seed42`
- 이유: 새 구조 중 validation MAE 최저, RMSE와 R2도 가장 좋음.

## 10. Final Test-Set Metrics

기존 전체 best final test:

- MAE 2.229539
- RMSE 3.005608
- R2 0.564486

새 구조 대표 `state_space_feedback` final test:

- MAE 2.284592
- RMSE 3.012905
- R2 0.562369

해석: 새 구조 대표는 validation에서 기존 best에 근접했지만 test MAE도 기존 best보다 낮지 않았다. 따라서 현재 결과만으로는 모델 구조 교체를 채택할 근거가 부족하다.

## 11. Generated Plots and Tables

- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/online_current_validation_leaderboard.csv`
- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/plot_judgment_summary.csv`
- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/plot_judgment_sessions.csv`
- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/trajectory_*.png`
- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/prediction_scatter_all.png`
- `runs/online_fms_current_tracking_0509_arch_shift/analysis_val/trend_metric_summary.png`
- `runs/online_fms_current_tracking_0509_arch_shift/arch_state_space_delta_feedback_seed42/eval_test/metrics.json`
- `runs/online_fms_current_tracking_0509_arch_shift/arch_state_space_delta_feedback_seed42/eval_test/test_predictions.csv`

## 12. Git Status Summary

작업 전부터 repository에는 많은 modified/untracked 파일이 있었다. 이번 작업에서 건드린 핵심 파일은 위 1번 목록이다. commit/push는 하지 않았다.

## 13. Remaining Issues or Warnings

- 네 구조 모두 hard-case plot 총평은 4 good / 0 medium / 8 bad로 기존 best와 같았다.
- state-space feedback은 flat prediction 비율을 줄이고 range ratio는 좋아졌지만, 5s delta correlation과 direction accuracy는 기존 best보다 낮았다.
- 단순한 head 교체만으로 plot hard case가 해결되지는 않았다. 특히 wrong-regime, large-centered-error가 남아 있어 calibration/person prior 또는 session-level regime identification 쪽 병목이 더 커 보인다.
