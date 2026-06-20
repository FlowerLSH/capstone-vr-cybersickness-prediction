# LC-SA-TCNFormer 작업 보고서

작성 시점: 2026-05-02 KST
작업 범위: DenseFMS 기반 future FMS forecasting을 위한 LC-SA-TCNFormer 구현, leakage-safe windowing 보강, validation-only full search, adaptive diagnosis/search, 최종 validation-selected test evaluation

## 1. 작업 요약

이번 작업에서는 DenseFMS 시계열 데이터에서 미래 FMS를 예측하기 위한 `lc_sa_tcnformer` 모델을 추가하고, calibration 길이, recent window 길이, forecast horizon, anchor policy, static feature 사용 여부를 CLI/config에서 제어할 수 있도록 학습 및 평가 파이프라인을 확장했다.

검색과 선택은 validation metric만 사용했다. Test set은 최종 configuration이 validation 기준으로 선택된 뒤에만 평가했다.

최종 선택된 configuration은 adaptive 단계에서 나온 short-horizon 모델이다.

- run name: `final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta`
- calibration: `120.0s`
- recent window: `30.0s`
- horizon: `1.0s`
- anchor mode: `sparse_observed`
- anchor interval: `60.0s`
- static features: `age`, `gender`, `mssq`
- prediction mode: anchor delta prediction enabled
- validation MAE: `1.7570`
- final test MAE: `1.7076`

중요한 해석상 주의점은 최종 선택 모델이 `horizon_seconds=1.0`인 short-horizon 모델이라는 점이다. 이 결과를 5s, 10s, 15s, 20s, 30s horizon 결과와 같은 난이도의 task로 직접 비교하면 안 된다. 15s 배포형 후보 중 가장 좋은 모델은 `stage2_c120_w30_h15_sparse_observed_ai60_static_d64_l1_mean_delta`이며 validation MAE는 `2.1045`였다.

## 2. 수정 및 추가 파일

### 수정한 파일

- `src/densefms_forecast/data.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`
- `src/densefms_forecast/evaluate.py`
- `src/densefms_forecast/losses.py`
- `scripts/run_densefms_sanity_tests.py`

### 추가한 파일

- `configs/lc_sa_tcnformer.yaml`
- `scripts/run_lc_sa_tcnformer_full_search.py`
- `docs/codex/LC_SA_TCNFORMER_WORK_REPORT_KO.md`

## 3. 구현 내용

### 3.1 Dataset/windowing

`future_sequence_targets`, `future_sequence_times`, `current_sequence_times`에 `prediction_start_steps` 인자를 추가했다. 기존에는 prediction start가 calibration step 기준으로 고정되는 구조였지만, LC-SA-TCNFormer는 recent window가 충분히 확보된 시점부터 예측해야 하므로 `max(calibration_steps, recent_steps - 1)`를 prediction start로 사용할 수 있게 했다.

Leakage-safe 규칙은 다음 기준으로 맞췄다.

- calibration input은 첫 `calibration_steps`의 motion/FMS만 사용한다.
- calibration 이후 FMS는 calibration input으로 들어가지 않는다.
- recent motion은 current time `t` 이하만 사용한다.
- target은 `FMS[t + horizon_steps]`이다.
- target FMS는 input으로 사용하지 않는다.
- sparse/recent anchor는 current index `t` 이하의 FMS만 사용한다.
- `recent_start_observed`는 upper-bound 조건으로 표시한다.

### 3.2 Model

`src/densefms_forecast/model.py`에 `LCSATCNFormer`를 추가하고 model registry에 `lc_sa_tcnformer`를 등록했다.

모델 구조는 다음과 같다.

- calibration branch: motion + FMS 입력, causal/dilated TCN, Transformer encoder, pooling
- recent branch: motion-only 입력, causal TCN 기반 recent representation
- anchor encoder: `anchor_fms`, `time_since_anchor`
- static encoder: optional `age/gender/mssq`
- horizon encoder: single-horizon embedding
- fusion MLP: branch representations를 결합해 future FMS 예측

지원 anchor mode:

- `none`
- `calibration_end`
- `sparse_observed`
- `recent_start_observed`

`sparse_observed`와 `recent_start_observed`는 full FMS가 필요한 anchor mode로 표시해, 학습/평가에서만 full FMS가 전달되도록 했다. 그 외 모드는 calibration FMS까지만 전달한다.

Sparse anchor에서 scheduled anchor index의 FMS가 missing인 경우, current time 이하의 최신 finite FMS observation으로 fallback하도록 수정했다. 이 변경으로 sparse anchor 학습 중 NaN이 발생하던 문제를 제거했다.

### 3.3 Loss

`FutureSequenceLoss`에 `loss_type`을 추가했다.

- `smooth_l1`
- `mse`

또한 CLI/runner에서 쓰기 쉬운 `level_plus_trend` alias를 `level_trend_raw`와 호환되도록 처리했다.

### 3.4 Training/evaluation

`train.py` 주요 변경:

- `lc_sa_tcnformer` CLI 지원
- `--no_test_eval` 추가
- validation-only search 중 test evaluation 비활성화
- `--skip_existing` 추가
- `--save_predictions/--no-save_predictions`
- `--save_plots/--no-save_plots`
- `--loss_type`
- anchor/static/model hyperparameter CLI 추가
- prediction CSV에 anchor/time/static/common-window metadata 추가
- `prediction_start`를 사용한 target shift 계산
- all-val-MAE non-finite 상황에서 명시적 error 처리

`evaluate.py` 주요 변경:

- prediction CSV 저장
- final val/test evaluation 시 동일 metadata가 포함되도록 `collect_predictions` 인자 확장
- `run_dir` 초기화 순서 버그 수정

## 4. 검색 runner

`scripts/run_lc_sa_tcnformer_full_search.py`를 추가했다. 이 runner는 다음 흐름을 자동화한다.

1. hardware summary 저장
2. sanity tests 실행
3. Stage 1 validation-only coarse search
4. Stage 2 top validation configs refinement
5. validation leaderboard 생성
6. mediocre 조건 진단
7. adaptive diagnosis/report 생성
8. 최대 8개 adaptive experiments 실행
9. validation MAE 기준 final configuration 선택
10. final model 재학습
11. 최종 선택 후 val/test evaluation
12. final test metrics, plots, final report 생성

Search 학습 command에는 `--no_test_eval`이 들어가며, test evaluation은 `selected_config.json` 및 `final_training_spec.json` 생성 후 final evaluation 단계에서만 수행된다.

## 5. Sanity test 결과

실행 command:

```bash
/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe scripts/run_densefms_sanity_tests.py
```

결과: 전체 PASS.

확인 항목:

- import/compile check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- anchor policy check
- sparse anchor latest finite fallback
- LC-SA-TCNFormer forward shape/start index
- LC-SA target shift with `prediction_start`
- recent motion no-future-leakage
- dynamic dilation receptive field
- dry-run full-search command generation

추가로 다음 파일들에 대해 `py_compile`도 통과했다.

- `src/densefms_forecast/data.py`
- `src/densefms_forecast/model.py`
- `src/densefms_forecast/train.py`
- `src/densefms_forecast/evaluate.py`
- `src/densefms_forecast/losses.py`
- `scripts/run_densefms_sanity_tests.py`
- `scripts/run_lc_sa_tcnformer_full_search.py`

## 6. Full-training 검색 예산

Full training은 `docs/codex/LC_SA_TCNFORMER_FULL_SEARCH_GOAL.md`와 `AGENTS.md`에 `FULL_TRAINING_ALLOWED = true`가 있어 허용된 범위에서 실행했다.

Leaderboard 기준 완료 run 수:

- Stage 1: `17`
- Stage 2: `8`
- Adaptive: `8`
- Final: `1`
- 합계: `34`

Epoch 합계:

- Stage 1: `518`
- Stage 2: `391`
- Adaptive: `192`
- Final: `19`
- 합계: `1120`

Adaptive run은 정책 한도인 최대 `8`개를 넘지 않았다.

## 7. Validation leaderboard 요약

생성 파일:

- `runs/lc_sa_tcnformer_full_search/leaderboard_val.csv`
- `runs/lc_sa_tcnformer_full_search/leaderboard_val.md`

상위 주요 결과:

| 순위 | run_name | stage | horizon | anchor | val MAE | val RMSE | val R2 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta` | final | 1.0s | sparse_observed | 1.7570 | 2.9476 | 0.5916 |
| 2 | `adaptive_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta` | adaptive | 1.0s | sparse_observed | 1.7570 | 2.9476 | 0.5916 |
| 3 | `stage2_c120_w30_h15_sparse_observed_ai60_static_d64_l1_mean_delta` | stage2 | 15.0s | sparse_observed | 2.1045 | 2.9963 | 0.5695 |
| 4 | `adaptive_c120_w30_h15_sparse_observed_ai60_static_d64_l1_attention_delta` | adaptive | 15.0s | sparse_observed | 2.1134 | 3.0581 | 0.5516 |
| 5 | `stage2_c90_w30_h10_recent_start_observed_no_static_d64_l1_mean_delta` | stage2 | 10.0s | recent_start_observed | 2.1136 | 2.9782 | 0.5891 |

Horizon별 best validation MAE:

| horizon | best run | val MAE |
| ---: | --- | ---: |
| 1.0s | `final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta` | 1.7570 |
| 5.0s | `stage1_c90_w30_h5_calibration_end_no_static_d64_l1_mean` | 2.4202 |
| 10.0s | `stage2_c90_w30_h10_recent_start_observed_no_static_d64_l1_mean_delta` | 2.1136 |
| 15.0s | `stage2_c120_w30_h15_sparse_observed_ai60_static_d64_l1_mean_delta` | 2.1045 |
| 20.0s | `stage1_c90_w30_h20_calibration_end_no_static_d64_l1_mean` | 2.4294 |
| 30.0s | `stage1_c90_w30_h30_calibration_end_no_static_d64_l1_mean` | 2.4746 |

## 8. Adaptive diagnosis

생성 파일:

- `runs/lc_sa_tcnformer_full_search/adaptive_diagnosis.md`
- `runs/lc_sa_tcnformer_full_search/adaptive_diagnosis.json`
- `runs/lc_sa_tcnformer_full_search/adaptive_manifest.json`
- `runs/lc_sa_tcnformer_full_search/adaptive_report.md`

Adaptive stage가 trigger된 이유:

- best val MAE의 strongest baseline 대비 개선폭이 10% 미만이었다.
- no-anchor 모델이 anchor-based 모델보다 훨씬 나빴다.

Baseline 요약:

- global train mean MAE: `3.8258`
- calibration-end anchor MAE: `2.9250`
- sparse observed anchor MAE: `2.2272`
- recent-start observed upper-bound MAE: `2.7815`
- ridge small feature MAE: `2.7762`
- strongest baseline: sparse observed anchor, MAE `2.2272`

Adaptive 결과는 8개 고유 run으로 기록됐다. `level_plus_trend` 후보는 run name 충돌을 피하도록 runner를 수정해 별도 run으로 실행했다.

## 9. Final selected configuration

생성 파일:

- `runs/lc_sa_tcnformer_full_search/selected_config.json`
- `runs/lc_sa_tcnformer_full_search/final_training_spec.json`

최종 선택 configuration:

```text
run_name: final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta
model: lc_sa_tcnformer
calibration_seconds: 120.0
recent_window_seconds: 30.0
horizon_seconds: 1.0
anchor_mode: sparse_observed
anchor_interval_seconds: 60.0
use_static: true
static_features: age, gender, mssq
predict_delta_from_anchor: true
d_model: 64
transformer_layers: 1
pooling: mean
loss_type: smooth_l1
loss_mode: level_only
```

선택 기준은 validation MAE다. `recent_start_observed`는 upper-bound anchor 조건이므로 최종 deployment model로 선택하지 않았다.

## 10. Final test metrics

생성 파일:

- `runs/lc_sa_tcnformer_full_search/final_test_metrics.csv`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/eval_test/metrics.json`

최종 test metrics:

| metric | value |
| --- | ---: |
| test MAE | 1.7076 |
| test RMSE | 2.5656 |
| test R2 | 0.7173 |
| test sMAPE | 30.7486 |
| common test MAE | 1.6159 |
| common test RMSE | 2.4555 |

Test metrics는 최종 validation-based selection 이후에만 계산했다. 이 test 결과를 보고 모델 선택을 바꾸지 않았다.

## 11. 생성된 주요 산출물

검색 및 보고서:

- `runs/lc_sa_tcnformer_full_search/leaderboard_val.csv`
- `runs/lc_sa_tcnformer_full_search/leaderboard_val.md`
- `runs/lc_sa_tcnformer_full_search/final_test_metrics.csv`
- `runs/lc_sa_tcnformer_full_search/final_report.md`
- `runs/lc_sa_tcnformer_full_search/adaptive_report.md`
- `runs/lc_sa_tcnformer_full_search/adaptive_diagnosis.md`
- `runs/lc_sa_tcnformer_full_search/hardware_summary.json`

최종 prediction CSV:

- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/eval_val/val_predictions.csv`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/eval_test/test_predictions.csv`

최종 plots:

- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/plots/validation_leaderboard_bar.png`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/plots/val_predicted_vs_target.png`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/plots/val_residual_histogram.png`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/plots/test_predicted_vs_target.png`
- `runs/lc_sa_tcnformer_full_search/final_final_c120_w30_h1_sparse_observed_ai60_static_d64_l1_mean_delta/plots/test_residual_histogram.png`

추가 eval plots:

- `eval_val/plots/val_level_only_00.png` through `val_level_only_11.png`
- `eval_test/plots/test_level_only_00.png` through `test_level_only_11.png`

## 12. 작업 중 발견하고 수정한 문제

### 12.1 Recent branch 성능 문제

초기 LC recent branch는 각 prediction window를 unfold한 뒤 window별로 TCN을 다시 적용하는 구조라 epoch 시간이 매우 길었다. 이를 full sequence를 한 번 causal TCN으로 encode하고 rolling pooling하는 구조로 바꿔 epoch 시간을 크게 줄였다.

### 12.2 Sparse anchor NaN 문제

Sparse anchor에서 scheduled anchor FMS가 missing인 경우 NaN이 anchor encoder로 들어갈 수 있었다. current time 이하의 최신 finite FMS observation으로 fallback하도록 수정해 학습 NaN을 제거했다.

### 12.3 Final evaluation run_dir 버그

`evaluate.py`에서 `run_dir`를 `collect_predictions` 호출 전에 사용해 final evaluation이 실패했다. `run_dir` 초기화를 prediction collection 전에 수행하도록 수정했다.

### 12.4 Adaptive run name 충돌

Adaptive `level_plus_trend` 후보가 기존 `level_only` 후보와 같은 run name을 만들 수 있었다. non-default `loss_mode`를 run name에 포함하도록 수정해 8개 adaptive run이 모두 고유하게 기록되도록 했다.

## 13. Git 상태 요약

Commit/push는 하지 않았다.

현재 `git status --short` 기준 주요 변경:

- 수정됨: `scripts/run_densefms_sanity_tests.py`
- 수정됨: `src/densefms_forecast/data.py`
- 수정됨: `src/densefms_forecast/evaluate.py`
- 수정됨: `src/densefms_forecast/losses.py`
- 수정됨: `src/densefms_forecast/model.py`
- 수정됨: `src/densefms_forecast/train.py`
- 추가됨: `configs/lc_sa_tcnformer.yaml`
- 추가됨: `scripts/run_lc_sa_tcnformer_full_search.py`
- 추가됨: `docs/codex/LC_SA_TCNFORMER_WORK_REPORT_KO.md`

작업 시작 전부터 존재하던 unrelated dirty/untracked 파일도 남아 있다.

- `README_densefms_forecast.md`
- `configs/coff_lstm*.yaml`
- `AGENTS.md`
- `docs/codex/`
- `scripts/run_densefms_optimization.py`

이 파일들은 이번 작업에서 되돌리지 않았다.

## 14. 남은 이슈 및 권고

1. 최종 선택은 `horizon_seconds=1.0` short-horizon 모델이다. 장기 forecast 성능으로 일반화해서 보고하면 안 된다.
2. 15s 이상 horizon에서는 sparse FMS anchor 의존도가 크다. head/motion-only만으로 충분한 성능을 내는지는 아직 제한적이다.
3. Multi-horizon은 모델/손실 쪽 일부 기반은 있으나 prediction/evaluation path가 single-horizon 중심이라 이번 검색에서는 사용하지 않았다.
4. `recent_start_observed`는 deployment-realistic setting이 아니라 upper-bound ablation으로만 해석해야 한다.
5. Test set은 최종 선택 이후에만 평가했으며, test 결과로 configuration을 다시 선택하지 않았다.
