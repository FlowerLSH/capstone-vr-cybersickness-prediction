# Online Current Calibration Transformer 실험 보고서 - 2026-05-09

## 1. Modified or Added Files

- `src/densefms_forecast/model.py`
  - `OnlineFMSRiskTracker`에 `calibration_encoder_mode`를 추가했다.
  - 지원 모드: `tcn_transformer`(기존), `transformer`, `transformer_cls`.
  - `transformer_cls`는 calibration sequence 앞에 learnable CLS token을 붙이고 해당 token representation을 calibration summary로 쓴다.
  - `transformer`/`transformer_cls`에서는 TCN stem 대신 linear projection + learned positional embedding + Transformer encoder stack을 쓴다.
- `src/densefms_forecast/train.py`
  - CLI/config override에 `--calibration_encoder_mode`를 추가했다.
  - checkpoint/model kwargs에 calibration encoder mode가 저장되도록 연결했다.
- `scripts/run_densefms_sanity_tests.py`
  - `test_online_current_calibration_transformer_modes`를 추가했다.
  - transformer calibration mode의 forward shape, metadata, post-calibration FMS leakage 방지, calibration-only FMS 입력 일관성을 검사한다.
- `docs/codex/online_current_calibration_transformer_final_report_0509.md`
  - 본 보고서.

참고: 현재 worktree에는 이전 실험에서 생긴 다른 수정/미추적 파일도 남아 있다. 이번 calibration-Transformer 실험에서 핵심적으로 추가한 파일은 위 목록이다.

## 2. New CLI / Config Options

- `--calibration_encoder_mode {tcn_transformer,transformer,transformer_cls}`
  - `tcn_transformer`: 기존 calibration branch. TCN stem + Transformer encoder.
  - `transformer`: TCN stem 없이 calibration sequence를 linear projection 후 Transformer encoder stack으로 처리하고 기존 pooling을 적용.
  - `transformer_cls`: TCN stem 없이 learnable CLS token + positional embedding + Transformer encoder stack을 적용하고 CLS representation을 summary로 사용.
- 기존 옵션 `--transformer_layers`, `--pooling`과 조합해 calibration branch depth/pooling을 바꿨다.

## 3. Dataset / Windowing Changes

- 데이터셋/windowing 코드는 이번 실험에서 변경하지 않았다.
- 사용 데이터: `DenseFMS/Dataset`
- split: 기존 validation-best run의 `split.json` 재사용.
  - train/val/test session count: 316 / 60 / 52
  - participant group split 유지.
  - 이전 best test 평가에 기록된 split 파일과 현재 split 파일의 JSON 내용은 동일함을 확인했다.
- sampling interval: 0.5s
- max session points: 420
- calibration: 120s = 240 steps
- recent window: 10s
- current-FMS task horizon/rise auxiliary:
  - current FMS tracking task.
  - rapid-rise auxiliary: 5s, 10s.
- static features: `age`, `mssq`, `gender` with 4D static vector.
- validation search 중 `--no_test_eval`을 사용했고, test는 validation 선택 이후에만 final-report-only로 평가했다.

## 4. Model Changes

이번 질문의 핵심인 “TCN + Transformer 대신 Transformer 층을 쌓는 방식”을 다음 세 후보로 비교했다.

| Label | Calibration encoder | Transformer layers | Pooling / summary | Notes |
|---|---:|---:|---|---|
| `previous_best` | 기존 default `tcn_transformer` | 2 | mean | 기존 validation-best 기준선 |
| `tcn_transformer4_attention` | `tcn_transformer` | 4 | attention | 기존 TCN stem 유지, Transformer depth만 증가 |
| `transformer4_attention` | `transformer` | 4 | attention | TCN stem 제거, pure Transformer stack |
| `transformer4_cls_summary` | `transformer_cls` | 4 | CLS summary | TCN stem 제거, CLS token summary |

결론: pure Transformer stack은 동작했고 leakage-safe sanity도 통과했지만, validation MAE와 trajectory trend 지표 기준으로 기존 TCN+Transformer best를 넘지 못했다.

## 5. Anchor / Static / Multi-Horizon Support Status

- Anchor/FMS policy: 변경 없음. post-calibration FMS가 calibration input에 들어가지 않도록 유지했다.
- Static support: 유지. 이번 실험은 `age`, `mssq`, `gender` static feature를 사용했다.
- Multi-horizon support: 기존 future auxiliary/rise auxiliary 경로는 유지. 이번 calibration encoder ablation 자체는 current-FMS tracking 모델에서 수행했다.
- Current head architecture 4종(`trajectory_decoder`, `regime_gated`, `state_space_delta`, `range_scaled_delta`)은 이전 실험에서 구현/검증/학습까지 완료했고, 이번 추가 실험은 calibration branch encoder 구조만 바꿔 본 것이다.

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
  - 새 `test_online_current_calibration_transformer_modes`
- 경고:
  - PyTorch `enable_nested_tensor`/`norm_first` 관련 warning이 출력됐지만 실행 실패는 아니었다.

Full-training artifact 확인:

- `best.pt`, `final.pt`, `metrics.json`, `training_curves.csv`, `val_predictions.csv`, plots 생성 확인.
- 새 후보 best의 final-only test 평가에서 `eval_test/metrics.json`, `eval_test/test_predictions.csv`, `eval_test/plots` 생성 확인.
- `--resume`는 CLI compatibility용으로만 받아들이며 실제 이어학습은 지원하지 않는다고 trainer가 경고한다. 따라서 resume 검증 대상은 아니었다.

## 7. Full-Training Search Budget Actually Used

Full training은 사용자 직접 허락에 따라 진행했다.

공통 설정:

- config: `configs/online_current/selected_fds_static4.yaml`
- split file: `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42/split.json`
- runs dir: `runs/online_fms_current_tracking_0509_calib_transformer`
- GPU: NVIDIA GeForce RTX 4070
- max epochs: 80
- patience: 10
- batch size: 48
- seed: 42
- validation search 중 test evaluation 비활성화: `--no_test_eval`

실제 사용 budget:

| Run | Max epochs | 실제 epochs | Best epoch | Trainable params |
|---|---:|---:|---:|---:|
| `calib_tcn_transformer4_attention_seed42` | 80 | 27 | 17 | 2,437,210 |
| `calib_transformer4_attention_seed42` | 80 | 35 | 25 | 2,126,170 |
| `calib_transformer4_cls_summary_seed42` | 80 | 75 | 65 | 2,126,265 |

총 training epoch 사용량: 137 epochs.

추가로 validation 기준 후보군 내부 best인 `calib_transformer4_cls_summary_seed42`에 대해서만 final-only test evaluation을 1회 실행했다.

## 8. Validation Leaderboard

Validation selection metric은 MAE min이다. test는 search/selection에 사용하지 않았다.

| Rank | Label | Val MAE | Val RMSE | Val R2 | Acc <= 1.0 | Warning F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `previous_best` | 1.922834 | 2.798767 | 0.612003 | 0.415648 | 0.704575 |
| 2 | `transformer4_cls_summary` | 2.053563 | 3.113267 | 0.519904 | 0.418611 | 0.706983 |
| 3 | `transformer4_attention` | 2.133747 | 3.128161 | 0.515299 | 0.352315 | 0.700124 |
| 4 | `tcn_transformer4_attention` | 2.157230 | 3.134886 | 0.513213 | 0.351204 | 0.738476 |

해석:

- calibration branch를 deeper Transformer로 바꾼 후보들은 기존 validation-best를 넘지 못했다.
- `transformer4_cls_summary`가 새 후보 중 가장 좋았다.
- `tcn_transformer4_attention`은 warning F1은 높았지만, current-FMS MAE/RMSE가 크게 나빠졌다.

## 9. Plot / Trajectory Result

분석 출력:

- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/online_current_validation_leaderboard.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/plot_judgment_summary.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/plot_judgment_sessions.csv`
- trajectory PNG 12개와 scatter/trend summary PNG 생성.

Trend metric summary:

| Label | Pearson session mean | Centered MAE session mean | Delta corr 5s | Direction acc 5s | Flat range <25% session rate |
|---|---:|---:|---:|---:|---:|
| `previous_best` | 0.468224 | 1.362749 | 0.463442 | 0.723072 | 0.050847 |
| `transformer4_cls_summary` | 0.388216 | 1.442747 | 0.344457 | 0.687758 | 0.084746 |
| `transformer4_attention` | 0.443510 | 1.417546 | 0.388392 | 0.733373 | 0.067797 |
| `tcn_transformer4_attention` | 0.464479 | 1.404219 | 0.364140 | 0.721307 | 0.101695 |

Plot judgment sample summary:

| Label | Good | Medium | Bad | Total |
|---|---:|---:|---:|---:|
| `previous_best` | 4 | 0 | 8 | 12 |
| `tcn_transformer4_attention` | 4 | 0 | 8 | 12 |
| `transformer4_attention` | 4 | 0 | 8 | 12 |
| `transformer4_cls_summary` | 4 | 0 | 8 | 12 |

Plot 관점 결론:

- deeper/pure Transformer calibration branch가 bad trajectory case 수를 줄이지 못했다.
- `transformer4_attention`은 5s direction accuracy만 기준선보다 약간 높았지만, MAE/RMSE/centered MAE/delta correlation은 기준선보다 나빴다.
- `transformer4_cls_summary`는 pointwise MAE는 새 후보 중 최고였지만 shape/trend 지표가 기준선보다 떨어졌다.
- 따라서 plot 개선 목적으로 calibration encoder만 pure Transformer stack으로 바꾸는 것은 현재 근거가 약하다.

## 10. Final Selected Configuration

Validation 기준 최종 선택은 변경하지 않는다.

선택 모델:

- `runs/online_fms_current_tracking_0509_param_search/psearch_causal_dyn_fds075_ord015_seed42`
- 기존 calibration encoder default: `tcn_transformer`
- Transformer layers: 2
- pooling: mean
- stream context: `deep_tcn_latent_gru`
- state feedback: `none`
- motion features: `causal_dynamics_v1`
- FDS blend: 0.75
- ordinal combine weight: 0.15
- static features: `age`, `mssq`, `gender`

선택 이유:

- validation MAE 1.922834로 이번 calibration-Transformer 후보 전체보다 낮다.
- trajectory trend 지표도 전체적으로 더 낫다.
- test는 selection 이후 final-report-only로만 참고했다.

## 11. Final Test-Set Metrics

최종 선택 모델(`previous_best`) test metrics:

| Metric | Value |
|---|---:|
| Test MAE | 2.229539 |
| Test RMSE | 3.005608 |
| Test R2 | 0.564486 |
| Acc <= 1.0 | 0.348825 |
| Warning F1 | 0.689286 |
| Rapid-rise-any F1 | 0.283066 |

새 calibration 후보군 내부 best(`transformer4_cls_summary`)도 validation 기준으로 선택한 뒤 final-only test를 1회 확인했다. 이는 전체 최종 선택 모델이 아니다.

| Metric | `transformer4_cls_summary` test |
|---|---:|
| Test MAE | 2.335336 |
| Test RMSE | 3.048219 |
| Test R2 | 0.552050 |
| Acc <= 1.0 | 0.315278 |
| Warning F1 | 0.665812 |
| Rapid-rise-any F1 | 0.287655 |

해석:

- 새 후보는 rapid-rise-any F1만 아주 소폭 높았고, 핵심 current-FMS MAE/RMSE/R2 및 warning F1은 최종 선택 모델보다 나빴다.

## 12. Generated Plots / Tables

주요 생성물:

- `runs/online_fms_current_tracking_0509_calib_transformer/calib_tcn_transformer4_attention_seed42/`
- `runs/online_fms_current_tracking_0509_calib_transformer/calib_transformer4_attention_seed42/`
- `runs/online_fms_current_tracking_0509_calib_transformer/calib_transformer4_cls_summary_seed42/`
- `runs/online_fms_current_tracking_0509_calib_transformer/calib_transformer4_cls_summary_seed42/eval_test/metrics.json`
- `runs/online_fms_current_tracking_0509_calib_transformer/calib_transformer4_cls_summary_seed42/eval_test/test_predictions.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/online_current_validation_leaderboard.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/online_current_validation_leaderboard.json`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/plot_judgment_summary.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/plot_judgment_sessions.csv`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/trend_metric_summary.png`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/prediction_scatter_all.png`
- `runs/online_fms_current_tracking_0509_calib_transformer/analysis_val/trajectory_*.png`

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

- calibration encoder를 pure Transformer stack으로 바꾸는 것만으로는 plot bad case가 줄지 않았다.
- 현재 hard case의 핵심 문제는 calibration branch 표현력 부족만이 아니라, post-calibration online latent dynamics/transition supervision/rare high-FMS regime 처리 쪽일 가능성이 더 크다.
- 기존 기준선보다 direction accuracy가 일부 좋아지는 후보는 있었지만, MAE/RMSE와 shape metrics가 같이 나빠졌다. 따라서 지금 단계에서 이 방향을 mainline으로 채택할 근거는 부족하다.
- PyTorch Transformer nested-tensor warning은 남아 있지만 학습/평가 실패는 아니다.
- `--resume`는 실제 checkpoint resume이 아니라 CLI compatibility 수준이다.
