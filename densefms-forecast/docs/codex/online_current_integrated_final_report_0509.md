# Online Current Integrated Improvement Final Report - 2026-05-09

## 1. 수정/추가 파일

- `docs/codex/online_current_integrated_improvement_plan_0509.md`
  - 목표 문서 상단에 `FULL_TRAINING_ALLOWED = true` 추가.
- `src/densefms_forecast/model.py`
  - `motion_feature_mode=causal_dynamics_v1` 추가.
- `src/densefms_forecast/train.py`
  - 새 motion feature mode CLI 선택지 연결.
- `src/densefms_forecast/evaluate.py`
  - online current risk 평가에서 checkpoint/config의 future auxiliary horizon을 전달하도록 보완.
- `scripts/diagnose_online_current_motion_dynamics.py`
  - causal motion dynamics 진단 스크립트 추가.
- `scripts/run_online_current_integrated_improvement.py`
  - 통합 개선 후보 dry-run/execute runner 추가.
- `scripts/analyze_online_current_tracking.py`
  - 고정 validation plot set 기반 PLOT 프록시 요약 CSV 추가.
- `scripts/run_densefms_sanity_tests.py`
  - causal dynamics feature bank 및 integrated runner dry-run sanity 추가.
- `docs/codex/online_current_improvement_live_report.md`
  - dry-run, validation-only full training 진행 로그 기록.
- `docs/codex/online_current_integrated_final_report_0509.md`
  - 본 최종 보고서 추가.

## 2. 신규 CLI/config 옵션

- `--motion_feature_mode causal_dynamics_v1`
  - 기존 raw motion에 causal derivative/energy/complexity proxy를 추가하는 feature mode.
- `scripts/run_online_current_integrated_improvement.py`
  - `--phase`, `--execute`, `--trajectory_count`, `--python` 기반으로 후보 명령 생성 및 실행.
- 목표 문서 플래그
  - `FULL_TRAINING_ALLOWED = true`

## 3. Dataset/windowing 변경

- 학습/검증/최종 test 모두 기존 fixed split을 사용했다.
  - `runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42/split.json`
- 실제 사용 config:
  - `sampling_interval=0.5`
  - `max_session_points=420`
  - `calibration_seconds=120.0`
  - `recent_window_seconds=10.0`
  - `horizon_seconds=10.0`
  - static features: `age`, `mssq`, `gender(binary2)`
- leakage-safe 규칙은 유지했다.
  - calibration input은 calibration 구간 FMS만 사용.
  - recent motion은 current time 이하만 사용.
  - target은 미래/current target 위치로 shift.
  - test set은 validation 선택 완료 후 최종 1회 평가에만 사용.

## 4. Model 변경

- `causal_dynamics_v1` feature bank:
  - raw 6D motion에 accel/gyro/motion norm, causal delta/jerk norm, short/long causal rolling energy, energy ratio, sign-change rate, spectral proxy, channel energy entropy, participation ratio 등을 추가.
  - rolling feature는 현재 시점까지의 정보만 사용한다.
- 기존 baseline model과 config는 제거하지 않았다.
- 최종 선택 모델:
  - `online_fms_risk_tracker`
  - base config: `configs/online_current/selected_fds_static4.yaml`
  - override: `--motion_feature_mode causal_dynamics_v1`

## 5. Anchor/static/multi-horizon 지원 상태

- anchor policy:
  - 기존 online current risk sanity에서 calibration/recent anchor leakage 정책 통과.
- static features:
  - age/MSSQ/gender binary2 사용.
  - train/val/test 모두 static feature availability 정상 출력 확인.
- multi-horizon:
  - rapid-rise horizon 5s/10s 유지.
  - phase2 후보에서 future auxiliary horizon 5s/10s/15s 경로 학습/평가 sanity 통과.
  - 최종 선택 모델은 auxiliary head 후보가 아니라 `future_aux/delta_aux/event_aux` test metric은 비어 있음.

## 6. Sanity test 결과

- `py_compile` 통과:
  - `model.py`, `train.py`, `evaluate.py`
  - `run_densefms_sanity_tests.py`
  - `run_online_current_integrated_improvement.py`
  - `diagnose_online_current_motion_dynamics.py`
  - `analyze_online_current_tracking.py`
- 전체 lightweight sanity suite 통과:
  - import/check path
  - seconds-to-steps conversion
  - target shift correctness
  - calibration leakage check
  - recent-window leakage check
  - anchor policy check
  - model forward shape check
  - integrated sweep dry-run command generation
  - future/delta/event auxiliary paths
  - `causal_dynamics_v1` feature bank
- checkpoint/output 확인:
  - 선택 run에 `best.pt`, `final.pt`, `metrics.json`, `training_curves.csv`, `val_predictions.csv`, `eval_test/test_predictions.csv` 생성 확인.
- resume:
  - trainer의 `--resume`는 CLI 호환용 경고만 출력하며 실제 resume 학습은 지원하지 않음.
  - 중복 방지는 `--skip_existing` 경로로만 지원.

## 7. Full-training search budget

- full training 허용 근거:
  - 목표 문서에 `FULL_TRAINING_ALLOWED = true` 추가 후 진행.
- 실행 범위:
  - validation-only candidate training 9개.
  - 모든 candidate는 `--no_test_eval`로 학습 중 test 평가 생략.
- 실제 사용량:

| run | epochs | seconds | best epoch | best val MAE |
| --- | ---: | ---: | ---: | ---: |
| `integrated_p1_risk015_seed42` | 36 | 187.7 | 26 | 2.086855 |
| `integrated_p1_risk035_seed42` | 60 | 308.5 | 50 | 1.943569 |
| `integrated_p1_ordblend015_seed42` | 23 | 120.3 | 13 | 2.182090 |
| `integrated_p1_ordblend025_seed42` | 60 | 313.1 | 50 | 1.973466 |
| `integrated_p1_fdsblend075_seed42` | 37 | 193.0 | 27 | 2.119247 |
| `integrated_p2_future_delta_event_light_seed42` | 39 | 206.2 | 29 | 2.080329 |
| `integrated_p2_delta_only_light_seed42` | 39 | 207.8 | 29 | 2.079648 |
| `integrated_p2_trajectory_w003_d5_seed42` | 40 | 207.7 | 30 | 2.121088 |
| `integrated_p4_causal_dynamics_v1_seed42` | 48 | 250.3 | 38 | 1.930335 |

- total: 382 epochs, 1994.5 seconds, 약 33.24 minutes.

## 8. Validation leaderboard

선택은 validation metrics로만 수행했다. PLOT 프록시는 모든 후보가 동일하게 `good=4`, `medium=0`, `bad=8`이라 tie-breaker로 MAE/shape metrics를 사용했다.

| rank | label | val MAE | RMSE | session Pearson | centered MAE | delta corr 5s | direction acc 5s | flat rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `integrated_p4_causal_dynamics_v1_seed42` | 1.930335 | 2.837943 | 0.457169 | 1.372215 | 0.444166 | 0.725721 | 0.067797 |
| 2 | `integrated_p1_risk035_seed42` | 1.943569 | 2.786587 | 0.463472 | 1.372416 | 0.432055 | 0.700412 | 0.033898 |
| 3 | `fds_static4` | 1.945173 | 2.771334 | 0.462414 | 1.378370 | 0.433324 | 0.716892 | 0.033898 |
| 4 | `integrated_p1_ordblend025_seed42` | 1.973466 | 2.845643 | 0.454598 | 1.392693 | 0.447248 | 0.724250 | 0.016949 |
| 5 | `integrated_p2_delta_only_light_seed42` | 2.079648 | 2.958658 | 0.431384 | 1.412204 | 0.396430 | 0.728958 | 0.067797 |

## 9. Final selected configuration

- selected by validation only:
  - `integrated_p4_causal_dynamics_v1_seed42`
- selection reason:
  - lowest validation MAE: 1.930335
  - centered MAE roughly tied/best: 1.372215
  - 5s delta corr improved over baseline: 0.444166 vs 0.433324
  - 5s direction accuracy improved over baseline: 0.725721 vs 0.716892
  - PLOT 프록시는 baseline과 동일한 4/0/8이라 선택 기준에서 차이를 만들지 못함.
- caution:
  - RMSE와 session Pearson은 기존 baseline보다 낮지 않다.
  - flat-rate는 baseline보다 높다.

## 10. Final test-set metrics

test set은 위 configuration을 validation으로 확정한 뒤 최종 보고용으로만 평가했다.

- checkpoint:
  - `runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42/best.pt`
- output:
  - `runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42/eval_test/metrics.json`
  - `runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42/eval_test/test_predictions.csv`

| metric | value |
| --- | ---: |
| test MAE | 2.293558 |
| test RMSE | 3.084873 |
| test R2 | 0.541212 |
| within 1.0 | 0.337714 |
| within 2.0 | 0.562073 |
| caution high-FMS F1 | 0.820912 |
| warning high-FMS F1 | 0.685268 |
| ordinal accuracy | 0.194017 |
| ordinal off-by-one accuracy | 0.438889 |
| rapid-rise 5s AUROC | 0.753502 |
| rapid-rise 5s F1 | 0.237033 |
| rapid-rise 10s AUROC | 0.722684 |
| rapid-rise 10s F1 | 0.244153 |
| final warning F1 | 0.283230 |

## 11. Generated plots/tables

- validation leaderboard:
  - `runs/online_fms_current_tracking_0509_integrated/analysis/online_current_validation_leaderboard.csv`
  - `runs/online_fms_current_tracking_0509_integrated/analysis/online_current_validation_leaderboard.json`
- PLOT 프록시:
  - `runs/online_fms_current_tracking_0509_integrated/analysis/plot_judgment_summary.csv`
  - `runs/online_fms_current_tracking_0509_integrated/analysis/plot_judgment_sessions.csv`
- trajectory/scatter plots:
  - `runs/online_fms_current_tracking_0509_integrated/analysis/trajectory_*.png`
  - `runs/online_fms_current_tracking_0509_integrated/analysis/prediction_scatter_all.png`
  - `runs/online_fms_current_tracking_0509_integrated/analysis/trend_metric_summary.png`
- selected run plots:
  - `runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42/plots/`
  - `runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42/eval_test/plots/`
- motion diagnostic:
  - `runs/online_fms_current_tracking_0509_integrated/motion_dynamics_diagnostic/`

## 12. git status summary

- commit/push 하지 않았다.
- `runs/**` 산출물은 git 추적 대상에 추가하지 않았다.
- 작업 전부터 worktree에 다수의 modified/untracked 파일이 있었고 그대로 보존했다.
- 이 작업에서 의도적으로 수정/추가한 주요 파일은 1번 목록에 정리했다.

## 13. 남은 이슈/경고

- PLOT 결과는 자동 metric-derived proxy이며 사람의 시각적 판정이 아니다.
- 최종 선택 모델은 validation MAE와 일부 shape metric은 개선됐지만, validation RMSE/session Pearson/flat-rate는 baseline 대비 명확히 우월하지 않다.
- test MAE는 validation MAE보다 높다. test metric은 최종 보고용이며 후보 선택에 사용하지 않았다.
- `causal_dynamics_v1`는 단일 seed 결과다. 안정성 판단에는 seed 반복이 필요하다.
- `--resume`는 실제 resume 기능이 아니라 CLI compatibility placeholder다.
