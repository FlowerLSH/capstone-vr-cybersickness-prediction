# PROVISIONAL_SELECTION_REPORT

작성 시각: 2026-05-08 13:25 KST

## 결론

`OVERNIGHT_ONLINE_CURRENT_FMS_GOAL_0508.md`의 final lock 조건을 만족하는 단일 deployable winner를 찾지 못했다. 따라서 `FINAL_SELECTION_LOCK.md`는 작성하지 않고, final test도 실행하지 않는다.

## Best Candidate So Far

목적별 provisional best는 다음과 같다.

| 목적 | run | validation MAE | 판단 |
|---|---|---:|---|
| MAE-first | `deep_tcn_latent_gru_420_large_calib240_lds_gamma05_seed42` | 1.943267 | MAE 최저지만 flatness와 delta 지표가 약함 |
| shape-balanced deployable | `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_no_fds_no_lds_seed42` | 1.952795 | MAE는 약간 높지만 session/delta/flatness 지표가 가장 균형적 |

## Why Not Locked

- MAE 최저 run은 shape/trajectory 지표가 좋지 않다.
  - flat_range_lt25pct_session_rate: 0.254237
  - delta_corr_5s: 0.271848
  - direction_acc_5s: 0.682166
- shape-balanced run은 MAE 최저 run보다 약 0.49% 높다.
- 새로 구현한 후보들은 current-FMS validation MAE를 reference보다 개선하지 못했다.
- Calibration-FiLM seed42는 일부 shape/RMSE 지표가 좋았지만 seed123 split-fixed 재학습에서 MAE 2.115939로 불안정했다.
- validation plot의 80% trend-following 목표를 충족한다고 말할 수 있는 수동 plot evidence가 없다. 기존 FDS/static4 calibration count도 pass 5 / partial 1 / fail 6이다.

## Evidence Missing

- validation trajectory plot에 대한 pass/partial/fail 수동 분류가 새 후보별로 필요하다.
- recovery/drop segment에 대한 explicit event-weighted 진단이 부족하다.
- seed stability가 확인된 near-winner가 없다.
- privileged teacher가 deployable student representation을 개선할 수 있는지 아직 검증하지 않았다.

## Dominant Failure Mode

- low-FMS recovery/drop 구간에서 과대예측이 남아 있다.
- 일부 MAE-best 후보는 central/flat trajectory로 수렴하면서 shape 지표가 나빠진다.
- auxiliary future/delta/event supervision은 direction_acc를 올릴 수 있지만 current-FMS MAE를 훼손했다.
- calibration/static 정보가 motion-to-FMS transition을 안정적으로 개인화하기에는 아직 부족하다.

## Tried And Rejected

| run | validation MAE | reject reason |
|---|---:|---|
| `future_delta_event_seed42` | 2.121242 | direction_acc 개선 대비 MAE 악화가 큼 |
| `decoder_tcn2_seed42` | 2.167417 | MAE/shape 모두 reference보다 약함 |
| `future_delta_event_light_seed42` | 2.073373 | auxiliary weight 축소 후에도 MAE gap 큼 |
| `calib_film_seed42` | 1.957564 | 근접하지만 seed stability 부족 |
| `calib_film_seed123_split42` | 2.115939 | seed stability 실패 |
| `fds_calib_film_seed42` | 2.085593 | FDS/static4 대비 악화 |
| `ordmix010_seed42` | 1.980109 | ordinal mix 축소 단독 개선 없음 |

## Next Recommended Experiments

1. Recovery/drop event 가중 loss:
   - low-to-high보다 high-to-low/recovery miss를 더 강하게 벌점화.
   - selection은 계속 validation MAE primary, shape metrics secondary.

2. Privileged teacher distillation:
   - `start_only` upper-bound teacher는 teacher-only로만 사용.
   - student inference는 calibration FMS + current/past head motion만 유지.
   - representation distillation 또는 delta/event logits distillation부터 시작.

3. Calibration-conditioned transition 재시도:
   - 단순 FiLM은 seed stability가 약했다.
   - GRU gate bias/adaptor처럼 더 직접적인 state-transition conditioning을 작게 테스트.

4. Plot classification protocol:
   - `final_validation_analysis/trajectory_*`에서 후보별 pass/partial/fail을 수동 기록.
   - 80% trend-following에 못 미치면 final test를 계속 보류.

## Test Policy

이번 provisional 결정에서는 final test를 실행하지 않았다. 기존 run directory에 과거 `eval_test` 산출물이 있는 경우에도 이번 selection evidence에는 사용하지 않았다.
