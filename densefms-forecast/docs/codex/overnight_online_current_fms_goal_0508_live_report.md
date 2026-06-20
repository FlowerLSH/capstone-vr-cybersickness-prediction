# Overnight Online Current FMS Goal 0508 라이브 리포트

작성 시각: 2026-05-08 13:25 KST

## 상단 요약

| 항목 | 현재 판단 |
|---|---|
| 현재 best deployable run(순수 MAE) | `deep_tcn_latent_gru_420_large_calib240_lds_gamma05_seed42` |
| best validation MAE | 1.943267 |
| best shape/trajectory 후보 | `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_no_fds_no_lds_seed42` |
| 새로 가장 근접한 후보 | `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_calib_film_seed42` |
| 주요 failure mode | MAE와 trajectory shape가 분리됨. MAE-best는 flat-range failure가 높고, shape-best는 MAE가 약간 밀림. |
| 다음 권장 실험 | recovery/drop event 가중 loss 또는 privileged teacher distillation, 단 student inference는 calibration-only 유지 |
| final test 실행 여부 | 실행하지 않음 |
| 현재 상태 | provisional |

## 구현 및 검증 요약

- `future_aux_horizon_seconds` 기반 +5s/+10s/+15s future FMS, delta, rise/drop/plateau event 보조 target/loss/metrics/CSV 수집을 연결했다.
- optional shallow causal decoder TCN을 추가했다.
- calibration embedding으로 DeepTCN latent feature를 FiLM 방식으로 조절하는 `stream_calib_condition_mode=film`을 추가했다.
- 모든 신규 strict 후보는 calibration FMS window와 현재/과거 head motion만 사용한다. post-calibration real FMS anchor는 사용하지 않았다.
- test set은 이번 선택/검색에 사용하지 않았다. 일부 기존 run directory에 과거 `eval_test` 산출물이 있었지만, 이번 판단/leaderboard에는 `val_predictions.csv`만 사용했다.

## 검증 결과

| 체크 | 결과 |
|---|---|
| import check | 통과: torch 2.6.0+cu124, CUDA 사용 가능 |
| seconds-to-steps | sanity suite 통과 |
| target shift | current/rise/future/delta/event target alignment 통과 |
| calibration leakage | sanity suite 통과 |
| recent-window leakage | sanity suite 통과 |
| anchor policy | strict online tracker는 post-calibration anchor 미노출 |
| model forward shape | current/risk/ordinal/future/event/decoder TCN/FiLM shape 통과 |
| prediction CSV 생성 | smoke 및 full validation run에서 `val_predictions.csv` 생성 |
| checkpoint/metrics 생성 | full run별 `best.pt`, `metrics.json`, `training_curves.csv` 생성 |
| sweep dry-run | `run_online_current_long_search.py --dry_run`: pending 0 |

## 실험별 기록

### 1. future/delta/event auxiliary

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_future_delta_event_seed42`
- config: `configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_future_delta_event.yaml`
- hypothesis: multi-horizon future/delta/event supervision이 trajectory direction encoding을 강화할 수 있다.
- validation: MAE 2.121242, RMSE 2.987511, delta_corr_5s 0.409425, direction_acc_5s 0.748381.
- decision: reject as final. Direction accuracy는 올랐지만 MAE가 크게 악화되고 flatness도 악화됐다.

### 2. shallow causal decoder TCN

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_decoder_tcn2_seed42`
- config: `configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_decoder_tcn2.yaml`
- hypothesis: output-side causal temporal context가 trajectory shape를 부드럽게 맞출 수 있다.
- validation: MAE 2.167417, RMSE 3.020075, delta_corr_5s 0.398886, flat_range_lt25pct 0.101695.
- decision: reject. MAE와 shape 지표 모두 reference보다 약하다.

### 3. future/delta/event light weights

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_future_delta_event_light_seed42`
- config: `configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_future_delta_event_light.yaml`
- hypothesis: auxiliary weight를 낮추면 direction supervision을 유지하면서 current MAE 악화를 줄일 수 있다.
- validation: MAE 2.073373, RMSE 2.914740, delta_corr_5s 0.435054, direction_acc_5s 0.728370.
- decision: reject/refine. heavy보다 낫지만 still MAE gap이 크다.

### 4. calibration-conditioned FiLM transition

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_calib_film_seed42`
- config: `configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_calib_film.yaml`
- hypothesis: calibration embedding이 motion-to-state transition을 조절하면 subject/session bias와 delta behavior가 개선될 수 있다.
- validation: MAE 1.957564, RMSE 2.725296, pearson_session_mean 0.479211, delta_corr_5s 0.457486, flat_range_lt25pct 0.016949.
- decision: investigate but not lock. RMSE와 delta_corr는 좋지만 MAE는 best/reference보다 낮지 않고 seed123 재학습에서 불안정했다.

### 5. calibration-FiLM seed stability

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_calib_film_seed123_split42`
- config: same as calibration-FiLM, split fixed to seed42 reference split.
- validation: MAE 2.115939, RMSE 2.997301.
- decision: reject as stable winner. seed stability가 약하다.

### 6. FDS/LDS + calibration-FiLM

- run: `deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_calib_film_seed42`
- config: `configs/online_fms_current_tracker_deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_calib_film.yaml`
- validation: MAE 2.085593, RMSE 2.979036.
- decision: reject. FDS/static4 기준보다 크게 악화됐다.

### 7. ordinal mix 0.10

- run: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_no_fds_no_lds_ordmix010_seed42`
- config: no-FDS/static4 config with CLI override `--fms_combine_weight_ordinal 0.10`.
- validation: MAE 1.980109, RMSE 2.739334, centered_mae 1.349908.
- decision: reject. ordinal mix 축소만으로는 MAE/shape 동시 개선이 없다.

## 최종 validation leaderboard

| label | val MAE | RMSE | session Pearson | centered MAE | delta_corr_5s | direction_acc_5s | flat_range<25% |
|---|---:|---:|---:|---:|---:|---:|---:|
| lds_gamma05 | 1.943267 | 2.977258 | 0.420627 | 1.444262 | 0.271848 | 0.682166 | 0.254237 |
| fds_static4 | 1.945173 | 2.771334 | 0.462414 | 1.378370 | 0.433324 | 0.716892 | 0.033898 |
| no_fds_no_lds | 1.952795 | 2.745454 | 0.484028 | 1.339728 | 0.454214 | 0.728370 | 0.016949 |
| calib_film_s42 | 1.957564 | 2.725296 | 0.479211 | 1.379114 | 0.457486 | 0.685697 | 0.016949 |
| ordmix010 | 1.980109 | 2.739334 | 0.466601 | 1.349908 | 0.427170 | 0.728664 | 0.033898 |
| future_delta_event_light | 2.073373 | 2.914740 | 0.429641 | 1.393433 | 0.435054 | 0.728370 | 0.033898 |
| fds_calib_film | 2.085593 | 2.979036 | 0.428776 | 1.416976 | 0.393589 | 0.723661 | 0.067797 |
| calib_film_s123 | 2.115939 | 2.997301 | 0.444450 | 1.396344 | 0.405191 | 0.724250 | 0.016949 |
| future_delta_event | 2.121242 | 2.987511 | 0.435382 | 1.399589 | 0.409425 | 0.748381 | 0.084746 |
| decoder_tcn2 | 2.167417 | 3.020075 | 0.445287 | 1.402897 | 0.398886 | 0.682460 | 0.101695 |

## 해석

- MAE만 보면 `lds_gamma05`가 1.943267로 가장 낮다. 하지만 flat-range failure가 25.4%이고 delta_corr_5s가 0.272라 trajectory model로는 약하다.
- `fds_static4`는 MAE가 거의 동일하면서 shape 지표가 훨씬 안정적이다.
- `no_fds_no_lds`는 MAE가 0.49% 정도 나쁘지만 session Pearson, centered MAE, delta_corr, direction_acc, flatness가 가장 균형적이다.
- `calib_film_s42`는 RMSE와 delta_corr가 좋았지만 seed123에서 성능이 무너져 final lock에 충분하지 않다.
- future/delta/event heads는 event/direction 쪽 일부 signal은 제공하지만 current MAE 손실이 너무 크다.
- 현재 evidence로는 80% trajectory trend-following 목표를 달성했다고 말할 수 없다. 목표 파일에 기록된 기존 FDS/static4 calibration plot count도 pass 5 / partial 1 / fail 6으로 80%에 못 미친다.

## Provisional 결정

최종 선택 lock은 작성하지 않는다. test는 실행하지 않는다.

provisional best는 목적에 따라 둘로 나뉜다.

- MAE-first: `deep_tcn_latent_gru_420_large_calib240_lds_gamma05_seed42`
- deployable shape-balanced: `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_no_fds_no_lds_seed42`

다음 실험은 recovery/drop event 가중 loss 또는 privileged `start_only` teacher의 representation/distillation을 권장한다. 단 student inference boundary는 calibration-only로 유지해야 한다.
