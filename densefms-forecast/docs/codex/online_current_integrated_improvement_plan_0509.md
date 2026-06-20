# Online Current FMS Integrated Improvement Plan - 2026-05-09

FULL_TRAINING_ALLOWED = true

## 기준 모델

현재 기준 모델:

`runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42`

Canonical config:

`configs/online_current/selected_fds_static4.yaml`

핵심 구성:

- `stream_context_mode`: `deep_tcn_latent_gru`
- `max_session_points`: 420
- `calibration_seconds`: 120.0 / 240 steps
- `recent_window_seconds`: 10.0
- `horizon_seconds`: 10.0
- `decoder_context_mode`: `state`
- static features: `age`, `mssq`, `gender_male`, `gender_female`
- current head: basic regression + cumulative ordinal blend
- `fms_combine_weight_ordinal`: 0.20
- `risk_head_enabled`: true
- `risk_loss_weight`: 0.25
- LDS enabled, `lds_gamma`: 0.5
- FDS enabled

기준 validation 성능:

| model | MAE | RMSE | session Pearson | centered MAE | delta corr 5s | direction acc 5s | flat rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected FDS/static4/risk-aux | 1.9452 | 2.7713 | 0.4624 | 1.3784 | 0.4333 | 0.7169 | 0.0339 |

## 평가 및 실험 원칙

- 동일 split, 동일 `max_session_points=420`을 우선 사용한다.
- Model selection은 validation metric과 validation plot으로만 한다.
- Test metric은 최종 선택 후 final report 용도로만 평가한다.
- 기존 기준선과 같은 fixed validation plot protocol을 사용한다.
- Smoke-test metric을 실제 성능으로 해석하지 않는다.
- Full training이 허용된 경우에도 실험 범위를 명확히 기록한다.
- 리소스 여유가 있으면 OOM이 나지 않는 선에서 병렬 학습을 허용하되, memory pressure가 보이면 즉시 concurrency를 낮춘다.
- 언제든 실험을 중단, 재개, 보고할 수 있도록 결과 보고서는 마지막에 한 번에 쓰지 않고 매 run 종료 시점마다 갱신한다.

필수 sanity:

- import check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- anchor policy check
- model forward shape check
- dry-run sweep command generation if sweep scripts are modified

## Change isolation rule

Unless explicitly justified in the live report, each run should change only one experimental factor family at a time.

Examples:

- risk_loss_weight sweep only
- ordinal blend sweep only
- future/delta/event auxiliary bundle only
- causal dynamics feature bank only

Do not combine a new architecture, new feature bank, new loss, and changed hyperparameters in the same first trial.

If a combined run is needed, first establish the individual component evidence.

## Live reporting rule

실험 보고서는 append-only 방식으로 관리한다. 각 training run 또는 diagnostic run이 끝날 때마다 즉시 live report를 갱신한다.

권장 live report 경로:

`docs/codex/online_current_improvement_live_report.md`

각 run이 끝날 때마다 기록할 내용:

- run name
- 실행 시각
- 사용 config와 CLI override
- 실험 목적
- 기대 효과
- 변경된 모델/데이터/손실 구성
- sanity check 결과
- 학습 budget: epoch 수, early stopping 여부, best epoch, 실제 소요 시간
- validation 주요 지표: MAE, RMSE, session Pearson, centered MAE, delta corr 5s, direction acc 5s, flat rate
- PLOT 판정: 좋음/보통/나쁨 개수와 주요 fail/pass 관찰
- 기준 모델 대비 판단: keep, reject, rerun-needed, or promote-later
- 생성 산출물 경로: checkpoint, metrics, predictions, plots, leaderboard
- 경고 또는 blocker

중단/재개 규칙:

- 사용자가 중단을 요청하면 새 run 시작을 멈추고, 이미 완료된 run까지의 live report를 기준으로 현재 best 후보와 남은 후보를 요약한다.
- 재개할 때는 live report에서 마지막 completed run과 pending run을 확인하고 이어서 진행한다.
- test set 평가는 live report 중간 업데이트에 포함하지 않는다. test set은 validation으로 최종 선택한 뒤 final report 용도로만 평가한다.

## 성능 향상 판단 기준

성능 향상은 PLOT 개선을 primary로 판단하고, MAE는 guardrail로 사용한다.

기준 모델의 fixed validation plot 분포:

- 좋음: 4 / 12
- 보통: 1 / 12
- 나쁨: 7 / 12

기준 validation MAE:

- 1.9452

PLOT 개선에서 가장 중요한 것은 `좋음` 개수를 바로 늘리는 것이 아니라, 명확한 실패인 `나쁨` plot을 `보통` 이상으로 끌어올리는 것이다. 현재 기준 모델은 반대 방향 움직임, plateau 구간의 false U-shape, high/low regime mismatch, 큰 rise/drop miss가 많기 때문에 `나쁨` 감소 자체가 의미 있는 개선이다.

PLOT 판정:

- PLOT 개선 후보: `나쁨 <= 5`
- PLOT 확실한 개선: `나쁨 <= 4` and `좋음 + 보통 >= 8`
- PLOT 강한 개선: `나쁨 <= 3` and `좋음 + 보통 >= 9`
- 좋음 증가 보조 기준: `좋음 >= 6`
- PLOT 악화: `나쁨 >= 8` 또는 기존 좋음 plot 중 2개 이상이 보통/나쁨으로 하락

MAE guardrail:

- MAE 안전권: `validation MAE <= 1.975`
- MAE 조건부 허용권 A: `1.975 < validation MAE <= 2.10`, 단 PLOT 확실한 개선이면 `promote-later` 가능
- MAE 조건부 허용권 B: `2.10 < validation MAE <= 2.20`, 단 PLOT 강한 개선일 때만 `promote-later` 가능. 이 경우 current-FMS 성능 개선 모델이 아니라 trajectory-shape candidate로 별도 표시한다.
- MAE reject권: `validation MAE > 2.20`

최종 후보 우선순위:

1. PLOT 개선 후보 이상 + `validation MAE <= 1.975`
2. `validation MAE <= 1.915` + PLOT 악화 없음
3. PLOT 확실한 개선 + `1.975 < validation MAE <= 2.10`: `promote-later`
4. PLOT 강한 개선 + `2.10 < validation MAE <= 2.20`: trajectory-shape candidate로 별도 표시

모델 비교 시 우선순위는 `나쁨 plot 감소 -> 기존 좋음 plot 유지 -> 보통 이상 plot 증가 -> 좋음 plot 증가 -> MAE 개선` 순서로 본다.

## 기존 코드로 바로 가능한 실험

### Risk Auxiliary Weight

관찰:

- `risk_loss_weight=0.25`가 현재 기준선이다.
- `risk_head_enabled=false`, `risk_loss_weight=0.0`은 validation MAE와 shape metric이 나빠졌다.
- `risk_head_enabled=true`, `risk_loss_weight=0.0`도 나빠졌다.

해석:

- risk head 모듈 자체보다 rapid-rise auxiliary loss가 shared representation을 regularize하는 효과가 있는 것으로 보인다.

후보:

- `risk_loss_weight`: 0.15, 0.35

주의:

- risk metric을 primary selection으로 쓰지 않는다.
- current FMS와 trajectory plot 품질을 같이 본다.

### Ordinal / Regression Blend

현재:

- `fms_combine_weight_ordinal=0.20`
- 최종 current FMS는 regression output과 ordinal-derived value를 blend한다.

후보:

- `fms_combine_weight_ordinal`: 0.15, 0.25

주의:

- ordinal 비중이 너무 높으면 bin 중심으로 뭉개질 수 있다.
- flat rate와 plot lag를 확인한다.

### FDS 강도 완화

현재:

- FDS enabled, `fds_blend=1.0`

후보:

- `fds_blend`: 0.75

주의:

- FDS가 너무 강하면 평균적인 trajectory로 끌릴 수 있으므로 완화값을 확인한다.

### Future / Delta / Event Auxiliary

현재 코드에는 future auxiliary, delta auxiliary, event auxiliary 경로가 이미 있다.

후보:

- `future_aux_horizon_seconds`: 5.0, 10.0, 15.0
- `future_aux_loss_weight`: 0.05
- `delta_aux_loss_weight`: 0.05 또는 0.10
- `event_aux_loss_weight`: 0.03
- `event_delta_threshold`: 1.0

의도:

- current FMS 한 점 예측을 넘어 near-future trajectory 정보를 latent state에 학습시킨다.
- 현재 risk auxiliary가 도움이 된 것처럼, future/rise/drop 신호가 shared representation을 개선할 수 있다.

주의:

- future target은 label로만 사용하고 input으로 들어가면 안 된다.
- target은 첫 420 step 안에 있어야 한다.

### 약한 Trajectory-Aware Loss

이전 관찰:

- DILATE-lite와 transition-weighted loss는 강하게 넣었을 때 MAE/RMSE를 악화시켰다.

재시도 방향:

- 강한 대체 loss가 아니라 약한 auxiliary로만 사용한다.
- current MAE path는 유지한다.

후보:

- `trajectory_loss_weight`: 0.03 또는 0.05
- `trajectory_delta_seconds`: 5.0, 10.0
- `trajectory_delta_weight`: 1.0
- `trajectory_centered_weight`: 0.3
- `trajectory_range_weight`: 0.1

주의:

- smoothness를 과하게 주면 그럴듯하지만 반응이 늦은 plot이 된다.

## 구조 개선 후보

### 1. Future Trajectory Auxiliary Head

가장 먼저 시도할 구조 개선 후보.

제안:

- current FMS head는 유지한다.
- 같은 latent state에서 짧은 future FMS trajectory를 auxiliary로 예측한다.

예측 대상:

- `FMS_t`
- `FMS_{t+5s}`
- `FMS_{t+10s}`
- `FMS_{t+15s}`

또는 delta 형태:

- `FMS_{t+5s} - FMS_t`
- `FMS_{t+10s} - FMS_t`
- `FMS_{t+15s} - FMS_t`

기대:

- latent state가 near-future rise/drop 정보를 담게 된다.
- plot lag와 direction miss를 줄일 수 있다.

주의:

- future label은 supervision으로만 사용한다.
- current prediction path는 유지한다.

### 2. Predicted-State Residual Update Head

문제의식:

- hidden state에서 매 시점 FMS 값을 독립적으로 뽑으면 trajectory continuity가 약해질 수 있다.
- FMS는 독립적인 점 예측이라기보다 누적 sickness state의 관측값에 가깝다.

제안:

```text
pred_fms_t = pred_fms_{t-1} + bounded_delta_t
bounded_delta_t = delta_head(hidden_t, motion_context_t)
```

기대:

- local rise/drop 변화가 더 자연스러워질 수 있다.
- sudden jump와 excessive flatness를 동시에 줄일 가능성이 있다.
- plot continuity가 좋아질 수 있다.

주의:

- `delta_from_calibration`처럼 calibration FMS에 계속 묶는 방식은 피한다.
- post-calibration 실제 FMS를 recurrent input으로 쓰면 안 된다.
- recurrent update는 모델의 이전 prediction 또는 latent state만 사용한다.
- delta scale이 너무 작으면 반응이 늦고, 너무 크면 plot이 noisy해질 수 있다.

### 3. Explicit Latent Sickness State

문제의식:

- 현재 hidden state가 motion representation, person/session prior, sickness state를 모두 암묵적으로 담당한다.
- risk auxiliary가 도움이 되는 이유도 구조적으로 명확히 드러나지 않는다.

제안:

```text
calibration encoder -> person/session prior p
motion encoder -> motion feature m_t
latent sickness state z_t = update(z_{t-1}, m_t, p)
z_t -> current FMS
z_t -> rapid-rise risk
z_t -> future trajectory auxiliary
```

기대:

- current FMS, risk, future trajectory가 같은 sickness state를 공유한다.
- risk auxiliary가 단순 side head가 아니라 latent sickness state를 학습시키는 신호가 된다.
- plot에서 변화 감지와 state continuity가 좋아질 수 있다.

주의:

- 변경 폭이 크므로 future trajectory auxiliary와 event auxiliary에서 먼저 신호를 확인한 뒤 진행한다.
- 기존 checkpoint compatibility와 config flag를 명확히 유지한다.

### 4. Causal Motion Dynamics Feature Bank

문헌 기반으로 가장 즉시 적용 가능한 추가 후보.

구성:

- raw 6D motion
- velocity norm
- acceleration norm
- jerk norm
- rolling motion energy
- rolling jerk energy
- sign-change rate
- short-window spectral proxy
- rolling covariance/eigenvalue complexity

모델 적용 방식:

- 우선 기존 raw sequence에 feature를 append한다.
- branch fusion은 이후 단계로 미룬다.

기대:

- head-motion-only 상황에서 derivative/statistical/temporal feature를 모델이 더 쉽게 사용한다.
- rise/drop timing과 high-FMS 구간 반응이 개선될 가능성이 있다.

주의:

- 모든 feature는 causal해야 한다.
- centered window나 future motion을 사용하면 안 된다.

### 5. Motion Complexity Diagnostic 및 Feature

문헌 힌트:

- VR motion tracking에서 cybersickness/discomfort가 높을수록 movement complexity가 감소하는 경향이 보고되었다.

후보:

- causal rolling PCA complexity proxy
- covariance eigenvalue entropy
- participation ratio
- d95-like explained variance feature
- complexity drop score

진행 순서:

1. 먼저 offline diagnostic으로 FMS level, future delta, rapid-rise label과의 관계를 확인한다.
2. 의미 있는 proxy만 feature bank에 포함한다.

주의:

- PCA/statistics는 train split 기준 normalization만 사용한다.
- 각 시점 feature는 과거 window로만 계산한다.

### 6. Multi-Timescale Motion Response

문제의식:

- 멀미는 motion에 즉시 반응하지 않고 누적 및 지연 반응을 보인다.

제안:

- 5s / 15s / 30s / 60s causal motion summary
- short/long energy ratio
- recent jerk burst score
- complexity drop score
- 필요하면 person prior가 timescale gate를 조절

기대:

- response lag를 더 잘 잡을 수 있다.
- session/person별 response speed 차이를 반영할 수 있다.

주의:

- 모든 summary는 current time t 이하만 사용한다.
- receptive field와 actual input window의 의미를 명확히 기록한다.

### 7. Rise/Fall/Plateau Event Auxiliary

관찰:

- risk auxiliary를 제거하면 current FMS validation 성능도 나빠졌다.
- 변화 이벤트 학습이 current FMS representation에 도움을 줄 가능성이 있다.

제안:

- rapid-rise binary만 쓰지 말고 trajectory event를 직접 예측한다.

후보 label:

- rise 시작 여부
- fall 시작 여부
- plateau 여부
- local slope sign
- absolute delta magnitude bin

주의:

- event label은 target window에서만 계산하고 input으로 들어가면 안 된다.
- class imbalance가 클 수 있으므로 loss weight는 작게 시작한다.

### 8. Person Prior as Bias / Scale / Response-Speed

문제의식:

- 개인차는 크고, static/calibration 정보는 단순 concat보다 susceptibility prior로 쓰는 편이 자연스럽다.

제안:

```text
calibration + static -> person prior p
p -> baseline bias
p -> amplitude scale
p -> response speed gate
online motion state -> dynamic sickness signal
prediction = bias + scale * dynamic_signal
```

기대:

- MSSQ와 calibration FMS trajectory를 더 의미 있게 사용할 수 있다.
- 같은 motion에도 민감한 사람과 둔감한 사람을 다르게 반응시킬 수 있다.

주의:

- calibration FMS에 직접 delta로 묶지 않는다.
- static shortcut/overfit 여부를 확인한다.

### 9. Coarse Severity Band Auxiliary

제안:

- continuous FMS regression은 유지한다.
- 기존 cumulative ordinal head 외에 coarse severity band head를 추가한다.

후보:

- 데이터 분포 기반 low / mid / high band
- high-FMS underprediction 완화용 coarse risk band

주의:

- FMS-20을 FMS-10 banding으로 단순 변환하지 않는다.
- threshold가 임의적이면 실험 해석이 흐려진다.

### 10. Self-Supervised Motion Pretraining

제안:

- FMS label 없이 head motion sequence만으로 pretraining한다.

후보 objective:

- masked motion reconstruction
- next-window motion prediction
- contrastive predictive coding
- future motion energy prediction

주의:

- 구현 비용이 크다.
- pretraining objective가 sickness와 무관하면 효과가 약할 수 있다.
- 구조 개선 1-2개를 먼저 본 뒤 시도한다.

### 11. Content / Scenario Prior

제안:

- VR scene, optical flow type, density, motion condition 같은 metadata가 안정적으로 있으면 session/content embedding으로 넣는다.

주의:

- deployment에서 사용 불가능한 metadata면 기본 경로에 넣지 않는다.
- content shortcut으로 participant response를 덮지 않는지 확인한다.

### 12. Uncertainty / Mixture-of-Regimes Head

제안:

- single mean prediction 대신 uncertainty 또는 regime mixture를 예측한다.

후보 regime:

- stable low
- slow rise
- rapid rise
- plateau high
- recovery/fall

주의:

- MAE를 직접 개선하지 않을 수 있다.
- mixture collapse를 방지해야 한다.
- mean-regression으로 계속 뭉개지는 문제가 확인될 때 검토한다.

## 권장 우선순위

1. Future trajectory auxiliary head
2. Causal motion dynamics feature bank
3. Motion complexity diagnostic, 이후 complexity feature
4. Multi-timescale motion response
5. Rise/fall/plateau event auxiliary
6. Person prior as bias/scale/response-speed
7. Predicted-state residual update head
8. Explicit latent sickness state
9. Coarse severity band auxiliary
10. Self-supervised motion pretraining
11. Content/scenario prior, metadata가 있을 때만
12. Uncertainty 또는 mixture-of-regimes head

## 권장 진행 흐름

### Phase 0. 기준선 고정

- `selected_fds_static4.yaml`을 기준 config로 유지한다.
- 기준 validation leaderboard와 fixed plot set을 고정한다.
- risk-head ablation 결과를 registry에 남겨둔다.

### Phase 1. 최소 파라미터 확인

목표:

- 현재 기준선 주변의 핵심 loss/blend 민감도만 확인한다.

후보:

- `risk_loss_weight`: 0.15, 0.35
- `fms_combine_weight_ordinal`: 0.15, 0.25
- `fds_blend`: 0.75

### Phase 2. 기존 Auxiliary 경로 활용

목표:

- 구현 부담 없이 future/delta/event/trajectory auxiliary가 도움이 되는지 확인한다.

후보:

- future + delta + event light
- delta-only light
- weak trajectory loss

### Phase 3. Motion Dynamics Diagnostic

목표:

- velocity, acceleration, jerk, energy, complexity feature가 FMS level/delta/rise와 관련 있는지 먼저 확인한다.

산출물:

- feature correlation table
- high/low FMS feature distribution
- rise event 전후 motion dynamics plot
- complexity-vs-FMS diagnostic plot

### Phase 4. Causal Dynamics Feature Bank

목표:

- diagnostic에서 의미 있었던 feature만 causal하게 추가한다.

방식:

- `motion_feature_mode=causal_dynamics_v1`
- raw append 방식으로 시작한다.
- branch fusion은 다음 단계로 미룬다.

### Phase 5. Multi-Timescale / Event / Person Prior

순서:

1. multi-timescale motion response
2. rise/fall/plateau event auxiliary
3. person prior as bias/scale/response-speed

이 단계들은 Phase 2-4에서 신호가 나온 뒤 진행한다.

### Phase 6. 최종 분석

- 이번 개선 탐색에서는 여러 seed 반복을 수행하지 않는다.
- 모든 실험은 기본적으로 `seed=42` 기준으로 비교한다.
- 좋은 후보가 충분히 모이면, 나중에 별도 안정성 검증 단계에서만 여러 seed 반복을 수행한다.
- 최종 선택은 validation 기준으로만 한다.
- test set은 최종 선택 이후 final report 용도로만 평가한다.

## Reject 기준

아래 중 하나라도 해당하면 개선 후보로 보지 않는다.

- Validation 기준으로 선택 근거가 없다.
- Validation MAE가 좋아졌지만 fixed plot이 명확히 나빠졌다.
- Direction metric은 좋아졌지만 RMSE/centered MAE가 크게 나빠졌다.
- Target plateau에서 큰 false rise/drop을 만든다.
- 주요 rise/drop timing이 일관되게 늦거나 반대로 움직인다.
- high-FMS 또는 low-FMS regime을 크게 잘못 예측한다.
- leakage-safe windowing 검증을 통과하지 못한다.

## 참고 문헌 링크

- Islam et al. 2022, ISMAR, forecasting onset with multimodal deep fusion: https://doi.org/10.1109/ISMAR55827.2022.00026
- Islam et al. 2021, ISMAR, integrated HMD sensor deep fusion: https://doi.org/10.1109/ISMAR52148.2021.00017
- Islam et al. 2021 arXiv preprint: https://arxiv.org/abs/2108.06437
- Salehi et al. 2024, head movement patterns: https://arxiv.org/abs/2402.02725
- IEEE VR 2025 motion complexity paper: https://doi.org/10.1109/VR59515.2025.00040
- Setu et al. 2024, Mazed and Confused dataset: https://arxiv.org/abs/2409.06898
- Kim et al. 2021, clinical predictors: https://www.nature.com/articles/s41598-021-91573-w
- Kelly et al. 2026 OSF preprint, FMS/SSQ tolerance banding: https://doi.org/10.31234/osf.io/p3xy4_v1
