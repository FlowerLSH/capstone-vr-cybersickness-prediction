# Online Current FMS Improvement Plan - 2026-05-09

## 기준 모델

현재 계속 볼 기준 모델은 아래 run이다.

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

이 모델은 MAE 단독 최저 모델은 아니지만, MAE-best보다 RMSE, trajectory correlation, flatness가 훨씬 안정적이어서 기준선으로 유지한다.

## 성능 향상의 정의

이 프로젝트에서 "성능 향상"은 둘 중 하나 이상을 만족해야 한다.

1. Validation MAE가 기준선 대비 유의미하게 낮아진다.
2. MAE가 비슷하더라도 validation trajectory plot에서 멀미 변화의 상승, 하강, plateau, lag를 더 잘 따라간다.

실험 선택은 validation 기준으로만 한다. Test set은 최종 선택 이후 final report 용도로만 평가한다.

권장 판정 기준:

- MAE 개선: validation MAE가 최소 0.03 이상 좋아지면 후보, 0.05 이상 좋아지면 강한 후보로 본다.
- Shape 개선: MAE가 기준선 대비 0.03 이내로 유지되면서 centered MAE, delta corr, direction acc, flat rate 중 2개 이상이 좋아지면 후보로 본다.
- Plot 개선: 고정 validation session plot에서 pass/partial/fail을 수동 기록한다. 단순히 예측이 더 smooth해진 것은 개선으로 보지 않는다.
- Seed 안정성: 최종 후보는 최소 3개 seed에서 반복 확인한다.

## 평가 프로토콜

모든 실험은 아래 원칙을 따른다.

- 동일 split, 동일 `max_session_points=420`을 우선 사용한다.
- Model selection은 validation metric과 validation plot으로만 한다.
- Test metric은 최종 선택 후 1회 보고한다.
- 기존 기준선과 같은 plot protocol을 사용한다.
- Smoke-test metric을 실제 성능으로 해석하지 않는다.
- Full training이 허용된 경우에도 좁은 실험 범위를 유지한다.
- 리소스 여유가 있으면 OOM이 나지 않는 선에서 병렬 학습을 허용하되, memory pressure가 보이면 즉시 concurrency를 낮춘다.

필수 sanity:

- import check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- anchor policy check
- model forward shape check
- sweep command dry-run if sweep script is changed

## 파라미터 튜닝 후보

아래 항목들은 구조를 크게 바꾸지 않고 기준 모델의 균형점을 찾는 실험이다. "큰 아이디어"라기보다는 선택 모델의 민감도를 확인하는 용도다.

### 1. Risk Auxiliary Weight Sweep

관찰:

- `risk_loss_weight=0.25`가 현재 기준선이다.
- `risk_head_enabled=false`, `risk_loss_weight=0.0`은 validation MAE와 shape metric이 나빠졌다.
- `risk_head_enabled=true`, `risk_loss_weight=0.0`도 나빠졌다.

해석:

- risk head 모듈 자체보다 rapid-rise auxiliary loss가 shared representation을 regularize하는 효과가 있는 것으로 보인다.

시도값:

- `risk_loss_weight`: 0.10, 0.15, 0.25, 0.35, 0.50

기대:

- 0.25 근처에서 MAE와 trajectory shape의 균형점이 있을 가능성이 높다.
- 0.10 또는 0.15가 MAE를 조금 개선하면서 plot 품질을 유지할 수 있다.

주의:

- risk metric을 primary selection으로 쓰지 않는다.
- current FMS MAE와 plot 품질을 같이 본다.

### 2. Ordinal/Regression Blend 조정

현재:

- `fms_combine_weight_ordinal=0.20`
- 최종 current FMS는 regression output과 ordinal-derived value를 blend한다.

시도값:

- `fms_combine_weight_ordinal`: 0.10, 0.15, 0.20, 0.25, 0.30

기대:

- ordinal 비중이 낮아지면 세밀한 회귀 MAE가 좋아질 수 있다.
- ordinal 비중이 높아지면 high-FMS 구간과 단계적 변화가 안정될 수 있다.

주의:

- 너무 높이면 예측이 bin 중심으로 뭉개질 수 있다.
- flat rate와 plot lag를 반드시 확인한다.

### 3. LDS/FDS 강도 조정

현재:

- LDS enabled, `lds_gamma=0.5`
- FDS enabled, `fds_blend=1.0`

시도값:

- `lds_gamma`: 0.3, 0.5, 0.7
- `fds_blend`: 0.5, 0.75, 1.0
- `fds_start_smooth`: 2, 4, 6

기대:

- tail FMS와 high-FMS 구간 반응을 개선할 수 있다.
- FDS가 너무 강하면 평균적인 trajectory로 끌릴 수 있으므로 완화값도 본다.

주의:

- MAE만 좋아지고 rise/drop이 둔해지는 경우는 reject한다.
- high-FMS 구간 MAE, delta corr, trajectory plot을 같이 본다.

### 4. 약한 Trajectory-Aware Loss

이전 관찰:

- DILATE-lite와 transition-weighted loss는 강하게 넣었을 때 MAE/RMSE를 악화시켰다.

재시도 방향:

- 강한 대체 loss가 아니라 약한 auxiliary로만 사용한다.
- current MAE를 primary로 유지한다.

후보:

- 약한 delta consistency loss
- 약한 centered trajectory loss
- 약한 rise direction auxiliary
- 매우 작은 smoothness regularization

기대:

- plot에서 상승/하강 방향성이 좋아질 수 있다.

주의:

- smoothness를 과하게 주면 그럴듯하지만 반응이 늦은 plot이 된다.
- selection은 MAE leaderboard와 shape leaderboard를 분리해서 본다.

### 5. Recent Window / Receptive Field 조정

현재:

- `recent_window_seconds=10.0`
- deep TCN dilation `[1, 2, 4, 8, 16]`
- 출력 로그상 deep TCN receptive field는 62.5초 수준이다.

시도값:

- `recent_window_seconds`: 15.0, 20.0
- `deep_tcn_dilations`: `[1, 2, 4, 8]`
- `deep_tcn_dilations`: `[1, 2, 4, 8, 16]`
- `deep_tcn_dilations`: `[1, 2, 4, 8, 16, 32]`

기대:

- motion response lag와 누적 motion effect를 더 잘 잡을 수 있다.

주의:

- recent motion은 current time t까지만 사용해야 한다.
- recent window나 dilation을 바꾸면 leakage sanity를 다시 확인한다.

### 6. Seed 반복

현재 후보 간 차이가 작은 경우가 많다. MAE 0.02에서 0.05 수준 차이는 seed noise일 수 있다.

권장 seed:

- 42
- 7
- 123

최종 후보는 평균뿐 아니라 seed별 rank 안정성을 본다.

## 구조 개선 후보

아래 항목들은 단순 튜닝보다 더 큰 개선 가능성이 있는 방향이다. 실패 리스크는 높지만, MAE와 plot quality를 동시에 개선할 가능성이 있다.

### 1. Future Trajectory Auxiliary Head

가장 먼저 시도할 구조 개선 후보.

현재 문제:

- 모델은 dense current FMS를 잘 맞추려 하지만, plot에서 중요한 것은 "지금 상승 중인지, 하강 중인지, plateau인지"다.
- 한 점 예측만으로는 latent state가 미래 변화 방향을 충분히 담지 않을 수 있다.

제안:

- current FMS head는 유지한다.
- 같은 latent state에서 짧은 future FMS trajectory를 auxiliary로 예측한다.

예측 대상 예:

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
- plot lag가 줄어들 수 있다.
- delta corr와 direction acc가 개선될 수 있다.
- MAE도 좋아질 가능성이 있다.

주의:

- future target은 label로만 사용하고 input으로 들어가면 안 된다.
- target은 첫 420 step 안에 있어야 한다.
- current FMS selection은 계속 validation MAE 중심으로 하되, shape metric을 같이 본다.

판정:

- MAE가 기준선과 비슷하거나 좋아야 한다.
- delta corr, direction acc, fixed validation plot이 좋아져야 한다.

### 2. Predicted-State Residual Update Head

현재 문제:

- hidden state에서 매 시점 FMS 값을 직접 뽑는다.
- FMS는 독립적인 점 예측이라기보다 누적 sickness state의 관측값에 가깝다.

제안:

```text
pred_fms_t = pred_fms_{t-1} + bounded_delta_t
bounded_delta_t = delta_head(hidden_t, motion_context_t)
```

중요한 점:

- `delta_from_calibration`처럼 calibration FMS에 계속 묶는 방식은 피한다.
- post-calibration 실제 FMS를 recurrent input으로 쓰면 안 된다.
- recurrent update는 모델의 이전 prediction 또는 latent state를 사용한다.

기대:

- plot continuity가 좋아질 수 있다.
- local rise/drop 변화가 더 자연스러워질 수 있다.
- sudden jump와 excessive flatness를 동시에 줄일 가능성이 있다.

위험:

- delta scale이 너무 작으면 반응이 늦어진다.
- delta scale이 너무 크면 plot이 noisy해진다.
- MAE가 좋아져도 실제 변화 timing이 밀리면 reject해야 한다.

### 3. Explicit Latent Sickness State

현재 문제:

- hidden state가 motion representation, person/session prior, sickness state를 모두 암묵적으로 담당한다.
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

위험:

- 구조 변경 폭이 커서 implementation risk가 있다.
- 기존 checkpoint compatibility와 config flag를 신경 써야 한다.
- 먼저 future trajectory auxiliary가 효과 있는지 확인한 뒤 진행하는 것이 낫다.

### 4. Rise/Fall/Plateau Event Auxiliary

현재 관찰:

- risk auxiliary를 제거하면 current FMS validation 성능도 나빠졌다.
- 이는 변화 이벤트 학습이 current FMS representation에 도움을 준다는 신호다.

제안:

rapid-rise binary만 쓰지 말고 trajectory event를 더 직접적으로 예측한다.

후보 label:

- rise 시작 여부
- fall 시작 여부
- plateau 여부
- local slope sign
- absolute delta magnitude bin

기대:

- 상승 시작 timing이 개선될 수 있다.
- flat prediction으로 평균에 머무르는 문제가 줄어들 수 있다.
- plot 평가 기준과 직접 연결된다.

주의:

- event label은 target window에서만 계산하고 input으로 들어가면 안 된다.
- class imbalance가 클 수 있으므로 loss weight를 작게 시작한다.
- event metric이 좋아졌더라도 current FMS MAE와 plot이 나쁘면 reject한다.

### 5. Person Prior와 Online Dynamic State 분리

현재 문제:

- calibration summary, static feature, online motion state가 decoder에서 섞인다.
- calibration은 현재 값 anchor라기보다 개인별 susceptibility, bias, response speed를 설명하는 prior에 가깝다.

제안:

```text
person/session prior p = f(calibration, static)
online dynamic state s_t = g(motion up to t)
prediction = head([s_t, p, s_t * gate(p)])
```

기대:

- 같은 motion에도 민감한 사람과 둔감한 사람을 다르게 반응시킬 수 있다.
- MSSQ/static feature 활용이 명확해진다.
- baseline bias와 dynamic change를 분리할 수 있다.

위험:

- 데이터 크기가 작아 overfit 가능성이 있다.
- static feature가 강한 shortcut으로 작동하지 않는지 확인해야 한다.

### 6. Multi-Timescale Motion Response Encoder

현재 문제:

- 멀미는 motion에 즉시 반응하지 않고 누적 및 지연 반응을 보인다.
- 현재 TCN/GRU가 이를 암묵적으로 배우지만, response lag가 명시적으로 모델링되지는 않는다.

제안:

- 5s / 15s / 30s / 60s motion response bank
- learnable temporal decay accumulator
- motion energy / jerk energy의 multi-timescale causal summary
- attention over past causal motion features

기대:

- response lag를 더 잘 잡을 수 있다.
- rise/drop timing이 좋아질 수 있다.
- session/person별 response speed 차이를 반영할 수 있다.

주의:

- 모든 summary는 causal해야 한다.
- window 끝 이후 motion을 보면 안 된다.
- receptive field와 actual input window의 의미를 명확히 기록해야 한다.

## 권장 진행 순서

### Phase 0. 기준선 고정

- `selected_fds_static4.yaml`을 기준 config로 유지한다.
- 기준 validation leaderboard와 fixed plot set을 고정한다.
- risk-head ablation 결과를 registry에 남겨둔다.

### Phase 1. 작은 비용의 튜닝 실험

목표:

- 현재 기준선이 어느 loss/blend 구간에서 안정적인지 확인한다.

우선순위:

1. `risk_loss_weight`: 0.10, 0.15, 0.35
2. `fms_combine_weight_ordinal`: 0.15, 0.25
3. `lds_gamma` / `fds_blend` 소규모 조정

선택 기준:

- MAE가 최소 0.03 이상 좋아지거나
- MAE가 유지되면서 shape metric과 fixed plot이 좋아져야 한다.

### Phase 2. Future Trajectory Auxiliary

목표:

- latent state가 near-future FMS 변화 방향을 담도록 만든다.

권장 구현:

- 기존 current head를 유지한다.
- auxiliary future FMS 또는 future delta head를 추가한다.
- loss weight는 작게 시작한다.
- current MAE selection을 유지한다.

이 단계가 가장 먼저 해볼 구조 개선이다.

### Phase 3. Event Auxiliary

목표:

- rise/fall/plateau timing을 직접 학습시킨다.

권장 구현:

- rapid-rise head를 확장하거나 별도 event head를 둔다.
- local slope sign, rise onset, fall onset label을 만든다.
- small auxiliary weight로 시작한다.

### Phase 4. Latent Sickness State / Person Prior 분리

목표:

- current/risk/future/event head가 공유하는 명시적 sickness state를 만든다.
- calibration/static은 susceptibility prior로 분리한다.

이 단계는 변경 폭이 크므로 Phase 2/3에서 신호가 나온 뒤 진행한다.

## Reject 기준

아래 중 하나라도 해당하면 성능 향상으로 보지 않는다.

- Validation MAE가 좋아졌지만 plot이 더 flat해졌다.
- Direction metric은 좋아졌지만 RMSE/centered MAE가 크게 나빠졌다.
- Test metric만 좋아지고 validation 기준으로는 선택 근거가 없다.
- Fixed validation plot에서 상승/하강 timing이 실제보다 일관되게 늦다.
- 고-FMS 구간을 계속 과소예측한다.
- seed 하나에서만 좋아지고 반복 seed에서 rank가 무너진다.
- leakage-safe windowing 검증을 통과하지 못한다.

## 현재 판단

단기적으로는 risk/ordinal/LDS/FDS 튜닝이 비용 대비 효율적이다. 하지만 이것들은 대부분 파라미터 튜닝이다.

구조적으로 가장 먼저 해볼 만한 것은 `future trajectory auxiliary head`다. 현재 목표가 "한 점 MAE"뿐 아니라 "plot에서 멀미 변화가 의미 있게 따라가는 것"이므로, 모델의 latent state가 near-future trajectory를 예측하도록 압력을 주는 방향이 가장 직접적이다.

그 다음은 `rise/fall/plateau event auxiliary`와 `explicit latent sickness state`다. 특히 risk auxiliary loss가 도움이 된 ablation 결과를 보면, 변화 이벤트를 더 명시적으로 학습시키는 쪽은 충분히 시도할 가치가 있다.
