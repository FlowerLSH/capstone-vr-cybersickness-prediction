# DenseFMS 구현 모델 아키텍처 설명서

이 문서는 현재 저장소에 구현된 DenseFMS 미래 FMS 예측 모델의 종류와 아키텍처를 코드 기준으로 정리한다. 주요 구현 위치는 `src/densefms_forecast/model.py`, 학습 CLI 노출 위치는 `src/densefms_forecast/train.py`, classical feature baseline 구현 위치는 `scripts/run_densefms_long_target_search.py`이다.

## 1. 공통 문제 설정

모든 neural forecaster는 head/motion time-series와 calibration FMS를 사용해 현재 시점 `t` 이후의 FMS, 즉 `FMS[t + horizon_steps]`를 예측한다.

공통 입력:

- `head`: `[B, T, 6]` head/motion feature sequence.
- `y_calib` 또는 `fms`: FMS sequence. 모델/anchor mode에 따라 `[B, C]` calibration-only 또는 `[B, T]` full observed FMS가 들어간다.
- `lengths`: session별 유효 길이.
- `static`: optional static covariates. 현재 static feature 구성은 보통 `age`, `gender`, `mssq`에서 만들어지는 `[B, 5]` 벡터이다.

공통 출력:

- `future`: 예측 FMS. 단일 horizon이면 `[B, pred_steps]`, multi-horizon이면 `[B, pred_steps, num_horizons]`.
- `mask`: 유효 target 위치 mask. 단일 horizon은 `[B, pred_steps]`, multi-horizon은 `[B, pred_steps, num_horizons]`.
- `prediction_start`: compact prediction sequence의 실제 시작 index.
- `horizon_steps_list`: multi-horizon 또는 단일 horizon의 step 목록.

FMS는 학습 파이프라인에서 정규화된 값으로 다뤄지며, neural head는 일반적으로 `sigmoid` 또는 anchor-delta clamp를 통해 `[0, 1]` 범위 예측을 만든다.

## 2. 공통 leakage-safe windowing 규칙

모델 구현은 아래 제약을 기준으로 설계되어 있다.

- calibration branch는 첫 `calibration_steps`의 head/FMS만 사용한다.
- recent branch는 현재 시점 `t` 이하의 motion window만 사용한다.
- target `FMS[t + horizon_steps]`는 forward 입력 feature로 사용하지 않는다.
- `sparse_observed`와 `recent_start_observed` anchor는 current index 이하의 FMS만 참조한다.
- `recent_start_observed`는 배포 현실 설정이 아니라 upper-bound diagnostic이다.

## 3. Anchor 정책

Anchor는 모델이 현재 시점 근처의 관측 FMS 상태를 참조할 수 있게 하는 옵션이다.

| Anchor mode | 의미 | 사용 가능성 |
|---|---|---|
| `none` | FMS anchor 없이 예측 | 완전 motion/calibration 중심 |
| `calibration_end` | calibration 마지막 FMS, `calibration_steps - 1` | 배포 친화적 |
| `sparse_observed` | 일정 간격으로 관측된 가장 최근 FMS | 사용자 입력 주기 필요 |
| `recent_start_observed` | recent window 시작점 FMS | upper-bound 전용, 배포 후보 아님 |

`predict_delta_from_anchor=True`일 때 모델은 절대 FMS를 직접 예측하지 않고 anchor FMS에 제한된 delta를 더한다.

```text
pred = clamp(anchor_fms + delta_scale * tanh(raw_pred), 0, 1)
```

## 4. 공통 building block

### CalibrationEncoder

COFFLSTM과 CalibOnly가 사용하는 calibration encoder이다.

1. calibration head `[B, C, 6]`, calibration FMS `[B, C]`, FMS 1-step delta를 결합한다.
2. `Linear -> GELU -> LayerNorm`으로 token embedding을 만든다.
3. 1-layer LSTM으로 calibration sequence를 인코딩한다.
4. attention pooling 결과와 final hidden state를 concat한다.
5. `Linear -> GELU -> LayerNorm`으로 128-d calibration context를 만든다.

### TCNBlock / LCBranchTCN

causal convolution 기반 sequence encoder이다.

- `CausalConv1d`는 왼쪽 padding만 사용해 미래 시점이 섞이지 않게 한다.
- `TCNBlock`은 causal convolution 2개, GELU, dropout, residual, LayerNorm으로 구성된다.
- `LCBranchTCN`은 입력 projection 뒤 dilation 목록에 따라 여러 `TCNBlock`을 쌓는다.

### RecentWindowEncoder

COFFLSTM과 Recent10TCN의 recent window encoder이다.

- recent head window `[B, W, 6]`를 64차원으로 projection한다.
- dilation `[1, 2, 4]` TCN block을 통과한다.
- 마지막 token을 64차원 recent representation으로 변환한다.

### RecentTransformerEncoder

COFFLSTM에서 선택 가능한 recent encoder이다.

- recent head window를 embedding하고 positional parameter를 더한다.
- causal self-attention mask를 적용한다.
- feed-forward block과 LayerNorm을 거쳐 마지막 token을 recent representation으로 사용한다.

### StaticEncoder

static covariate를 MLP로 인코딩한다.

- 입력: `[B, static_dim]`.
- 구조: `Linear -> GELU -> LayerNorm -> Dropout -> Linear -> GELU -> LayerNorm`.
- COFFLSTM에서는 calibration context와 fusion되고, LC-SA-TCNFormer에서는 fusion branch 중 하나로 사용된다.

## 5. 모델별 아키텍처

## 5.1 COFFLSTM (`coff_lstm`)

COFFLSTM은 calibration-conditioned online recurrent forecaster이다.

구조:

1. `CalibrationEncoder`가 첫 calibration 구간의 head/FMS/delta-FMS를 128차원 context로 인코딩한다.
2. static feature를 쓰는 경우 `StaticEncoder`와 `ContextFusion`이 calibration context를 static-aware context로 바꾼다.
3. `StateInitializer`가 calibration context로 LSTMCell 초기 hidden/cell state를 만든다.
4. current head `head[t]`는 `head_projection`을 거친다.
5. FiLM conditioning이 켜져 있으면 calibration context에서 `gamma`, `beta`를 만들고 head embedding에 적용한다.
6. online LSTMCell이 calibration 이후 각 현재 시점 `t`를 causal하게 업데이트한다.
7. optional recent encoder가 `head[t-recent_steps+1:t+1]` window를 인코딩한다.
8. `ForecastHead`가 calibration context, online hidden state, recent representation을 합쳐 future FMS를 예측한다.

특징:

- post-calibration FMS는 recurrent update에 들어가지 않는다.
- `no_film`, `no_recent_encoder`, `recent_encoder=tcn|transformer`, static on/off ablation을 지원한다.
- legacy multihead 모드에서는 now head, level head, delta head, gate를 함께 계산할 수 있다.

## 5.2 Recent10TCN (`recent10_tcn`)

recent motion window만 사용하는 단순 neural baseline이다.

구조:

1. 각 현재 시점 `t`에 대해 head recent window를 만든다.
2. `RecentWindowEncoder`가 window를 64차원 representation으로 인코딩한다.
3. 작은 MLP head 뒤 `now`, `future` linear head를 둔다.

특징:

- calibration FMS와 static feature를 사용하지 않는다.
- recent motion만으로 예측 가능한 정도를 확인하는 baseline이다.
- window는 항상 current time 이하로 구성된다.

## 5.3 CalibOnly (`calib_only`)

calibration 정보와 time embedding만 사용하는 baseline이다.

구조:

1. `CalibrationEncoder`가 첫 calibration 구간의 head/FMS/delta-FMS를 128차원 context로 만든다.
2. 전체 sequence index에 대한 learned time embedding을 만든다.
3. calibration context를 모든 time step으로 broadcast한다.
4. calibration context와 time embedding을 concat한 뒤 MLP로 now/future를 예측한다.

특징:

- calibration 이후 motion은 사용하지 않는다.
- session별 초기 calibration profile과 경과 시간만으로 가능한 예측 성능을 보는 baseline이다.

## 5.4 LC-SA-TCNFormer (`lc_sa_tcnformer`)

LC-SA-TCNFormer는 Long-Calibrated State-Anchored TCN-Transformer forecaster이다. 현재 best-score 검색에서 가장 강한 neural architecture로 사용됐다.

구조:

1. Calibration branch
   - 첫 `calibration_steps`의 `[head, FMS]`를 concat한다.
   - `LCBranchTCN`으로 causal/dilated feature를 만든다.
   - `TransformerEncoder`를 통과시킨다.
   - mean, last, attention 중 하나의 pooling으로 calibration embedding `z_calib`를 만든다.

2. Recent motion branch
   - 전체 head sequence를 `LCBranchTCN`으로 causal 인코딩한다.
   - 각 prediction position에서 recent window를 mean/last/attention pooling해 `z_recent[t]`를 만든다.

3. Anchor branch
   - anchor FMS와 `time_since_anchor / 120`을 2차원 feature로 만든다.
   - `Linear -> GELU -> LayerNorm -> Dropout -> Linear -> GELU -> LayerNorm`으로 `z_anchor[t]`를 만든다.

4. Static branch
   - static on이면 `StaticEncoder`로 static representation을 만들고, 필요시 `d_model`로 projection한다.

5. Horizon branch
   - horizon seconds를 `horizon_seconds / 60`으로 scaling한 뒤 MLP로 horizon embedding을 만든다.

6. Fusion/head
   - `z_calib`, `z_recent`, horizon embedding, optional anchor embedding, optional static embedding을 concat한다.
   - fusion MLP 후 linear head가 raw prediction을 만든다.
   - `predict_delta_from_anchor=True`면 anchor-delta 방식으로 최종 FMS를 만든다. 아니면 sigmoid 절대값 예측을 사용한다.

특징:

- calibration은 long context, recent motion은 causal TCN, state anchor는 FMS 상태를 반영한다.
- `calib_dilations`, `recent_dilations`, `transformer_layers`, `transformer_heads`, `pooling`, static, anchor mode, delta prediction을 조합할 수 있다.
- `multi_horizon=True`이면 같은 branch representation에 horizon embedding만 바꿔 여러 horizon을 동시에 예측한다.

## 5.5 AnchorDeltaMLP (`anchor_delta_mlp`)

AnchorDeltaMLP는 sequence encoder 없이 hand-crafted summary feature와 anchor-delta를 쓰는 강한 cheap baseline이다.

구조:

1. Calibration summary feature
   - calibration FMS의 first, last, mean, std, max, min, last-first, slope.
   - calibration head의 mean, std.

2. Recent motion summary feature
   - recent window head의 mean, std, min, max.
   - 앞 3축과 뒤 3축 magnitude의 mean/std/max.
   - first 3축 frame difference magnitude의 mean/std/max.

3. Anchor feature
   - anchor FMS.
   - `time_since_anchor / 120`.

4. Optional static feature
   - age/gender/mssq 기반 static vector.

5. Horizon scalar
   - `horizon_seconds / 60`.

6. MLP
   - 기본 hidden layout은 `[hidden_dim, hidden_dim, max(32, hidden_dim // 2)]`.
   - output은 raw scalar이며, 보통 anchor-delta clamp로 최종 FMS를 만든다.

특징:

- 학습 비용이 낮고 해석하기 쉽다.
- motion sequence를 learned temporal encoder로 직접 학습하지 않는다.
- early search에서 강한 baseline으로 유용하다.

## 5.6 AnchorDeltaGRU (`anchor_delta_gru`)

AnchorDeltaGRU는 AnchorDeltaMLP feature에 learned GRU motion state를 추가한 모델이다.

구조:

1. AnchorDeltaMLP와 같은 calibration summary, recent summary, anchor, static, horizon feature를 만든다.
2. full head sequence를 GRU로 인코딩한다.
3. prediction position `t`의 GRU hidden state를 feature에 concat한다.
4. GRU hidden dimension 기반 MLP가 raw prediction을 만든다.
5. anchor-delta 또는 sigmoid 방식으로 최종 FMS를 만든다.

특징:

- hand-crafted summary feature와 learned recurrent motion state를 결합한다.
- LC-SA-TCNFormer보다 단순하지만 AnchorDeltaMLP보다 temporal representation이 풍부하다.

## 5.7 RecentTCN+SummaryCalib (`recent_tcn_summary_calib`)

RecentTCN+SummaryCalib는 AnchorDeltaMLP의 summary feature에 learned recent TCN representation을 추가한 모델이다.

구조:

1. AnchorDeltaMLP와 같은 calibration summary, recent summary, anchor, static, horizon feature를 만든다.
2. head sequence를 `LCBranchTCN`으로 causal 인코딩한다.
3. prediction position마다 recent window의 TCN representation을 mean/last/attention pooling한다.
4. pooled recent TCN vector를 summary feature에 concat한다.
5. MLP가 raw prediction을 만들고 anchor-delta 또는 sigmoid 방식으로 최종 FMS를 만든다.

특징:

- calibration은 hand-crafted summary로 처리하고, recent motion만 learned TCN으로 강화한다.
- 장기 검색의 deployment-realistic H=5/H=10/H=15 후보에서 강하게 동작했다.
- LC-SA-TCNFormer보다 작고 빠른 편이다.

## 5.8 Gated Fusion (`gated_fusion`)

Gated Fusion은 RecentTCN+SummaryCalib에 feature-wise gating을 추가한 변형이다.

구조:

1. RecentTCN+SummaryCalib와 같은 base feature를 만든다.
2. `gate = sigmoid(Linear(base_features))`를 계산한다.
3. `base_features * gate`를 MLP에 넣는다.
4. training 중 `branch_dropout`이 설정되면 base feature 일부를 무작위로 drop한다.

특징:

- feature branch 간 중요도를 학습적으로 조정하려는 모델이다.
- `anchor_dropout` 인자는 CLI/생성 spec에 포함되어 있으나, 현재 클래스의 forward에서는 명시적인 anchor-only dropout으로 사용되지는 않는다.

## 6. MultiHorizon 모드

MultiHorizon은 별도 모델 클래스라기보다 `lc_sa_tcnformer`, `anchor_delta_mlp`, `recent_tcn_summary_calib` 계열에서 켤 수 있는 출력 모드이다.

동작:

- `multi_horizon=True`와 `horizon_set=[1.0, 2.5, 5.0, 10.0, 15.0, 30.0]` 같은 설정을 사용한다.
- prediction mask와 output shape는 `[B, pred_steps, num_horizons]`가 된다.
- LC-SA-TCNFormer는 horizon embedding을 horizon별로 바꿔 fusion/head를 반복한다.
- FeatureAnchor 계열은 horizon scalar를 바꿔 MLP를 반복 적용한다.

주의:

- multi-horizon aggregate MAE는 horizon별 성능을 평균적으로 반영하므로, 단일 H=1 최적화와 직접 같은 의미가 아니다.
- 최종 보고에서는 horizon별 metrics를 따로 봐야 한다.

## 7. Classical feature baselines

`scripts/run_densefms_long_target_search.py`에는 PyTorch 모델이 아닌 scikit-learn 기반 classical baseline도 구현되어 있다.

사용 모델:

- Ridge
- ElasticNet
- RandomForestRegressor
- HistGradientBoostingRegressor
- GradientBoostingRegressor

feature vector:

- anchor FMS.
- `time_since_anchor / 120`.
- `horizon_seconds / 60`.
- calibration FMS summary 8개.
- recent motion statistics.
- static feature(age, mssq, gender one-hot).

특징:

- validation-only 비교를 위한 cheap baseline이다.
- checkpoint는 `model.pkl`, scaler 통계, prediction CSV 형태로 저장된다.
- neural `build_model()` registry에는 포함되지 않는다.

## 8. 모델 비교 표

| 모델 | Calibration 사용 | Recent motion 사용 | Learned temporal encoder | Anchor 지원 | Static 지원 | MultiHorizon |
|---|---|---|---|---|---|---|
| COFFLSTM | LSTM+attention | optional TCN/Transformer window | online LSTMCell | 직접 anchor 없음 | 지원 | 미지원 |
| Recent10TCN | 미사용 | recent window | causal TCN | 없음 | 미사용 | 미지원 |
| CalibOnly | LSTM+attention | 미사용 | calibration LSTM | 없음 | 미사용 | 미지원 |
| LC-SA-TCNFormer | TCN+Transformer | causal TCN | TCN+Transformer | 지원 | 지원 | 지원 |
| AnchorDeltaMLP | summary stats | summary stats | 없음 | 지원 | 지원 | 지원 |
| AnchorDeltaGRU | summary stats | summary stats + GRU | GRU | 지원 | 지원 | 지원 |
| RecentTCN+SummaryCalib | summary stats | summary stats + TCN | causal TCN | 지원 | 지원 | 지원 |
| Gated Fusion | summary stats | summary stats + TCN | causal TCN + feature gate | 지원 | 지원 | 상속 구조상 지원 |
| Classical baselines | summary stats | summary stats | 모델별 tree/linear | sparse anchor feature | 지원 | 별도 horizon별 학습 |

## 9. 현재 학습 CLI에서 선택 가능한 모델명

`src/densefms_forecast/train.py`의 `--model` choices:

- `coff_lstm`
- `recent10_tcn`
- `calib_only`
- `lc_sa_tcnformer`
- `anchor_delta_mlp`
- `anchor_delta_gru`
- `recent_tcn_summary_calib`
- `gated_fusion`

## 10. 해석상 주의

- H=1 성능이 좋아도 H=5/H=10/H=15 일반 forecasting 성공을 의미하지 않는다.
- anchor interval이 짧을수록 예측은 쉬워질 수 있지만 사용자 입력 부담이 커진다.
- `recent_start_observed`는 현재 배포 상황에서 사용 가능한 설정이 아니라 upper-bound diagnostic이다.
- test metric은 validation으로 선택된 최종 모델에 대한 final-report-only 결과로만 해석해야 한다.
