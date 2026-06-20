# d128 구조 중간 보고서

- 작성 시각: 2026-05-05 15:57 KST
- 최신 갱신: 2026-05-05 18:30 KST
- 대상 run 계열: `lcsa_per_horizon_heads`, `model_type=lc_sa_tcnformer`
- 목적: 현재 adaptive search에서 사용하는 `d_model=128` 후보가 실제로 어떤 구조인지, 그리고 현재까지 validation에서 어떤 신호를 보였는지 정리한다.
- 주의: d128 관련 수치는 validation 기준이다. final test는 validation으로 선택된 최종 모델 1개에 대해서만 수행했고, d128 후보는 최종 선택되지 않았다.

## 한 줄 요약

`d128`은 새 모델 종류가 아니라 `LCSATCNFormer`의 내부 표현 폭을 `d_model=128`로 키운 설정이다. calibration TCN, recent-motion TCN, Transformer encoder, start-FMS anchor encoder, horizon encoder, fusion MLP, horizon별 출력 head가 모두 128차원 표현을 중심으로 동작한다.

최종 search 기준으로 `d128` 자체가 최고 성능을 만들지는 않았다. 전체 최고 validation primary mean MAE는 `d_model=96`, recent window 20초, seed 7, lr 3e-4, ff192 조합의 `2.0448`이다. d128 계열 최고는 seed123/drop0.05의 `2.1417`이며, d128은 완전히 무의미한 방향은 아니지만 recent window 단축과 d96/ff192 조합보다 개선 폭이 작았다.

## 입력 정책

현재 d128 후보는 main track 정책을 유지한다.

- `fms_context_mode=start_only`
- `anchor_mode=none`
- `anchor_interval_seconds=0`
- `sparse_observed=false`
- `recent_start_observed=false`
- `use_static=false`
- identity feature 미사용
- multi-horizon: h=5, h=10, h=15
- selection metric: validation MAE mean over h=5/10/15

`start_only`에서는 calibration FMS history는 calibration phase 입력으로 들어가고, post-calibration FMS context는 recent motion window의 시작 FMS 하나만 anchor context로 들어간다.

## 구조

코드 기준 위치:

- 모델 정의: `src/densefms_forecast/model.py`, `LCSATCNFormer`
- 모델 생성: `src/densefms_forecast/model.py`, `build_model("lc_sa_tcnformer", ...)`
- search spec: `scripts/run_goal_mae_search_v2.py`
- 기본 config: `configs/lc_sa_tcnformer.yaml`

현재 d128 후보의 핵심 구조는 다음과 같다.

- Head motion input: `[B, T, 6]`
- Calibration window: 90초, 180 steps
- Recent window: 30초, 60 steps
- Sampling interval: 0.5초
- Calibration encoder: `LCBranchTCN(input_dim=7, d_model=128)`
  - 6개 head/motion feature + calibration FMS 1개
  - dilation: `[1, 2, 4, 8, 16]`
  - kernel size: 3
- Calibration transformer:
  - 1 encoder layer
  - 4 attention heads
  - head dimension: 32
  - feed-forward dim: 128
- Recent encoder: `LCBranchTCN(input_dim=6, d_model=128)`
  - recent motion만 사용
  - auto dilation 결과: `[1, 2, 4, 8]`
  - receptive field: 61 steps, 30.5초
- Pooling: mean
- Start-FMS anchor encoder:
  - 입력: `[start_fms_value, time_since_start / 120]`
  - 2 -> 128 -> 128 MLP
  - `anchor_mode=none`이지만 `start_only`라서 recent-window 시작 FMS context는 별도 encoder로 들어간다.
- Horizon encoder:
  - 입력: horizon seconds / 60
  - 1 -> 128
- Fusion:
  - concat: calibration 128 + recent 128 + horizon 128 + start-FMS anchor 128
  - fusion input dim: 512
  - MLP: 512 -> 256 -> 128
- Output:
  - `per_horizon_heads=true`
  - h=5, h=10, h=15 각각 `Linear(128, 1)` head
  - 최종 예측은 sigmoid로 0..1 범위

`hidden_dim=128`도 experiment log에 보이지만, `lc_sa_tcnformer` build path에서는 이 값이 구조를 직접 바꾸는 핵심 인자가 아니다. 이 계열의 실제 capacity knob은 주로 `d_model`, `transformer_ff_dim`, layer/head 수다.

## d96/d112/d128 파라미터 규모

동일한 start_only, no-static, multi-horizon, per-horizon-heads 조건에서 계산한 trainable parameter 수:

| d_model | trainable params | fusion dim | attention head dim | recent RF |
|---:|---:|---:|---:|---:|
| 96 | 668,772 | 384 | 24 | 30.5s |
| 112 | 903,860 | 448 | 28 | 30.5s |
| 128 | 1,174,276 | 512 | 32 | 30.5s |

d128은 d96 대비 parameter가 약 1.76배다. recent receptive field는 그대로라서 temporal coverage가 늘어나는 게 아니라, 같은 입력 범위를 더 큰 hidden width로 표현하는 변경이다.

## 현재까지 validation 결과

주요 비교값:

- 최종 전체 best: `d_model=96`, recent window 20초, seed 7, lr 3e-4, ff192, dropout 0.05
  - primary mean MAE: `2.0448`
  - h5/h10/h15: `1.9791 / 2.0491 / 2.1062`
- 이전 d96 seed7 lr3.5e-4:
  - primary mean MAE: `2.1315`
  - h5/h10/h15: `2.0718 / 2.1430 / 2.1796`
- 이전 d96 seed7 기본 lr 3e-4:
  - primary mean MAE: `2.1517`

d128 완료 후보:

| run 요약 | lr | dropout | seed | primary mean MAE | h5 | h10 | h15 | 판단 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| d128 lr2e-4 drop0.03 | 0.0002 | 0.03 | 42 | 2.1895 | 2.1139 | 2.2069 | 2.2476 | 느린/약한 수렴 |
| d128 lr1e-4 drop0.03 | 0.0001 | 0.03 | 42 | 2.1961 | 2.1181 | 2.2177 | 2.2526 | underfit 성향 |
| d128 lr3e-4 drop0.05 | 0.0003 | 0.05 | 42 | 2.1587 | 2.0817 | 2.1751 | 2.2192 | 괜찮지만 best 아님 |
| d128 lr3e-4 drop0.05 | 0.0003 | 0.05 | 7 | 2.1775 | 2.1150 | 2.1923 | 2.2253 | seed7에서는 악화 |
| d128 lr3e-4 drop0.05 | 0.0003 | 0.05 | 123 | 2.1417 | 2.0836 | 2.1567 | 2.1849 | d128 중 현재 best |
| d128 lr3e-4 drop0.05 | 0.0003 | 0.05 | 202 | 2.1819 | 2.1123 | 2.2055 | 2.2280 | seed 민감 |
| d128 lr2.5e-4 drop0.05 | 0.00025 | 0.05 | 7 | 2.1851 | 2.1071 | 2.2066 | 2.2415 | LR 낮추면 악화 |
| d128 lr3.5e-4 drop0.05 | 0.00035 | 0.05 | 7 | 2.1902 | 2.1194 | 2.2047 | 2.2466 | 고LR도 악화 |

## 중간 해석

d128은 표현 폭을 키우기 때문에 long-horizon h10/h15 개선을 기대한 후보였다. 실제로 seed123에서는 h10/h15가 `2.1567 / 2.1849`까지 내려와 꽤 강했지만, d96 seed7 lr3.5e-4의 `2.1430 / 2.1796`에도 못 미쳤고, 최종 best인 d96/recent20의 `2.0491 / 2.1062`와는 차이가 더 컸다.

현재 관찰상 성능을 가장 크게 좌우한 것은 `d_model=128` 자체보다 seed와 learning rate다. d96에서 lr3.5e-4가 새 best를 만들었고, d128은 더 큰 capacity 때문에 seed와 LR에 더 민감해진 것으로 보인다.

따라서 다음 판단 기준은 명확하다.

- d128 lr3.5e-4도 best를 넘기지 못했으므로, d128은 보조 후보로만 기록한다.
- 최종적으로는 d96/ff192에서 recent window를 20초로 줄인 후보가 가장 유망했다.
- d112 결과가 좋지 않았으므로 단순 중간 width 탐색은 중단하는 편이 낫다.

## 누수/평가 상태

- 일반 adaptive run은 `--no_test_eval`로 실행했다.
- validation lock 이전 pre-lock test output은 발견되지 않았다.
- final test는 `FINAL_SELECTION_LOCK.md` 작성 이후 validation-selected 최종 모델 1개에 대해서만 수행했다.
- d128 후보도 main track과 동일하게 start_only/no sparse/no recent_start_observed 조건을 유지한다.
