# Online Current FMS Literature Follow-up - 2026-05-09

## 목적

이 문서는 현재 선택 모델을 기준으로 추가 개선 후보를 문헌 기반으로 정리한 것이다.

기준 모델:

`runs/online_fms_current_tracking_0508/deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42`

기준 목표:

- Validation MAE를 유의미하게 낮춘다.
- 또는 MAE를 유지하면서 validation trajectory plot에서 멀미 상승, 하강, plateau, lag를 더 잘 예측한다.

이 문서는 `online_current_improvement_plan_0509.md`의 후속 메모다. 앞 문서가 파라미터 튜닝과 구조 개선의 1차 후보를 정리했다면, 여기서는 선행 연구에서 나온 추가 힌트를 현재 DenseFMS 설정에 맞게 변환한다.

## 탐색한 선행 연구의 공통 패턴

비슷한 문제를 다룬 연구들은 대체로 아래 접근을 썼다.

1. HMD에 내장된 head/eye tracking을 적극적으로 사용한다.
2. 단일 시점 분류보다 onset forecasting, severity regression, multi-horizon prediction을 시도한다.
3. motion의 원시값만 쓰지 않고 velocity, acceleration, jerk, spectral/temporal/statistical feature를 만든다.
4. eye tracking, physiological signal, content metadata가 있으면 multimodal fusion으로 성능을 올린다.
5. 개인차가 크므로 demographic/clinical/person prior를 별도로 고려한다.
6. FMS/SSQ를 단순 연속값이 아니라 severity band나 risk/tolerance와 연결하려는 흐름이 있다.

현재 우리 데이터는 head/motion-only 중심이므로, 바로 적용 가능한 것은 2, 3, 5, 6번이다. Eye tracking, GSR, HR, content metadata는 현재 데이터에 없으면 당장 적용하기 어렵지만, 구조 설계의 방향성으로는 참고할 수 있다.

## 주요 문헌과 시사점

### 1. Islam et al. 2022 - Cybersickness Onset Forecasting

논문:

- Rifatul Islam, Kevin Desai, John Quarles, "Towards Forecasting the Onset of Cybersickness by Fusing Physiological, Head-tracking and Eye-tracking with Multimodal Deep Fusion Network", ISMAR 2022.
- DOI: https://doi.org/10.1109/ISMAR55827.2022.00026

핵심 접근:

- Cybersickness를 onset 이후 탐지하는 대신 30-60초 앞서 forecasting하는 문제로 설정했다.
- Eye tracking, head tracking, HR, GSR을 multimodal fusion했다.
- LSTM, N-BEATS, DeepTCN 계열을 forecasting backbone으로 비교했다.
- FMS를 ground truth로 사용했다.
- DeepTCN 기반 multimodal fusion이 강한 결과를 보였다.

현재 모델에 주는 시사점:

- `future trajectory auxiliary head`는 문헌 방향과 잘 맞는다.
- 단순 current FMS 예측보다 `t+5s`, `t+10s`, `t+15s`, 가능하면 `t+30s`를 auxiliary로 두는 것이 타당하다.
- 현재 risk auxiliary가 도움이 된 것도 onset/rise forecasting 신호가 shared representation을 개선했기 때문으로 해석할 수 있다.
- DeepTCN 계열이 유효했다는 점은 현재 `deep_tcn_latent_gru` 기준선을 유지할 근거가 된다.

추가 실험 후보:

- current FMS + future FMS multi-horizon auxiliary
- current FMS + future delta auxiliary
- rapid-rise risk head를 future trajectory head와 함께 사용
- horizon별 loss weight를 작게 시작: 5s > 10s > 15s

### 2. Islam et al. 2021 - Integrated HMD Sensor Deep Fusion

논문:

- Rifatul Islam, Kevin Desai, John Quarles, "Cybersickness Prediction from Integrated HMD's Sensors: A Multimodal Deep Fusion Approach using Eye-tracking and Head-tracking Data", ISMAR 2021.
- DOI: https://doi.org/10.1109/ISMAR52148.2021.00017
- arXiv: https://arxiv.org/abs/2108.06437

핵심 접근:

- 외부 생체 센서 대신 HMD 내장 sensor에서 얻을 수 있는 eye tracking, head tracking, gameplay video를 사용했다.
- Heterogeneous sensor branch를 deep fusion했다.
- 30명, 1755개 stereoscopic video segment와 eye/head tracking, self-reported severity를 사용했다.
- Eye + head tracking만으로도 강한 severity prediction 결과를 보고했다.

현재 모델에 주는 시사점:

- 현재 데이터에 eye tracking이 없더라도, head tracking branch를 단일 raw sequence로만 보지 말고 여러 motion representation branch로 나눌 수 있다.
- 예: raw head motion branch, velocity/acceleration/jerk branch, frequency/complexity branch.
- Deep fusion은 꼭 여러 센서가 있어야만 가능한 것이 아니라, 같은 head motion에서 만든 여러 view를 fusion하는 방식으로도 쓸 수 있다.

추가 실험 후보:

- `motion_raw_branch` + `motion_derivative_branch` + `motion_complexity_branch`
- branch-level gating 또는 attention fusion
- branch dropout으로 특정 feature view에 과의존하지 않게 만들기

### 3. Salehi et al. 2024 - Head Movement Patterns

논문:

- Masoud Salehi, Nikoo Javadpour, Brietta Beisner, Mohammadamin Sanaei, Stephen B. Gilbert, "Cybersickness Detection through Head Movement Patterns: A Promising Approach", arXiv 2024 / HCII 2024.
- arXiv: https://arxiv.org/abs/2402.02725

핵심 접근:

- Head movement를 cybersickness marker로 사용했다.
- 6축 head movement와 그 derivative인 velocity, acceleration, jerk를 분석했다.
- Statistical, temporal, spectral feature를 추출했다.
- Recursive Feature Elimination으로 중요한 feature를 고르고 여러 ML 모델을 비교했다.

현재 모델에 주는 시사점:

- 현재 `motion_feature_mode=norm_delta_energy`는 좋은 시작이지만, derivative와 spectral/temporal feature를 더 적극적으로 넣을 여지가 있다.
- Deep model에 raw signal을 넣어도, 작은 데이터에서는 handcrafted causal feature가 여전히 도움이 될 수 있다.
- 특히 plot에서 rise/drop timing을 맞추려면 jerk, motion energy burst, spectral power change가 유용할 가능성이 있다.

추가 실험 후보:

- Causal rolling velocity/acceleration/jerk feature
- 5s/15s/30s window별 motion energy, jerk energy
- Causal spectral proxy: short-window band power, zero-crossing, sign-change rate
- Feature importance diagnostic: lightweight model 또는 permutation importance로 어떤 motion feature가 FMS change와 연결되는지 확인

주의:

- Spectral feature는 centered window를 쓰면 leakage가 생긴다.
- 모든 feature는 current time t 이하의 motion만 사용해야 한다.

### 4. IEEE VR 2025 - Motion Complexity

논문:

- "Reduction of Motion Complexity as an Objective Indicator of Cybersickness in Virtual Reality", IEEE VR 2025.
- DOI: https://doi.org/10.1109/VR59515.2025.00040

핵심 접근:

- Loss of complexity hypothesis를 VR motion tracking에 적용했다.
- PCA 기반 d95 score로 movement complexity를 계산했다.
- Cybersickness/discomfort가 높을수록 movement complexity가 감소하는 경향을 보고했다.
- Physical head movement 6-DOF보다 virtual camera motion까지 포함한 9-DOF complexity가 더 민감했다고 보고했다.

현재 모델에 주는 시사점:

- 단순 motion magnitude보다 "movement complexity가 줄어드는 상태"가 cybersickness marker일 수 있다.
- 현재 모델이 FMS 상승 구간을 못 따라간다면, head motion의 다양성/복잡도 저하가 clue가 될 수 있다.
- 데이터에 virtual camera motion이 없으면 9-DOF는 어렵지만, 6-DOF head motion complexity는 시도 가능하다.

추가 실험 후보:

- Causal rolling PCA complexity proxy
- Sliding-window covariance eigenvalue entropy
- Effective dimensionality / participation ratio
- d95-like explained variance feature
- Complexity decrease event auxiliary: 최근 complexity가 감소 중인지 예측하거나 feature로 제공

주의:

- PCA/statistics는 train split 기준 normalization만 사용한다.
- 각 시점 feature는 과거 window로만 계산한다.

### 5. Setu et al. 2024 - Mazed and Confused Dataset

논문:

- Jyotirmay Nag Setu et al., "Mazed and Confused: A Dataset of Cybersickness, Working Memory, Mental Load, Physical Load, and Attention During a Real Walking Task in VR", arXiv 2024.
- arXiv: https://arxiv.org/abs/2409.06898

핵심 접근:

- Real walking VR 상황에서 head position/orientation, eye tracking, images, physiological readings, cybersickness severity, physical/mental load를 수집했다.
- Case study에서 cybersickness severity classification을 수행했다.
- SHAP 분석에서 eye tracking과 physiological measure의 중요성을 보고했다.

현재 모델에 주는 시사점:

- Cybersickness는 motion만이 아니라 task load, attention, physical load와 얽힐 수 있다.
- 현재 데이터에 task/content metadata가 있다면 session-level covariate로 넣을 가치가 있다.
- 현 데이터가 motion-only라면, future dataset integration을 염두에 둔 modular multimodal input 구조가 유리하다.

추가 실험 후보:

- Session/content embedding이 가능한 metadata가 있는지 조사
- Scenario/domain label이 있으면 content prior로 추가
- 동일 participant/session split에서 content imbalance가 selection을 왜곡하는지 확인

### 6. Kim et al. 2021 - Clinical Predictors

논문:

- Hyewon Kim et al., "Clinical predictors of cybersickness in virtual reality (VR) among highly stressed people", Scientific Reports 2021.
- URL: https://www.nature.com/articles/s41598-021-91573-w

핵심 접근:

- 83명 대상, psychiatric/ophthalmologic/otologic evaluation과 physiological parameters를 함께 조사했다.
- FMS/SSQ 변화와 나이, 심리 지표, 흡연 등 개인 요인의 관계를 분석했다.
- 40-59세 그룹이 19-39세 그룹보다 FMS 증가가 컸고, affect 관련 지표도 cybersickness와 관련이 있었다.

현재 모델에 주는 시사점:

- 개인차는 무시하기 어렵다.
- 현재 static feature인 age, gender, MSSQ를 단순 concatenation보다 "person/session prior"로 분리하는 구조가 타당하다.
- Calibration FMS trajectory도 개인별 susceptibility와 response speed를 추정하는 prior로 써야 한다.

추가 실험 후보:

- Person prior branch: bias, scale, response speed를 별도 예측
- Static/calibration prior가 online state update gate를 조절
- Calibration summary에서 slope, range, volatility, max, final FMS를 더 명시적으로 사용

주의:

- Static feature는 데이터가 작으면 shortcut/overfit 위험이 있다.
- Validation split에서 participant leakage가 없어야 한다.

### 7. FMS/SSQ Tolerance Banding - Kelly et al. 2026 Preprint

논문:

- Jonathan W. Kelly, Michael C. Dorneich, Stephen B. Gilbert, "Interpreting FMS and SSQ Cybersickness Ratings via User Tolerance in Virtual Reality", OSF Preprint 2026.
- DOI: https://doi.org/10.31234/osf.io/p3xy4_v1

핵심 접근:

- FMS/SSQ score를 early termination/dropout probability와 연결해 mild/moderate/severe/extreme risk band를 만들었다.
- FMS-10 기준으로 risk band와 time-to-dropout을 연결했다.

현재 모델에 주는 시사점:

- FMS를 0-20 연속 회귀로만 보는 대신 behaviorally meaningful severity band auxiliary를 둘 수 있다.
- 현재 cumulative ordinal head는 0-20 각 level을 예측하지만, 별도의 coarse severity/risk band head가 더 안정적일 수 있다.

추가 실험 후보:

- Fine ordinal head 유지 + coarse severity band auxiliary 추가
- High-risk transition band에 loss를 조금 더 주기
- MAE와 별도로 severe/extreme recall 또는 high-FMS underprediction rate를 validation report에 추가

주의:

- 이 논문은 preprint이므로 threshold를 그대로 가져오기보다 validation-derived band로 조정하는 것이 낫다.

## 추가 개선 후보

### A. Causal Motion Dynamics Feature Bank

가장 즉시 적용 가능한 추가 후보.

구성:

- raw 6D motion
- velocity norm
- acceleration norm
- jerk norm
- rolling energy
- rolling jerk energy
- sign-change rate
- short-window spectral proxy
- rolling covariance/eigenvalue complexity

모델 적용 방식:

- 기존 raw sequence에 feature를 append
- 또는 별도 motion dynamics branch로 encode 후 fusion

기대:

- head-motion-only 상황에서 문헌상 중요한 derivative/statistical/temporal feature를 모델이 더 쉽게 사용한다.
- rise/drop timing과 high-FMS 구간 반응이 개선될 가능성이 있다.

위험:

- feature 수가 많아지면 overfit 가능성이 있다.
- 모든 feature가 causal인지 강하게 테스트해야 한다.

우선순위:

높음.

### B. Motion Complexity Auxiliary

구성:

- 최근 window의 effective dimensionality 또는 PCA complexity를 label/feature로 계산한다.
- current FMS 예측과 함께 complexity decrease state를 auxiliary로 예측하게 한다.

기대:

- 멀미가 올라갈 때 motion complexity가 줄어드는 패턴을 모델이 잡을 수 있다.
- plot에서 plateau/rise 상태 구분에 도움이 될 수 있다.

위험:

- Complexity-FMS 관계가 데이터셋별로 다를 수 있다.
- 먼저 offline correlation 진단을 해야 한다.

우선순위:

중간-높음. 먼저 diagnostic부터 한다.

### C. Multi-Timescale Learnable Response Bank

구성:

- 5s, 15s, 30s, 60s causal motion summary branch를 만든다.
- 각 timescale branch를 gate/attention으로 fusion한다.
- Person prior가 timescale gate를 조절하게 할 수 있다.

기대:

- 멀미의 누적/지연 반응을 명시적으로 모델링한다.
- 특정 participant는 빠르게, 다른 participant는 느리게 반응하는 차이를 반영할 수 있다.

우선순위:

높음. Future trajectory auxiliary 다음 구조 개선 후보로 적합하다.

### D. Coarse Severity Band Auxiliary

구성:

- Continuous FMS regression은 유지한다.
- 기존 cumulative ordinal head 외에 coarse band head를 추가한다.
- 예: low / mild / moderate / severe / extreme, 또는 데이터 분포 기반 quantile band.

기대:

- MAE만 최적화할 때 high-FMS 구간을 과소예측하는 문제를 줄일 수 있다.
- Plot에서 "위험 구간에 들어갔는지" 판단이 안정될 수 있다.

위험:

- Band threshold가 임의적이면 실험 해석이 흐려진다.
- FMS-20을 FMS-10 banding으로 단순 변환하면 안 된다.

우선순위:

중간. Current ordinal head가 이미 있으므로, high-FMS underprediction 문제가 확인되면 시도한다.

### E. Person Prior as Bias/Scale/Response-Speed Head

구성:

```text
calibration + static -> person prior p
p -> baseline bias
p -> amplitude scale
p -> response speed gate
online motion state -> dynamic sickness signal
prediction = bias + scale * dynamic_signal
```

기대:

- Static/calibration 정보가 단순 feature가 아니라 개인별 민감도 조절자로 작동한다.
- MSSQ와 calibration FMS trajectory를 더 의미 있게 사용할 수 있다.

위험:

- 잘못 만들면 calibration anchor에 과하게 묶여 post-calibration 변화를 못 따라갈 수 있다.
- `delta_from_calibration`처럼 calibration에 직접 묶는 구조는 피한다.

우선순위:

중간-높음. Multi-timescale response와 결합하면 가치가 크다.

### F. Self-Supervised Motion Pretraining

구성:

- FMS label 없이 head motion sequence만으로 pretraining한다.
- 예: masked motion reconstruction, next-window motion prediction, contrastive predictive coding, future motion energy prediction.

기대:

- DenseFMS label 수가 제한적일 때 motion encoder가 더 안정적이 될 수 있다.
- 특히 derivative/spectral/complexity feature와 결합 가능하다.

위험:

- Pretraining objective가 sickness와 무관하면 효과가 약할 수 있다.
- 구현 비용이 크고 검증 루프가 길어진다.

우선순위:

중간. 구조 개선 1-2개를 먼저 본 뒤 시도한다.

### G. Content/Scenario Prior

구성:

- VR scene, optical flow type, density, motion condition 같은 metadata가 있으면 session/content embedding으로 넣는다.
- Content-specific baseline risk와 dynamic response를 분리한다.

기대:

- 같은 head motion이라도 content가 다르면 sickness response가 달라질 수 있다.
- Content imbalance가 모델을 흔드는지 진단할 수 있다.

위험:

- Content label이 test/generalization에서 사용 불가능하면 deployment assumption과 맞지 않는다.
- Content shortcut으로 participant response를 덮어버릴 수 있다.

우선순위:

조건부. Metadata가 안정적으로 있고 deployment에서도 사용 가능할 때만 한다.

### H. Uncertainty / Mixture-of-Regimes Head

구성:

- 모델이 single mean만 예측하지 않고 uncertainty 또는 regime mixture를 예측한다.
- Regime 예: stable low, slow rise, rapid rise, plateau high, recovery/fall.

기대:

- 동일 motion에서 participant response가 갈리는 구간을 평균으로 뭉개는 문제를 완화할 수 있다.
- Plot에서 flat mean prediction으로 가는 현상을 줄일 수 있다.

위험:

- Negative log-likelihood류는 MAE를 직접 개선하지 않을 수 있다.
- Mixture collapse를 방지해야 한다.

우선순위:

중간-낮음. Current model이 계속 mean-regression으로 뭉개질 때 검토한다.

## 권장 우선순위

현재 데이터와 코드 상태를 기준으로, 새로 추가된 아이디어까지 포함하면 우선순위는 다음과 같다.

1. Future trajectory auxiliary head
2. Causal motion dynamics feature bank
3. Multi-timescale learnable response bank
4. Motion complexity diagnostic, 이후 complexity feature/auxiliary
5. Rise/fall/plateau event auxiliary
6. Person prior as bias/scale/response-speed
7. Coarse severity band auxiliary
8. Self-supervised motion pretraining
9. Content/scenario prior, metadata가 있을 때만
10. Uncertainty 또는 mixture-of-regimes head

## 바로 해볼 수 있는 실험 묶음

### Experiment Set 1. Motion Dynamics Diagnostic

목표:

- 실제 DenseFMS 데이터에서 velocity/acceleration/jerk/complexity가 FMS level 또는 FMS delta와 관련 있는지 확인한다.

할 일:

- Train split에서 causal rolling feature 계산
- FMS level, FMS delta, future rise label과 correlation/MI 분석
- Validation split에서는 diagnostic만 확인, selection에는 사용하지 않는다.

산출물:

- feature correlation table
- high-FMS vs low-FMS feature distribution
- rise event 전후 motion dynamics plot

### Experiment Set 2. Dynamics Feature Branch

목표:

- 문헌에서 쓰인 head movement feature를 현재 deep model에 추가한다.

할 일:

- `motion_feature_mode`에 extended causal dynamics mode 추가
- raw append 방식과 branch fusion 방식을 비교
- 기준 모델과 동일 seed/split으로 1차 비교

선택 기준:

- MAE가 기준선보다 좋아지거나, MAE 유지 + plot/shape 개선.

### Experiment Set 3. Future Trajectory + Risk Auxiliary

목표:

- onset forecasting 문헌의 방향을 현재 DenseFMS dense-label 설정에 맞게 적용한다.

할 일:

- t+5s, t+10s, t+15s future FMS 또는 future delta auxiliary 추가
- 기존 rapid-rise risk auxiliary는 유지
- small loss weight에서 시작

선택 기준:

- current MAE 유지/개선
- delta corr, direction acc, fixed plot 개선

### Experiment Set 4. Complexity Feature/Auxiliary

목표:

- motion complexity 감소가 DenseFMS에서도 sickness marker인지 확인한다.

할 일:

- rolling PCA/eigenvalue complexity 계산
- feature로 append하거나 auxiliary target으로 사용
- high-FMS underprediction과 complexity 감소의 관계 확인

선택 기준:

- rise/plateau 구간 plot 개선
- high-FMS 구간 과소예측 감소

## 현재 판단

추가 문헌 탐색까지 반영하면, "그냥 head를 더 깊게 만든다"보다 다음 두 방향이 더 설득력 있다.

1. Head motion의 derivative, spectral, complexity 정보를 causal feature/branch로 명시적으로 제공한다.
2. Current FMS 한 점 예측을 넘어 future trajectory, rise/fall/plateau event, severity band를 auxiliary로 같이 학습한다.

현재 선택 모델에서 risk auxiliary가 도움이 된 것은 중요한 신호다. 선행 연구도 onset forecasting과 multimodal temporal fusion을 강조한다. 따라서 다음 구조 개선은 `future trajectory auxiliary`를 중심으로 두고, 동시에 head-motion-only 문헌에서 강하게 나온 `velocity/acceleration/jerk/complexity` feature bank를 붙이는 방향이 가장 타당하다.

## 참고 문헌 링크

- Islam et al. 2022, ISMAR, forecasting onset with multimodal deep fusion: https://doi.org/10.1109/ISMAR55827.2022.00026
- Islam et al. 2021, ISMAR, integrated HMD sensor deep fusion: https://doi.org/10.1109/ISMAR52148.2021.00017
- Islam et al. 2021 arXiv preprint: https://arxiv.org/abs/2108.06437
- Salehi et al. 2024, head movement patterns: https://arxiv.org/abs/2402.02725
- IEEE VR 2025 motion complexity paper: https://doi.org/10.1109/VR59515.2025.00040
- Setu et al. 2024, Mazed and Confused dataset: https://arxiv.org/abs/2409.06898
- Kim et al. 2021, clinical predictors: https://www.nature.com/articles/s41598-021-91573-w
- Kelly et al. 2026 OSF preprint, FMS/SSQ tolerance banding: https://doi.org/10.31234/osf.io/p3xy4_v1
