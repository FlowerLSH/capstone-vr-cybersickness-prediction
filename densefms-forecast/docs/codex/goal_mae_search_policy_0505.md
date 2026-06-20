/goal

DenseFMS future forecasting 코드베이스에서 corrected start_only 조건의 validation MAE를 낮추는 장기 adaptive search를 진행해줘.

이번 goal은 단순 run 개수 늘리기가 아니라,
corrected start_only 조건에서 무누수 / 허용 입력 / 공정 평가를 지키면서
더 나은 temporal modeling / calibration usage / fusion 구조 / optimization setting을 찾는 것이다.

출력 디렉터리:
- runs/goal_mae_search_v2/

============================================================
0. CRITICAL SUMMARY — 반드시 먼저 따를 것
============================================================

이번 goal은 validation 기반 장기 adaptive search를 수행한 뒤,
모든 선택을 validation 기준으로 lock한 다음,
마지막에 frozen best config로 final test를 딱 1회만 수행하는 것이다.

Non-negotiable rules:

1. Adaptive search 중에는 test set을 절대 열지 마라.
2. Primary selection metric은 h=5/10/15 validation MAE 평균이다.
3. Main track은 corrected start_only, anchor_mode=none이다.
4. current FMS, future FMS, target FMS, sparse_observed, recent_start_observed는 input으로 절대 사용하지 마라.
5. validation 기준 최종 후보를 lock하기 전에는 test metric / test prediction / test plot을 생성하지 마라.
6. final test는 모든 validation-based selection이 끝난 뒤 frozen best config로 1회만 허용한다.
7. final test 이후에는 어떤 architecture / hyperparameter / preprocessing / window / loss / static usage도 변경하지 마라.
8. run 개수 제한은 없다. 전체 search budget은 wall-clock time으로만 제한한다.
9. 30분 미만 실행 후 goal complete 처리하지 마라.
10. primary metric 개선폭이 10% 미만이면, 명시적 사용자 중단이나 환경 blocker가 없는 한 최소 max(4시간, wall-clock budget의 70%)를 사용하기 전 final selection lock으로 넘어가지 마라.
11. hyperparameter tuning은 허용되지만, hyperparameter tuning만으로 architecture-first goal을 만족한 것으로 보지 마라.
12. architecture family는 최소 7개 이상 검토하라. 3개만 검토하고 종료하지 마라.
13. 7개 architecture family를 검토하기 전에는, 명시적 사용자 중단 / GPU 문제 / 데이터 문제 / 필수 audit 실패 같은 blocker가 없는 한 final selection lock으로 넘어가지 마라.
14. 새 architecture/model class는 full run 전에 forward shape, leakage, alignment sanity test를 통과해야 한다.
15. RUN_STATE.md, resume_manifest.csv, experiment_log.csv, leaderboard.csv는 run마다 갱신하라.
16. interrupt / timeout / crash가 발생해도 중간 결과를 사람이 해석하고 resume할 수 있어야 한다.
17. leakage 방지와 공정한 평가는 MAE 개선보다 우선한다.

============================================================
1. 반드시 먼저 확인할 문서
============================================================

코드 수정이나 학습 실행 전에 반드시 다음을 확인하라.

- AGENTS 관련 규칙
- README
- docs/codex/goal_mae_search_policy*.md가 있으면 반드시 읽고 준수
- docs/codex/test.md가 있으면 읽고 준수

주의:
- docs/codex/test.md는 현재 없을 수 있다.
- docs/codex/test.md가 없으면 blocker로 처리하지 말고 RUN_STATE.md에 “docs/codex/test.md absent”라고 기록한 뒤 진행하라.

우선순위:
- 시스템 지시, AGENTS, repository safety rule, leakage/safety rule은 항상 따른다.
- 기존 goal_mae_search_policy.md 계열 문서와 이 goal의 세부 조건이 충돌할 경우, 이번 0505/v2 search에 대한 task-specific 조건은 이 프롬프트를 우선한다.
- 단, 이 프롬프트는 AGENTS나 safety/no-leakage 원칙보다 우선하지 않는다.

============================================================
2. 핵심 목표
============================================================

목표는 많은 실험을 수행하는 것이 아니다.
목표는 corrected start_only 조건에서 validation MAE를 의미 있게 낮추는 것이다.

모든 실험은 이전 결과에서 관찰된 실패 원인을 해결하기 위한 가설을 가져야 한다.

이번 v2에서 우선적으로 해결하려는 실패 양상:

- 예측값이 평균적인 FMS 구간으로 수축되는 문제
- high-FMS 구간을 제대로 예측하지 못하는 문제
- horizon이 길어질수록 예측이 급격히 무너지는 문제
- calibration 정보를 충분히 활용하지 못하는 문제
- start_only FMS에 의존하되, current/future FMS leakage 없이 미래 FMS를 예측해야 하는 문제
- static feature가 도움이 될 수 있으나 subject/session identity leakage 없이 활용해야 하는 문제

성능 개선 가능성이 낮은 방향은 중단할 수 있다.
하지만 시간이 남아 있고 아직 탐색하지 않은 유망한 branch family가 있다면
search를 complete 처리하지 말고 더 유망한 방향으로 확장하라.

MAE 목표 달성보다 leakage 방지와 공정한 평가가 절대 우선이다.

============================================================
3. Budget / wall-clock tracking
============================================================

FULL_TRAINING_ALLOWED = true

이번 goal에서는 full training / long adaptive search를 허용한다.

Budget:
- MAX_WALL_CLOCK_HOURS = 8
- MAX_TOTAL_RUNS는 없다.
- MAX_EPOCHS_PER_RUN = 80
- EARLY_STOPPING_PATIENCE = 8

run 개수 제한은 두지 않는다.
이번 search는 run 개수가 아니라 wall-clock time으로 제한한다.

단, run 개수를 늘리는 것이 목표가 아니다.
의미 없는 micro-run 반복은 금지한다.
각 run은 명확한 가설과 이전 결과에 근거한 선택 이유를 가져야 한다.

Soft run-count guard:
- 총 run 수가 60개를 초과하면 RUN_STATE.md에 초과 이유를 기록하라.
- 60개 초과 시 micro-run 반복이 아닌지 감사하라.
- 같은 family에서 작은 hyperparameter 변형만 반복되고 있다면 즉시 branch 전환 여부를 판단하라.
- run cap은 아니지만, 60개 이후의 run은 selection_reason과 expected learning value를 더 명확히 기록하라.

Wall-clock 관리 규칙:

1. goal 시작 시각을 RUN_STATE.md에 기록하라.
2. 각 run 시작 전 elapsed time / remaining time을 RUN_STATE.md에 기록하라.
3. 각 run 종료 후 elapsed time / remaining time을 RUN_STATE.md에 갱신하라.
4. 남은 시간이 부족하면 새 training run을 시작하지 말고 결과 정리와 final selection lock을 우선하라.
5. final selection lock, final test 1회, 최종 보고를 위해 마지막 30~45분 정도는 남겨두는 것을 목표로 하라.
6. 단, wall-clock 상황상 final test를 안전하게 수행할 수 없으면 FINAL_SELECTION_LOCK.md까지 작성하고, final test 미수행 사유를 명확히 기록하라.

Staged epoch budget:
- 새 architecture family representative branch: 우선 20~30 epoch 수준으로 평가하라.
- 가능하면 7개 architecture family 각각에서 smoke/sanity test 후 representative validation run을 수행하라.
- promising refinement branch: 필요하면 최대 80 epoch까지 허용한다.
- finalist rerun / seed check: 최대 80 epoch까지 허용한다.
- 모든 후보를 무조건 80 epoch로 돌려 architecture coverage를 줄이지 마라.
- 단, 기존 CLI나 early stopping 구조상 정확히 staged epoch 적용이 어렵다면 가장 가까운 방식으로 구현하고 RUN_STATE.md에 기록하라.

GPU가 없거나 runtime이 과도하거나 training environment가 불안정하면
full training을 강행하지 말고 smoke/audit만 수행한 뒤 중단하라.

============================================================
4. v1 결과 해석
============================================================

이전 runs/goal_mae_search 결과는 유의미한 성능 개선으로 보지 않는다.

v1 요약:
- v1 no-static baseline primary val MAE mean over h=5/10/15 ≈ 2.4943
- v1 best primary val MAE mean over h=5/10/15 ≈ 2.4747
- 개선폭 ≈ 0.0196 MAE, 약 0.8%

해석:
- v1은 누수/평가 조건은 지켰지만 유의미한 성능 개선 후보를 찾지 못한 1차 탐색이다.
- v1에서 final test를 너무 일찍 실행한 것은 절차상 한계로 기록한다.
- v2 adaptive search에서는 v1 test 결과를 사용하지 않는다.
- v1 test 파일을 열지 마라.
- 이미 이전 context나 문서에 남아 있는 v1 test 수치가 있더라도, v2의 model selection, hyperparameter selection, branch selection, 실험 방향 선택에는 명시적으로 사용하지 마라.
- v2에서는 validation 기준으로만 후보를 고른 뒤, 마지막에 frozen best config로 final test를 1회만 수행한다.

============================================================
5. Test 사용 정책
============================================================

Adaptive search 중에는 test set을 절대 열지 않는다.

Adaptive search 중 금지:
- test metric 생성 금지
- test prediction CSV 생성 금지
- test plot 생성 금지
- v1 test 결과 파일 읽기 금지
- v1 test 결과를 실험 선택에 사용 금지
- test-driven tuning 금지

일반 adaptive search run에서는 train/val metric과 train/val prediction CSV만 생성한다.

Final test 허용 조건:

Final test는 모든 validation-based model selection이 끝난 뒤,
frozen best config로 딱 1회만 허용한다.

Final test 전에 반드시 수행:
1. validation 기준 best candidate 확정
2. best_model_summary.md 작성
3. FINAL_SELECTION_LOCK.md 작성
4. FINAL_SELECTION_LOCK.md에 다음을 명시:
   - 선택된 run_name
   - 선택된 checkpoint / config
   - 선택 이유
   - primary validation metric
   - h=5/10/15 validation MAE
   - static 사용 여부
   - architecture family
   - final test command
   - final test 이후 추가 튜닝을 하지 않겠다는 명시적 선언

Final test 이후 금지:
- 추가 training run 금지
- architecture 변경 금지
- hyperparameter 변경 금지
- window setting 변경 금지
- calibration_seconds 변경 금지
- recent_window_seconds 변경 금지
- loss 변경 금지
- static usage 변경 금지
- preprocessing 변경 금지
- split 변경 금지
- final test 결과를 보고 다시 후보 선택 금지

Final test는 held-out evaluation으로만 보고한다.
Final test 결과가 나쁘더라도 그 결과를 근거로 다시 tuning하지 않는다.

만약 final test command가 mechanical error로 실패하면:
- frozen config를 바꾸지 않는 범위에서만 경로/스크립트 호출 문제를 수정할 수 있다.
- 모델, 하이퍼파라미터, 데이터, 전처리, split은 바꾸지 마라.
- 실패와 수정 내용을 final_test_audit.md에 기록하라.

============================================================
6. Main track 정의
============================================================

이번 v2의 main deployment-realistic track은 다음 설정을 기준으로 한다.

- fms_context_mode = start_only
- anchor_mode = none
- anchor_interval_seconds = 0
- sparse_observed 사용 금지
- recent_start_observed 사용 금지

Static feature는 허용한다.

허용 static:
- Age
- Gender
- MSSQ

static 정보가 validation MAE를 개선한다면 main track 성능으로 인정한다.

단, 다음 정보는 금지한다.
- participant_id
- session_id
- condition id
- trial id
- experiment id
- file name에서 유도한 subject/session 식별자
- train/val/test split 정보를 암시하는 feature
- participant/session identity와 동등하게 작동할 수 있는 derived identifier

static-enabled 결과와 no-static 결과는 반드시 분리해서 기록한다.
static 사용 여부를 숨기지 마라.

============================================================
7. FMS input policy
============================================================

“calibration 이후 FMS 전면 금지”가 아니다.
FMS input은 fms_context_mode에 따라 제한한다.

Main track은 fms_context_mode=start_only이다.

start_only에서 허용되는 FMS:

1. calibration encoder에 들어가는 calibration 구간 FMS history
2. 각 prediction 시점 recent motion window의 시작 FMS:
   FMS[t - recent_window_steps + 1]
3. 단, 해당 start FMS가 missing / NaN / non-finite이면
   Start FMS missing policy에 따른 fallback FMS

start_only에서 금지되는 FMS:

1. target FMS:
   FMS[t + horizon_steps]
2. current index t 이후의 future FMS
3. current FMS:
   FMS[t]
4. recent window 내부의 dense FMS sequence 전체
5. sparse_observed
6. recent_start_observed
7. calibration_end anchor
8. sparse_anchor
9. start_only 정책 밖의 추가 anchor-assisted FMS
10. sparse_observed / recent_start_observed flag 또는 그와 동등한 관측 여부 정보
11. validation/test target이나 future FMS에서 파생한 feature

============================================================
8. Start FMS naming / metadata policy
============================================================

원칙 이름은 다음을 사용한다.

- start_fms_index
- start_fms_time
- start_fms_value

현재 CSV의 기존 컬럼명은 backward-compatible alias로만 해석한다.

- anchor_index = start_fms_index
- anchor_time = start_fms_time
- anchor_fms = start_fms_value

anchor_mode=none이어도 기존 코드 호환을 위해 CSV에 anchor_* metadata가 있을 수 있다.
이 경우 anchor_*는 추가 anchor-assisted FMS가 아니라
start_only의 start FMS metadata alias로만 해석한다.

============================================================
9. Start FMS missing policy
============================================================

start_only의 원칙적 start FMS index는:

FMS[t - recent_window_steps + 1]

이다.

단, 해당 index의 FMS가 missing / NaN / non-finite인 경우에는
현재 구현의 fallback 정책을 허용한다.

fallback 정책:
- nominal start index 이하에서 가장 가까운 최신 finite FMS를 사용한다.
- fallback FMS index는 nominal start index 이하이어야 한다.
- current index t보다 이후의 FMS는 절대 fallback으로 사용할 수 없다.
- target index t + horizon_steps의 FMS는 절대 fallback으로 사용할 수 없다.
- future FMS를 사용한 backward fill은 금지한다.

가능하면 prediction CSV 또는 metadata에 다음을 기록한다.
- anchor_index 또는 start_fms_index
- anchor_time 또는 start_fms_time
- anchor_fms 또는 start_fms_value
- anchor_is_fallback
- nominal_start_index
- nominal_start_time

해당 컬럼 추가가 어렵다면 leakage_audit.md에서
actual anchor_index가 nominal start index 이하인지 검증하라.

============================================================
10. Prediction indexing / alignment
============================================================

recent motion window가 [t - recent_window_steps + 1, ..., t]라면,
start_only FMS는 원칙적으로 FMS[t - recent_window_steps + 1]만 허용한다.

target은 정확히 FMS[t + horizon_steps]이다.

head/motion input은 current time t까지만 사용할 수 있다.
t 이후의 head/motion/FMS는 input으로 절대 사용하지 않는다.

검증:
- horizon_seconds=5이면 target은 정확히 t+5초여야 한다.
- horizon_seconds=2.5이면 target은 정확히 t+2.5초여야 한다.
- horizon_seconds=1이면 target은 정확히 t+1초여야 한다.

Prediction CSV alignment에서 다음을 검증한다.
- current_time
- target_time
- start_fms_time 또는 anchor_time
- start_fms_index 또는 anchor_index
- start_fms_value 또는 anchor_fms
- predicted_fms
- target_fms

anchor_index가 current index t 이후이거나 target index를 참조하면 leakage로 간주한다.

============================================================
11. Primary selection metric
============================================================

Primary selection metric:

validation MAE mean over h=5, h=10, h=15

최종 main candidate는 h=5/10/15 validation MAE 평균이 가장 낮은 후보를 기준으로 선택한다.

해석:
- h=5, h=10, h=15는 main forecasting 성능으로 본다.
- h=2.5는 valid forecasting result로 별도 기록한다.
- h=1은 diagnostic / lower-bound / sanity-check로만 사용한다.
- h=1은 목표 달성 여부와 final model selection에 반영하지 않는다.

horizon별 best result도 별도로 기록한다.
- best h=2.5
- best h=5
- best h=10
- best h=15
- best diagnostic h=1

h=1에서 좋은 MAE가 나오더라도 “목표 달성”으로 보고하지 마라.

============================================================
12. Improvement target
============================================================

corrected start_only baseline primary validation MAE를 기준으로 개선율을 계산한다.

현재 참고 baseline:
- primary val MAE mean over h=5/10/15 ≈ 2.4943

v2에서 corrected baseline을 재현하면 v2에서 재현된 baseline을 기준으로 삼는다.

목표 수준:
- weak improvement: baseline 대비 5% 이상 개선
- minimum meaningful improvement: baseline 대비 10% 이상 개선
- strong improvement: baseline 대비 15% 이상 개선
- stretch improvement: baseline 대비 20% 이상 개선 또는 primary val MAE <= 2.0
- MAE 1.2 / 1.0은 매우 공격적인 long-shot stretch target으로만 둔다.

해석:
- 5% 미만 개선은 유의미한 성능 개선으로 보지 않는다.
- 5~10% 개선은 weak lead로 기록한다.
- 10% 이상 개선하면 minimum candidate로 기록한다.
- 15% 이상 개선하면 strong candidate로 기록한다.
- 20% 이상 개선하거나 primary val MAE <= 2.0이면 stretch success로 기록한다.

중요:
- 10% 미만 개선은 successful final candidate로 보지 않는다.
- 하지만 search 종료 시점의 best observed validation candidate는 반드시 보고한다.
- 목표 달성에 실패했더라도 best validation result, 실패 원인, 다음 우선순위를 기록하라.

조기 종료 방지:
- primary metric 개선폭이 10% 미만이면, 명시적 사용자 중단이나 환경 blocker가 없는 한 최소 max(4시간, wall-clock budget의 70%)를 사용하기 전 final selection lock으로 넘어가지 마라.
- primary metric 개선폭이 10% 미만이면 “아직 유의미한 후보를 찾지 못함”으로 보고하고, 남은 branch를 계속 탐색한다.
- primary metric 개선폭이 10~15% 사이이면 candidate는 찾은 것이지만, 시간이 충분히 남아 있고 미검토 branch가 있다면 추가 개선을 계속 시도한다.
- primary metric 개선폭이 15% 이상이면 strong candidate로 보고 재현성 확인을 수행한다.
- primary metric 개선폭이 20% 이상이면 stretch success로 기록한다.
- 그래도 final test는 validation-based final selection lock 이후 딱 1회만 수행한다.

============================================================
13. Architecture-first adaptive search policy
============================================================

이번 v2의 핵심은 cheap hyperparameter sweep이 아니라 architecture-level adaptive search다.

다만 hyperparameter tuning도 모델 튜닝의 중요한 일부로 인정한다.
Hyperparameter tuning은 다음 용도로 사용하라.

1. corrected baseline 안정화
2. architecture family 간 공정한 비교
3. promising architecture candidate refinement
4. 실패한 구조가 optimization 문제로 실패했는지 확인
5. best candidate의 재현성 및 안정성 확인

하지만 hyperparameter tuning만으로 architecture-first goal을 만족한 것으로 보지 않는다.
같은 모델 계열에서 작은 learning rate/dropout/hidden size 변형만 반복하지 마라.

단순 변경만으로 architecture-level 변경으로 인정하지 않는 예:
- hidden size만 변경
- dropout만 변경
- learning rate만 변경
- weight decay만 변경
- batch size만 변경
- layer 수만 소폭 변경
- loss weight만 변경
- scheduler만 변경

architecture-level 변경으로 인정되는 예:
- temporal encoder family 변경: LSTM, GRU, TCN, causal Transformer, multi-scale encoder
- calibration encoder와 recent encoder의 구조적 분리
- calibration summary를 recurrent initial state 또는 gating으로 주입
- static feature를 concat이 아닌 FiLM/gating/attention 방식으로 fusion
- start FMS를 scalar concat이 아닌 embedding/gating branch로 처리
- statistical summary feature와 neural temporal representation의 hybrid fusion
- horizon-specific head 또는 multi-horizon shared encoder + horizon head 구조
- ordinal/regression dual-head 구조
- high-FMS 대응을 위한 density-aware 또는 ordinal-aware output structure

architecture family coverage requirement:

- 최소 7개 이상의 서로 다른 architecture family를 검토하라.
- “최소 3개”가 아니다. 3개만 검토하고 종료하지 마라.
- final selection lock 전에는 서로 다른 architecture family 7개 이상을 smoke/sanity test하고, 가능한 경우 각 family별 representative candidate를 20~30 epoch 수준으로 validation run하라.
- 7개 family를 모두 검토하기 전에는, 명시적 사용자 중단 / GPU 문제 / 데이터 문제 / 필수 audit 실패 같은 blocker가 없는 한 final selection lock으로 넘어가지 마라.
- 7개 family를 모두 실행하지 못했다면 RUN_STATE.md와 final_report.md에 다음을 기록하라.
  - 실행하지 못한 architecture family
  - 실행하지 못한 이유
  - 그 판단이 단순 조기 종료가 아닌 이유
  - 해당 family가 현재 best 개선에 덜 유망하다고 판단한 근거 또는 환경상 실행 불가능했던 근거

우선 검토할 architecture family:

1. GRU / LSTM recurrent family
- compact GRU
- stacked GRU/LSTM
- attention pooling over hidden states
- calibration summary를 hidden state initialization에 사용하는 구조

2. TCN / dilated causal CNN family
- dilated causal TCN
- recent-window TCN + calibration summary fusion
- causal padding만 허용하고 current time t 이후 input은 금지

3. multi-scale TCN family
- kernel sizes 3/5/9 등 multi-scale temporal pattern 추출
- short-term motion burst와 longer trend를 함께 포착
- multi-branch TCN 또는 inception-style temporal convolution 검토

4. lightweight causal Transformer family
- calibration tokens + recent motion tokens 분리
- temporal pooling attention
- horizon-aware query 또는 horizon embedding
- t 이후 input을 볼 수 없는 causal structure 유지

5. calibration-summary gated fusion family
- calibration encoder와 recent encoder를 구조적으로 분리
- calibration summary를 gating, recurrent initial state, FiLM-like modulation으로 주입
- calibration 정보를 단순 concat보다 적극적으로 활용

6. static FiLM / personality-aware fusion family
- Age/Gender/MSSQ branch를 단순 concat이 아니라 FiLM/gating/attention으로 사용
- static dropout을 적용해 static feature 과의존 방지
- participant/session identity는 절대 사용 금지

7. hybrid statistical + neural encoder family
- calibration/recent window의 mean, std, min, max, range, slope, delta, volatility 추출
- neural encoder output과 concat 또는 gating fusion
- 평균 수축과 high-FMS 실패를 완화할 수 있는지 검토

8. ordinal / high-FMS-aware auxiliary head family
- FMS가 ordinal score라는 점을 반영
- regression head와 ordinal/high-FMS-aware auxiliary head를 함께 검토
- primary metric은 기존 h=5/10/15 validation MAE로 유지

9. multi-horizon shared encoder + horizon-specific head family
- h=5/10/15를 하나의 shared encoder로 처리하되 horizon-specific head 또는 horizon embedding을 사용
- horizon이 길어질수록 무너지는 문제를 구조적으로 다룬다.
- h=1은 diagnostic으로만 사용하고 selection에는 반영하지 않는다.

Architecture search loop:

1. 새 architecture family는 먼저 smoke test를 수행한다.
2. smoke test가 통과하면 representative candidate 1개를 20~30 epoch 수준으로 validation training한다.
3. 가능한 경우 7개 architecture family 각각에서 representative validation result를 확보한다.
4. primary metric과 horizon별 failure pattern을 비교한다.
5. promising family만 expand한다.
6. promising refinement는 최대 80 epoch까지 허용한다.
7. 같은 family에서 2~3회 연속 개선이 없으면 중단하고 다른 branch로 전환한다.
8. best architecture 후보는 가능하면 최소 1회 rerun 또는 seed 확인을 수행한다.
9. 모든 구조 변경은 왜 현재 실패 양상을 해결할 수 있는지 hypothesis를 기록한다.

새 model class / 새 architecture sanity requirement:

새 architecture 또는 새 model class를 full run하기 전에 다음을 확인하라.

- forward pass shape sanity
- batch dimension / time dimension alignment
- output shape가 horizon별 target shape와 일치하는지 확인
- no future motion input 확인
- no current FMS input 확인
- no target FMS input 확인
- no future FMS input 확인
- start_fms_index / anchor_index가 nominal start index 이하인지 확인
- fallback start FMS가 current index 또는 target index를 참조하지 않는지 확인
- sparse_observed / recent_start_observed가 input feature로 들어가지 않는지 확인
- validation/test statistics가 train preprocessing fit에 사용되지 않는지 확인

가능하면 이 sanity check를 자동화된 script/test 또는 audit command로 남겨라.
자동화가 어렵다면 leakage_audit.md와 RUN_STATE.md에 확인 근거를 기록하라.

필요하다면 새 model class, 새 config, 새 runner helper를 구현해도 된다.
단, 기존 baseline 경로를 깨지 말고 additive 방식으로 구현하라.
새 구조는 leaderboard의 model_type과 run_name에서 명확히 구분되어야 한다.

============================================================
14. Literature-informed search policy
============================================================

Literature-informed search는 optional이며 secondary이다.

목적은 문헌조사가 아니라,
현재 corrected start_only 조건에 적용 가능한 architecture candidate를 얻는 것이다.

규칙:
- broad literature survey를 하지 마라.
- literature review에 총 20분 이상 쓰지 마라.
- repository 내부 docs, 기존 notes, local papers를 우선 확인하라.
- 인터넷 접근이 불가능하면 web search를 강행하지 마라.
- 외부 코드를 복사하지 마라.
- 논문 구조를 그대로 복제하지 말고, 현재 input contract와 leakage policy에 맞게 최소 변형하라.
- 즉시 구현 가능한 아이디어가 없으면 literature review를 중단하고 architecture search로 돌아가라.

문헌 참고를 한 경우, RUN_STATE.md 또는 experiment_log.md에 다음을 간단히 기록하라.

- 참고한 자료 이름
- 핵심 architecture idea
- 현재 task에 적용 가능한 이유
- leakage risk
- 구현 여부
- 구현하지 않았다면 이유

별도의 literature_architecture_notes.md는 필수는 아니다.
다만 문헌 기반 아이디어를 여러 개 사용했다면 생성해도 된다.

============================================================
15. Adaptive search branch coverage
============================================================

v2를 종료하기 전에는 아래 branch family를 검토해야 한다.
각 branch는 반드시 실행하거나, 실행하지 않은 이유를 RUN_STATE.md에 기록한다.

단, branch coverage는 checklist가 아니라 adaptive search를 위한 safety net이다.
모든 branch를 동일 비중으로 실행하지 마라.
초기 representative experiment 이후 promising branch에 시간을 집중하라.

Priority A:
1. baseline / runner consistency branch
2. leakage / input audit branch
3. architecture branch
4. window branch

Priority B:
5. static branch
6. loss / imbalance branch
7. optimization / hyperparameter branch

Priority C:
8. curriculum / robustness branch

Branch family:

1. baseline / runner consistency branch
- corrected start_only baseline 재확인
- single-horizon vs multi-horizon 차이 확인
- h=5/10/15 primary metric 계산 방식 확인
- v2 baseline이 v1 baseline과 크게 다르면 원인 분석

2. leakage / input audit branch
- train/val/test subject split 확인
- normalization / scaling / imputation fit 범위 확인
- target/current/future FMS 미사용 확인
- anchor_index / start_fms_index alignment 확인
- fallback FMS가 nominal start index 이하인지 확인
- sparse_observed / recent_start_observed 미사용 확인

3. architecture branch
- 기존 구조가 평균 수축을 보이면 temporal encoder 또는 fusion head를 바꿔본다.
- 최소 7개 이상의 서로 다른 architecture family 검토를 기본 요구사항으로 둔다.
- 3개만 검토하고 종료하지 마라.
- 새 구조를 도입하기 전, 왜 그 구조가 현재 문제를 해결할 수 있는지 기록한다.
- 새 architecture는 full run 전에 sanity check를 통과해야 한다.
- 7개 architecture family를 모두 검토하지 못했다면 구체적 사유를 RUN_STATE.md와 final_report.md에 기록한다.

4. window branch
- calibration_seconds
- recent_window_seconds
- horizon_seconds

주의:
- primary selection metric은 항상 h=5/10/15 validation MAE 평균이다.
- main candidate는 반드시 h=5, h=10, h=15 validation metric을 모두 제공해야 한다.
- h=1은 diagnostic으로만 사용하고 selection에서 제외한다.
- h=2.5는 valid forecasting result로 기록하되 primary selection에서는 제외한다.
- horizon_seconds를 바꾼 실험은 primary metric과 충돌하지 않도록 h=5/10/15 기준 결과를 별도로 기록하라.
- calibration_seconds나 recent_window_seconds 변경으로 평가 구간이 달라지는 경우 common-window metric도 함께 기록하라.
- 평가 구간이 쉬워져서 좋아진 것인지 반드시 확인하라.

5. static branch
- no-static
- Age/Gender/MSSQ static
- static concat
- static FiLM/gating/attention fusion
- static dropout
- static이 악화되면 그 이유를 분석하고 중단
- static 사용 여부는 반드시 leaderboard에 명시

6. loss / imbalance branch
- MAE
- SmoothL1
- Huber
- FMS-density-aware weighting
- high-FMS weighting
- ordinal-aware auxiliary loss
- trend-aware auxiliary loss는 필요시 검토

주의:
- target smoothing으로 정답 자체를 쉽게 만들지 마라.
- evaluation code를 지표가 좋아지도록 바꾸지 마라.

7. optimization / hyperparameter branch
- learning rate
- weight decay
- dropout
- batch size
- hidden size / d_model
- scheduler
- SmoothL1 beta 또는 Huber beta
- gradient clipping
- optimizer
- early stopping patience

주의:
- optimization branch는 중요하지만 architecture search를 대체할 수 없다.
- 같은 실패 양상이 반복되면 같은 계열의 작은 변형을 계속 늘리지 말고 branch를 전환한다.

8. curriculum / robustness branch
- horizon curriculum
- calibration curriculum
- temporal masking
- sensor corruption
- static dropout
- input corruption
- 단, 최종 validation은 원래 조건에서 수행한다.

============================================================
16. Evaluation policy
============================================================

일반 adaptive search run에서는 train/val metric과 train/val prediction CSV만 생성한다.

Adaptive search 중 금지:
- test metric 생성 금지
- test prediction CSV 생성 금지
- test plot 생성 금지

val/test의 기존 dense rolling evaluation 정책은 유지한다.
성능 개선 목적으로 evaluation sample density, stride, window overlap 정책을 바꾸지 않는다.

validation sample을 제거하거나 쉬운 구간만 평가하는 방식으로 지표를 좋게 만들지 않는다.

calibration_seconds나 horizon_seconds를 바꿔 실험할 경우,
natural metric과 common-window metric을 모두 기록한다.

common-window metric은 calibration/horizon 변화로 평가 구간이 달라지는 문제를 보정하기 위한 것이며,
cherry-picking 용도로 사용하지 않는다.

Calibration sweep에서:
- common_target_start = max(calibration_seconds_list) + horizon_seconds
- 모든 run의 metrics를 target_time >= common_target_start 구간에서도 별도 계산

Horizon sweep에서:
- common_current_end = session_end_time - max(horizon_seconds_list)
- 모든 run의 metrics를 current_time <= common_current_end 구간에서도 별도 계산

Grid sweep에서:
- common_current_start = max(calibration_seconds_list)
- common_current_end = session_end_time - max(horizon_seconds_list)
- 모든 run의 metrics를 common current time range에서 별도 계산

============================================================
17. Leakage 금지
============================================================

다음은 절대 금지한다.

- train/val subject 혼합
- validation/test를 train preprocessing에 사용
- validation/test statistics로 normalization/scaling/imputation fit
- target FMS를 input에 사용
- future FMS를 input에 사용
- current FMS FMS[t]를 input에 사용
- recent window 내부 dense FMS sequence를 input에 사용
- future head/motion input 사용
- sparse_observed 사용
- recent_start_observed 사용
- calibration_end anchor 사용
- sparse_anchor 사용
- participant/session/condition/trial identity 사용
- file-derived identity 사용
- train/val/test split을 암시하는 feature 사용
- target smoothing으로 정답 자체를 쉽게 만들기
- evaluation code를 지표가 좋아지도록 바꾸기
- validation sample을 쉬운 구간만 남기도록 제거하기
- test-driven tuning
- v1 test 결과 파일 참조
- 이미 알려진 v1 test 수치를 selection/branch 판단에 사용
- final test 이후 추가 실험

모든 새 architecture는 input contract를 따라야 한다.
새 구조가 추가 feature를 만든다면 해당 feature가 current time t 이후 정보를 포함하지 않는지 반드시 확인한다.

============================================================
18. Interrupt / resume
============================================================

중간 interrupt, timeout, crash, OOM, 사용자 중단이 발생할 수 있음을 전제로 진행한다.

반드시 실시간 유지:
- runs/goal_mae_search_v2/RUN_STATE.md
- runs/goal_mae_search_v2/resume_manifest.csv
- runs/goal_mae_search_v2/experiment_log.csv
- runs/goal_mae_search_v2/leaderboard.csv

중단 시점까지의 결과를 사람이 해석할 수 있어야 한다.

재개 시:
- completed run은 중복 실행하지 않는다.
- interrupted run은 checkpoint/log 유효성을 확인해 resume한다.
- resume가 위험하거나 불가능하면 새 run_name으로 재실행할지 판단한다.
- failed/interrupted run을 기록 없이 숨기지 않는다.

RUN_STATE.md에는 항상 다음을 최신 상태로 유지한다.

- goal 시작 시각
- elapsed wall-clock time
- remaining wall-clock budget
- total run count
- 60 run 초과 여부와 초과 이유
- 현재까지 완료된 run
- 현재 진행 중인 run
- 현재 best primary validation result
- h=5/10/15 각각의 best validation result
- h=2.5 best validation result
- h=1 diagnostic result
- 시도한 branch family
- 아직 시도하지 않은 branch family와 이유
- 시도한 architecture family 수
- 실행한 architecture family 목록
- 미실행 architecture family 목록과 이유
- 7개 architecture family coverage 달성 여부
- 실패한 방향과 중단 이유
- 현재까지의 해석
- 다음에 이어서 실행할 후보
- architecture family별 현재 판단
- optimization branch별 현재 판단
- static branch 결과
- leakage audit 상태
- docs/codex/test.md 존재 여부
- final selection lock 여부
- final test 수행 여부

============================================================
19. Required outputs
============================================================

Output은 실시간 / 초기 / 최종 산출물로 나누어 관리한다.

A. 반드시 실시간 갱신할 파일:
- runs/goal_mae_search_v2/RUN_STATE.md
- runs/goal_mae_search_v2/resume_manifest.csv
- runs/goal_mae_search_v2/experiment_log.csv
- runs/goal_mae_search_v2/leaderboard.csv

B. 초기에 작성하고 필요시 갱신할 파일:
- runs/goal_mae_search_v2/input_contract.md
- runs/goal_mae_search_v2/leakage_audit.md
- runs/goal_mae_search_v2/baseline_summary.md

C. 최종 selection 직전에 작성할 파일:
- runs/goal_mae_search_v2/best_model_summary.md
- runs/goal_mae_search_v2/FINAL_SELECTION_LOCK.md

D. final test 이후 작성할 파일:
- runs/goal_mae_search_v2/final_test_audit.md
- runs/goal_mae_search_v2/final_report.md

E. 최종 정리 파일:
- runs/goal_mae_search_v2/experiment_log.md
- runs/goal_mae_search_v2/leaderboard.md

F. 선택적 파일:
- runs/goal_mae_search_v2/literature_architecture_notes.md
  문헌 기반 아이디어를 여러 개 사용한 경우에만 생성해도 된다.

leaderboard에는 반드시 다음을 포함한다.

- run_name
- status
- fms_context_mode
- anchor_mode
- anchor_interval_seconds
- use_static
- static_feature_set
- recent_start_observed
- sparse_observed
- predict_delta_from_anchor
- calibration_seconds
- recent_window_seconds
- horizon_seconds
- model_type
- architecture_family
- architecture_hypothesis
- architecture_family_index_or_group
- total_architecture_families_covered_so_far
- literature_inspired
- literature_source
- loss_type
- optimizer
- lr
- weight_decay
- dropout
- batch_size
- hidden_size_or_d_model
- max_epochs_planned
- epochs_completed
- seed
- split_file
- h=5 val MAE
- h=10 val MAE
- h=15 val MAE
- mean val MAE over h=5/10/15
- h=2.5 val MAE if available
- h=1 diagnostic val MAE if available
- common-window val MAE if available
- improvement percent vs corrected baseline
- branch family
- selection_reason
- failure_or_stop_reason
- checkpoint_path
- prediction_csv_path
- validation_plot_path if available
- sanity_check_status
- leakage_audit_status

Adaptive search leaderboard에는 test metric을 기록하지 않는다.
Final test 결과는 final_test_audit.md와 final_report.md에만 기록한다.

============================================================
20. Completion criteria
============================================================

Adaptive search를 멈추고 final selection lock으로 넘어갈 수 있는 조건은 다음 중 하나다.

1. MAX_WALL_CLOCK_HOURS에 근접했다.
2. corrected baseline 대비 primary validation MAE를 15% 이상 개선했고, 재현성 확인까지 완료했으며, 가능한 architecture family coverage를 충분히 수행했다.
3. corrected baseline 대비 primary validation MAE를 20% 이상 개선했고, final selection lock으로 넘어갈 이유를 RUN_STATE.md에 기록했다.
4. 주요 branch family를 대부분 검토했고, 남은 branch가 왜 유망하지 않은지 RUN_STATE.md에 구체적으로 기록했다.
5. 환경 문제, GPU 문제, 데이터 문제, 필수 audit 실패로 추가 학습이 불가능하다.
6. 사용자가 명시적으로 중단을 지시했다.

단, 4번 조건으로 종료하려면 아래를 반드시 만족해야 한다.

- baseline / runner consistency 확인
- leakage / input audit 확인
- 최소 7개 architecture family 검토 또는 미실행 사유 기록
- 새 architecture sanity check 수행 또는 미수행 사유 기록
- window branch 검토 또는 미실행 사유 기록
- static branch 검토 또는 미실행 사유 기록
- loss/imbalance branch 검토 또는 미실행 사유 기록
- optimization/hyperparameter branch 검토 또는 미실행 사유 기록
- literature-informed idea를 사용할 수 있었는지 확인하거나, 생략 사유 기록

다음 경우에는 adaptive search complete 처리하지 마라.

- 30분 미만만 실행했다.
- primary metric 개선폭이 10% 미만인데 명시적 사용자 중단/환경 blocker 없이 max(4시간, wall-clock budget의 70%) 미만만 사용했다.
- 5% 미만의 미세 개선만 있다.
- hyperparameter tuning만 조금 보고 architecture/window/loss/static branch를 거의 보지 않았다.
- 7개 architecture family를 검토하지 않았고, 그 이유도 구체적으로 기록하지 않았다.
- 3개 architecture family만 검토하고 충분하다고 판단했다.
- 새 architecture sanity check 없이 full run을 반복했다.
- 아직 시도하지 않은 주요 branch family가 있는데도 “개선 가능성이 낮다”는 추상적 이유만으로 종료하려 한다.
- final test를 먼저 실행하고 종료하려 한다.
- branch별 근거 없이 search complete 처리하려 한다.

Adaptive search 종료 후에는 반드시:
1. validation 기준 best candidate 선택
2. FINAL_SELECTION_LOCK.md 작성
3. frozen best config로 final test 1회 수행
4. final test 이후 추가 search 금지
5. 최종 보고 작성

============================================================
21. Final test protocol
============================================================

Final test는 adaptive search가 아니라 held-out evaluation이다.

절차:

1. validation 기준 best candidate를 선택한다.
2. best_model_summary.md를 작성한다.
3. FINAL_SELECTION_LOCK.md를 작성한다.
4. FINAL_SELECTION_LOCK.md 작성 이후에는 어떤 선택도 바꾸지 않는다.
5. frozen config / frozen checkpoint로 test evaluation을 1회 수행한다.
6. test metric, test prediction CSV, test plot을 생성한다면 final_test 디렉터리에만 저장한다.
7. final_test_audit.md에 다음을 기록한다.
   - 사용한 run_name
   - checkpoint
   - config
   - test command
   - test metric
   - test prediction CSV 위치
   - test plot 위치
   - test 이후 추가 튜닝을 하지 않았다는 확인
8. final test 이후 어떤 결과가 나오더라도 추가 실험을 수행하지 않는다.

주의:
- final test 결과가 validation 결과보다 나빠도 다시 tuning하지 마라.
- final test 결과가 좋아도 그것을 근거로 다른 후보를 고르지 마라.
- final test 결과는 최종 보고의 held-out evaluation으로만 사용한다.

============================================================
22. Git / repository safety
============================================================

- commit 하지 마라.
- push 하지 마라.
- 기존 baseline 코드를 깨지 마라.
- 새 구조는 가능하면 additive 방식으로 구현하라.
- runs/, artifacts/, DenseFMS/Dataset, checkpoint/model weight 파일은 git에 포함하지 마라.
- 최종 보고 전에 git status를 확인하고 변경 파일을 보고하라.
- 생성된 실험 산출물과 코드 변경 사항을 구분해서 보고하라.

============================================================
23. 최종 보고
============================================================

최종 보고는 한국어로 작성한다.

반드시 포함:

1. v1 결과를 실패/미미한 개선으로 재정리
2. v1 test 결과 파일을 열지 않았다는 확인
3. 이미 알려진 v1 test 수치를 v2 선택에 사용하지 않았다는 확인
4. 사용한 wall-clock time
5. 실행한 총 run 수
6. 60 run 초과 여부와 초과했다면 이유
7. baseline 성능
8. leakage audit 결과
9. input contract 요약
10. main track 정의
11. FMS input policy와 start FMS missing fallback 검증 결과
12. static-enabled vs no-static 비교
13. primary selection metric 기준 best validation model
14. corrected baseline 대비 validation 개선율
15. weak / minimum / strong / stretch 기준 중 어디에 해당하는지
16. horizon별 best validation result
17. h=1 diagnostic result
18. h>=2.5 valid forecasting result
19. h>=5 main forecasting result
20. branch family별 결론
21. architecture family별 결론
22. 총 몇 개 architecture family를 검토했는지
23. 7개 architecture family coverage 달성 여부
24. 7개 미만이면 각 미실행 family와 이유
25. 새 architecture sanity check 결과
26. optimization / hyperparameter tuning 결론
27. loss / imbalance branch 결론
28. window branch 결론
29. static branch 결론
30. literature-informed search를 수행했다면 참고한 자료와 반영한 아이디어
31. literature search를 생략했다면 생략 이유
32. 실행하지 않은 branch가 있다면 그 이유
33. interrupted/failed run 목록과 원인
34. resume 가능한 checkpoint가 있는지
35. 실패한 방향과 중단 이유
36. best config
37. best model 구조 요약
38. 재현 명령어
39. validation prediction plot / CSV 위치
40. FINAL_SELECTION_LOCK.md 작성 여부
41. final test 수행 여부
42. final test metric
43. final test prediction plot / CSV 위치
44. final test 이후 추가 튜닝을 하지 않았다는 확인
45. 현재까지 결과만 기준으로 해석했을 때의 결론
46. 추가 실행이 필요하다면 우선순위가 높은 다음 실험
47. 남은 리스크
48. 다음에 사람이 직접 판단해야 할 선택지
49. git status 요약

다시 강조:
- adaptive search 중에는 test를 절대 열지 않는다.
- final test는 frozen best config 확정 후 딱 1회만 수행한다.
- final test 이후 추가 튜닝은 절대 금지한다.
- run 개수 제한은 없다.
- 전체 실행은 wall-clock time으로 제한한다.
- run이 60개를 초과하면 RUN_STATE.md에 이유와 micro-run 감사 결과를 남긴다.
- 10% 미만 개선이면 successful final candidate로 인정하지 않는다.
- 그래도 best observed validation candidate는 반드시 보고한다.
- 10% 미만 개선이면 명시적 사용자 중단/환경 blocker 없이 max(4시간, wall-clock budget의 70%) 이전에 final selection lock으로 넘어가지 않는다.
- hyperparameter tuning은 허용하지만, 그것만으로 complete 처리하지 않는다.
- architecture family는 최소 7개 이상 검토해야 한다.
- 3개 architecture family만 검토하고 종료하지 마라.
- 7개 architecture family를 검토하지 못했다면 구체적인 이유를 RUN_STATE.md와 final_report.md에 기록해야 한다.
- 새 architecture는 full run 전에 shape/leakage/alignment sanity check를 통과해야 한다.
- h=1 결과는 diagnostic으로만 사용한다.
- h=2.5는 valid forecasting result로 기록하되 primary selection에서는 제외한다.
- h=5/10/15 validation MAE 평균이 primary selection metric이다.
- main candidate는 반드시 h=5, h=10, h=15 validation metric을 모두 제공해야 한다.
- 논문 구조는 optional 참고만 하고, 현재 corrected start_only input contract에 맞게 변형해야 한다.
- 목표는 corrected start_only 조건에서 validation MAE를 공격적으로 낮추는 것이다.