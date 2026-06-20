# DenseFMS 6h Adaptive Improvement Search Goal Spec

이 문서는 `/goal`에서 읽어 실행할 장시간 adaptive search 명세다.

권장 호출 예시:

```text
/goal

docs/codex/goal_mae_6h_adaptive_improvement_prompt.md를 읽고, 그 문서의 지시를 엄격히 준수해서 진행해줘.
```

DenseFMS future forecasting 코드베이스에서 corrected start_only 조건의 validation MAE를 낮추기 위한 6시간 adaptive improvement search를 진행해줘.

이 goal은 고정 시간표를 따르는 작업이 아니다.
먼저 필수 audit과 사전 정의 first-pass 실험 목록을 끝내고,
그 이후에는 현재까지의 validation 결과, horizon별 실패 양상, validation plot/trend review를 보고 다음 실험을 adaptive하게 선택한다.

중요:
- 이 프롬프트는 /goal 장시간 실행용이다.
- 이전 대화 맥락에 의존하지 말고, 아래 지시와 repository 파일을 기준으로 독립적으로 진행하라.
- interrupt, timeout, crash, context compaction, 재개 상황을 전제로 RUN_STATE와 resume_manifest를 계속 유지하라.

출력 디렉터리:
- runs/goal_mae_6h_adaptive_improvement_0506/

============================================================
0. Must Read First
============================================================

코드 수정, 학습, 평가 전에 반드시 다음을 읽어라.

- AGENTS.md
- docs/codex/goal_mae_search_policy_0505.md
- docs/codex/goal_mae_overnight_hybrid_prompt.md, 존재하면 참고하라.
- runs/goal_mae_overnight_hybrid_0505/final_report.md, 존재하면 읽어라.
- runs/goal_mae_overnight_hybrid_0505/best_model_summary.md, 존재하면 읽어라.
- runs/goal_mae_overnight_hybrid_0505/validation_trend_plot_audit.md, 존재하면 읽어라.

주의:
- runs/**/eval_test/ 아래 파일은 pre-lock 단계에서 읽지 마라.
- 기존 test metric / test prediction / test plot은 이번 search의 어떤 판단에도 사용하지 마라.
- 기존 final test 결과나 test plot은 현재 search의 branch 선택에 사용하지 마라.
- 이미 문서에 남아 있는 test 결과는 archived final report context로만 취급한다.
- 필요한 plot review는 validation prediction CSV/plot만 새로 생성하거나 기존 validation artifact만 사용해 수행한다.
- 이번 search의 모든 실험 선택은 validation 결과와 validation prediction/plot만 기준으로 한다.

============================================================
1. Non-Negotiable Rules
============================================================

FULL_TRAINING_ALLOWED = true

1. Adaptive search 중에는 test set을 열지 마라.
2. 일반 run에서는 train/val metric과 train/val prediction CSV만 생성한다.
3. test metric / test prediction CSV / test plot은 validation 기준 최종 후보를 lock한 뒤 1회만 허용한다.
4. 이번 goal이 exploratory validation search로 끝나는 경우 final test를 생략할 수 있다. 생략하면 이유를 final_report.md에 명확히 기록하라.
5. final test를 수행했다면, 이후 어떤 architecture / hyperparameter / preprocessing / window / loss / static usage도 변경하지 마라.
6. Primary selection metric은 validation MAE mean over h=5, h=10, h=15다.
7. h=2.5는 auxiliary forecasting result로 별도 기록한다.
8. h=1은 diagnostic/lower-bound/sanity-check로만 기록하고 final selection에는 쓰지 않는다.
9. Main track은 corrected start_only, anchor_mode=none이다.
10. current FMS, target FMS, future FMS, recent window 내부 dense FMS sequence, sparse_observed, recent_start_observed는 금지한다.
11. 새 모델/구조는 기존 main run들이 받던 허용 입력 이상을 받으면 안 된다.
12. participant_id/session_id/condition_id/trial_id/experiment_id/file-derived identity는 금지한다.
13. validation/test를 train preprocessing fit에 사용하지 마라.
14. evaluation code를 지표가 좋아지도록 바꾸지 마라.
15. completed run을 불필요하게 중복 실행하지 마라.
16. failed/interrupted run은 숨기지 말고 resume_manifest.csv와 experiment_log에 기록하라.
17. commit 하지 마라.
18. push 하지 마라.

============================================================
2. Budget And Execution Style
============================================================

Wall-clock budget:
- TARGET_WALL_CLOCK_HOURS = 6
- MAX_WALL_CLOCK_HOURS = 6.5
- MIN_ACTIVE_SEARCH_HOURS = 5
- 마지막 30분은 결과 정리, leaderboard, plot audit, final_report 작성을 위해 남기는 것을 목표로 한다.

MIN_ACTIVE_SEARCH_HOURS는 GPU training 시간만 뜻하지 않는다.
audit, validation plot review, error analysis, negative-result analysis, resume documentation도 active search/audit 시간에 포함한다.
따라서 5시간을 채우기 위해 끝나지 못할 training run을 무리하게 시작하지 말고,
남은 시간이 애매하면 분석과 문서화를 active search로 수행하라.

이 goal은 hour-by-hour schedule을 따르지 않는다.

Execution style:

1. 필수 audit과 tiered first-pass experiment plan을 먼저 수행한다.
2. 각 run 이후 leaderboard와 experiment_log를 갱신한다.
3. 이후에는 현재까지의 결과를 보고 다음 실험을 adaptive하게 선택한다.
4. wall-clock budget은 최대 예산이지 고정 시간표가 아니다.
5. per-run epoch, patience, max run duration은 validation behavior와 남은 시간에 따라 자유롭게 조정할 수 있다.
6. 단, docs/codex/goal_mae_search_policy_0505.md가 더 엄격한 epoch/run cap을 요구하면 더 엄격한 제한을 따른다.
7. 시간을 채우기 위해 의미 없는 run을 하지 마라.
8. 하지만 tiered first-pass가 끝났다는 이유만으로 일찍 goal complete 처리하지 마라.
9. 시간이 남고 유망한 branch가 있으면 best branch를 계속 파고들어라.
10. 명시적 사용자 중단, leakage/split blocker, 환경 blocker, 또는 training 자체가 불가능한 상황이 아니면 최소 5시간은 active search/audit을 지속하라.
11. 5시간 전에 "유망한 branch가 없다"는 이유만으로 goal complete 처리하지 마라.

Run length guidance:
- 새 구조/새 feature는 full run 전에 import, shape, leakage sanity를 통과해야 한다.
- 새 branch 첫 확인은 cheap run으로 시작한다. 예: 25~40 epoch, patience 5~8.
- promising branch는 80 epoch 또는 policy가 허용하는 longer confirmation run으로 승격한다.
- current best reproduction, promising candidate confirmation, finalist seed check에는 더 긴 run을 우선 배정한다.
- 실패 원인이 optimization인지 구조 자체인지 애매하면 learning rate / patience / seed를 한 번 확인할 수 있다.

Stop conditions:
- wall-clock budget이 거의 소진됨
- 더 이상 유망한 branch가 남지 않음
- 필수 보고 파일을 마무리해야 함
- leakage/split/input contract 문제가 발견되어 training을 계속하면 안 됨
- 환경 blocker, OOM, dependency 문제 등으로 의미 있는 진행이 불가능함

중요:
- "더 이상 유망한 branch가 남지 않음"은 즉시 종료 조건이 아니다.
- 아래 Negative-result adaptive mode까지 수행하거나, 수행하지 못한 이유를 기록한 뒤에만 training 중단 사유로 사용할 수 있다.
- 아직 검토하지 않은 major branch가 있거나, error audit에서 설명 가능한 실패 원인이 남아 있으면 search를 계속한다.
- elapsed time이 MIN_ACTIVE_SEARCH_HOURS보다 작으면, "더 이상 유망한 branch가 남지 않음"을 종료 사유로 사용할 수 없다.
- 5시간 전에는 very weak but plausible 후보라도 fallback search queue에서 골라 계속 진행한다.
단, fallback 후보도 아래 expected learning value 중 하나를 가져야 한다.
- h15 병목 원인 확인
- delta branch collapse 여부 확인
- motion-derived feature의 단독 효과 확인
- static 정보의 subgroup/worst-session 개선 여부 확인
- high-change auxiliary가 smoothing/lag에 주는 영향 확인
- optimization failure 가능성 확인
- secondary metric 개선 후보의 조합 가능성 확인
위 learning value를 설명할 수 없는 run은 5시간을 채우기 위해서라도 실행하지 마라.

Do not stop early merely because:
- tiered first-pass list가 끝남
- 첫 번째 개선 후보가 나옴
- 한 branch에서 두세 개 run이 실패함
- MAE가 조금 개선됐지만 smoothing/lag 문제가 여전히 남아 있음

If time remains after the tiered first-pass plan:

남은 시간은 아래 우선순위로 사용하라. 이 순서는 고정 시간표가 아니라 adaptive priority order다.

1. 새로운 best 또는 best 대비 +0.03 이내인 후보를 80 epoch / longer patience / rerun으로 확인한다.
2. h=15가 여전히 병목이면 h15 head capacity branch를 한 단계 더 확장한다.
3. smoothing/lag 또는 large rise/drop 실패가 남아 있으면 level+delta dual-head branch를 확장한다.
4. dual-head가 delta를 제대로 못 잡거나 plot이 평탄하면 motion-derived causal feature branch를 확장한다.
5. 단독 효과가 확인된 구성만 조합한다:
   - best h15 head + best dual-head
   - best dual-head + best motion feature
   - best h15 head + best motion feature
   - all combined는 앞 조합들이 유망할 때만 실행한다.
6. primary MAE가 동급인데 trend/high-FMS/worst-session이 개선된 후보가 있으면 robustness candidate로 유지하고 한 번 더 확인한다.
7. static subgroup 또는 worst-session 문제가 뚜렷하면 static raw/direct, tiny encoder, gated residual 중 아직 확인하지 않은 가장 단순한 후보를 실행한다.
8. 위 branch들이 모두 부진하고 large-delta sample이 충분하면 high-change auxiliary classification을 낮은 loss weight로 1~2개만 확인한다.
9. 명확한 best가 있으면 추가 seed 또는 rerun으로 robustness를 확인한다.
10. 더 이상 유망한 training branch가 없으면 새 run을 만들지 말고 error audit, plot review, 실패 원인, 다음 우선순위를 문서화한다.

Negative-result adaptive mode:

tiered first-pass와 위 우선순위 후보들이 모두 부진해 보여도, 시간이 충분히 남아 있으면 바로 종료하지 마라.
대신 아래 순서로 "왜 안 되는지"를 좁히는 cheap exploratory run 또는 audit을 수행한다.

1. current best reproduction / metric parsing / validation sample count / input contract를 다시 확인해 비교 기준이 흔들리지 않았는지 점검한다.
2. 가장 덜 구현 위험이 큰 branch부터 아직 안 해본 representative cheap run을 1개씩 실행한다:
   - h15 residual/deeper head 중 미실행 후보
   - level+delta에서 delta-only 또는 simple average 후보
   - motion norm-only feature 후보
   - static raw/direct 또는 tiny encoder 후보, subgroup/worst-session 문제가 남아 있을 때
   - high-change auxiliary lambda 0.05 후보, large-delta sample이 충분할 때
3. 모든 branch가 primary MAE를 악화시키더라도, h15 / trend / high-FMS / worst-session 중 하나를 개선한 후보가 있는지 확인한다.
4. 개선 후보가 전혀 없으면 optimization failure 가능성을 한 번만 점검한다:
   - seed 변경
   - patience 증가
   - learning rate 한 단계 조정
   단, 같은 branch에서 근거 없는 micro-tuning 반복은 하지 마라.
5. 그래도 개선 신호가 없으면, 그때 training을 멈추고 실패 원인, negative results, 다음 연구 우선순위를 충분히 문서화한다.

Minimum 5-hour fallback search queue:

elapsed time이 5시간 미만이고 training 환경이 정상이라면, Negative-result adaptive mode 이후에도 아래 후보 중 아직 하지 않은 것을 계속 선택한다.
이 후보들은 "확실히 유망한 후보"가 아니라 "가능성이 조금이라도 있는 fallback 후보"로 기록하라.
각 run에는 hypothesis와 expected learning value를 남긴다.
expected learning value는 반드시 다음 중 하나와 연결되어야 한다.
- h15 병목 원인 확인
- delta branch collapse 여부 확인
- motion-derived feature의 단독 효과 확인
- static 정보의 subgroup/worst-session 개선 여부 확인
- high-change auxiliary가 smoothing/lag에 주는 영향 확인
- optimization failure 가능성 확인
- secondary metric 개선 후보의 조합 가능성 확인

1. h15-focused fallback:
   - h15 residual head의 hidden dim/activation/dropout 소폭 변경
   - h15-only mild loss weight
   - h10/h15 shared residual head
2. dual-head fallback:
   - delta-only with smaller head
   - delta-only with larger head
   - level+delta average without gate
   - gated fusion with alpha/beta 0.1, 0.2, 0.3 중 미실행 값
   - gate regularization 또는 gate temperature, 구현이 간단할 때만
3. motion-feature fallback:
   - norm-only
   - norm + delta-norm
   - gyro-only rolling energy
   - accel+gyro rolling energy
   - feature dropout 또는 weaker feature encoder, 과적합이 의심될 때
4. static fallback:
   - static raw/direct
   - static tiny encoder 8
   - static tiny encoder 16
   - static gated residual/context bias
   - static dropout, static 과의존이 의심될 때
5. high-change fallback:
   - threshold 2 / 3 / 4 중 미실행 값
   - quantile threshold
   - lambda 0.05 또는 0.1
   - class weight, class imbalance가 심할 때
6. conservative optimization fallback:
   - current best family에서 seed 변경
   - learning rate 한 단계 낮춤 또는 높임
   - patience 증가
   - weight decay/dropout 소폭 변경
   - 단, 같은 작은 hyperparameter 변형을 3회 이상 반복하지 마라.
7. combination fallback:
   - primary MAE가 best 대비 +0.05~0.08 이내이고 secondary metric이 좋아진 후보끼리만 조합한다.
   - h15 improvement, trend improvement, high-FMS improvement 중 하나라도 있어야 한다.

5시간을 채우기 위한 fallback search에서도 금지되는 것:
- test set 확인
- recent window 10/30 외 sweep
- input policy 위반
- identity feature 추가
- evaluation metric 변경
- completed run 중복 실행
- failed run을 기록 없이 숨기기
- 아무 가설 없는 random run

Training을 멈추기 전에 다음 중 하나는 반드시 만족해야 한다.
- tiered first-pass branch와 Negative-result adaptive mode를 모두 수행했다.
- 남은 시간이 meaningful run을 끝내기에 부족하다.
- leakage/input/split/audit 문제 때문에 training을 계속하면 안 된다.
- 환경 blocker 때문에 training을 계속할 수 없다.
- 사용자가 명시적으로 중단을 요청했다.

추가 종료 제한:
- elapsed time이 5시간 미만이면, 위 조건만으로는 부족하다.
- 5시간 미만 종료는 leakage/split blocker, 환경 blocker, training 불가능, 사용자 명시 중단, 또는 required sanity failure처럼 계속 진행하면 안 되는 사유가 있을 때만 허용한다.
- 단순히 "best를 못 찾았다", "branch가 약하다", "first-pass가 끝났다"는 이유로 5시간 전에 종료하지 마라.

Do not use remaining time for:
- recent window 10/30 외 sweep
- h=1/h=2.5 diagnostic만 좋아지는 방향의 확장
- 같은 branch에서 근거 없는 learning rate/dropout micro-tuning 반복
- full Cartesian product
- test set 확인

Before starting any new run with less than 45 minutes remaining:
- 예상 runtime과 reporting 시간을 RUN_STATE.md에 적어라.
- 새 run이 끝나지 못할 가능성이 높으면 시작하지 말고 reporting/audit을 우선하라.
- 이미 유망한 후보가 있으면 새 branch보다 confirmation/reporting을 우선하라.

============================================================
3. Current Validation Baseline
============================================================

다음 후보는 현재 validation-only starting point로 사용한다.
test selection에는 사용하지 마라.

Current validation best:
- run: next1h_recent10_e80_s7
- checkpoint: runs/goal_mae_next1h_0505/next1h_recent10_e80_s7/best.pt
- model family: lc_sa_tcnformer / lcsa_per_horizon_heads
- recent_window_seconds = 10
- fms_context_mode = start_only
- anchor_mode = none
- static = no static
- primary validation MAE mean(h=5/10/15): 1.8001924350
- h5/h10/h15 validation MAE: 1.6699913779 / 1.8279172394 / 1.9026686877

Interpretation caveat:
- recent10은 "최근 10초 head motion + 약 10초 전 start FMS" 조건이다.
- recent30은 "최근 30초 head motion + 약 30초 전 start FMS" 조건이다.
- recent window를 바꾸면 head motion 길이와 start-FMS recency가 동시에 바뀐다.
- 따라서 recent10/recent30은 pure motion-window 비교가 아니라 motion-history length vs FMS-anchor recency tradeoff로 해석한다.
- 이번 goal에서는 recent_window_seconds를 10과 30만 사용한다.
- recent3/5/7.5/12.5/15 sweep은 하지 마라.

Before expanding search, verify or document:
- current best가 같은 train/val split을 사용했는지
- denormalized MAE 계산이 기존 기준과 같은지
- no test files are read or generated pre-lock
- no current/target/future FMS in input
- no recent dense FMS sequence in input
- start-FMS anchor definition, fallback rule, and observed lag
- validation sample count, or explain any difference

============================================================
4. Main Input Policy
============================================================

Main track:
- fms_context_mode = start_only
- anchor_mode = none
- anchor_interval_seconds = 0
- sparse_observed 사용 금지
- recent_start_observed 사용 금지

start_only에서 허용되는 post-calibration FMS:
- recent motion window의 시작 FMS 하나만 허용한다.
- 원칙 이름은 start_fms_index/start_fms_time/start_fms_value로 사용한다.
- 기존 CSV의 anchor_index/anchor_time/anchor_fms는 start_fms_*의 backward-compatible alias로만 해석한다.
- start FMS가 missing이면 nominal start index 이하의 최신 finite FMS fallback만 허용한다.

금지:
- target FMS
- current FMS FMS[t]
- future FMS
- recent window 내부 dense FMS sequence
- sparse_observed
- recent_start_observed
- calibration_end anchor
- sparse_anchor
- identity-derived feature

Static policy:
- Age/Gender/MSSQ는 허용된다.
- Age와 MSSQ는 train split에서만 fit한 mean/std로 z-score normalize한다.
- Gender는 2D one-hot [gender_male, gender_female]을 primary static encoding으로 사용한다.
- unknown/missing gender는 [0, 0]으로 처리한다.
- static scaler, imputer, encoder fit에 validation/test 값을 사용하지 마라.
- static은 이번 goal의 primary branch가 아니라 secondary branch다. 단, error audit에서 subgroup/worst-session 문제가 뚜렷하거나 시간이 남으면 active 후보로 승격할 수 있다.

============================================================
5. Tiered First-Pass And Conditional Experiment Plan
============================================================

아래 목록은 단순 체크리스트가 아니라 tiered execution plan이다.
단, 이미 동일 조건의 completed run과 충분한 audit 문서가 있으면 중복 실행하지 말고 재사용 사유를 기록하라.

Tier 원칙:
- Tier 0은 반드시 수행한다.
- Tier 1은 우선 수행하되, 각 branch의 모든 후보를 다 돌리는 것이 아니라 representative 후보부터 실행한다.
- Tier 2는 Tier 1 결과와 validation plot/error audit을 보고 수행한다.
- Tier 3은 조건부 fallback이다. smoothing/lag, subgroup, large-delta 문제가 남거나 5시간 active search floor를 채워야 할 때 수행한다.
- 어떤 tier에서도 full Cartesian product를 돌리지 마라.

5.1 Tier 0: Mandatory audit and setup

- input_contract.md 작성
- leakage_audit.md 작성
- current best comparability 확인
- no pre-lock test output guard 확인
- h=5/h=10/h=15 validation plot review 기준 확정
- delta distribution audit 작성:
  - target_fms - start_fms_value 분포
  - horizon별 delta mean/std/quantile
  - large rise/drop threshold 후보: absolute 2/3/4 또는 train quantile
  - high-FMS bucket과 large-delta bucket의 sample count

5.2 Tier 1: h=15 head capacity branch

목적:
- h=15에서 smoothing/lag와 높은 MAE가 계속 병목인지 확인한다.
- shared encoder를 크게 키우기보다 horizon-specific head capacity를 제한적으로 보강한다.

Tier 1 first-pass:
- 아래 후보 중 1~2개 representative run을 먼저 실행한다.
- 모든 후보를 한 번에 다 돌리지 마라.

후보:
- baseline/current best reproduction or reuse
- h15 deeper head: h=15 head만 depth +1
- h15 residual MLP head
- h10/h15 residual MLP head
- h15 mild widening, 단 전체 모델 대형화는 피한다.

기록:
- primary mean(h=5/10/15)
- h15 MAE 변화
- h5/h10 악화 여부
- validation plot에서 h15 rise/drop lag 완화 여부

5.3 Tier 1: Level + Delta dual-head branch

목적:
- 모델이 FMS level과 future change를 한 head에서 뭉개지 않도록 분리한다.
- target level 예측과 target-start delta 예측을 함께 학습해 상승/하강 반응을 개선한다.

Tier 1 first-pass:
- delta-only는 우선 실행한다.
- simple average 또는 gated fusion 중 하나를 먼저 실행한다.
- 둘 중 하나가 유망할 때만 horizon-specific gated fusion이나 h15-only delta/residual fusion으로 확장한다.

후보 순서:
- delta-only: final_pred = start_fms_value + delta_head(z)
- level + delta simple average
- level + delta gated fusion
- horizon-specific gated fusion
- h15-only delta/residual fusion, h15 병목이 뚜렷할 때만

권장 loss:
- pred_level = level_head(z)
- pred_delta = delta_head(z)
- pred_delta_value = start_fms_value + pred_delta
- gate = sigmoid(gate_head(z))
- final_pred = gate * pred_level + (1 - gate) * pred_delta_value
- loss = L(final_pred, target) + alpha * L(pred_level, target) + beta * L(pred_delta, target - start_fms_value)

초기 alpha/beta:
- alpha = 0.2 또는 0.3
- beta = 0.2 또는 0.3

필수 기록:
- gate mean/std
- horizon별 gate mean
- large rise/drop 구간 gate behavior
- true_delta distribution
- pred_delta distribution
- pred_delta가 0 근처로 붕괴하는지
- level head와 delta head 중 어느 쪽에 과의존하는지

5.4 Tier 2: Motion-derived causal feature branch

목적:
- raw head motion만으로 모델이 직접 학습해야 하는 motion intensity, jerk, rolling energy를 안전한 causal transform으로 보강한다.
- 새 정보를 추가하는 것이 아니라 기존 head motion의 현재 시점 이하 데이터만 사용한 파생 feature여야 한다.

실행 조건:
- Tier 1 이후에도 smoothing/lag 또는 large rise/drop 실패가 남아 있음
- dual-head가 delta를 충분히 잡지 못함
- 남은 시간이 충분하거나 5시간 active search floor를 채우기 위한 plausible branch가 필요함

후보:
- norm features only:
  - accel_norm
  - gyro_norm
- norm + delta norm:
  - accel_delta_norm
  - gyro_delta_norm
- norm + delta norm + rolling energy:
  - rolling_gyro_energy_3s
  - 필요하면 rolling_accel_energy_3s

규칙:
- rolling feature는 반드시 causal해야 한다. feature[t]는 motion[<=t]만 사용한다.
- normalization/statistics fit은 train split에서만 한다.
- validation/test 값으로 scaler를 fit하지 마라.
- feature가 너무 많아져 과적합하면 더 단순한 feature group으로 후퇴한다.

5.5 Tier 3: High-change auxiliary classification branch

이 branch는 우선순위가 낮다.
다음 조건에서만 실행한다.
- h15 head / dual-head / motion feature 이후에도 smoothing/lag가 dominant failure로 남아 있음
- delta distribution audit에서 large rise/drop sample이 충분함
- 남은 wall-clock이 충분함

목적:
- 회귀 loss만으로는 큰 상승/하강을 평균화하는 문제를 완화한다.
- target-start delta에서 파생한 label만 사용하므로 input leakage가 아니다.

후보:
- 3-class classification: rise / stable / drop
- threshold = 2, 3, 4 비교 또는 train split quantile threshold
- aux loss weight lambda = 0.05, 0.1, 0.2
- class imbalance가 심하면 class weight 또는 focal loss 검토

필수 metric:
- rise F1
- drop F1
- macro F1
- rise recall
- drop recall
- primary validation MAE 변화

Auxiliary head가 좋아도 primary MAE와 trend plot이 나빠지면 main 후보로 승격하지 마라.

5.6 Tier 3: Static branch

이 branch는 조건부 fallback이다.
다음 조건에서 수행한다.
- static subgroup 또는 worst-session 문제가 error audit에서 뚜렷함
- no-static 후보들이 primary MAE는 좋지만 특정 subgroup/high-FMS/worst-session에서 반복적으로 실패함
- 5시간 active search floor를 채워야 하고 아직 static representative 후보를 확인하지 않았음

후보:
- static raw/direct
- static tiny encoder 8
- static tiny encoder 16
- static gated residual/context bias, 구현이 단순할 때만

승격 조건:
- no-static 대비 primary MAE 개선
- 또는 primary MAE가 best 대비 +0.03 이내이고 h15/high-FMS/worst-session/subgroup 중 하나가 개선됨

============================================================
6. Adaptive Continuation Rules
============================================================

Tiered first-pass plan 이후에는 결과 기반으로 다음 실험을 고른다.

Promote a branch if:
- primary validation MAE가 개선됨
- primary MAE가 best 대비 +0.03 이내이고 h15가 의미 있게 개선됨
- h5/h10 손실이 작고 h15 smoothing/lag가 validation plot에서 개선됨
- primary MAE가 best 대비 +0.05 이내이고 flow-following metric이 명확히 개선됨
- high-FMS bucket 또는 large rise/drop bucket error가 줄어듦
- static subgroup이나 worst-session error가 뚜렷하게 개선됨

Stop or deprioritize a branch if:
- 두 개 이상의 변형이 연속으로 primary MAE를 명확히 악화시킴
- h=1 또는 h=2.5만 좋아지고 h=5/10/15가 나빠짐
- validation plot에서 더 심한 smoothing/lag가 보임
- dynamic range가 더 수축하거나 rise/drop direction을 더 못 따라감
- pred_delta가 0 근처로 붕괴해 delta branch가 사실상 start_fms copy가 됨
- gate가 항상 한쪽 head만 선택해 dual-head 의미가 사라짐
- 구현 위험이 search budget을 과도하게 소모함

Combination rule:
- full Cartesian product를 돌리지 마라.
- 단독 효과가 있는 후보만 조합한다.
- 우선 조합:
  1. best h15 head + best dual-head
  2. best dual-head + best motion feature
  3. best h15 head + best motion feature
  4. all combined, 단 앞의 조합들이 유망할 때만

Seed/confirmation rule:
- 새로운 best candidate가 나오면 가능하면 추가 seed 또는 rerun으로 확인한다.
- 남은 시간이 부족하면 seed robustness보다 final reporting과 audit을 우선한다.
- seed result는 mean/variance를 기록한다.

Recent-window rule:
- recent_window_seconds는 10과 30만 사용한다.
- recent10이 best여도 recent3/5/7.5로 확장하지 마라.
- recent30이 유망해지는 경우에도 recent12.5/15/20 sweep으로 확장하지 마라.
- 이 goal의 핵심은 window sweep이 아니라 prediction dynamics 개선이다.

Static rule:
- static은 secondary지만 완전히 배제하지 않는다.
- no-static 대비 primary MAE가 개선되거나, primary MAE가 동급(+0.03 이내)이면서 h15/high-FMS/worst-session 중 하나가 개선되면 static candidate로 유지한다.
- static이 과적합하는 것처럼 보이면 더 큰 encoder가 아니라 raw/direct, tiny encoder, gated residual 중 더 단순한 방식으로 후퇴한다.

============================================================
7. Validation Plot And Error Audit
============================================================

MAE 외에 validation prediction plot을 보고 상승/하강 추세를 따라가는지 확인한다.

Plot review는 secondary diagnostic이다.
Final selection의 primary metric은 여전히 validation MAE mean over h=5/10/15다.
하지만 이번 search의 중요한 secondary objective는 "절대값을 맞추는 것뿐 아니라 대략적인 FMS 흐름을 따라가는 모델"을 찾는 것이다.
MAE가 비슷하다면 더 나은 flow-following 후보를 우선 유지하고 추가 확인한다.

Strong candidate마다 다음을 수행한다.
- h=5, h=10, h=15를 분리한 validation plot 생성 또는 기존 prediction CSV 기반 review
- representative sessions와 worst sessions를 분리해서 확인
- peak/trough smoothing 여부 기록
- 상승 구간 lag 기록
- 하강 구간 lag 또는 overprediction 기록
- high-FMS underprediction 기록

Flow-following secondary metrics:
- direction agreement: 같은 session/horizon 안에서 sign(delta predicted_fms)와 sign(delta target_fms)가 일치하는 비율
- large-rise recall: target_fms가 일정 threshold 이상 상승한 구간에서 predicted_fms도 상승 방향을 잡는 비율
- large-drop recall: target_fms가 일정 threshold 이상 하락한 구간에서 predicted_fms도 하락 방향을 잡는 비율
- slope correlation 또는 delta correlation: delta predicted_fms와 delta target_fms의 correlation
- dynamic range ratio: std(predicted_fms) / std(target_fms), 또는 range(predicted_fms) / range(target_fms)
- lag estimate: target과 prediction의 cross-correlation peak lag, 계산이 간단한 경우
- high-FMS bias: target high-FMS bucket에서 prediction이 얼마나 낮게 수축되는지

Flow-following 해석 규칙:
- direction agreement, large-rise/drop recall, slope/delta correlation이 높을수록 흐름을 잘 따라간다.
- dynamic range ratio가 너무 낮으면 prediction이 과도하게 평탄한 것이다.
- lag estimate가 크면 상승/하강을 늦게 따라가는 것이다.
- high-FMS bias가 음수로 크면 peak/plateau underprediction이 심한 것이다.
- primary MAE가 best 대비 +0.03 이내이고 flow-following이 좋아진 후보는 robustness candidate로 유지한다.
- primary MAE가 best 대비 +0.03~0.05 사이여도 flow-following이 크게 좋아지면 후속 confirmation run을 고려한다.
- primary MAE가 best 대비 +0.08 이상 나쁘면 flow-following이 좋아도 main candidate로 승격하지 말고 별도 qualitative lead로만 기록한다.

필수 error audit:
- horizon별 MAE/RMSE/bias
- target FMS bucket별 MAE/bias
- start FMS bucket별 MAE/bias
- target-start delta bucket별 MAE/bias
- current_time bucket별 MAE/bias
- high-FMS underprediction
- large-rise underprediction
- large-drop overprediction
- direction agreement
- large-rise/drop recall
- slope or delta correlation
- dynamic range ratio
- lag estimate, 계산한 경우
- worst validation sessions
- static을 사용한 경우 age/MSSQ/gender subgroup audit

Trend/plot audit을 adaptive planning에 반영하라.
예:
- MAE가 비슷한데 trend를 더 잘 따라가면 robustness candidate로 유지한다.
- example.png처럼 정확한 절대값은 다소 낮아도 전체 상승/하강 흐름을 잡는 후보는 qualitative lead로 기록한다.
- MAE가 좋아도 plot이 지나치게 평탄하면 추가 dual-head/motion-feature 보강을 검토한다.

============================================================
8. Required Verification
============================================================

코드 수정 또는 새 구조 추가 시 반드시 확인:

- import check
- model forward shape check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- start-FMS/anchor policy check
- static preprocessing train-only fit check, static 사용 시
- derived motion feature causal check, motion feature 사용 시
- no current/target/future FMS input check
- no recent dense FMS sequence input check
- no pre-lock test output guard
- dry-run command generation, runner/sweep script를 수정한 경우

Full training 중 확인:
- checkpoint saving
- metrics JSON/CSV generation
- train/val prediction CSV generation
- leaderboard generation
- resume behavior or skip-completed behavior
- failed/interrupted run visibility

새 model class / 새 output head sanity:
- batch dimension과 horizon dimension 정합성 확인
- output shape가 target shape와 일치하는지 확인
- loss가 denormalized metric 계산을 바꾸지 않는지 확인
- dual-head auxiliary loss가 train target에서만 label을 만들고 input에 target을 넣지 않는지 확인
- high-change label이 input feature로 들어가지 않는지 확인

============================================================
9. Required Outputs
============================================================

필수로 다음을 생성/유지하라.

- runs/goal_mae_6h_adaptive_improvement_0506/PLAN.md
- runs/goal_mae_6h_adaptive_improvement_0506/input_contract.md
- runs/goal_mae_6h_adaptive_improvement_0506/leakage_audit.md
- runs/goal_mae_6h_adaptive_improvement_0506/delta_distribution_audit.md
- runs/goal_mae_6h_adaptive_improvement_0506/experiment_log.csv
- runs/goal_mae_6h_adaptive_improvement_0506/leaderboard.csv
- runs/goal_mae_6h_adaptive_improvement_0506/RUN_STATE.md
- runs/goal_mae_6h_adaptive_improvement_0506/resume_manifest.csv
- runs/goal_mae_6h_adaptive_improvement_0506/best_model_summary.md
- runs/goal_mae_6h_adaptive_improvement_0506/final_report.md

권장 output:
- runs/goal_mae_6h_adaptive_improvement_0506/model_change_audit.md
- runs/goal_mae_6h_adaptive_improvement_0506/adaptive_queue.csv
- runs/goal_mae_6h_adaptive_improvement_0506/experiment_log.md
- runs/goal_mae_6h_adaptive_improvement_0506/leaderboard.md
- runs/goal_mae_6h_adaptive_improvement_0506/error_audit_by_run/
- runs/goal_mae_6h_adaptive_improvement_0506/validation_plot_reviews/
- runs/goal_mae_6h_adaptive_improvement_0506/branch_summaries.md
- runs/goal_mae_6h_adaptive_improvement_0506/VALIDATION_SELECTION_LOCK.md, final test를 수행할 경우 필수
- runs/goal_mae_6h_adaptive_improvement_0506/final_test_audit.md, final test를 수행한 경우

시간이 부족하면 필수 output을 우선하고, branch별 상세 요약은 branch_summaries.md 하나로 합쳐라.
문서 파일을 많이 만드는 것보다 experiment_log.csv, leaderboard.csv, RUN_STATE.md, final_report.md의 정확성을 우선한다.

RUN_STATE.md에는 항상 다음을 남겨라.
- goal 시작 시각
- elapsed/remaining time
- 현재 active run
- completed run
- failed/interrupted run
- next planned experiment
- adaptive decision reason
- resume 가능 여부

experiment_log.csv에는 최소 다음 컬럼을 포함하라.
- run_name
- status
- parent_run
- branch
- hypothesis
- changed_fields
- recent_window_seconds
- static_usage
- model_changes
- loss_changes
- motion_feature_changes
- seed
- epochs_or_max_epochs
- primary_val_mae_h5_h10_h15_mean
- val_mae_h5
- val_mae_h10
- val_mae_h15
- h15_change_vs_parent
- trend_plot_review
- direction_agreement
- large_rise_recall
- large_drop_recall
- delta_or_slope_correlation
- dynamic_range_ratio
- lag_estimate
- high_fms_or_large_delta_notes
- promotion_decision
- stop_reason
- test_usage

============================================================
10. Final Selection And Final Test Policy
============================================================

Final selection은 validation 기준으로만 한다.

Primary score:
- mean validation MAE over h=5, h=10, h=15

Secondary checks:
1. h=15 validation MAE
2. validation plot/trend-following
3. flow-following metrics: direction agreement, large-rise/drop recall, delta correlation, dynamic range ratio, lag
4. high-FMS and large-delta bucket error
5. seed robustness
6. model simplicity and input-policy clarity
7. static subgroup robustness, static 사용 시

Final selection note:
- 최종 main model은 여전히 primary validation MAE mean(h=5/10/15)가 가장 낮은 후보를 기준으로 한다.
- 단, primary MAE가 매우 근접한 후보들 사이에서는 flow-following과 h15 behavior를 tie-break로 사용한다.
- primary MAE는 조금 낮지 않더라도 flow-following이 뚜렷하게 좋은 후보는 "best trend-following candidate"로 별도 보고한다.

Final test는 이번 6시간 goal에서 필수가 아니다.
단, 아래 조건을 모두 만족하면 validation-selected frozen config로 딱 1회 수행할 수 있다.

- validation 기준 best candidate가 확정됨
- leaderboard와 best_model_summary.md가 작성됨
- VALIDATION_SELECTION_LOCK.md가 작성됨
- pre-lock test output이 없음을 확인함
- final test 이후 추가 tuning을 하지 않겠다고 명시함
- 남은 시간이 final test와 report 작성에 충분함

Final test를 수행하지 않는 경우:
- final_report.md에 "이번 goal은 validation-only exploratory search로 종료했고 final test는 수행하지 않았다"라고 명시한다.
- best validation candidate와 후속 final-test 조건을 기록한다.

Final test를 수행하는 경우:
- final_test_audit.md를 작성한다.
- final test 결과와 validation selection 결과를 분리해 보고한다.
- final test 결과를 보고 추가 실험을 선택하지 않는다.

============================================================
11. Final Report
============================================================

최종 보고는 한국어로 작성하라.

반드시 포함:

1. 수정/추가 파일
2. 새 CLI/config 옵션
3. dataset/windowing/preprocessing 변경
4. model/head/loss 변경
5. 허용 입력 정책 준수 여부
6. sanity test 결과
7. 실제 사용한 full-training search budget
8. validation leaderboard
9. primary metric 기준 best model
10. horizon별 best validation result
11. h=1 diagnostic result, 있으면 기록
12. h=2.5 auxiliary result, 있으면 기록
13. h15 head branch 결과
14. level+delta dual-head branch 결과
15. motion-derived feature branch 결과
16. high-change auxiliary 결과, 실행한 경우
17. validation plot/trend-following review
18. flow-following metrics 요약: direction agreement, large-rise/drop recall, delta correlation, dynamic range ratio, lag estimate
19. best trend-following candidate, primary MAE best와 다르면 별도 표시
20. interrupted/failed run
21. resume 가능 여부
22. final test 수행 여부
23. final test를 수행했다면 validation 기준 선택과 final test 1회 결과를 명확히 분리
24. generated plots/tables
25. git status summary
26. 남은 문제와 다음 우선순위

금지:
- test-driven tuning
- 일반 run에서 test metric/test prediction/test plot 생성
- final test 결과를 보고 다시 실험 선택
- 허용 입력을 넘어서는 정보 추가
- identity feature 추가
- validation/test를 train preprocessing에 사용
- evaluation code를 지표가 좋아지도록 변경
- completed run 중복 실행
- failed/interrupted run 숨기기
- commit
- push
