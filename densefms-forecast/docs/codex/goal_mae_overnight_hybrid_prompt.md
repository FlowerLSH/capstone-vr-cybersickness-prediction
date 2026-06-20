# /goal Prompt: DenseFMS Overnight Hybrid MAE Search

아래 내용을 `/goal`에 그대로 붙여넣어 사용한다.

```text
/goal

DenseFMS future forecasting 코드베이스에서 corrected start_only 조건의 validation MAE를 낮추는 overnight hybrid search를 진행해줘.

반드시 먼저 다음 문서를 읽고 준수해라.

- AGENTS.md
- docs/codex/goal_mae_search_policy_0505.md
- runs/goal_mae_next1h_0505/NEXT_SEARCH_PLAN.md, 존재하면 읽어라.
- runs/goal_mae_next1h_0505/compact_context_ablation.md, 존재하면 읽어라.
- docs/codex/test.md, 존재하면 읽어라. 없으면 blocker로 보지 말고 RUN_STATE.md에 absent로 기록하라.

이번 goal은 "사전 정의 fixed comparison"과 "결과 기반 adaptive branch"를 둘 다 수행하는 overnight search다.
단순히 run 개수를 채우는 것이 아니라, 무누수 / 허용 입력 / 공정 평가 조건에서 validation MAE를 낮추고,
동시에 구조적으로 설명 가능한 모델을 찾는 것이 목표다.

출력 디렉터리:
- runs/goal_mae_overnight_hybrid_0505/

============================================================
0. Non-negotiable Rules
============================================================

FULL_TRAINING_ALLOWED = true

1. Adaptive/fixed search 중에는 test set을 절대 열지 마라.
2. 일반 run에서는 train/val metric과 train/val prediction CSV만 생성한다.
3. test metric / test prediction CSV / test plot은 validation 기준 최종 후보를 lock한 뒤 1회만 생성한다.
4. final test 이후에는 어떤 architecture / hyperparameter / preprocessing / window / loss / static usage도 변경하지 마라.
5. Primary selection metric은 validation MAE mean over h=5, h=10, h=15다.
6. h=2.5는 auxiliary forecasting result로 별도 기록한다.
7. h=1은 diagnostic/lower-bound/sanity-check로만 기록하고 final selection에는 쓰지 않는다.
8. Main track은 corrected start_only, anchor_mode=none이다.
9. current FMS, target FMS, future FMS, recent window 내부 dense FMS sequence, sparse_observed, recent_start_observed는 금지한다.
10. participant_id/session_id/condition_id/trial_id/experiment_id/file-derived identity는 금지한다.
11. 새 모델/구조는 기존 run들이 받던 허용 입력 이상을 받으면 안 된다.
12. validation/test를 train preprocessing fit에 사용하지 마라.
13. evaluation code를 metric이 좋아지도록 바꾸지 마라.
14. commit 하지 마라.
15. push 하지 마라.
16. completed run은 불필요하게 중복 실행하지 마라.
17. failed/interrupted run은 숨기지 말고 resume_manifest.csv와 experiment_log에 기록하라.

============================================================
1. Budget
============================================================

Overnight budget:
- TARGET_WALL_CLOCK_HOURS = 12
- MAX_WALL_CLOCK_HOURS = 13
- MAX_EPOCHS_PER_RUN = 80
- EARLY_STOPPING_PATIENCE = 8
- cheap check는 40 epoch / patience 6을 기본으로 사용한다.
- final selection lock, final test 1회, final report를 위해 마지막 45~60분을 남겨라.

시간이 부족하면 우선순위는 다음과 같다.

1. leakage/input contract audit
2. current best와 follow-up result 재현/요약
3. recent10 vs recent30 paired verification
4. best recent 후보 seed robustness
5. compact context / static 구조 탐색
6. 시간이 남으면 strict recent-window 검증
7. validation lock
8. final test 1회
9. final report

============================================================
2. Known Validation-Only Starting Point
============================================================

다음은 test selection에 쓰지 말고, validation-only prior로만 사용하라.

기존 locked v2 validation best:
- run: v2_lcsa_per_horizon_heads_adaptive_seed7_ff192_recent20_mh_nostatic_lr0p0003_wd0p0001_drop0p05_e80_s7
- primary validation MAE mean(h=5/10/15): 2.0447891590
- h5/h10/h15: 1.9790893717 / 2.0490536479 / 2.1062244574

follow-up validation-only best:
- run: next1h_recent10_e80_s7
- config: lcsa_per_horizon_heads, d_model=96, transformer_ff_dim=192, no static, start_only/no anchor, recent_window_seconds=10
- primary validation MAE mean(h=5/10/15): 1.8001924350
- h5/h10/h15: 1.6699913779 / 1.8279172394 / 1.9026686877
- 이 후보는 final test를 아직 받지 않았다.

중요 caveat:
- LCSA TCN에서 recent_window_seconds는 pooling window다.
- recent_window_seconds=10에서도 causal recent TCN receptive field는 14.5초일 수 있다.
- 미래 누수는 아니지만, "정확히 최근 10초만 사용"이라고 주장하려면 strict recent-window variant를 검증해야 한다.
- start_only에서 recent_window_seconds를 바꾸면 recent head-motion 길이와 start-FMS anchor recency가 동시에 바뀐다.
- 따라서 recent10/recent30 비교는 pure motion-window 비교가 아니라 motion-history length vs FMS-anchor recency tradeoff 비교로 해석해야 한다.
- recent10은 "최근 10초 head motion + 약 10초 전 start FMS" 조건이다.
- recent30은 "최근 30초 head motion + 약 30초 전 start FMS" 조건이다.

compact context quick check:
- horizon scalar만 쓰는 것은 손실이 비교적 작았다.
- start FMS를 값 1개 scalar only로 주는 것은 손실이 더 컸다.
- raw static은 빠른 check에서 손실을 회복하지 못했다.
- 다만 static은 Age/Gender/MSSQ가 허용 입력이고 실제 사용자 차이를 설명할 수 있으므로 이번 goal에서 더 적극적으로 검증한다.
- recent10/recent30 paired verification의 base는 no static으로 두어 window/FMS-anchor tradeoff를 먼저 분리한다.
- 그 다음 best recent condition에서 static branch를 필수 ablation으로 수행한다.
- horizon branch는 full d_model expansion을 지양하고 scalar/none/small encoder를 one-factor-at-a-time으로 검증한다.
- start FMS branch는 pure scalar only를 바로 main으로 채택하지 말고 scalar_time, small encoder 8/16/32, original encoded를 비교한다.
- static branch도 full Cartesian product는 금지하되, no-static과 동급의 candidate로 적극 비교한다.

Before expanding any search, verify next1h_recent10_e80_s7:
- same train/val split as prior locked runs
- same target normalization and denormalized MAE calculation
- no test files created or read
- no current/target/future FMS in input
- no recent dense FMS sequence in input
- start-FMS anchor definition, nominal index, fallback rule, and observed lag
- same validation sample count, or explain any difference
- If any mismatch is found, mark the result as non-comparable and do not use it as current best.

============================================================
3. Main Track Input Policy
============================================================

Main track 기본 설정:

- fms_context_mode = start_only
- anchor_mode = none
- anchor_interval_seconds = 0
- sparse_observed 사용 금지
- recent_start_observed 사용 금지
- Age/Gender/MSSQ static은 허용하되 identity leakage 없이만 사용한다.

Static input policy:
- Age와 MSSQ는 반드시 train split에서만 fit한 mean/std로 z-score normalize한다.
- Age/MSSQ missing 값이 있으면 train split 통계만 사용해 impute하고, missing count와 처리 방식을 static_branch_summary.md에 기록한다.
- Gender는 static branch의 primary implementation에서 2차원 one-hot [is_male, is_female]으로 인코딩한다.
- unknown/missing gender는 [0, 0]으로 처리한다.
- 현재 코드가 gender_unknown 3차원을 기본으로 쓰면, static branch 실행 전에 가능하면 2D gender option을 구현한다.
- 3D gender fallback은 구현 시간이 부족하거나 backward compatibility가 필요한 경우에만 허용하며, 그 경우 primary static result가 2D가 아니라는 사실을 input_contract.md와 static_branch_summary.md에 명확히 기록한다.
- static scaler/encoder fit에 validation/test 값을 사용하지 마라.
- participant_id/session_id/file-derived identity, condition/trial identity, source path-derived feature는 static에 절대 포함하지 마라.

start_only에서 허용되는 post-calibration FMS:
- recent motion window의 시작 FMS 하나만 허용한다.
- 원칙 이름은 start_fms_index/start_fms_time/start_fms_value로 사용한다.
- 현재 CSV의 anchor_index/anchor_time/anchor_fms는 start_fms_*의 backward-compatible alias로만 해석한다.
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

============================================================
4. Fixed Track
============================================================

Fixed track은 사전에 정한 비교 축이다.
중간 결과가 나빠도 해석상 필요한 최소 비교는 끝까지 수행하라.
단, 명백한 OOM/bug/leakage/환경 blocker가 있으면 기록하고 중단한다.

4.1 Recent10 vs Recent30 paired verification

Base config:
- model = lc_sa_tcnformer
- architecture family = lcsa_per_horizon_heads
- d_model = 96
- transformer_ff_dim = 192
- per_horizon_heads = true
- no static
- start_only/no anchor
- loss_type = smooth_l1
- loss_mode = level_only
- learning_rate = 0.0003
- weight_decay = 0.0001
- dropout = 0.05
- horizons = 5, 10, 15

Run only these two recent-window conditions:
- recent_window_seconds = 10
- recent_window_seconds = 30

Both conditions must be run as a paired comparison:
- seed = 7 first
- if budget allows, add seed = 42 for both conditions
- epochs = 80
- patience = 8

Interpretation rule:
- This is not a pure recent-motion-window comparison.
- Label it "motion-history length vs FMS-anchor recency tradeoff".
- recent10 means less head motion but closer start FMS.
- recent30 means more head motion but older start FMS.
- If recent10 wins, write: "recent10 condition was better; this may reflect the closer start-FMS anchor, not just the shorter motion window."
- If recent30 wins, write: "recent30 condition was better; this may reflect the value of longer head-motion history despite the older start-FMS anchor."

If runtime is too high:
- run 40 epoch cheap checks for recent10 and recent30.
- promote the better condition to 80 epoch only after both cheap checks complete.
- do not expand to recent3/5/7.5/12.5/15 in this goal.

4.2 Seed robustness

After paired recent10/recent30 verification:
- choose the better condition by primary validation MAE.
- run seeds 123 and 202 for the better condition if budget allows.
- if the loser is within +0.03 primary MAE, also add the same seeds for the loser.
- do not give one condition extra seed budget unless the comparison is already clearly decided and documented.
- compare mean and variance across seeds.

4.3 Strict recent-window variant

Because nominal recent window and TCN receptive field differ, implement or use a strict recent-window variant:
- the recent encoder must not see motion older than the nominal recent window for each prediction time.
- no future motion is allowed.
- compare strict vs existing causal-TCN behavior for best recent windows.
- document whether the best result relies on older-than-window past context.

4.4 Compact context fixed checks

Use the better verified condition from recent10/recent30.
If the paired verification is incomplete, use current recent10 only as validation-only prior and label it as unverified.

Do not run the full Cartesian product.
Run one factor at a time:
- Step 1: fix start-FMS context and static; sweep horizon context only.
- Step 2: fix best horizon context; sweep start-FMS context only.
- Step 3: fix best horizon/start-FMS context; run static-active ablation.
- Step 4: combine only top 1-2 settings if budget remains.

Horizon context candidates:
- encoded d_model default
- horizon_encoder_dim = 32
- horizon_encoder_dim = 16
- horizon_encoder_dim = 8
- horizon_context_mode = scalar
- horizon_context_mode = none
- Prefer scalar/none/small encoder if performance loss is small.
- Do not keep full d_model horizon expansion solely by default; justify it by validation MAE.

Start-FMS context candidates:
- original [start_fms, time_since] -> d_model
- [start_fms, time_since] -> 32
- [start_fms, time_since] -> 16
- [start_fms, time_since] -> 8
- direct [start_fms, time_since]
- start_fms scalar_time/direct를 확인해라.
- start_fms scalar only는 낮은 우선순위로 둔다. 이미 빠른 check에서 손실이 컸으므로 필요한 경우에만 재확인한다.
- time_since를 제거하면 성능이 얼마나 떨어지는지 기록하라.
- start FMS 값은 normalized scalar로 직접 전달하거나 small encoder로만 확장하는 방향을 우선한다.

Static candidates:
- no static baseline
- static raw/direct: [age_z, mssq_z, gender_male, gender_female]
- static tiny encoder 8
- static tiny encoder 16
- static gated residual/context bias if simple to implement
- Age/MSSQ는 train-only z-score normalized 값을 사용한다.
- Gender는 2D one-hot [male, female]을 primary static path로 사용한다. unknown/missing은 [0,0]으로 둔다.
- 현재 코드가 gender_unknown 3차원을 기본으로 쓰면, static branch 전에 2D gender option을 구현하는 것을 우선한다. 3D fallback은 예외로만 허용하고 static_branch_summary.md와 input_contract.md에 기록하라.
- static은 main fusion 전체를 지배하지 않도록 compact하게 결합하되, 성능 개선 가능성이 있으면 적극적으로 승격한다.
- static 후보는 primary mean(h=5/10/15), h=15, high-FMS, worst-session audit을 모두 보고 판단한다.
- static이 primary mean을 개선하지 않더라도 high-FMS underprediction 또는 worst-session error를 명확히 줄이면 별도 static-robust candidate로 유지한다.
- static raw/direct가 약하면 즉시 폐기하지 말고 tiny encoder, gated residual/context bias처럼 static이 motion/context representation을 보정하는 구조를 최소 1개 이상 확인한다.
- static 후보는 no-static보다 파라미터가 커졌다는 이유만으로 제외하지 말고, validation primary metric과 robustness audit으로 판단한다.

============================================================
5. Adaptive Track
============================================================

Adaptive track은 중간 validation 결과와 error audit에서 관찰된 실패 원인을 바탕으로 다음 후보를 생성한다.

Adaptive 후보는 반드시 다음 필드를 기록한다.

- run_name
- hypothesis
- trigger_result
- validation_plot_review
- trend_following_issue
- expected_improvement
- stop_condition
- parent_run
- changed_fields
- allowed_input_check
- test_usage = none_prelock

Adaptive branching rules:

1. recent10이 계속 best이면
   - recent30과 같은 seed/config로 비교됐는지 먼저 확인한다.
   - "가까운 start-FMS anchor 이득이 짧은 head-motion 손실보다 컸을 수 있음"으로 해석한다.
   - recent7.5 / recent5 / recent3 쪽으로 확장하지 마라.
   - compact horizon/start-FMS branch 탐색을 우선한다.
   - 시간이 남으면 strict-window 검증을 수행한다.

2. recent30이 best이면
   - recent10과 같은 seed/config로 비교됐는지 먼저 확인한다.
   - "긴 head-motion history 이득이 오래된 start-FMS anchor 손실보다 컸을 수 있음"으로 해석한다.
   - compact context와 static-active ablation을 recent30 기준으로 진행한다.

3. recent10이 h=5만 개선하고 h=15를 악화시키면
   - horizon별 recent window를 새로 도입하지 말고,
   - h=15 head capacity
   - h=15 mild loss weight
   를 검토한다.

4. horizon scalar/none/small encoder의 성능 손실이 작으면
   - horizon scalar 또는 small encoder + start small encoder 조합을 확장한다.
   - 모델 설명 가능성과 parameter reduction도 같이 기록한다.

5. start scalar_time 또는 start small encoder가 기존 start encoder를 거의 따라잡으면
   - start encoder 8/16/32 중 best를 80 epoch와 multi-seed로 승격한다.
   - start scalar only가 나쁘면 time_since 또는 small encoder가 필요한 이유를 기록한다.

6. static branch는 더 적극적으로 탐색한다.
   - Age/MSSQ normalized scalar와 gender 2D one-hot 입력을 static primary path로 구현/검증한다.
   - recent10/recent30 paired verification 이후, best recent condition에서 static raw/direct와 tiny encoder를 우선 실행한다.
   - no-static 대비 primary mean이 개선되면 즉시 80 epoch와 추가 seed로 승격한다.
   - primary mean이 동급(+0.03 MAE 이내)이면서 h=15, high-FMS, worst-session 중 하나라도 개선하면 static-robust candidate로 유지하고 80 epoch 승격을 고려한다.
   - raw/direct가 나빠도 static 자체를 포기하지 말고, tiny encoder 또는 gated residual/context bias 중 최소 1개 구조를 추가로 검증한다.
   - static이 과적합하는 것처럼 보이면 더 큰 encoder로 키우지 말고 raw/direct, tiny encoder, gated residual 중 더 단순한 결합으로 후퇴한다.

7. high-FMS underprediction 또는 large target-start delta 문제가 남으면
   - target-start bucket별 error audit을 바탕으로 mild loss/head 보강을 검토한다.
   - 단 target/future FMS를 input feature로 추가하지 마라.

Promotion rules:

- cheap 40 epoch 후보가 current best 대비 +0.03 MAE 이내이거나 특정 horizon에서 의미 있게 개선하면 80 epoch로 승격할 수 있다.
- 80 epoch 후보가 best가 되면 최소 2개 이상의 추가 seed로 robustness를 확인하라.
- adaptive 후보가 두 번 연속 명확히 악화되면 해당 branch를 중단하고 기록하라.

============================================================
6. Error Audit
============================================================

Strong candidate마다 validation prediction CSV만 사용해 error audit을 생성하라.
MAE/RMSE 수치만 보지 말고 validation plot 또는 validation prediction 시계열을 함께 확인하라.
단, pre-lock 단계에서는 test plot/test prediction/test metric을 생성하거나 열람하지 마라.

필수 audit:
- horizon별 MAE/RMSE/bias
- target FMS bucket별 MAE/bias
- start FMS bucket별 MAE/bias
- target-start delta bucket별 MAE/bias
- current_time bucket별 MAE/bias
- static subgroup별 MAE/bias: age bucket, MSSQ bucket, gender
- worst validation sessions
- high-FMS underprediction
- large-drop overprediction
- validation plot trend review: 상승/하강 구간을 따라가는지, 지연(lag)이 있는지, peak/trough를 과도하게 smoothing하는지
- representative validation plots or plot-derived notes for strong candidates

특히 다음을 확인하라.

- h=15가 계속 병목인지
- target-start가 크게 양수인 구간에서 underprediction이 개선되는지
- target-start가 크게 음수인 구간에서 overprediction이 개선되는지
- static이 전체 평균, h=15, worst sessions, high-FMS 구간 중 어디에 도움되는지
- static feature별로 과적합이나 subgroup degradation이 있는지
- validation plot에서 상승을 늦게 따라가거나 하강을 못 따라가는 후보는, MAE가 비슷하면 adaptive planning에서 낮은 우선순위로 둔다.
- MAE가 약간 나빠도 상승/하강 추세, high-FMS peak, rapid drop을 더 잘 따라가는 후보는 robustness candidate로 기록하고 후속 실험을 고려한다.
- plot review는 qualitative secondary diagnostic이다. final selection의 primary metric은 여전히 validation mean MAE over h=5/10/15다.

============================================================
7. Required Verification
============================================================

코드 수정 또는 새 구조 추가 시 반드시 확인:

- import check
- model forward shape check
- seconds-to-steps conversion
- target shift correctness
- calibration leakage check
- recent-window leakage check
- anchor/start-FMS policy check
- static preprocessing train-only fit check
- dry-run command generation if runner/sweep script is modified
- no pre-lock test output guard

Full training 중 확인:
- checkpoint saving
- metrics JSON/CSV generation
- train/val prediction CSV generation
- leaderboard generation
- resume behavior or skip-completed behavior
- failed/interrupted run visibility

Strict-window variant를 구현하면 추가 확인:
- recent encoder가 nominal window 밖 past motion을 보지 않는지
- recent encoder가 future motion을 보지 않는지
- strict and non-strict candidates가 input contract 문서에 구분되어 기록되는지

============================================================
8. Required Outputs
============================================================

반드시 다음을 생성/유지하라.

- runs/goal_mae_overnight_hybrid_0505/PLAN.md
- runs/goal_mae_overnight_hybrid_0505/input_contract.md
- runs/goal_mae_overnight_hybrid_0505/leakage_audit.md
- runs/goal_mae_overnight_hybrid_0505/fixed_comparison.csv
- runs/goal_mae_overnight_hybrid_0505/adaptive_queue.csv
- runs/goal_mae_overnight_hybrid_0505/experiment_log.csv
- runs/goal_mae_overnight_hybrid_0505/experiment_log.md
- runs/goal_mae_overnight_hybrid_0505/leaderboard.csv
- runs/goal_mae_overnight_hybrid_0505/leaderboard.md
- runs/goal_mae_overnight_hybrid_0505/RUN_STATE.md
- runs/goal_mae_overnight_hybrid_0505/resume_manifest.csv
- runs/goal_mae_overnight_hybrid_0505/error_audit_by_run/
- runs/goal_mae_overnight_hybrid_0505/recent10_recent30_tradeoff_summary.md
- runs/goal_mae_overnight_hybrid_0505/strict_window_audit.md
- runs/goal_mae_overnight_hybrid_0505/compact_context_summary.md
- runs/goal_mae_overnight_hybrid_0505/static_branch_summary.md
- runs/goal_mae_overnight_hybrid_0505/best_model_summary.md
- runs/goal_mae_overnight_hybrid_0505/VALIDATION_SELECTION_LOCK.md 또는 FINAL_SELECTION_LOCK.md
- runs/goal_mae_overnight_hybrid_0505/final_test_audit.md
- runs/goal_mae_overnight_hybrid_0505/final_report.md

일반 adaptive/fixed run에서는 test 관련 파일을 만들지 마라.
final test 관련 파일은 validation lock 이후에만 생성하라.

============================================================
9. Final Selection
============================================================

Final selection은 validation 기준으로만 한다.

Primary score:
- mean validation MAE over h=5, h=10, h=15

Tie-break / secondary checks:
1. h=15 validation MAE
2. seed robustness mean/variance
3. strict-window result
4. error audit 개선
5. 구조 설명 가능성
6. parameter efficiency

Final selection 전에:
- leaderboard를 최신화하라.
- best 후보의 validation prediction audit을 작성하라.
- failed/interrupted runs를 숨기지 마라.
- test output이 pre-lock에 생성되지 않았는지 검사하라.
- VALIDATION_SELECTION_LOCK.md 또는 FINAL_SELECTION_LOCK.md를 작성하라.

Final test:
- validation-selected frozen config/checkpoint로 딱 1회만 수행하라.
- final test 이후 어떤 추가 튜닝도 하지 마라.

============================================================
10. Final Report
============================================================

최종 보고는 한국어로 작성하라.

반드시 포함:

1. validation 기준 선택과 final test 1회 결과를 분리
2. primary selection metric 기준 best model
3. horizon별 best validation result
4. h=2.5 auxiliary result, 있으면 기록
5. h=1 diagnostic result, 있으면 기록
6. recent10/recent30 tradeoff 결과
7. strict-window 검증 결과
8. compact context 결과
9. static branch 결과
10. validation plot/trend-following review 결과
10. seed robustness
11. error audit 요약
12. failed/interrupted runs
13. resume 가능 여부
14. 생성된 주요 파일
15. git status summary
16. final test 이후 추가 튜닝을 하지 않았다는 확인

금지:
- test-driven tuning
- 일반 run에서 test metric/test prediction/test plot 생성
- final test 결과를 보고 다시 실험 선택
- 허용 입력을 넘어서는 정보 추가
- identity feature 추가
- validation/test를 train preprocessing에 사용
- completed run 중복 실행
- failed/interrupted run 숨기기
- commit
- push
```
