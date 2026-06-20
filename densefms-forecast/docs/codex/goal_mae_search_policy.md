# DenseFMS MAE Search Policy

현재 DenseFMS future forecasting 코드베이스에서, 현재 허용된 입력 정보만 사용하여 MAE를 최대한 낮추는 장기 adaptive search를 진행한다.

## 1. 가장 중요한 원칙

- 정해진 실험 목록을 수행하는 것이 목표가 아니다.
- 실험 개수 채우기가 목표가 아니다.
- 모델 종류를 많이 시도했다는 사실은 성과가 아니다.
- 목표는 “무누수, 허용 입력, 공정 평가 조건을 지키면서 MAE를 낮추는 것”이다.
- 모든 실험은 이전 결과에서 관찰된 실패 원인을 해결하기 위한 가설을 가져야 한다.
- 성능 개선 가능성이 낮다고 판단되는 방향은 과감히 중단하고, 더 유망한 방향으로 전환한다.
- 목표 MAE 달성보다 leakage 방지와 공정한 평가가 절대 우선이다.

## 2. Execution permission

FULL_TRAINING_ALLOWED = true

- 이 goal에서는 full training / long adaptive search를 허용한다.
- 단, 아래 budget을 초과하지 않는다.
- 먼저 AGENTS 관련 규칙, README, docs/codex/test.md를 읽어라.
- repository 내부 AGENTS, README, docs/codex/test.md 규칙과 이 문서가 충돌하면 충돌 지점을 보고하고, 안전/무누수/실험통제 원칙을 우선한다.
- GPU가 없거나 runtime이 과도하거나 training environment가 불안정하면 full training을 강행하지 말고 smoke/audit만 수행한 뒤 중단한다.

## 3. Budget / execution policy

- MAX_WALL_CLOCK_HOURS = 8
- MAX_TOTAL_RUNS = 75
- MAX_EPOCHS_PER_RUN = 10
- EARLY_STOPPING_PATIENCE = 12
- FINAL_RETRAIN_MAX_EPOCHS = 100
- FINAL_RETRAIN_PATIENCE = 15
- FINAL_RETRAIN도 MAX_TOTAL_RUNS와 MAX_WALL_CLOCK_HOURS 안에 포함한다.

- MAX_TOTAL_RUNS는 안전 상한이지 목표 run 수가 아니다.
- 특정 run 수를 채우는 것이 목표가 아니다.
- 실험 수가 아니라 validation 결과, 실패 양상, 개선 가설에 따라 다음 run을 결정한다.
- 이전 실험 결과를 분석해 개선 가능성이 있는 동안 adaptive search를 진행한다.
- 성능 개선 가능성이 낮거나 같은 실패 양상이 반복되면 해당 방향은 중단하고 다른 방향으로 전환한다.
- 더 이상 유의미한 개선 가설이 없다고 판단되면 MAX_TOTAL_RUNS에 도달하지 않았더라도 중단하고 지금까지의 결과와 한계를 정리한다.

## 4. Interrupt / resume policy

- 중간에 interrupt, timeout, crash, GPU OOM, 사용자 중단이 발생할 수 있음을 전제로 작업한다.
- 어떤 시점에서 중단되더라도, 그때까지의 결과를 사람이 해석할 수 있어야 한다.
- 어떤 시점에서 재개하더라도, 이미 완료된 실험을 중복 실행하지 않고 이어서 진행할 수 있어야 한다.
- 각 run은 가능한 한 atomic하게 관리한다.
  - started
  - running
  - completed
  - failed
  - interrupted
  상태를 명확히 기록한다.

- 다음 파일을 유지한다.
  - runs/goal_mae_search/RUN_STATE.md
  - runs/goal_mae_search/resume_manifest.csv
  - runs/goal_mae_search/experiment_log.csv
  - runs/goal_mae_search/experiment_log.md
  - runs/goal_mae_search/leaderboard.csv
  - runs/goal_mae_search/leaderboard.md

- resume_manifest.csv에는 다음 정보를 기록한다.
  - run_name
  - status
  - command
  - config_path
  - checkpoint_path
  - metrics_path
  - prediction_csv_path
  - plot_dir
  - start_time
  - end_time
  - best_epoch
  - best_val_mae
  - failure_reason 또는 interrupt_reason
  - resume_action

- 재개 시에는 먼저 RUN_STATE.md, resume_manifest.csv, experiment_log.csv, leaderboard.csv를 읽고 현재까지의 결과를 요약한다.
- 이미 completed 상태인 run은 재실행하지 않는다.
- interrupted 상태인 run은 checkpoint와 log가 유효하면 resume를 시도한다.
- checkpoint/log가 불완전하거나 resume가 위험하면 해당 run을 failed/interrupted로 표시하고, 같은 설정을 새 run_name으로 재실행할지 판단한다.
- failed run도 숨기지 말고 failure reason을 기록한다.
- 중단 직전까지의 best validation 결과, 실패한 방향, 다음으로 이어서 할 실험 후보를 RUN_STATE.md에 남긴다.

## 5. Main track 정의

Main deployment-realistic track은 다음 설정을 기준으로 한다.

- fms_context_mode = start_only
- anchor_mode = none
- anchor_interval_seconds = 0
- sparse_observed 사용 금지
- recent_start_observed 사용 금지

Static feature는 적극적으로 사용할 수 있다.

- use_static = true 허용
- Age, Gender, MSSQ는 허용된 static input으로 간주한다.
- static 정보가 validation MAE를 개선한다면 main track 성능으로 인정한다.
- 단, participant_id, session_id, condition id, trial id, experiment id, file name에서 유도한 subject/session 식별자처럼 직접적인 ID 또는 조건 식별자는 사용하지 않는다.
- static feature는 실제 deployment에서도 사전에 수집 가능한 사용자 특성 정보로 해석한다.
- static feature 사용 여부는 모든 run name, leaderboard, final report에 명시한다.
- static-enabled 결과와 no-static 결과를 섞어서 보고하지 않는다.

## 6. FMS input policy

“calibration 이후 FMS 전면 금지”가 아니다.
FMS input은 fms_context_mode에 따라 제한한다.

Main track은 fms_context_mode=start_only이다.

start_only에서 허용되는 FMS:

1. calibration encoder에 들어가는 calibration 구간 FMS history
2. 각 prediction 시점 recent motion window의 시작 FMS:
   FMS[t - recent_window_steps + 1]
3. 단, 해당 start FMS가 missing / NaN / non-finite이면 Start FMS missing policy에 따른 fallback FMS

start_only에서 금지되는 FMS:

1. target FMS:
   FMS[t + horizon_steps]
2. current index t 이후의 future FMS
3. recent window 내부의 dense FMS sequence 전체
4. FMS[t] current FMS
5. sparse_observed, recent_start_observed, calibration_end anchor, sparse_anchor처럼 start_only 정책 밖의 추가 anchor-assisted FMS
6. sparse_observed / recent_start_observed flag 또는 그와 동등한 관측 여부 정보
7. validation/test target이나 future FMS에서 파생한 feature

## 7. Start FMS missing policy

- start_only의 원칙적 start FMS index는:
  FMS[t - recent_window_steps + 1]
  이다.
- 단, 해당 index의 FMS가 missing / NaN / non-finite인 경우에는 현재 구현의 fallback 정책을 허용한다.
- fallback 정책:
  - nominal start index 이하에서 가장 가까운 최신 finite FMS를 사용한다.
  - fallback FMS index는 nominal start index 이하이어야 한다.
  - current index t보다 이후의 FMS는 절대 fallback으로 사용할 수 없다.
  - target index t + horizon_steps의 FMS는 절대 fallback으로 사용할 수 없다.
  - future FMS를 사용한 backward fill은 금지한다.
- 현재 CSV의 anchor_index / anchor_time / anchor_fms는 start_fms_index / start_fms_time / start_fms_value의 backward-compatible alias로 해석한다.
- 가능하면 다음 컬럼을 추가하거나 기록한다.
  - anchor_index
  - anchor_time
  - anchor_fms
  - anchor_is_fallback
  - nominal_start_index
  - nominal_start_time
- 추가 구현 비용이 과도하면 최소한 leakage_audit.md에 fallback 발생 여부와 검증 방법을 기록한다.

## 8. Prediction indexing / CSV alignment

- recent motion window가 [t - recent_window_steps + 1, ..., t]라면, start_only FMS는 원칙적으로 FMS[t - recent_window_steps + 1]만 허용한다.
- 해당 값이 missing이면 Start FMS missing policy에 따라 nominal start index 이하의 최신 finite FMS로 fallback할 수 있다.
- target은 정확히 FMS[t + horizon_steps]이다.
- head/motion input은 current time t까지만 사용할 수 있다.
- t 이후의 head/motion/FMS는 input으로 절대 사용하지 않는다.
- horizon_seconds=5이면 target은 정확히 t+5초여야 한다.
- horizon_seconds=2.5이면 target은 정확히 t+2.5초여야 한다.
- horizon_seconds=1이면 target은 정확히 t+1초여야 한다.

Prediction CSV에서 다음 metadata alignment를 검증한다.

- current_time
- target_time
- start_fms_time 또는 anchor_time
- start_fms_index 또는 anchor_index
- start_fms_value 또는 anchor_fms
- predicted_fms
- target_fms

현재 코드가 start_fms_* 컬럼을 저장하지 않는다면, anchor_time / anchor_index / anchor_fms를 start FMS metadata alias로 해석한다.

fms_context_mode=start_only에서는 anchor_time / anchor_index / anchor_fms가 recent motion window의 시작 FMS를 의미해야 한다.

- missing fallback이 발생하지 않은 경우 anchor_index는 원칙적으로 t - recent_window_steps + 1이어야 한다.
- missing fallback이 발생한 경우 anchor_index는 nominal start index 이하의 최신 finite FMS index여야 한다.
- anchor_index가 current index t 이후이거나 target index를 참조하면 leakage로 간주한다.

## 9. Static feature 처리

허용 static input:

- Age
- Gender
- MSSQ

금지 static/identity input:

- participant_id
- session_id
- condition id
- trial id
- experiment id
- file name에서 유도한 subject/session 식별자
- train/val/test split 정보를 암시하는 feature

처리 원칙:

- Age, MSSQ는 train set에서만 fit한 scaler로 normalize하고 val/test에는 transform만 적용한다.
- Gender encoding 방식은 train/val/test에서 일관되게 적용한다.
- missing static value가 있으면 train set 기준 imputation 값만 사용한다.
- static missingness 자체를 feature로 사용할 경우, 그 mask가 subject/session identity를 암시하지 않는지 확인한다.
- static feature를 사용한 결과와 사용하지 않은 결과를 leaderboard에서 분리해서 볼 수 있게 기록한다.

## 10. Test 사용 제한

일반 adaptive search run에서는 test set을 사용하지 않는다.

- 일반 adaptive search run에서는 train/val metric과 train/val prediction CSV만 생성한다.
- 일반 run에서 test metric, test prediction CSV, test plot을 생성하지 않는다.
- test prediction/metric은 validation 기준 최종 후보가 확정된 이후 1회만 생성한다.
- test 결과를 보고 다음 실험을 선택하지 않는다.
- baseline 확인 시 이미 존재하는 test 결과를 기록할 수는 있지만, 그 결과를 기반으로 다음 실험을 고르지 않는다.
- 최종 보고에서 validation 기준 선택과 final test 결과를 명확히 구분한다.

## 11. Evaluation policy

- val/test의 기존 dense rolling evaluation 정책은 유지한다.
- 성능 개선 목적으로 evaluation sample density, stride, window overlap 정책을 바꾸지 않는다.
- val/test에서 기존보다 더 쉬운 평가 구간만 남기는 방식은 금지한다.
- calibration_seconds나 horizon_seconds를 바꿔 실험할 경우, natural metric과 common-window metric을 모두 기록한다.
- common-window metric은 calibration/horizon 변화로 평가 구간이 달라지는 문제를 보정하기 위한 것이며, cherry-picking 용도로 사용하지 않는다.

## 12. Primary selection metric

최종 main model은 다음 기준으로 선택한다.

- primary selection metric = mean validation MAE over h=5, h=10, h=15
- 즉, h=5/10/15 세 horizon의 validation MAE 평균이 가장 낮은 후보를 main final candidate로 선택한다.
- h=5, h=10, h=15는 main forecasting 성능으로 해석한다.
- h=2.5는 valid forecasting result로 별도 기록한다.
- h=1은 diagnostic / lower-bound / sanity-check로만 사용하며 목표 달성 여부와 final model selection에 반영하지 않는다.
- horizon별 best result도 별도로 기록한다.
  - best h=2.5
  - best h=5
  - best h=10
  - best h=15
  - best diagnostic h=1
- h=1에서 좋은 MAE가 나오더라도 “목표 달성”으로 보고하지 않는다.
- h=1이 가장 좋은 결과인 경우, “짧은 horizon에서는 예측 가능성이 있으나, 의미 있는 forecasting horizon에서는 아직 목표 미달”처럼 해석한다.

## 13. 성능 목표

최근 patched start_only baseline을 기준으로 개선을 목표로 한다.

참고 baseline:

- H5: val MAE 약 2.020, test MAE 약 2.121
- H10: val MAE 약 2.117, test MAE 약 2.227
- H15: val MAE 약 2.285, test MAE 약 2.371

목표:

- baseline은 no-static start_only 결과와 static-enabled start_only 결과를 구분해 기록한다.
- static-enabled model이 더 좋다면 main best result로 인정한다.
- 우선 목표는 start_only + allowed static 조건에서 H5/H10/H15 validation MAE를 유의미하게 낮추는 것이다.
- 1차 성공 기준은 h>=5에서 baseline 대비 명확한 개선을 보이는 것이다.
- 강한 성공 기준은 h=5/10/15 mean validation MAE에서 baseline 대비 10% 이상 개선이다.
- stretch target:
  - h>=2.5에서 MAE 1.2 이하
  - 최종 도전 목표는 h>=2.5에서 MAE 1.0 이하
- 단, MAE 1.2 또는 1.0은 현재 증거상 매우 공격적인 stretch target으로 취급한다.
- 목표를 달성하지 못해도 정당한 실험 중 가장 좋은 결과와 실패 원인을 명확히 보고한다.
- MAE 수치가 좋아졌더라도 leakage 가능성, 평가 조건 변경, 입력 정보 증가가 있으면 성과로 인정하지 않는다.

## 14. 허용되는 접근

### 14-1. 모델 구조 개선

현재 허용된 main track 입력 정보만 사용한다는 조건하에 모델 구조는 자유롭게 개선할 수 있다.

예:

- TCN
- GRU/LSTM variant
- causal Transformer
- multi-scale temporal encoder
- residual/gated block
- attention pooling
- lightweight state-space style block
- CNN+RNN hybrid

단, 어떤 구조든 현재 시점 t 이후의 정보를 보지 않는 causal 구조인지 검증한다.

새 구조를 구현하기 전, 왜 이 구조가 현재 실패 양상을 개선할 수 있는지 experiment_log.md에 기록한다.

### 14-2. 학습 방법론 개선

허용:

- curriculum learning
- horizon curriculum
- calibration curriculum
- input masking/corruption
- feature masking
- temporal masking
- Gaussian noise
- dropout
- bias drift
- spike/dropout injection
- regularization
- optimizer/scheduler 조정
- weight decay
- EMA/SWA
- early stopping 조정
- MAE / SmoothL1 / Huber beta sweep
- FMS-density-aware weighting
- trend-aware auxiliary loss

단, loss weighting이나 sampling weight는 train distribution에서만 계산한다.

### 14-3. 파생 feature / preprocessing

현재 허용된 입력으로부터, 현재 시점 t 이하만 사용해 계산되는 causal derivative/statistical feature는 후보로 둘 수 있다.

예:

- velocity
- acceleration
- rolling mean/std
- causal temporal difference

단, 이것은 same-input derived feature로 명확히 기록하고, 원본 입력만 사용한 모델과 별도 ablation을 남긴다.

미래 구간, target FMS, validation/test 통계를 사용한 feature는 금지한다.

## 15. predict_delta_from_anchor 관련

- predict_delta_from_anchor는 별도로 기록한다.
- predict_delta_from_anchor가 sparse anchor, current FMS, future FMS, dense FMS sequence, observed flag를 요구하면 main track에서 금지한다.
- 만약 predict_delta_from_anchor가 start_only에서 허용된 단일 start FMS 또는 Start FMS missing policy에 따른 fallback FMS만 사용한다면 후보로 검토할 수 있다.
- 이 경우에도 반드시 predict_delta_from_anchor=true/false ablation을 남긴다.
- delta anchor가 start_only 정책 밖의 추가 anchor-assisted diagnostic 성능을 만든 경우, 이를 main 성능으로 보고하지 않는다.

## 16. 작업 순서

### Phase 0. Baseline input contract 문서화

먼저 현재 코드와 실험 설정을 읽고 baseline의 정확한 input contract를 문서화한다.

문서 위치:

- runs/goal_mae_search/input_contract.md

반드시 확인할 것:

- input column
- fms_context_mode
- anchor_mode
- anchor_interval_seconds
- use_static 여부
- static feature 목록
- static preprocessing 방식
- recent_start_observed 사용 여부
- sparse_observed 사용 여부
- predict_delta_from_anchor 사용 여부
- calibration 구간
- calibration encoder에 들어가는 FMS
- 각 prediction 시점에서 start_only FMS index
- start FMS missing fallback 정책
- recent window 내부 dense FMS sequence가 input으로 들어가는지 여부
- prediction target index
- horizon_seconds와 target shift 일치 여부
- recent window가 current time t 이후를 포함하지 않는지
- val/test dense rolling evaluation 정책
- normalization fit 위치
- subject-wise split 유지 여부

### Phase 1. Leakage audit

성능 탐색보다 먼저 leakage audit을 수행한다.

문서 위치:

- runs/goal_mae_search/leakage_audit.md

검증 항목:

- train/val/test subject 분리
- target shift
- future head/FMS 미사용
- start_only FMS index
- start FMS missing fallback이 nominal start index 이하에서만 발생하는지
- fallback이 future FMS/backward fill을 사용하지 않는지
- recent window 내부 dense FMS sequence 미사용
- target FMS 미사용
- sparse_observed/recent_start_observed 미사용
- calibration_end anchor / sparse_anchor처럼 start_only 정책 밖의 추가 anchor-assisted FMS 미사용
- static feature가 허용 목록 안에 있는지
- static scaler/imputation이 train-only인지
- participant/session/condition/trial identity feature 미사용
- train-only normalization
- val/test dense rolling evaluation 정책 유지
- metric 계산 scale
- prediction CSV alignment

문제가 있으면 성능 탐색보다 먼저 수정한다.

### Phase 2. Patched start_only baseline 재현

현재 patched start_only baseline을 같은 split, 같은 seed, 같은 metric으로 재실행하거나 기존 best run을 확인한다.

문서 위치:

- runs/goal_mae_search/baseline_summary.md

기록할 것:

- baseline command
- split file
- seed
- input columns
- fms_context_mode
- anchor_mode
- anchor_interval_seconds
- use_static
- static feature 목록
- recent_start_observed
- sparse_observed
- predict_delta_from_anchor
- calibration_seconds
- recent_window_seconds
- horizon_seconds
- model summary
- val MAE/RMSE
- 기존 test 결과가 있다면 참고용으로만 기록
- prediction plot 또는 prediction CSV 위치
- 현재 실패 양상 분석

### Phase 3. Evidence-driven cheap improvement

먼저 낮은 비용의 개선부터 시도한다.

가능한 방향:

- learning rate
- weight decay
- dropout
- hidden size
- batch size
- scheduler
- early stopping
- SmoothL1 beta
- gradient clipping
- optimizer
- normalization 방식
- static feature scaling/imputation 방식
- static fusion 방식
- seed sensitivity 확인

단순 grid search가 아니다. 이전 결과를 보고 다음 실험을 선택한다.

### Phase 4. Evidence-driven architecture search

모델 구조는 결과 기반으로 선택한다.

- 특정 개수의 구조를 채우지 않는다.
- 현재 실패 양상에 근거해 가장 가능성 있는 구조 개선을 선택한다.
- 새 구조를 구현하기 전, 왜 이 구조가 현재 문제를 해결할 수 있는지 기록한다.
- 성능 개선이 없거나 과적합이 심하면 해당 방향을 빠르게 중단한다.

### Phase 5. Curriculum and robustness training

필요하다고 판단되면 curriculum 또는 robustness training을 시도한다.

가능한 방향:

- horizon curriculum
- calibration curriculum
- input masking curriculum
- temporal masking
- sensor corruption training
- noise/dropout/spike injection
- loss curriculum
- static dropout 또는 static corruption

주의:

- curriculum이 평가 조건을 유리하게 바꾸면 안 된다.
- 최종 평가는 원래 target horizon과 동일한 조건에서 수행한다.
- h=1을 curriculum의 쉬운 시작점으로 사용할 수는 있지만, 최종 목표 달성 여부는 h>=2.5에서만 판단한다.

### Phase 6. Promising direction refinement

유망한 방향이 발견되면 해당 방향을 중심으로 refinement를 진행한다.

- validation 성능, 안정성, leakage risk, 구현 복잡도를 기준으로 계속 진행할지 중단할지 판단한다.
- 최종 후보에 대해서는 budget 내에서 final retrain을 수행할 수 있다.
- FINAL_RETRAIN도 MAX_TOTAL_RUNS와 MAX_WALL_CLOCK_HOURS 안에 포함한다.
- test set은 최종 후보 확인에만 사용한다.
- test set 성능을 보고 다시 구조를 고르는 test-driven tuning은 금지한다.

## 17. Final retrain

Final retrain은 optional이다.

- validation으로 config와 best_epoch을 먼저 확정한다.
- final retrain을 할 경우 train+val로 재학습하되, epoch 수는 validation run에서 선택된 best_epoch으로 고정한다.
- final retrain 중 test를 보지 않는다.
- final test는 마지막에 1회만 수행한다.
- train+val retrain 구현이 복잡하거나 위험하면 final retrain을 생략하고 validation-selected checkpoint로 test를 1회 평가한다.

## 18. Required outputs

반드시 다음 파일을 유지한다.

- runs/goal_mae_search/input_contract.md
- runs/goal_mae_search/leakage_audit.md
- runs/goal_mae_search/baseline_summary.md
- runs/goal_mae_search/experiment_log.csv
- runs/goal_mae_search/experiment_log.md
- runs/goal_mae_search/leaderboard.csv
- runs/goal_mae_search/leaderboard.md
- runs/goal_mae_search/RUN_STATE.md
- runs/goal_mae_search/resume_manifest.csv
- runs/goal_mae_search/best_model_summary.md

leaderboard에는 반드시 다음을 포함한다.

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
- loss_type
- seed
- split_file
- h=5 val MAE
- h=10 val MAE
- h=15 val MAE
- mean val MAE over h=5/10/15
- h=2.5 val MAE if available
- h=1 diagnostic val MAE if available
- test metric은 final test 1회 결과에만 기록

## 19. 중간 저장 / 재개 요구사항

- 각 run 시작 전 config를 저장한다.
- 각 run 시작 시 resume_manifest.csv에 status=started로 기록한다.
- 학습 시작 후 status=running으로 갱신한다.
- epoch별 train/val metric을 저장한다.
- best checkpoint를 저장한다.
- 일반 run에서는 train/val prediction CSV만 저장한다.
- test prediction CSV는 최종 후보 확정 이후 final test 1회에서만 저장한다.
- run이 정상 종료되면 status=completed로 기록한다.
- run이 중단되면 status=interrupted로 기록하고, 중단 시점까지의 best metric과 checkpoint 경로를 남긴다.
- run이 오류로 실패하면 status=failed로 기록하고, 오류 원인과 stderr/log 위치를 남긴다.
- RUN_STATE.md에는 항상 현재까지의 결과 해석과 다음 실행 후보를 기록한다.

## 20. 평가 및 분석 요구사항

- 단순히 MAE만 보지 말고 prediction distribution도 확인한다.
- 실제 FMS 분포와 예측 FMS 분포를 비교한다.
- high FMS 구간에서 error가 큰지 확인한다.
- stable / increasing / decreasing 구간별로 성능을 확인한다.
- 예측이 평균으로 수축되는지 확인한다.
- 특정 subject에서만 좋아지는지 확인한다.
- validation 성능 개선이 전체적으로 안정적인지, 특정 구간/subject에만 의존하는지 확인한다.
- 모든 horizon 결과를 기록하되, 목표 달성 여부는 h>=2.5 기준으로만 판단한다.
- h=1은 diagnostic result로 분리한다.
- h=5 이상 결과를 주요 forecasting 성능으로 별도 요약한다.
- static-enabled 결과와 no-static 결과를 구분해 요약한다.
- main track 결과와 diagnostic/start_only 정책 밖의 추가 anchor-assisted/sparse_observed 결과를 섞지 않는다.
- 중간에 interrupt가 발생했다면, 중단 시점까지의 결과만 기준으로도 현재 결론과 다음 실험 우선순위를 정리한다.

## 21. 금지

- commit 하지 마라.
- push 하지 마라.
- runs/, artifacts/, DenseFMS/Dataset, checkpoint/model weight를 git에 추가하지 마라.
- test-driven tuning 하지 마라.
- 일반 run에서 test metric/test prediction/test plot을 생성하지 마라.
- 지표를 좋게 보이게 하려고 evaluation code를 바꾸지 마라.
- 허용 입력을 넘어서 정보를 추가하지 마라.
- validation/test를 train preprocessing에 사용하지 마라.
- target smoothing으로 정답 자체를 쉽게 만들지 마라.
- evaluation sample density, stride, window overlap 정책을 성능 개선 목적으로 바꾸지 마라.
- 좋은 결과만 남기고 실패한 실험을 숨기지 마라.
- h=1 결과를 목표 달성 근거로 사용하지 마라.
- sparse_observed, recent_start_observed, calibration_end anchor, sparse_anchor처럼 start_only 정책 밖의 추가 anchor-assisted FMS를 사용한 결과를 main track 성능으로 보고하지 마라.
- participant_id/session_id/condition_id/trial_id/file-derived identity를 사용하지 마라.
- static 사용 여부를 숨기지 마라.
- start FMS missing fallback에서 future FMS/backward fill을 사용하지 마라.
- completed run을 불필요하게 중복 실행하지 마라.
- interrupted/failed run을 기록 없이 숨기지 마라.

## 22. 최종 보고

최종 보고는 한국어로 작성한다.

포함할 것:

1. FULL_TRAINING_ALLOWED 확인 결과
2. 사용한 budget과 실제 사용량
3. baseline 성능
4. leakage audit 결과
5. input contract 요약
6. main track 정의
7. FMS input policy와 start FMS missing fallback 검증 결과
8. static feature 사용 여부와 처리 방식
9. static-enabled vs no-static 비교
10. primary selection metric 기준 best model
11. horizon별 best result
12. h=1 diagnostic result
13. h>=2.5 valid forecasting result
14. h>=5 main forecasting result
15. final test 1회 결과
16. interrupted/failed run 목록과 원인
17. resume 가능한 checkpoint가 있는지
18. 실패한 방향과 중단 이유
19. stretch target 달성 여부
20. best config
21. best model 구조 요약
22. 재현 명령어
23. prediction plot / CSV 위치
24. validation 기준 성능과 final test 성능의 구분
25. 현재까지 결과만 기준으로 해석했을 때의 결론
26. 추가 실행이 필요하다면 우선순위가 높은 다음 실험
27. 남은 리스크
28. 다음에 사람이 직접 판단해야 할 선택지