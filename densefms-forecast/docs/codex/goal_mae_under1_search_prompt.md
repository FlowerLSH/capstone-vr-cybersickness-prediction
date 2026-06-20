# DenseFMS Under-1 MAE Search Goal Spec

이 문서는 `/goal` 장시간 실행용 명세다.

Objective:
- DenseFMS future forecasting 코드베이스에서 validation primary MAE mean over h=5, h=10, h=15를 1.0 이하로 낮추는 것을 목표로 한다.
- 단, 누수, test-driven tuning, metric gaming, 허용 입력 확장은 절대 금지한다.
- 1.0 이하 달성이 불가능하면, 무누수 조건에서 시도한 유의미한 개선/실패 원인/다음 방향을 기록한다.

FULL_TRAINING_ALLOWED = true

Output directory:
- `runs/goal_mae_under1_search_0506/`

## Non-Negotiable Input Policy

이번 search의 main track은 직전 goal의 최종 선택 모델과 같은 입력 범위를 넘으면 안 된다.

Allowed inputs per prediction time `t`:
- calibration head motion in the calibration interval
- calibration FMS in the calibration interval
- recent head motion ending at current time `t`
- one start-FMS value at the start of the recent window, with the same fallback rule as previous corrected start_only runs

Dataset extent policy:
- Use at most the first 420 points of each session.
- Set `max_session_points = 420` for train, validation, and final test loading.
- The prediction target must also stay within the first 420 points.
- With zero-based indices, the maximum allowed `target_index` is 419.
- For h=5/10/15 at 0.5s sampling, the maximum current indices are 409/399/389 respectively, so that `target_index <= 419`.
- Do not treat labels after point 420 as train/validation/test targets unless a later explicit user instruction changes this policy.

Forbidden inputs:
- current FMS at `t`
- target FMS
- future FMS
- dense FMS sequence inside the recent post-calibration window
- sparse_observed
- recent_start_observed
- calibration_end anchor as an extra post-calibration anchor
- participant/session/condition/trial/experiment identity
- file-derived identity
- static features for this goal, unless a later explicit user message allows them again
- any new sensor or feature not causally derived from the already provided head-motion tensor

Allowed transforms:
- causal transforms of the already provided recent head motion, such as norm, delta norm, rolling energy, or learned causal encoders
- architectural changes that consume only the allowed tensors above
- validation-only ensembling or specialist models, if every component uses only the allowed tensors and the ensemble rule is selected on validation only

Disallowed shortcuts:
- shorter recent windows that make start-FMS materially closer than the current best unless explicitly labeled as a separate non-main analysis
- changing evaluation metric or denormalized MAE calculation
- sample filtering that improves validation MAE without a deployment-valid reason
- reading any `eval_test` artifact before validation lock
- using previous or new test results to choose experiments

## Selection And Test Policy

- Primary selection metric: validation MAE mean over h=5, h=10, h=15.
- h=1 is diagnostic only.
- h=2.5 is auxiliary only.
- Model selection must use validation metrics and validation prediction review only.
- Do not generate or read test predictions/metrics during adaptive search.
- Final test is allowed once after validation selection is locked.
- If validation primary MAE remains above 1.0, final test may be omitted unless there is a clearly improved, locked candidate worth final reporting.

## Required Outputs

Maintain:
- `runs/goal_mae_under1_search_0506/PLAN.md`
- `runs/goal_mae_under1_search_0506/input_contract.md`
- `runs/goal_mae_under1_search_0506/leakage_audit.md`
- `runs/goal_mae_under1_search_0506/experiment_log.csv`
- `runs/goal_mae_under1_search_0506/leaderboard.csv`
- `runs/goal_mae_under1_search_0506/RUN_STATE.md`
- `runs/goal_mae_under1_search_0506/resume_manifest.csv`
- `runs/goal_mae_under1_search_0506/run_report.md`
- `runs/goal_mae_under1_search_0506/final_report.md`

Live reporting rule:
- Do not defer detailed reporting until the end.
- Write and maintain `run_report.md` as a cumulative Korean run report.
- After every completed, failed, interrupted, or non-comparable run, immediately update `run_report.md` with:
  - why the run was designed
  - the validation result
  - whether it is comparable under the input policy
  - what insight was learned
  - whether the branch should continue, narrow, or stop
- Also update `experiment_log.csv`, `leaderboard.csv`, `resume_manifest.csv`, and `RUN_STATE.md` at the same time.
- Leave final ranking, TOP5 selection, validation lock, and any final test decision for the final report stage.

Recommended:
- validation ensemble audit
- single-horizon specialist audit
- validation plot reviews
- model change audit

## Search Priorities

Start from the prior locked validation best:
- `runs/goal_mae_6h_adaptive_improvement_0506/goal6h_delta_motion_norm_e80_s7`
- validation primary MAE: 1.727127

Prioritize:
1. Re-audit input contract and baseline comparability.
2. Validation-only ensemble of existing safe runs.
3. Single-horizon specialist models for h=5, h=10, h=15, all using the same allowed input policy.
4. Larger but still input-safe causal encoders over the same recent head window.
5. Conservative optimization/seed checks only when validation evidence justifies them.

Do not call the goal complete merely because the MAE target seems hard. Continue until either:
- validation primary MAE <= 1.0 is achieved and verified without leakage, or
- all high-value safe directions attempted in this goal are documented and no productive continuation remains under the input policy.
