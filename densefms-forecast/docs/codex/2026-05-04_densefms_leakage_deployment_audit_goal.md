# DenseFMS Leakage / Metric Gaming / Deployment Realism Audit Goal

## 0. Goal Metadata

- Project: 소융캡디2 / DenseFMS future FMS forecasting
- Date: 2026-05-04 KST
- Goal type: independent audit / red-team verification
- FULL_TRAINING_ALLOWED: false
- Main audit root: `runs/densefms_leakage_deployment_audit_20260504/`
- Primary objective: verify whether existing code, models, evaluation, and reports contain leakage, metric artifacts, or deployment-unrealistic assumptions
- Secondary objective: produce paper/report-safe wording and a prioritized fix list
- Git policy:
  - Do not commit.
  - Do not push.
  - Do not delete or overwrite existing runs.
  - Do not add `runs/`, `artifacts/`, checkpoints, prediction CSVs, or datasets to git.

---

## 1. Audit Mindset

This task is not a model improvement run.

Act as an independent reviewer trying to falsify the current results.

The audit should answer:

1. Is there any future-label, future-motion, target, validation, or test leakage?
2. Are any metrics computed in a way that makes results look better than they really are?
3. Are split/windowing/overlap choices inflating validation or test performance?
4. Are deployment-realistic claims actually deployment-realistic?
5. Are H=1 results being overgeneralized to longer horizons?
6. Are static features, FMS anchors, calibration inputs, or timestamps creating hidden shortcuts?
7. Which issues must be fixed before using results in a paper/report, and which are caveats?

Important principle:

- Do not assume the current report is correct just because sanity tests passed.
- Verify the actual code path used by training/evaluation/prediction CSV generation.
- If evidence is missing, mark the item as `UNVERIFIED`, not as safe.

---

## 2. Strict Execution Limits

Do not run long training.

Allowed:

- static code inspection
- artifact inspection
- metric recalculation from existing prediction CSVs
- small synthetic unit tests
- very short smoke tests only if necessary to exercise dataset/model code paths
- adding audit-only scripts if useful

Forbidden:

- full training
- hyperparameter search
- new model search
- selecting a better model
- using test metrics for model selection
- deleting old artifacts
- overwriting existing run directories
- commit/push

If a check would require long training, record it as `not executed due to audit scope` and propose a minimal future diagnostic.

---

## 3. Audit Output Directory

Create:

```text
runs/densefms_leakage_deployment_audit_20260504/
```

Required outputs:

```text
audit_report_ko.md
audit_findings.csv
audit_findings.jsonl
audit_checklist.md
code_inventory.md
artifact_inventory.json
data_split_audit.md
windowing_leakage_audit.md
normalization_preprocessing_audit.md
metric_recalculation_audit.md
validation_test_protocol_audit.md
deployment_realism_audit.md
claim_risk_audit.md
synthetic_leakage_tests.log
metric_recalc_results.csv
prediction_csv_recalc_results.csv
split_overlap_results.csv
common_window_recalc_results.csv
git_status.txt
```

If some files cannot be produced because required artifacts are missing, still create the file and explain what was missing.

---

## 4. Inputs to Inspect

Inspect these directories if they exist:

```text
src/densefms_forecast/
scripts/
docs/
README.md
README_ko.md
README_densefms_forecast.md
artifacts/
runs/densefms_long_target_search/
runs/densefms_long_horizon_improvement_20260503/
```

Prioritize these artifacts if present:

```text
runs/densefms_long_target_search/final_report.md
runs/densefms_long_target_search/leaderboard_val.csv
runs/densefms_long_target_search/final_selected_models.json
runs/densefms_long_target_search/final_test_metrics.csv
runs/densefms_long_target_search/progress_log.jsonl
runs/densefms_long_target_search/planned_manifest.json
runs/densefms_long_horizon_improvement_20260503/final_report.md
runs/densefms_long_horizon_improvement_20260503/leaderboard_live.csv
runs/densefms_long_horizon_improvement_20260503/final_selected_models.json
runs/densefms_long_horizon_improvement_20260503/final_test_metrics.csv
runs/densefms_long_horizon_improvement_20260503/progress_log.jsonl
```

For missing artifacts, record:

- path
- why it matters
- what audit became impossible or weaker because of the missing file

---

## 5. Severity Labels

Every finding must have one severity:

```text
critical
high
medium
low
info
```

Use this standard:

### critical

Use for issues that can invalidate core results:

- future FMS or target label enters model input
- future motion after current time enters model input
- train/validation/test split is mixed
- test metrics are used for search/model selection
- metric computation is wrong enough to change leaderboard ranking
- final report hides a known invalid setting as a main result

### high

Use for serious but possibly fixable issues:

- participant/session leakage is likely
- overlapping windows cross split boundaries
- scaler/normalizer is fit on all data instead of train only
- validation/test synthetic or oversampled samples are used
- deployment-realistic track uses information unavailable in real deployment
- static/test-session statistics are used in a way that assumes future knowledge

### medium

Use for issues that weaken interpretation:

- H=1 performance may be overclaimed as long-horizon performance
- timestamp/content-phase memorization is not ruled out
- common-window metrics are missing for calibration/horizon comparisons
- static feature burden or missingness is not reported
- validation overfitting risk due to many runs is not discussed
- per-participant/session aggregation bias is not reported

### low

Use for documentation or reproducibility issues:

- ambiguous variable names
- incomplete logs
- unclear report wording
- missing exact command for reproduction

### info

Use for checks that pass or caveats that only need reporting.

---

## 6. Finding Categories

Every finding must use one category:

```text
future_label_leakage
future_motion_leakage
target_shift_bug
off_by_one_horizon_bug
split_leakage
overlapping_window_leakage
normalization_leakage
imputation_or_smoothing_leakage
oversampling_leakage
metric_recalculation_mismatch
metric_filtering_bias
metric_clipping_or_rounding_bias
test_set_selection_leakage
validation_overfitting_risk
deployment_unrealistic_input
deployment_anchor_policy_issue
static_feature_practicality_issue
timestamp_or_content_memorization
h1_overclaim_risk
multi_horizon_aggregation_bias
common_window_fairness_issue
reporting_or_claim_risk
reproducibility_issue
other
```

---

## 7. audit_findings.csv Schema

Create `audit_findings.csv` with these columns:

```text
finding_id
severity
category
title
description
evidence_type
evidence_path
evidence_line_or_function
affected_models
affected_runs
affected_horizons
is_confirmed
is_deployment_blocker
is_paper_blocker
recommended_fix
recommended_report_wording
reproduction_steps
```

Also append equivalent JSON objects to `audit_findings.jsonl`.

---

## 8. Stage 0 — Code and Artifact Inventory

Write `code_inventory.md`.

Include:

1. dataset classes
2. window/target creation functions
3. split generation/loading functions
4. normalization/scaling functions
5. imputation/smoothing/interpolation functions
6. oversampling/resampling/sample-weighting code
7. model classes
8. training entrypoints
9. evaluation entrypoints
10. metric calculation functions
11. prediction CSV writers
12. search runners
13. final report generators
14. sanity test files

For each item, include:

- path
- function/class name
- short role
- whether it is used by the actual run artifacts
- confidence level: `confirmed`, `likely`, `unknown`

Write `artifact_inventory.json` with:

- discovered run directories
- leaderboards
- metrics files
- prediction CSVs
- checkpoints
- logs
- manifests
- reports
- missing expected artifacts

---

## 9. Stage 1 — Dataset Split and Overlap Audit

Write `data_split_audit.md` and `split_overlap_results.csv`.

### 9.1 Participant/session split

Verify whether train/validation/test splits are:

- participant-wise
- session-wise
- random-window-wise
- unknown

Check whether any participant or session appears in more than one split.

If split artifacts exist, calculate:

```text
train_participants ∩ val_participants
train_participants ∩ test_participants
val_participants ∩ test_participants
train_sessions ∩ val_sessions
train_sessions ∩ test_sessions
val_sessions ∩ test_sessions
```

Save exact counts and IDs where possible.

### 9.2 Window overlap leakage

Check whether overlapping windows are generated before or after splitting.

Risk patterns:

- windows from the same participant/session/time range appear in multiple splits
- nearly identical overlapping windows cross split boundaries
- calibration window from one split overlaps target/recent window from another split
- split is performed at sample/window level instead of participant/session level

Where possible, compute overlap using:

```text
participant_id
session_id
window_start_idx
window_end_idx
current_idx
target_idx
```

If exact indices are unavailable, record this as an `UNVERIFIED` risk.

### 9.3 Repeated session / carry-over risk

If participants have multiple sessions or experiments:

- determine whether the split target is unseen participant generalization or unseen session generalization
- check whether the same participant's different sessions are split across train/test
- explain whether this is acceptable for the paper claim being made

If the report claims unseen-user deployment, participant-wise separation is required.

---

## 10. Stage 2 — Future Leakage and Windowing Audit

Write `windowing_leakage_audit.md`.

This is the most important audit stage.

Expected target definition:

```text
target = FMS[t + horizon_steps]
```

Expected input rule:

```text
input may use data at time <= t only
input must not use FMS after current allowed anchor/current time
input must not use motion after current time t
```

### 10.1 Horizon conversion

Verify:

```text
sampling_interval = 0.5 seconds
H=1s   -> 2 steps
H=2.5s -> 5 steps
H=5s   -> 10 steps
H=10s  -> 20 steps
H=15s  -> 30 steps
H=30s  -> 60 steps
```

Check:

- round/floor/ceil consistency
- off-by-one target indexing
- target index `t+h-1` or `t+h+1` bugs
- final valid range bug near sequence end
- multi-horizon target index correctness

### 10.2 FMS input leakage

Check every dataset batch and model forward path.

Look for:

- full FMS sequence included in input batch
- future FMS included as a feature
- target FMS included in calibration/recent/anchor tensors
- teacher forcing with future FMS
- hidden state updated using future FMS
- target delta computed and accidentally passed as input
- trend label or derivative label passed as input

If future FMS is present in the batch object but not used by the model, verify by reading forward signatures and tensor references.

### 10.3 Recent window leakage

For every model using recent windows, verify:

- recent motion contains only samples `<= t`
- recent window does not include `t+1`, `t+h`, or target-near motion
- padding is not filled from future values
- slicing uses Python exclusive end safely
- variable horizon does not shift recent window toward target time

### 10.4 Sparse anchor leakage

For every sparse-anchor setting, verify:

- anchor index is always `<= t`
- anchor is computed relative to current time, not target time
- long horizons do not accidentally place anchor closer to target
- `recent_start_observed` is clearly marked upper-bound-only
- `recent_start_observed` is never included in deployment-realistic track

### 10.5 Calibration leakage

Verify:

- calibration motion/FMS ends at the intended calibration boundary
- calibration does not include prediction target or post-calibration future FMS
- longer calibration settings are interpreted correctly
- calibration branch does not compute whole-session summary statistics

### 10.6 Smoothing/interpolation/imputation leakage

Search for centered or bidirectional operations.

Risk patterns:

- centered rolling average over FMS
- interpolation using future FMS to fill past FMS
- imputation using full session statistics
- filtering/smoothing over the entire sequence before split
- derivative/trend computed using future labels and later used as input

Classify as high or critical depending on whether target/future information reaches model input or only evaluation diagnostics.

---

## 11. Stage 3 — Normalization, Preprocessing, and Oversampling Audit

Write `normalization_preprocessing_audit.md`.

### 11.1 Normalization/scaling

Verify:

- input scalers are fit on train only
- validation/test scalers are not fit separately
- full dataset statistics are not used for learned normalization
- target normalization is train-only or fixed-scale
- static feature normalization is train-only
- per-session normalization does not use future session data unavailable at inference
- per-participant normalization does not use the full test participant trajectory

Acceptable examples:

```text
fixed known FMS scale 0-20 normalization
train-only scaler fit
model-internal BatchNorm trained on train batches only
```

Risky examples:

```text
full dataset mean/std
per-session full-trajectory normalization
per-test-participant full-session standardization
normalization range determined after reading all splits
```

### 11.2 Imputation/outlier removal

Verify:

- outlier removal is fit/decided without using validation/test labels in a way that changes evaluation distribution
- imputation is train-only or online-safe
- no future values are used to fill current inputs
- removed rows/windows are reported by split

### 11.3 Oversampling/sample weighting

If SMOTE, oversampling, Tomek cleaning, balanced sampler, or sample weighting exists, verify:

- applied only to training split
- validation/test remain real samples only
- synthetic samples are not generated from validation/test participants
- synthetic samples are not evaluated
- validation MAE/RMSE are unweighted unless explicitly reported as weighted diagnostics
- class/target distribution from validation/test is not used to tune resampling

If oversampling exists in earlier or external code but not this DenseFMS forecasting code, record it as not applicable with evidence.

---

## 12. Stage 4 — Metric Recalculation Audit

Write `metric_recalculation_audit.md`.

Independently recalculate metrics from existing prediction CSV files.

Inspect:

```text
val_predictions.csv
test_predictions.csv
leaderboard_val.csv
leaderboard_live.csv
final_test_metrics.csv
metrics.json
metrics.csv
```

Recalculate where columns are available:

```text
MAE
RMSE
R2
sMAPE
common-window MAE
common-window RMSE
trend metrics
derivative metrics
high-FMS false positive rate
```

Save:

```text
metric_recalc_results.csv
prediction_csv_recalc_results.csv
common_window_recalc_results.csv
```

Required comparison columns:

```text
run_name
split
horizon
metric_name
official_value
recalculated_value
absolute_difference
relative_difference
status
source_path
```

### 12.1 Filtering bias

Check whether metric rows are filtered by:

- NaN/inf
- missing prediction
- missing target
- time range
- participant/session
- target value
- high error
- post-hoc common window

Verify that filtering is reported and not selected to improve metrics.

### 12.2 Clipping/rounding bias

If prediction CSV includes raw and clipped predictions, or if clipping is done in code, compute:

```text
raw prediction MAE/RMSE
clipped prediction MAE/RMSE
rounded prediction MAE/RMSE
clipped + rounded prediction MAE/RMSE
```

Check:

- whether leaderboard uses raw or clipped values
- whether clipping is legitimate for FMS scale
- whether reported values hide raw model instability
- whether 0-20, 0-10, normalized scale, or z-score scale are confused

### 12.3 Aggregation bias

Check whether metrics are:

- per-sample average
- per-session average
- per-participant average
- horizon aggregate
- weighted by sequence length

Report whether long sessions or participants dominate the metric.

For multi-horizon metrics, verify:

- H=1 does not hide poor H=5/H=10/H=15 performance
- per-horizon metrics are available
- aggregate metric weighting is documented

---

## 13. Stage 5 — Validation/Test Protocol Audit

Write `validation_test_protocol_audit.md`.

Verify:

- validation metrics are used for model selection
- test metrics are computed only after final role selection
- test metrics do not appear in progress logs before selection
- final selected models can be traced to validation leaderboard rows
- final role selection is not manually based on test performance
- ensembles, if any, are selected using validation only
- failed runs are not silently excluded in a way that biases conclusions

Search these artifacts:

```text
progress_log.jsonl
planned_manifest.json
final_selected_models.json
final_test_metrics.csv
final_report.md
leaderboard_val.csv
leaderboard_live.csv
```

If test was evaluated multiple times, classify carefully:

- If there is evidence it was not used for selection: `medium` or `low` reporting caveat.
- If test influenced model choice: `critical`.
- If evidence is insufficient: `UNVERIFIED` risk.

Also assess validation overfitting risk:

- number of validation runs
- number of model families/hyperparameter variants
- number of repeated selections on the same validation set
- whether multi-seed confirmation exists
- whether final report explains validation search budget

---

## 14. Stage 6 — Deployment Realism Audit

Write `deployment_realism_audit.md`.

Strictly separate:

```text
best-score track
deployment-realistic track
upper-bound-only track
diagnostic-only track
```

### 14.1 FMS availability and user burden

For each selected or reported model, record:

- calibration FMS requirement
- sparse anchor interval
- whether current FMS is required
- whether recent FMS is required
- how often the user must report FMS
- whether the model can run without future labels
- whether the setting is realistic for online VR mitigation

Flag as not deployment-realistic if it requires:

- future FMS
- target-time FMS
- whole-session statistics
- evaluation-start FMS unavailable in online use
- very frequent FMS input without reporting user burden
- `recent_start_observed` or any upper-bound convenience label

### 14.2 Static feature practicality

Audit use of:

```text
age
gender
MSSQ
other personal/static features
```

Check:

- whether these are available before deployment
- whether MSSQ requires extra user survey burden
- whether missing static features are handled
- whether static-only or timestamp-only baselines exist
- whether static features may act as participant identity/susceptibility proxy
- whether reports overclaim model learning motion dynamics when static features dominate

### 14.3 Timestamp/content memorization

DenseFMS uses a controlled content/motion setup. Therefore audit:

- whether timestamp is an input feature
- whether content phase can be inferred from motion profile
- whether all participants saw the same or very similar motion profile
- whether timestamp-only or content-phase-only baseline exists
- whether the model may be predicting average sickness trajectory by time rather than individual response

If no diagnostic exists, record as `medium` unverified risk and recommend:

```text
timestamp-only baseline
static-only baseline
motion-only baseline
shuffle-label sanity test
cross-content validation if future data exists
```

### 14.4 Real-time feasibility

Check:

- inference uses only data available by current time
- preprocessing can be done online
- required buffer length is reported
- calibration length is reported
- inference latency is acceptable or at least measured
- horizon is long enough to be useful for mitigation
- H=1 is not overclaimed as practical early warning

---

## 15. Stage 7 — Baseline and Shortcut Audit

Check whether these baselines exist in previous results or can be computed cheaply from existing CSVs:

```text
last_observed_fms persistence baseline
calibration_end_fms constant baseline
sparse_anchor constant baseline
timestamp_only baseline
static_only baseline
motion_only baseline
mean_train_target baseline
shuffled_label sanity baseline
```

Do not run long training to create missing baselines.

If missing, record as `missing diagnostic` and recommend future minimal runs.

Key interpretation:

- If last-observed or anchor baseline is close to the model, the model may mostly copy FMS state.
- If timestamp-only is strong, content-phase memorization is a risk.
- If static-only is strong, participant susceptibility dominates and motion-dynamics claims need caution.
- If shuffled-label is strong, leakage or metric artifact is likely.

---

## 16. Stage 8 — Synthetic Leakage Tests

Write results to `synthetic_leakage_tests.log`.

Implement audit-only synthetic tests if feasible.

### 16.1 Target sentinel test

Create a tiny synthetic sequence where FMS values are unique and monotonic.

Verify:

```text
target index = t + horizon_steps
```

for multiple horizons.

Also verify that target/future FMS sentinel values do not appear in model input tensors.

### 16.2 Future motion sentinel test

Place a sentinel value only in motion samples after current time `t`.

Verify the input tensor for prediction at `t` does not contain the sentinel.

### 16.3 Anchor policy test

Verify sparse anchor index is always:

```text
anchor_idx <= current_idx
```

and is not computed from target time.

### 16.4 Common-window test

Verify automatic common-window calculations for calibration/horizon comparisons.

Examples:

```text
calibration sweep: common_target_start = max(calibration_seconds_list) + horizon_seconds
horizon sweep: common_current_end = session_end_time - max(horizon_seconds_list)
grid sweep: common_current_start = max(calibration_seconds_list)
           common_current_end = session_end_time - max(horizon_seconds_list)
```

### 16.5 Split hash test

Use synthetic participant/session/window IDs to ensure the overlap detector catches split leakage.

---

## 17. Stage 9 — Existing Sanity Test Audit

Inspect `scripts/run_densefms_sanity_tests.py` if it exists.

Evaluate whether it checks:

- seconds-to-steps conversion
- target shift for all horizons
- calibration leakage
- recent-window leakage
- sparse anchor policy
- multi-horizon target indexing
- model forward shape
- actual dataset/model code path, not only mocks
- negative controls that would fail if leakage were introduced

Record:

- what is well covered
- what is not covered
- what PASS does and does not prove

If useful, add audit-only tests, but do not commit.

---

## 18. Stage 10 — Report and Claim Risk Audit

Write `claim_risk_audit.md`.

Read existing reports/docs and classify important claims as:

```text
safe
safe_with_caveat
needs_rewording
unsupported
misleading
```

Pay special attention to claims like:

```text
H=1 performance implies long-horizon prediction success
validation success implies deployment readiness
deployment-realistic without reporting FMS input burden
real-time prediction despite unavailable current/future information
generalizable despite single content or limited content/device validation
motion-dynamics learning despite no timestamp/static-only baseline
robustness without masking/corruption/missing-modality evaluation
static feature benefit without ablation
```

For each risky claim, include:

- original wording or paraphrase
- problem
- safer replacement wording
- whether it is paper-blocking

---

## 19. Required Final Audit Report

Write final report in Korean:

```text
runs/densefms_leakage_deployment_audit_20260504/audit_report_ko.md
```

It must include:

1. 감사 목적
2. 감사 대상 파일/산출물
3. 감사 범위와 한계
4. 한 줄 결론
5. 최종 판정
6. Critical findings
7. High findings
8. Medium findings
9. Low/info findings
10. 통과한 검사 목록
11. 확인 불가 항목 목록
12. split/windowing 검사 결과
13. target/future leakage 검사 결과
14. normalization/preprocessing/oversampling 검사 결과
15. metric 재계산 결과
16. validation/test protocol 검사 결과
17. deployment-realistic 판정
18. static feature / timestamp / content memorization risk
19. H=1 및 long-horizon claim risk
20. 논문/보고서에 써도 되는 안전한 표현
21. 피해야 할 표현
22. 반드시 고쳐야 할 항목
23. caveat로 보고하면 되는 항목
24. 추가로 있으면 좋은 진단 실험
25. 재현 명령어
26. `git status --short`

Final verdict must be one of:

```text
CLEAR
CLEAR_WITH_CAVEATS
NOT_CLEAR_NEEDS_FIXES
UNVERIFIED_INSUFFICIENT_EVIDENCE
```

Verdict guidance:

- `CLEAR`: no confirmed serious issue and enough evidence exists for all major checks.
- `CLEAR_WITH_CAVEATS`: no confirmed critical/high issue, but caveats remain.
- `NOT_CLEAR_NEEDS_FIXES`: one or more confirmed critical/high issues, or paper/deployment-blocking medium issues.
- `UNVERIFIED_INSUFFICIENT_EVIDENCE`: artifacts/code are insufficient to verify major claims.

---

## 20. Acceptance Criteria

The audit is successful if:

1. No long training is run.
2. Existing code and artifacts are inventoried.
3. Split/windowing/future leakage paths are inspected.
4. Existing metrics are recalculated from prediction CSVs where possible.
5. Validation/test protocol is checked from logs/manifests/reports.
6. Deployment-realistic claims are separated from best-score and upper-bound tracks.
7. Findings are written to both CSV and JSONL.
8. Korean final audit report is produced.
9. Missing evidence is explicitly labeled as missing or unverified.
10. No commit/push is performed.

The audit is highly successful if it also produces:

1. synthetic sentinel tests for target/future motion leakage,
2. split overlap detector output,
3. raw vs clipped/rounded metric comparison,
4. safe paper wording suggestions,
5. a prioritized fix list with severity and blocker status.

---

## 21. Suggested Audit Commands

Use these as a starting point. Adjust to the actual repository structure.

```bash
python scripts/run_densefms_sanity_tests.py
```

If an audit script is added:

```bash
python scripts/audit_densefms_leakage_deployment.py \
  --run_roots runs/densefms_long_target_search runs/densefms_long_horizon_improvement_20260503 \
  --output_dir runs/densefms_leakage_deployment_audit_20260504 \
  --data_dir ./DenseFMS/Dataset \
  --no_long_training
```

Metric recalculation from artifacts should run without training:

```bash
python scripts/audit_densefms_metrics.py \
  --run_roots runs/densefms_long_target_search runs/densefms_long_horizon_improvement_20260503 \
  --output_dir runs/densefms_leakage_deployment_audit_20260504
```

Synthetic leakage tests should be lightweight:

```bash
python scripts/audit_densefms_synthetic_leakage_tests.py \
  --output_dir runs/densefms_leakage_deployment_audit_20260504
```

If these scripts do not exist, either implement minimal audit-only versions or document the manual inspection results.

---

## 22. Required Final Response

Final response must be in Korean.

Include:

1. 최종 판정
2. critical/high/medium finding 개수
3. 가장 중요한 발견 3~5개
4. 당장 고쳐야 할 항목
5. caveat로 보고하면 되는 항목
6. 생성된 audit report 경로
7. 생성된 CSV/JSONL 경로
8. `git status --short` 요약

Do not overclaim.

If no critical leakage is found, say:

```text
감사 범위 내에서는 critical leakage를 발견하지 못했다.
```

Do not say:

```text
leakage가 절대 없다.
```

If evidence is missing, clearly say which claim remains unverified.
