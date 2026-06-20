# Final Model Decision

Date: 2026-05-20

This file records the currently selected final model/policy for DenseFMS online high-risk warning.

## Final Selection

The final warning system is:

`zero_anchor_highgate_delta2_onset10_high12_hrisk100_full_seed42`

with the validation-selected false-alert-reduction alert policy.

This supersedes the fixed `p >= 0.5` warning rule. The fixed `p >= 0.5` rule should be treated as a baseline comparison only.

## Model Artifact

- Checkpoint: `runs/online_current_onset_warning_0514/zero_anchor_highgate_delta2_onset10_high12_hrisk100_full_seed42/best.pt`
- Main onset-warning report: `reports/online_current_onset_warning_0514/onset_warning_experiment_report.md`
- Alert policy report: `reports/online_current_alert_policy_0514/alert_policy_tuning_report.md`
- Selected validation metrics: `reports/online_current_alert_policy_0514/zero_anchor_highgate_delta2_onset10/selected_policy_validation_metrics.csv`
- Selected test metrics: `reports/online_current_alert_policy_0514/zero_anchor_highgate_delta2_onset10/selected_policy_test_metrics.csv`

## Final Alert Policy

Policy was selected using validation predictions only, then applied once to the test split for final reporting.

| Risk target | On threshold | Off threshold | Min duration | Cooldown |
|---|---:|---:|---:|---:|
| FMS >= 8 within 20s | 0.50 | 0.30 | 0.5s | 0.0s |
| FMS >= 12 within 20s | 0.80 | 0.60 | 0.5s | 0.0s |

The policy uses hysteresis: turn alert on when probability reaches the on threshold, and keep it on until probability drops below the off threshold.

## Final Test Event Metrics

| Risk target | Event precision | Event recall | Event F1 | Alert episodes | False alert episodes | Mean lead time |
|---|---:|---:|---:|---:|---:|---:|
| FMS >= 8 | 0.4043 | 0.9667 | 0.5701 | 47 | 28 | 17.29s |
| FMS >= 12 | 0.5172 | 0.8214 | 0.6348 | 29 | 14 | 15.48s |

## Baseline Comparison

Compared with the fixed `p >= 0.5` baseline:

| Risk target | Policy | Event precision | Event recall | Event F1 | Alert episodes | False alert episodes | Mean lead time |
|---|---|---:|---:|---:|---:|---:|---:|
| FMS >= 8 | fixed `p >= 0.5` | 0.2721 | 0.9667 | 0.4247 | 147 | 107 | 17.21s |
| FMS >= 8 | final tuned policy | 0.4043 | 0.9667 | 0.5701 | 47 | 28 | 17.29s |
| FMS >= 12 | fixed `p >= 0.5` | 0.1593 | 1.0000 | 0.2748 | 113 | 95 | 18.12s |
| FMS >= 12 | final tuned policy | 0.5172 | 0.8214 | 0.6348 | 29 | 14 | 15.48s |

## Interpretation

The final system is not a retrained model. It is the same best high-risk onset model with a validation-selected alert policy that reduces false alert episodes.

For FMS >= 12, false alert episodes decreased from 95 to 14, while event F1 improved from 0.2748 to 0.6348. Recall decreases from 1.0000 to 0.8214, so the final policy should be described as a lower-false-alert warning policy rather than a maximum-sensitivity warning rule.

## Conservative Policy Probe

A follow-up probe checked whether threshold/hysteresis/min-duration/cooldown tuning alone could improve precision further:

- `reports/online_current_alert_policy_0514/conservative_policy_probe_2026-05-20.md`

Conclusion: keep the final policy above. More conservative policy settings can reduce false alerts, especially for FMS >= 12, but they do not reliably improve test event precision and they substantially reduce recall.

A separate dual-head policy probe checked whether FMS >= 8 and FMS >= 12 probabilities can be combined:

- `reports/online_current_alert_policy_0514/dual_head_policy_probe_2026-05-20.md`

Conclusion: a dual-head FMS >= 8 caution candidate gives a small improvement, but FMS >= 12 should keep the current final `p12` policy because dual fusion reduces recall too much for only a tiny precision gain.

## Retraining Probe

Head-only fine-tuning and a train-only logistic calibrator probe were also attempted:

- `reports/risk_head_precision_0520/risk_head_precision_retraining_report_2026-05-20.md`

Conclusion: do not promote the new runs. The validation-selected fine-tuned challenger did not improve the final test event metrics, especially for FMS >= 12. The current final model/policy remains the final selection.

## Persistent Warning-Light UI Candidate

Date: 2026-05-21

For the planned UI where a yellow/red warning light remains visible until released, an additional state-aware high-risk head was trained with:

`high_risk_label_mode = current_or_future`

This target treats a row as positive when the current FMS is already above the threshold or when it will exceed the threshold within 20 seconds.

Selected persistent-light candidate:

- Run: `state_headonly_pos0p5_thr12_seed42`
- Checkpoint: `runs/risk_light_state_0521/state_headonly_pos0p5_thr12_seed42/best.pt`
- Report: `reports/risk_light_state_0521/persistent_warning_light_training_report_2026-05-21.md`
- Validation leaderboard: `reports/risk_light_state_0521/validation_persistent_light_leaderboard.csv`
- Final test comparison: `reports/risk_light_state_0521/final_persistent_light_test_comparison.csv`

Validation-selected UI policies:

| Light | Target | Policy |
|---|---|---|
| Yellow | FMS >= 8 | `on=0.40/off=0.10/min=1s/cool=0s` |
| Red | FMS >= 12 | `on=0.40/off=0.20/min=0.5s/cool=0s` |

Final test persistent-light metrics for this candidate:

| Light | UI precision | Pre-onset recall | High-state coverage | UI false alert episodes | Mean lead |
|---|---:|---:|---:|---:|---:|
| Yellow / FMS >= 8 | 0.8000 | 0.8333 | 0.9592 | 9 | 18.08s |
| Red / FMS >= 12 | 0.6857 | 0.7857 | 0.9093 | 11 | 15.57s |

Conclusion: use this as the persistent warning-light UI candidate. Keep the earlier onset-warning final above when the metric is strictly pre-onset event warning. The two rows answer different evaluation questions and should not be mixed in one table without explanation.

## 10s Persistent Warning-Light Candidate

Date: 2026-05-22

A follow-up retraining changed the persistent-light target from 20 seconds to 10 seconds:

`current FMS >= threshold OR FMS will exceed threshold within 10 seconds`

Selected 10s candidate:

- Run: `state10_headonly_pos2_thr12_seed42`
- Checkpoint: `runs/risk_light_state10_0522/state10_headonly_pos2_thr12_seed42/best.pt`
- Report: `reports/risk_light_state10_0522/risk_light_10s_training_report_2026-05-22.md`
- Validation leaderboard: `reports/risk_light_state10_0522/validation_persistent_light_leaderboard.csv`
- Frame-level test metrics: `reports/risk_light_state10_0522/selected_state10_pos2_thr12_frame_level_test_metrics.csv`

Validation-selected UI policies:

| Light | Target | Policy |
|---|---|---|
| Yellow | FMS >= 8 | `on=0.45/off=0.15/min=0.5s/cool=0s` |
| Red | FMS >= 12 | `on=0.75/off=0.45/min=0.5s/cool=0s` |

Final test frame-level metrics:

| Light | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Yellow / FMS >= 8 | 0.8060 | 0.7671 | 0.9635 | 0.8542 |
| Red / FMS >= 12 | 0.8245 | 0.6306 | 0.8210 | 0.7133 |

Conclusion: this is a shorter-lead 10s warning-light candidate. It should not automatically replace the 20s persistent-light UI candidate unless the UI requirement is explicitly changed from 20s anticipation to 10s anticipation.
