# Adaptive next-step policy for DenseFMS LC-SA-TCNFormer

## Purpose

This file defines what Codex should do if the planned LC-SA-TCNFormer search produces only mediocre validation results.

The goal is not to run unlimited experiments. The goal is to analyze validation results, diagnose likely failure modes, run a small bounded set of follow-up experiments, and produce an honest report.

Do not use test metrics during this adaptive stage.

Test evaluation is allowed only after a final configuration is selected using validation metrics.

## When to trigger adaptive stage

After Stage 1 and Stage 2 full-search results are complete, trigger adaptive analysis if any of the following conditions hold:

1. Best validation MAE improves less than 10% over the strongest naive baseline.
2. Best validation R2 <= 0.10.
3. Best validation prediction variance is less than 25% of target variance.
4. Prediction trajectories are visually over-smoothed or collapse toward the subject/global mean.
5. Best no-anchor model is much worse than anchor-based models.
6. Calibration length changes do not improve common-window validation metrics.
7. Recent window changes do not meaningfully change validation metrics.
8. Horizon 5s or 10s prediction is already poor enough that longer horizon results are not meaningful.
9. High-FMS detection recall is very low even when regression MAE is acceptable.
10. Most runs fail, diverge, or produce unstable validation curves.

If the results are clearly good, do not run adaptive stage. Proceed to final validation-based selection and test evaluation.

## Baselines required for judging mediocre results

Before declaring results mediocre, compute or verify at least these baselines on validation set:

### Baseline A. Global train mean

Predict the train-set mean FMS for all validation samples.

### Baseline B. Calibration-end anchor

Predict FMS[C_steps - 1] for every future target.

### Baseline C. Last available sparse anchor

For sparse_observed settings, predict the last observed anchor_fms.

### Baseline D. Recent-start observed anchor upper bound

For recent_start_observed settings only, predict recent_start FMS.

This is an upper-bound state baseline, not a deployment baseline.

### Baseline E. Linear or ridge baseline, if feasible

Use a small feature vector:

- calibration mean FMS
- calibration final FMS
- calibration FMS slope
- recent motion mean/std
- static features if enabled
- horizon_seconds

Do not tune this heavily.

## Required diagnostic report

Create:

runs/lc_sa_tcnformer_full_search/adaptive_diagnosis.md

The report must include:

1. Best validation runs.
2. Naive baseline comparison.
3. Whether predictions collapsed toward the mean.
4. Target variance vs prediction variance.
5. Calibration-length effect.
6. Recent-window effect.
7. Horizon degradation.
8. Anchor-mode effect.
9. Static-feature effect.
10. Training curve diagnosis:
   - underfitting
   - overfitting
   - unstable training
   - data leakage suspicion
   - target shift suspicion
11. Recommended adaptive experiments.
12. Budget to be used.

## Failure mode diagnosis

Classify the results into one or more of the following failure modes.

### Failure mode 1. Mean collapse / over-smoothing

Symptoms:

- prediction variance << target variance
- trajectories look flat
- validation MAE slightly better than mean baseline but R2 low
- high-FMS recall poor

Likely causes:

- model learns global content/time prior
- level-only regression encourages average prediction
- insufficient state information
- no delta/trend supervision

Recommended next experiments:

1. Use delta prediction from anchor.
2. Add trend auxiliary loss.
3. Predict multi-horizon outputs jointly.
4. Add quantile/Huber robustness if outliers dominate.
5. Add subject-wise calibration summary features.

### Failure mode 2. Anchor dependence

Symptoms:

- calibration_end or sparse_observed performs much better than no_anchor
- recent_start_observed upper bound is much better than all realistic settings
- no_anchor cannot track FMS level

Likely causes:

- head motion alone insufficient to infer current sickness state
- FMS state inertia dominates future FMS
- calibration embedding not enough to infer state drift

Recommended next experiments:

1. Sparse anchor interval sweep:
   - 10s
   - 30s
   - 60s
   - 90s
   - 120s
2. predict_delta_from_anchor=true.
3. Add time_since_anchor more explicitly.
4. Add anchor dropout during training:
   - sometimes provide anchor
   - sometimes mask anchor
5. Report realistic deployment requirement:
   - user self-report frequency needed.

### Failure mode 3. Calibration not useful

Symptoms:

- C=30, 60, 90, 120 show no common-window improvement
- attention pooling does not help
- calibration branch embeddings look unused
- static or anchor dominates

Likely causes:

- calibration encoder not extracting useful response pattern
- calibration too long and noisy
- FMS trajectory in calibration not aligned with later response
- model ignores calibration branch

Recommended next experiments:

1. Add calibration summary auxiliary features:
   - initial FMS
   - final FMS
   - mean FMS
   - FMS slope
   - max FMS
   - calibration delta
2. Compare calibration encoder variants:
   - TCN only
   - Transformer only
   - summary MLP only
   - TCN-Transformer
3. Use attention pooling instead of mean pooling.
4. Add branch dropout to prevent recent/anchor branches from dominating.
5. Add calibration contrast:
   - use calibration embedding to predict future trajectory class.

### Failure mode 4. Recent motion not useful

Symptoms:

- W=10, 30, 60 produce similar validation metrics
- recent TCN receptive field changes do not matter
- removing recent branch barely hurts

Likely causes:

- motion profile is similar across participants
- individual FMS differences dominate motion-trigger effects
- recent motion features are insufficiently expressive
- TCN receptive field not matched to window

Recommended next experiments:

1. Add hand-crafted recent motion statistics:
   - mean
   - std
   - min/max
   - jerk
   - acceleration magnitude
   - angular velocity magnitude
   - FFT power if easy
2. Compare recent encoder variants:
   - TCN
   - GRU/LSTM
   - TCN + attention pooling
   - temporal statistics MLP
3. Try W=5s, 15s, 30s, 60s.
4. Try kernel_size=5 only for recent TCN.
5. Try d_model=128 if not overfitting.

### Failure mode 5. Long horizon impossible

Symptoms:

- H=1s or 2.5s good
- H=10s+ degrades sharply
- H=30s very poor

Likely causes:

- usable warning horizon is short
- future FMS depends on unobserved future motion or subjective changes
- recursive state drift is hard

Recommended next experiments:

1. Multi-horizon model with horizon_set:
   - [1, 2.5, 5, 10, 15, 30]
2. Future trajectory prediction:
   - predict several future FMS points, not one point
3. Report horizon limit honestly:
   - identify usable horizon where MAE degradation remains acceptable
4. Use horizon embedding.
5. Use curriculum:
   - train short horizons first, then longer horizons.

### Failure mode 6. Overfitting

Symptoms:

- train loss decreases
- val loss increases
- larger models worse
- static models overfit

Recommended next experiments:

1. Increase dropout.
2. Reduce d_model.
3. Weight decay sweep.
4. Early stopping patience reduction.
5. Simpler model:
   - summary features + MLP
   - TCN only
6. Remove attention pooling if unstable.

### Failure mode 7. Underfitting

Symptoms:

- train loss and val loss both high
- model predictions flat
- larger model improves validation
- training stops without meaningful learning

Recommended next experiments:

1. Increase d_model to 128.
2. Add transformer_layers=2.
3. Increase epochs.
4. Lower learning rate if unstable.
5. Try MSE vs SmoothL1 if SmoothL1 under-penalizes large errors.
6. Add calibration summary features.

### Failure mode 8. Possible data/windowing bug

Symptoms:

- all models perform extremely poorly
- naive anchor baseline beats neural models by a lot
- metrics inconsistent across scripts
- target shift looks wrong in plots
- predictions/timestamps misaligned

Required action:

Stop adaptive model search.

Run leakage/windowing sanity tests again.

Manually inspect at least 3 session prediction CSVs and verify:

- current_time
- target_time
- target_index
- anchor_index
- recent window bounds
- calibration bounds
- target FMS alignment

Do not continue training until fixed.

## Adaptive experiment budget

Adaptive stage budget is limited.

Default:

- Maximum adaptive runs: 8
- Max epochs per adaptive run: 60
- Patience: 8
- Seeds: [42]
- Use validation metrics only.

If GPU/time budget is low:

- Maximum adaptive runs: 4
- Max epochs: 40
- Patience: 6

Do not exceed 8 adaptive runs unless the user explicitly approves.

## Adaptive experiment decision tree

After diagnosis, choose experiments using this priority order.

### Priority 1. Check baselines and bugs

Always run or verify naive baselines first.

If target/windowing bug is suspected, fix bug before any new model experiment.

### Priority 2. If mean collapse

Run:

1. best_config + predict_delta_from_anchor=true
2. best_config + trend auxiliary loss, lambda=0.1
3. best_config + multi_horizon, horizon_set=[1,2.5,5,10,15,30]
4. best_config + calibration summary features

### Priority 3. If anchor dependence

Run:

1. sparse_observed interval 30s
2. sparse_observed interval 60s
3. sparse_observed interval 90s
4. anchor dropout if implemented
5. predict_delta_from_anchor=true

### Priority 4. If calibration not useful

Run:

1. calibration summary MLP only
2. TCN-only calibration encoder
3. attention pooling calibration encoder
4. branch dropout

### Priority 5. If recent motion not useful

Run:

1. recent temporal statistics branch
2. recent TCN kernel_size=5
3. recent W=15s
4. recent W=60s with matching dilation
5. recent encoder ablation off/on

### Priority 6. If long horizon poor

Run:

1. multi-horizon model
2. short-horizon focused models:
   - H=1
   - H=2.5
   - H=5
3. report usable horizon threshold.

### Priority 7. If overfitting

Run:

1. smaller d_model=32 or 64
2. dropout=0.3
3. weight_decay=1e-3
4. simpler TCN-only model

### Priority 8. If underfitting

Run:

1. d_model=128
2. transformer_layers=2
3. learning_rate lower/higher check
4. MSE loss

## Adaptive model additions allowed

The following additions are allowed during adaptive stage if needed:

### Calibration summary feature branch

Add features:

- calibration_start_fms
- calibration_end_fms
- calibration_mean_fms
- calibration_std_fms
- calibration_max_fms
- calibration_delta_fms
- calibration_slope_fms

Encode with MLP and concatenate to fusion head.

### Recent temporal statistics branch

Add features over recent_seq:

- mean per motion channel
- std per motion channel
- min per motion channel
- max per motion channel
- motion magnitude mean/std
- angular velocity magnitude mean/std
- jerk magnitude mean/std if easy

Encode with MLP and concatenate to fusion head.

### Branch dropout

During training, randomly mask one or more branches:

- calibration branch
- recent branch
- anchor branch
- static branch

Use only if implemented safely.

Default branch dropout probability:

- 0.1

### Anchor dropout

During training, randomly replace anchor with none/calibration_end to reduce over-dependence.

Default:

- p=0.2

### Trend auxiliary loss

Use only as auxiliary.

Trend target:

- increase if target_fms - anchor_fms > 0.5
- stable if abs(target_fms - anchor_fms) <= 0.5
- decrease if target_fms - anchor_fms < -0.5

Loss:

total_loss = level_loss + lambda_trend * trend_loss

Default:

lambda_trend = 0.1

### Multi-horizon output

Output FMS for:

[1, 2.5, 5, 10, 15, 30]

Use mean SmoothL1 loss across horizons.

## Adaptive selection rule

After adaptive runs, select final configuration by validation MAE.

Tie-breakers:

1. validation RMSE
2. common-window validation MAE
3. high-FMS recall if clinically/safety relevant
4. simpler model
5. deployment-realistic anchor mode over upper-bound anchor mode

Never select recent_start_observed as final deployment model unless the report clearly labels it as upper-bound only.

If the best validation model is recent_start_observed, also select the best deployment-realistic model separately.

## Final test evaluation rule

Only after final validation-based selection:

1. Evaluate selected best deployment-realistic model on test set.
2. Optionally evaluate best upper-bound model on test set, clearly labeled upper-bound.
3. Do not change model choice after seeing test results.

## Required adaptive report

Create:

runs/lc_sa_tcnformer_full_search/adaptive_report.md

Include:

1. Why adaptive stage was triggered.
2. Baseline comparison.
3. Failure mode classification.
4. Adaptive experiments chosen and why.
5. Adaptive budget used.
6. Validation results of adaptive experiments.
7. Whether adaptive stage improved over original best validation result.
8. Final selected deployment-realistic model.
9. Optional upper-bound model.
10. Final test metrics after selection.
11. Remaining limitations.
12. Recommended future work if results are still mediocre.

## If results remain mediocre

If adaptive stage does not produce meaningful validation improvement:

Do not keep running more experiments.

Write a clear limitation-focused conclusion:

- head/motion-only may be insufficient for this forecasting horizon
- sparse or anchor-based FMS input may be required
- static features may not be enough to replace multimodal eye/physio signals
- usable horizon may be shorter than expected
- external transfer or multimodal dataset such as PRECYSE may be the next necessary step
- future work should compare against multimodal settings and/or use more explicit state-space modeling

Recommend one next research direction, not many.

Possible final recommendations:

1. Move to sparse-anchor deployment setting.
2. Shorten horizon to the empirically usable range.
3. Use multi-horizon trajectory prediction.
4. Transfer to PRECYSE for multimodal comparison.
5. Add eye/physio modality if available.