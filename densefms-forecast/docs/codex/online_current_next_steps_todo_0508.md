# Online Current FMS Next-Step TODO - 2026-05-08

This note records the next modeling ideas after the DeepTCN latent-GRU
state decoder experiments.

Current reference model:

- `deep_tcn_latent_gru_420_large_calib240_state_decoder_static4_no_fds_no_lds_seed42`
- Calibration: 120 seconds / 240 steps
- Online stream: DeepTCN dilations `[1, 2, 4, 8, 16]`
- State update: latent GRU initialized from calibration state
- Decoder input: `state_t[192] + static[4] = 196`
- Heads: direct regression + 0..20 cumulative ordinal
- FDS/LDS result: little trajectory-shape improvement; no-FDS/no-LDS looked cleaner on test.

## Current Failure Pattern

The main remaining problem does not look like label-density imbalance.

Observed issues:

- Recovery / low-FMS transition is weak.
- When target FMS drops sharply, predictions often stay too high.
- Predictions are smoother than the step-like DenseFMS labels.
- Some outputs look like a plausible continuous discomfort curve, but timing does not match the current FMS label.
- Subject/session bias remains visible, especially low-FMS overprediction and unstable high-FMS behavior.

Working interpretation:

- LDS/FDS addressed label imbalance, but the bottleneck is more likely temporal dynamics, recovery behavior, and subject/session-specific state transition.

## Priority 1 - Current + Future Multi-Horizon Joint Prediction

Idea:

```text
state_t -> current FMS
state_t -> FMS t+5s
state_t -> FMS t+10s
state_t -> FMS t+15s
```

Why it fits this project:

- Current-only prediction may allow the latent state to learn a short-cut for level.
- Multi-horizon prediction should pressure `state_t` to encode trajectory direction and near-future evolution.
- This is also the most natural bridge from current tracking to future FMS forecasting.

Literature basis:

- Ben Taieb et al. (2010), multiple-output multi-step forecasting.
  - Supports predicting multiple horizons jointly instead of treating each horizon independently.
- Lim et al. (2021), Temporal Fusion Transformers.
  - Established multi-horizon forecasting framing using static covariates and temporal dynamics.

Suggested first implementation:

- Add optional multi-horizon current tracker heads for `[0s, 5s, 10s, 15s]`.
- Keep current FMS as the primary validation metric.
- Report per-horizon MAE and shape metrics separately.
- Do not use test set for choosing horizon weights.

## Priority 2 - Event / Delta Auxiliary Loss

Idea:

```text
main: FMS_t regression
aux1: delta_5s = FMS[t+5s] - FMS[t]
aux2: delta_10s = FMS[t+10s] - FMS[t]
aux3: rise / drop / plateau classification
```

Important distinction:

- This is not an anchor/delta decoder.
- Delta targets are training-only auxiliary supervision.
- Inference still uses motion/state/static only.

Why it fits this project:

- The plot failure is mainly rise/drop timing and recovery, not only absolute level.
- A delta/event auxiliary target directly supervises the behavior that LDS/FDS did not change.

Literature basis:

- Caruana (1997), Multitask Learning.
  - Related auxiliary tasks can improve shared representations and generalization.

Suggested first implementation:

- Start with `delta_5s`, `delta_10s`, and 3-class event label: rise/drop/plateau.
- Use small auxiliary weights first so current MAE is not overwhelmed.
- Track `delta_corr_5s`, `direction_acc_5s`, and low-FMS recovery errors.

## Priority 3 - Causal Temporal Decoder Head

Idea:

```text
state_t sequence [B, P, 192]
+ static
-> shallow causal decoder TCN
-> regression / ordinal / risk heads
```

Why it fits this project:

- Current decoder reads each `state_t + static` with mostly stepwise MLP heads.
- A shallow causal temporal decoder can model output-side temporal patterns without future leakage.
- This may help trajectory shape while preserving causal deployment constraints.

Literature basis:

- Bai, Kolter, and Koltun (2018), TCN.
  - TCNs are a strong sequence-modeling reference with causal convolutions and long effective memory.

Suggested first implementation:

- Add optional decoder TCN with 2-4 causal blocks.
- Keep receptive field small enough to avoid simply over-smoothing.
- Compare against the no-FDS/no-LDS static4 baseline on both MAE and shape metrics.

## Priority 4 - Calibration-Conditioned State Transition

Idea:

```text
z_calib -> FiLM / gate bias / adapter
DeepTCN motion latent + previous state -> state_t
```

Why it fits this project:

- Re-concatenating `z_calib` into the decoder every step felt less natural.
- Conditioning the state transition itself is more consistent with the idea that calibration defines subject/session response dynamics.
- It may help different users map similar motion patterns to different FMS trajectories.

Literature basis:

- Perez et al. (2018), FiLM.
  - Conditioning information can modulate network computation through feature-wise affine transformations.

Suggested first implementation:

- Start with a simple FiLM/adaptor on DeepTCN latent features before the latent GRU.
- Avoid adding `z_calib` directly back into the final decoder feature.
- Compare with the current `z_calib -> h0 only` state initializer.

## Suggested Experiment Order

1. `no_fds_no_lds_static4 + multi_horizon`
2. `no_fds_no_lds_static4 + delta/event auxiliary`
3. Combine multi-horizon + delta/event auxiliary if both are stable.
4. Add shallow causal decoder TCN only if shape metrics remain weak.
5. Try calibration-conditioned state transition after the above baselines are clear.

## Reporting Checklist For Next Runs

- Validation-based model selection only.
- Test set final-report-only.
- Report final FMS, regression head, and ordinal head separately.
- Report MAE/RMSE plus shape metrics:
  - session correlation
  - centered session MAE
  - `delta_corr_5s`
  - `direction_acc_5s`
  - low-FMS recovery errors
- Include representative success and failure plots.
- Explicitly note that LDS/FDS were tried and did not materially improve trajectory shape.
