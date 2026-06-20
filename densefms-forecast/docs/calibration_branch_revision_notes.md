# Calibration Branch Revision Notes

## Context

Current baseline:

- model: `selected_deeptcn_risk035_static4`
- config: `configs/online_current/selected_deeptcn_risk035_static4.yaml`
- checkpoint: `runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/best.pt`

The current calibration branch is:

```text
head[:240]         [B,240,6]
calibration FMS    [B,240]

concat ->          [B,240,7]
DeepTCN 7->96 ->   [B,240,96]
mean pooling ->    [B,96]

summary stats ->   [B,20]
summary MLP ->     [B,96]

z_calib = mean_pool + summary_embedding
z_calib             [B,96]
init hidden state   Linear(96->192) -> [1,B,192]
```

`240` steps means `120s / 0.5s`.

## Current Concerns

1. To preserve calibration trajectory information, `DeepTCN output [B,240,96] -> mean pooling [B,96]` may be too aggressive.

Mean pooling is not obviously wrong because DeepTCN features already contain temporal context. With dilations `[1,2,4,8,16,32]`, kernel `3`, and two convs per TCN block:

```text
RF = 1 + 2 * (3 - 1) * (1+2+4+8+16+32)
   = 253 steps
   = 126.5 seconds
```

So late calibration features can see nearly the full 120s calibration history.

Still, the output remains a sequence:

```text
H = DeepTCN(calib_seq) = [B,240,96]
H[:, t, :] = t-specific representation with causal temporal context
```

Mean pooling removes the time axis:

```text
mean(H, dim=time) = [B,96]
```

This can blur distinctions such as:

- low early FMS then late rise
- high early FMS then recovery
- stable plateau
- sudden late rise near calibration end

2. To preserve the learned DeepTCN representation, adding summary embedding directly may be too blunt.

Current fusion:

```text
z_pool    [B,96]
z_summary [B,96]

z_calib = z_pool + z_summary
```

This assumes both vectors share the same latent semantics. But `z_pool` is learned temporal representation, while `z_summary` is projected hand-crafted statistics.

This can blur or overwrite the pooled representation.

3. To reduce bottlenecking, compressing to `[B,96]` and expanding to `[1,B,192]` may be unnecessarily narrow.

Current:

```text
z_calib [B,96]
Linear(96->192)
h0      [1,B,192]
```

This is a common pattern, but here calibration is the main person/session prior. A 96D bottleneck may be too tight, especially because decoder mode is `state`, so calibration is mostly injected through the stream initial state rather than directly concatenated into the decoder.

## Recommended First Revision

Use `mean + last + summary concat fusion`.

Rationale:

- `mean(H)` keeps overall calibration tendency / susceptibility.
- `last(H)` keeps calibration-end sickness state.
- `summary` keeps explicit FMS/motion statistics.
- concat fusion avoids forcing summary and pooled features into the same latent space before fusion.

Proposed shape:

```text
H = DeepTCN(calib_seq)          [B,240,96]

z_mean = mean(H, dim=1)         [B,96]
z_last = H[:, -1, :]            [B,96]

summary_raw                 [B,20]
z_summary = MLP(summary_raw)   [B,96]

concat = [z_mean, z_last, z_summary]
concat                         [B,288]

z_calib = MLP(concat)           [B,192]
h0 = Linear(192->192)           [1,B,192]
```

This is simpler and more explainable than fixed temporal pyramid pooling.

## Why Not Temporal Pyramid First

Temporal pyramid pooling would split calibration into fixed bins, for example:

```text
0-40s / 40-80s / 80-120s
```

This is not very natural for this task because important change points vary by session/person:

- some participants rise early
- some rise late
- some recover
- some stay flat

Fixed bins introduce an arbitrary assumption that those intervals are semantically meaningful. This may be harder to justify in a presentation.

## Attention Pooling Clarification

Attention pooling here means not a full Transformer stack, but a learned pooling over DeepTCN time outputs.

Basic attention pooling:

```text
H          [B,240,96]
score_t    = MLP(H_t)          -> [B,240,1]
alpha_t    = softmax(score_t)  -> [B,240,1]
z_attn     = sum(alpha_t * H_t)-> [B,96]
```

Mean pooling is the special case where every time step has weight `1/240`.

The project goal is not to learn a globally important timestamp. The goal is to detect which moments are important for each person/session.

Basic attention is input-dependent, so weights can differ by session. It may be enough, and should remain a valid experiment. A more explicitly change-aware variant is only an additional candidate, not a requirement.

## Attention Candidate Variants

Plain attention candidate:

```text
score_t = MLP(H_t)
alpha = softmax(score over calibration time)
z_attn = sum(alpha_t * H_t)
```

This keeps the model simple and leaves open the possibility that DeepTCN already encodes the relevant change information inside `H_t`.

Event-aware attention candidate:

```text
score_t = MLP([H_t, delta_H_t, FMS_t, delta_FMS_t, time_t])
alpha = softmax(score over calibration time)
z_event = sum(alpha_t * H_t)
```

This variant adds explicit change information to the attention scorer. It may help if plain attention does not reliably identify person/session-specific change moments, but it should be treated as one hypothesis rather than a fixed design constraint.

Suggested score input:

```text
H_t       [96]
delta_H_t [96]
FMS_t      [1]
delta_FMS  [1]
time_t     [1]
----------------
score input [195]
```

Proposed:

```text
delta_H[:, 0] = 0
delta_H[:, t] = H[:, t] - H[:, t-1]

delta_FMS[:, 0] = 0
delta_FMS[:, t] = FMS[:, t] - FMS[:, t-1]

score_t = MLP([H_t, delta_H_t, FMS_t, delta_FMS_t, time_t])
alpha = softmax(score over calibration time)
z_event = sum(alpha_t * H_t)
```

Final fusion candidate:

```text
z_mean    [B,96]
z_last    [B,96]
z_event   [B,96]
z_summary [B,96]

concat -> [B,384]
MLP ->    [B,192]
```

This may better match the target of identifying person/session-specific important calibration moments and change patterns, but plain attention remains worth testing because DeepTCN features may already contain enough temporal-change evidence.

## Suggested Experiment Order

1. Replace mean-only pooling with mean+last+summary concat fusion.

Expected change:

```text
z_calib: [B,96] -> [B,192]
summary addition -> concat fusion
init_state: Linear(192->192)
```

Keep the rest unchanged:

- same DeepTCN calibration encoder
- same stream DeepTCN latent GRU
- same `decoder_context_mode=state`
- same loss setup
- same validation-only selection

2. If step 1 helps or is neutral, test gated summary fusion.

Candidate:

```text
gate = sigmoid(MLP([z_mean, z_last, z_summary]))
z_calib = MLP([z_mean, z_last, gate * z_summary])
```

3. If step 1 is stable, test plain attention pooling first.

Candidate:

```text
z_calib = MLP([z_mean, z_last, z_attn, z_summary]) -> [B,192]
```

4. If plain attention is neutral or suggests partial benefit, test event-aware attention.

Candidate:

```text
z_calib = MLP([z_mean, z_last, z_event, z_summary]) -> [B,192]
```

5. Only after calibration representation improves, consider decoder direct conditioning.

Candidate:

```text
decoder input = [state_t, z_calib, static]
```

With `state_t [B,P,192]`, `z_calib [B,192]`, and `static [B,4]`:

```text
decoder input [B,P,388]
```

This should be tested separately because it changes how calibration affects every time step, not only the initial state.

## Current Preferred Next Step

The most defensible next experiment is:

```text
calibration pooling = mean + last
summary fusion = concat MLP
z_calib_dim = 192
decoder conditioning unchanged
```

Reason:

- directly addresses the mean-pooling bottleneck
- avoids arbitrary temporal pyramid bins
- avoids blunt summary addition
- keeps the change small enough to isolate the effect
