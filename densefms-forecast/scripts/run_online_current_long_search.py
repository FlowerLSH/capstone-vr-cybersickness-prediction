"""Run validation-only online current-FMS candidates sequentially.

The script is intentionally conservative: it always passes --no_test_eval and a
fixed --split_file, then refreshes validation analysis after each completed run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_CANDIDATES = [
    {
        "run_name": "current_fms_dual_auxrisk_static",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_static.yaml",
        "hypothesis": "static user/session features reduce subject-level current-FMS scale and bias error",
    },
    {
        "run_name": "current_fms_dual_auxrisk_recent30_stats",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_recent30_stats.yaml",
        "hypothesis": "30s causal motion statistics and temporal risk context improve long rise/drop tracking",
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_lowsmooth",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "MAE loss and lower smoothing reduce lag and amplitude flattening",
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale100_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "larger delta scale with weaker ordinal auxiliary improves amplitude without over-classifying levels",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_smooth005",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "a small delta-smoothness term may retain MAE gains while reducing noisy trajectory overshoot",
        "extra_args": [
            "--smoothness_weight",
            "0.005",
        ],
    },
    {
        "run_name": "current_fms_risknostats_mae_lowsmooth",
        "config": "configs/online_fms_current_tracker_risk_no_stats_mae_select.yaml",
        "hypothesis": "the strongest simple baseline may benefit from MAE loss and no smoothing more than the dual head",
        "extra_args": [
            "--loss_type",
            "mae",
            "--smoothness_weight",
            "0.0",
            "--current_reg_aux_weight",
            "0.3",
            "--ordinal_loss_weight",
            "0.5",
            "--learning_rate",
            "0.0008",
        ],
    },
    {
        "run_name": "current_fms_risknostats_mae_ord03",
        "config": "configs/online_fms_current_tracker_risk_no_stats_mae_select.yaml",
        "hypothesis": "weaker ordinal auxiliary on the simple risk baseline may improve raw level regression",
        "extra_args": [
            "--loss_type",
            "mae",
            "--smoothness_weight",
            "0.0",
            "--current_reg_aux_weight",
            "0.3",
            "--ordinal_loss_weight",
            "0.3",
            "--learning_rate",
            "0.0008",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_recent5",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "shorter causal recent context may reduce lag for current FMS tracking",
        "extra_args": [
            "--recent_window_seconds",
            "5.0",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_recent15",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "moderately longer causal recent context may improve trend shape without the 30s over-smoothing failure",
        "extra_args": [
            "--recent_window_seconds",
            "15.0",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "repeat the current best architecture with seed 7 to separate robust signal from seed variance",
        "extra_args": [
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_seed13",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "repeat the current best architecture with seed 13 to estimate validation stability",
        "extra_args": [
            "--seed",
            "13",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_seed21",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "repeat the current best architecture with seed 21 to find whether MAE gains are consistent",
        "extra_args": [
            "--seed",
            "21",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale090_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "test whether the best delta-scale gain peaks below 1.0 rather than at the current 1.0 setting",
        "extra_args": [
            "--current_delta_scale",
            "0.9",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale110_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "slightly larger delta scale may improve amplitude without the flatness seen in earlier baselines",
        "extra_args": [
            "--current_delta_scale",
            "1.1",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "a larger residual scale may reduce high-FMS under-amplitude if it does not overreact",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale130_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "scale120 became the best single model, so test whether the optimum continues above 1.2",
        "extra_args": [
            "--current_delta_scale",
            "1.3",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale140_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "a still larger residual scale tests the overreaction boundary after the scale120 gain",
        "extra_args": [
            "--current_delta_scale",
            "1.4",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_ord02",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "with scale120, weaker ordinal loss may keep the MAE gain while reducing bin-induced bias",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.2",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_ord04",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "with scale120, stronger ordinal supervision may stabilize absolute levels if the larger scale adds noise",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_combine035",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "scale120 may benefit from less ordinal blending to preserve continuous amplitude",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_combine055",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "scale120 with more ordinal blending may improve RMSE and level ordering",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.55",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale100_ord02",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "weaker ordinal auxiliary may let continuous MAE dominate scale calibration",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.2",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale100_ord04",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "slightly stronger ordinal auxiliary may stabilize level bins while preserving the MAE improvement",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_combine035_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "less ordinal blending may reduce high/low bin anchoring error in the best MAE family",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_combine055_ord03",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "more ordinal blending may improve robust level ordering without returning to the original smooth-l1 setup",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.55",
            "--ordinal_loss_weight",
            "0.3",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "weaker rapid-rise auxiliary may reduce conflict with current-FMS MAE while retaining useful movement signal",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risk010",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "risk025 improved MAE, so test an even weaker risk auxiliary before removing it entirely",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.1",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risk000",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "risk025 improved level accuracy, so verify whether the risk head should be disabled for current-FMS selection",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.0",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine the best residual scale family with the improved weak-risk auxiliary",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_ord04_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine the best ordinal-loss single model with the improved weak-risk auxiliary",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale120_ord04_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "test the best scale, stronger ordinal supervision, and weak risk auxiliary together",
        "extra_args": [
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risk075",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "stronger rapid-rise auxiliary may improve rise/drop tracking enough to help validation MAE",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.75",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_reg020",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "weaker direct-regression auxiliary may let the gated head fit residual dynamics more freely",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--current_reg_aux_weight",
            "0.2",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_reg050",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "stronger direct-regression auxiliary may stabilize absolute level while keeping the larger residual scale",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--current_reg_aux_weight",
            "0.5",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_lr0004_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "lower learning rate may reduce seed sensitivity and over-shooting around the best single model",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--learning_rate",
            "0.0004",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_lr0008_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "higher learning rate may find the good basin faster or escape the plateau seen in nearby runs",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--learning_rate",
            "0.0008",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_wd0003_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "stronger weight decay may improve validation stability for the best high-scale residual setup",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--weight_decay",
            "0.0003",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_longpatience_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "the best run selected epoch 40, so longer patience/epochs may capture a later validation improvement",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--epochs",
            "120",
            "--patience",
            "16",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_no_feedback_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "removing predicted-current feedback may reduce self-reinforcing level bias in online tracking",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--rollout_mode",
            "none",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_motiontcn_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "a shallow causal motion TCN stem may add useful short-term dynamics without the heavy-model overfit pattern",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--motion_encoder_context",
            "tcn",
            "--motion_encoder_layers",
            "1",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risktcn_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "temporal context on risk/current diagnostics may improve movement tracking without enabling motion stats",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_temporal_context",
            "tcn",
            "--risk_temporal_layers",
            "1",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_stats10_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "causal motion statistics may help if used with the best loss and 10s window instead of the failed 30s setup",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--motion_stats_branch",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_recent20_scale100",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "20s causal recent context may capture slower rises while avoiding the 30s stats over-smoothing failure",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--recent_window_seconds",
            "20.0",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "after enabling online loss_type, train the best weak-risk family with true MAE rather than SmoothL1",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_lr0008_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true MAE may prefer the faster learning-rate basin that previously added useful ensemble diversity",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--learning_rate",
            "0.0008",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_scale090_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true MAE with the high-correlation scale090 family may improve trend shape and ensemble complementarity",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "0.9",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_scale120_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true MAE may combine the best weak-risk auxiliary with the stronger residual-amplitude scale family",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.2",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_ord04_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true MAE with slightly stronger ordinal supervision tests whether level bins stabilize absolute scale",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "static subject features have ensemble value, so train a true-MAE static member instead of relying on the older smooth baseline",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_scale090_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine static calibration features with the high-correlation scale090 weak-risk setting",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "0.9",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_combine035_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "full ensemble kept the older combine035 member, so train a true-MAE weak-risk counterpart",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_combine055_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "test the opposite ordinal-blend side of true-MAE weak-risk ensemble diversity",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.55",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_ord04_combine035_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine true-MAE ord04's single-model gain with the ensemble-useful lower ordinal blend",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_ord04_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "full ensemble gave true-MAE static high weight; add stronger ordinal supervision to test static calibration diversity",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "test whether the ensemble-useful true-MAE static member improves with lower ordinal blending",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_lr0004_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "lower learning rate may make the ensemble-useful true-MAE static member less noisy",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--learning_rate",
            "0.0004",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_lr0008_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "higher learning rate tests whether true-MAE static's ensemble value comes from a sharper basin",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--learning_rate",
            "0.0008",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_dropout020_risk025",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "stronger static dropout may keep static diversity while reducing validation overfit",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.2",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale090_ord03_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "ext2 ensemble gave high weight to scale090; seed 7 tests whether its useful bias is stable or diversifies the ensemble",
        "extra_args": [
            "--current_delta_scale",
            "0.9",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_scale090_ord03_seed13",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "scale090 received substantial ensemble weight; seed 13 checks whether another basin adds validation diversity",
        "extra_args": [
            "--current_delta_scale",
            "0.9",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--seed",
            "13",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_lr0008_scale100_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "ext2 ensemble gave high weight to lr0008; seed 7 tests a second high-learning-rate basin",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--learning_rate",
            "0.0008",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_lr0008_scale100_seed13",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "repeat the ensemble-useful lr0008 family with seed 13 to test validation stability and complementarity",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--learning_rate",
            "0.0008",
            "--seed",
            "13",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_combine035_ord03_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine035 remained in the exact ensemble; seed 7 tests whether lower ordinal blending has reusable diversity",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_mae_risk025_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "weak-risk MAE retained a small exact-ensemble weight, so seed 7 tests whether its rise-sensitive bias is seed-specific",
        "extra_args": [
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.45",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_ord04_combine035_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true-MAE ord04 combine035 newly entered ext2, so seed 7 tests whether it adds independent validation-error corrections",
        "extra_args": [
            "--loss_type",
            "mae",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.4",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "true-MAE static combine035 received the largest ext2 weight; seed 7 checks whether static diversity improves the ensemble further",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_seed13",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "repeat the largest-weight true-MAE static combine035 family with seed 13 to test ensemble complementarity",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "13",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_seed21",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "static combine035 seed7 produced the ext3 gain, so seed 21 checks whether the improved static basin repeats",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "21",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_seed31",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "another static combine035 seed tests whether the largest ext3 member is a robust family or a single lucky seed",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "31",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine025_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "seed7 static combine035 improved strongly; lower ordinal blending to 0.25 may preserve continuous level scale further",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.1",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.25",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "7",
        ],
    },
    {
        "run_name": "current_fms_dual_auxrisk_true_mae_static_combine035_dropout020_seed7",
        "config": "configs/online_fms_current_tracker_dual_auxrisk_mae_lowsmooth.yaml",
        "hypothesis": "combine the strong static seed7 basin with higher static dropout to test whether regularized static diversity improves ext3",
        "extra_args": [
            "--loss_type",
            "mae",
            "--use_static",
            "--static_dropout",
            "0.2",
            "--current_delta_scale",
            "1.0",
            "--fms_combine_weight_ordinal",
            "0.35",
            "--ordinal_loss_weight",
            "0.3",
            "--risk_loss_weight",
            "0.25",
            "--seed",
            "7",
        ],
    },
]


REFERENCE_RUNS = [
    ("ensemble_weighted_with_lr_linprog", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_with_lr_linprog"),
    ("ensemble_weighted_with_lr", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_with_lr"),
    ("ensemble_weighted_with_risk025", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_with_risk025"),
    ("ensemble_weighted_with_ord04", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_with_ord04"),
    ("ensemble_weighted_scale_sweep", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_scale_sweep"),
    ("ensemble_weighted_with_scale090", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_with_scale090"),
    ("ensemble_weighted_top", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_weighted_top"),
    ("ensemble_scale_static_heavy_risk", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_scale_static_heavy_risk"),
    ("ensemble_scale_static_risk", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_scale_static_risk"),
    ("ensemble_scale_static", "runs/online_fms_current_tracking_0507/current_fms_ensemble_val_scale_static"),
    ("risk_no_stats_ref", "runs/online_fms_risk_tracking_0507/online_risk_no_stats"),
    ("base_basic", "runs/online_fms_current_tracking_0507/current_fms_base_basic"),
    ("base_v1weights", "runs/online_fms_current_tracking_0507/current_fms_base_v1weights"),
    ("dual_medium", "runs/online_fms_current_tracking_0507/current_fms_dual_medium"),
    ("dual_auxrisk", "runs/online_fms_current_tracking_0507/current_fms_dual_auxrisk"),
    ("encoder_heavy", "runs/online_fms_current_tracking_0507/current_fms_encoder_heavy"),
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_command(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return int(proc.wait())


def _read_leaderboard(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _completed(run_dir: Path) -> bool:
    return (run_dir / "best.pt").exists() and (run_dir / "metrics.json").exists() and (run_dir / "val_predictions.csv").exists()


def _write_checkpoint(
    out_dir: Path,
    completed: List[Dict[str, str]],
    failed: List[Dict[str, str]],
    leaderboard_path: Path,
    split_file: str,
) -> None:
    rows = _read_leaderboard(leaderboard_path)
    lines = [
        f"# Online current-FMS checkpoint - {_now()}",
        "",
        "## Split",
        "",
        f"- fixed split file: `{split_file}`",
        "- test evaluation: not run in this script",
        "",
        "## Completed candidates",
        "",
    ]
    if completed:
        for item in completed:
            lines.append(f"- `{item['run_name']}`: {item['hypothesis']}")
    else:
        lines.append("- none in this invocation")
    lines.extend(["", "## Failed candidates", ""])
    if failed:
        for item in failed:
            lines.append(f"- `{item['run_name']}` exit_code={item['exit_code']}: {item['hypothesis']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Current validation leaderboard", ""])
    if rows:
        columns = [
            "label",
            "mae",
            "rmse",
            "pearson_session_mean",
            "centered_mae_session_mean",
            "delta_corr_5s",
            "direction_acc_5s",
            "flat_range_lt25pct_session_rate",
        ]
        available = [col for col in columns if col in rows[0]]
        lines.append("| " + " | ".join(available) + " |")
        lines.append("| " + " | ".join(["---"] * len(available)) + " |")
        for row in rows[:12]:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in available) + " |")
    else:
        lines.append("- leaderboard not available")
    lines.append("")
    (out_dir / "checkpoint_latest_ko.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online current-FMS validation-only long search.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--split_file", default="runs/online_fms_risk_tracking_0507/online_risk_no_stats/split.json")
    parser.add_argument("--runs_dir", default="runs/online_fms_current_tracking_0507")
    parser.add_argument("--analysis_dir", default="runs/online_fms_current_tracking_0507/analysis_long_search")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true", help="Print pending training commands without running them.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, str]] = []
    failed: List[Dict[str, str]] = []
    dry_run_commands: List[Sequence[str]] = []

    for spec in DEFAULT_CANDIDATES:
        run_name = spec["run_name"]
        run_dir = runs_dir / run_name
        if args.skip_existing and _completed(run_dir):
            completed.append(spec)
            continue
        else:
            cmd = [
                args.python,
                "-m",
                "src.densefms_forecast.train",
                "--data_dir",
                args.data_dir,
                "--config",
                spec["config"],
                "--model",
                "online_fms_risk_tracker",
                "--run_name",
                run_name,
                "--no_test_eval",
                "--split_file",
                args.split_file,
                "--skip_existing",
            ]
            cmd.extend([str(value) for value in spec.get("extra_args", [])])
            if args.dry_run:
                dry_run_commands.append(cmd)
                print("$ " + " ".join(cmd))
                continue
            code = _run_command(cmd, analysis_dir / "logs" / f"{run_name}.log")
            if code == 0 and _completed(run_dir):
                completed.append(spec)
            else:
                failed.append({**spec, "exit_code": str(code)})
                _write_checkpoint(
                    analysis_dir,
                    completed,
                    failed,
                    analysis_dir / "online_current_validation_leaderboard.csv",
                    args.split_file,
                )
                break

        if args.dry_run:
            continue
        run_pairs = REFERENCE_RUNS + [(item["run_name"].replace("current_fms_", ""), str(runs_dir / item["run_name"])) for item in completed]
        cmd = [
            args.python,
            "scripts/analyze_online_current_tracking.py",
            "--run_dirs",
            *[path for _, path in run_pairs],
            "--labels",
            *[label for label, _ in run_pairs],
            "--out_dir",
            str(analysis_dir),
            "--split",
            "val",
            "--primary_label",
            run_name.replace("current_fms_", ""),
            "--trajectory_count",
            "6",
        ]
        code = _run_command(cmd, analysis_dir / "logs" / f"analyze_after_{run_name}.log")
        if code != 0:
            failed.append({**spec, "exit_code": f"analysis:{code}"})
            break
        _write_checkpoint(
            analysis_dir,
            completed,
            failed,
            analysis_dir / "online_current_validation_leaderboard.csv",
            args.split_file,
        )

    if args.dry_run:
        dry_run_path = analysis_dir / "dry_run_commands.txt"
        dry_run_path.write_text("\n".join(" ".join(cmd) for cmd in dry_run_commands) + "\n", encoding="utf-8")
        print(f"Dry-run pending commands: {len(dry_run_commands)}")
        print(f"Dry-run command file: {dry_run_path}")
        return

    print(f"Completed {len(completed)} candidates, failed {len(failed)} candidates.")
    print(f"Latest checkpoint: {analysis_dir / 'checkpoint_latest_ko.md'}")


if __name__ == "__main__":
    main()
