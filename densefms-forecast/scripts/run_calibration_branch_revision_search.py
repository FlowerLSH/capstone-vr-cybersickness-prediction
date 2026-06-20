"""Run calibration-branch revision candidates and append a live report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "configs/online_current/selected_deeptcn_risk035_static4.yaml"
BASE_SPLIT = "runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/split.json"
BASELINE_VAL_MAE = 1.7402
REPORT = Path("reports/calibration_branch_revision_experiments_0513.md")
RUNS_DIR = "runs/calibration_branch_revision_0513"


def candidates() -> List[Dict[str, Any]]:
    common_concat = [
        "--calibration_fusion_mode",
        "mean_last_summary_concat",
        "--calibration_fusion_output_dim",
        "192",
    ]
    best_base = [
        "--fms_combine_weight_ordinal",
        "0.10",
        "--ordinal_loss_weight",
        "0.05",
        "--weight_decay",
        "0.0001",
    ]
    return [
        {
            "name": "cbr_gated_summary_h288_o192_seed42",
            "notes": "Gated summary fusion; tests whether explicit summary should be attenuated before concat.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_gated_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "288",
            ],
        },
        {
            "name": "cbr_event_attn_h384_o192_seed42",
            "notes": "Event-aware attention scorer using H, delta_H, FMS, delta_FMS, and time.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_event_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
            ],
        },
        {
            "name": "cbr_concat_h192_o192_seed42",
            "notes": "Smaller fusion MLP hidden width.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "192"],
        },
        {
            "name": "cbr_concat_h384_o192_seed42",
            "notes": "Wider fusion MLP hidden width.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "384"],
        },
        {
            "name": "cbr_concat_h256_o128_seed42",
            "notes": "Reduced calibration representation dimension to test whether 192D overfits.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_summary_concat",
                "--calibration_fusion_output_dim",
                "128",
                "--calibration_fusion_hidden_dim",
                "256",
            ],
        },
        {
            "name": "cbr_concat_h384_o256_seed42",
            "notes": "Expanded calibration representation dimension.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_summary_concat",
                "--calibration_fusion_output_dim",
                "256",
                "--calibration_fusion_hidden_dim",
                "384",
            ],
        },
        {
            "name": "cbr_concat_h288_o192_lr3e4_seed42",
            "notes": "Lower learning rate for the larger calibration representation.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--learning_rate", "0.0003"],
        },
        {
            "name": "cbr_concat_h288_o192_lr6e4_seed42",
            "notes": "Higher learning rate for faster adaptation.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--learning_rate", "0.0006"],
        },
        {
            "name": "cbr_concat_h288_o192_drop005_seed42",
            "notes": "Lower dropout in calibration/stream model.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--dropout", "0.05"],
        },
        {
            "name": "cbr_concat_h288_o192_drop015_seed42",
            "notes": "Higher dropout regularization.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--dropout", "0.15"],
        },
        {
            "name": "cbr_concat_h288_o192_no_fds_seed42",
            "notes": "Disable FDS to see if feature smoothing conflicts with new calibration embedding.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--no_fds_enabled"],
        },
        {
            "name": "cbr_concat_h288_o192_fds05_seed42",
            "notes": "Weaker FDS blend.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--fds_blend", "0.5"],
        },
        {
            "name": "cbr_concat_h288_o192_lds03_seed42",
            "notes": "Weaker LDS target reweighting.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--lds_gamma", "0.3"],
        },
        {
            "name": "cbr_concat_h288_o192_lds08_seed42",
            "notes": "Stronger LDS target reweighting.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--lds_gamma", "0.8"],
        },
        {
            "name": "cbr_concat_h288_o192_ord010_seed42",
            "notes": "Lower ordinal blend and ordinal auxiliary pressure.",
            "args": [
                *common_concat,
                "--calibration_fusion_hidden_dim",
                "288",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
            ],
        },
        {
            "name": "cbr_concat_h288_o192_ord025_seed42",
            "notes": "Higher ordinal blend and auxiliary pressure.",
            "args": [
                *common_concat,
                "--calibration_fusion_hidden_dim",
                "288",
                "--fms_combine_weight_ordinal",
                "0.25",
                "--ordinal_loss_weight",
                "0.15",
            ],
        },
        {
            "name": "cbr_concat_h288_o192_risk020_seed42",
            "notes": "Lower rapid-rise auxiliary weight to focus MAE.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--risk_loss_weight", "0.20"],
        },
        {
            "name": "cbr_concat_h288_o192_risk050_seed42",
            "notes": "Higher rapid-rise auxiliary weight to regularize dynamics.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--risk_loss_weight", "0.50"],
        },
        {
            "name": "cbr_attn_h384_o192_lr3e4_drop005_seed42",
            "notes": "Plain attention fusion with lower LR/dropout.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
                "--learning_rate",
                "0.0003",
                "--dropout",
                "0.05",
            ],
        },
        {
            "name": "cbr_event_h384_o192_lr3e4_drop005_seed42",
            "notes": "Event-aware attention with lower LR/dropout.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_event_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
                "--learning_rate",
                "0.0003",
                "--dropout",
                "0.05",
            ],
        },
        {
            "name": "cbr_deeptcn_transformer_concat_l1_seed42",
            "notes": "Add one calibration Transformer layer after DeepTCN with concat fusion.",
            "args": [
                *common_concat,
                "--calibration_fusion_hidden_dim",
                "288",
                "--calibration_encoder_mode",
                "deep_tcn_transformer",
                "--transformer_layers",
                "1",
            ],
        },
        {
            "name": "cbr_deeptcn_transformer_attn_l1_seed42",
            "notes": "DeepTCN + one Transformer layer + plain attention fusion.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
                "--calibration_encoder_mode",
                "deep_tcn_transformer",
                "--transformer_layers",
                "1",
            ],
        },
        {
            "name": "cbr_baseline_add_attention_pool_seed42",
            "notes": "Original additive summary fusion but attention pooling over DeepTCN sequence.",
            "args": ["--pooling", "attention"],
        },
        {
            "name": "cbr_baseline_add_last_pool_seed42",
            "notes": "Original additive summary fusion but last pooling.",
            "args": ["--pooling", "last"],
        },
        {
            "name": "cbr_concat_h288_o192_seed7",
            "notes": "Seed robustness check for the primary concat candidate.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--seed", "7"],
        },
        {
            "name": "cbr_concat_h288_o192_seed123",
            "notes": "Second seed robustness check for the primary concat candidate.",
            "args": [*common_concat, "--calibration_fusion_hidden_dim", "288", "--seed", "123"],
        },
        {
            "name": "cbr_baseline_add_mean_seed7",
            "notes": "Seed robustness check for the original add/mean calibration branch.",
            "args": ["--seed", "7"],
        },
        {
            "name": "cbr_baseline_add_mean_seed123",
            "notes": "Second seed robustness check for the original add/mean calibration branch.",
            "args": ["--seed", "123"],
        },
        {
            "name": "cbr_baseline_add_mean_lr3e4_seed42",
            "notes": "Original add/mean branch with lower learning rate.",
            "args": ["--learning_rate", "0.0003"],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_seed42",
            "notes": "Original add/mean branch with higher learning rate.",
            "args": ["--learning_rate", "0.0006"],
        },
        {
            "name": "cbr_baseline_add_mean_drop005_seed42",
            "notes": "Original add/mean branch with lower dropout.",
            "args": ["--dropout", "0.05"],
        },
        {
            "name": "cbr_baseline_add_mean_drop015_seed42",
            "notes": "Original add/mean branch with higher dropout.",
            "args": ["--dropout", "0.15"],
        },
        {
            "name": "cbr_baseline_add_mean_risk020_seed42",
            "notes": "Original add/mean branch with weaker risk auxiliary loss.",
            "args": ["--risk_loss_weight", "0.20"],
        },
        {
            "name": "cbr_baseline_add_mean_risk050_seed42",
            "notes": "Original add/mean branch with stronger risk auxiliary loss.",
            "args": ["--risk_loss_weight", "0.50"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_seed42",
            "notes": "Original add/mean branch with weaker ordinal blend and auxiliary loss.",
            "args": [
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_ord025_seed42",
            "notes": "Original add/mean branch with stronger ordinal blend and auxiliary loss.",
            "args": [
                "--fms_combine_weight_ordinal",
                "0.25",
                "--ordinal_loss_weight",
                "0.15",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lds03_seed42",
            "notes": "Original add/mean branch with weaker LDS weighting.",
            "args": ["--lds_gamma", "0.3"],
        },
        {
            "name": "cbr_baseline_add_mean_lds08_seed42",
            "notes": "Original add/mean branch with stronger LDS weighting.",
            "args": ["--lds_gamma", "0.8"],
        },
        {
            "name": "cbr_baseline_add_mean_fds05_seed42",
            "notes": "Original add/mean branch with weaker FDS blend.",
            "args": ["--fds_blend", "0.5"],
        },
        {
            "name": "cbr_baseline_add_mean_no_fds_seed42",
            "notes": "Original add/mean branch without FDS smoothing.",
            "args": ["--no_fds_enabled"],
        },
        {
            "name": "cbr_baseline_add_deeptcn_transformer_l1_seed42",
            "notes": "Original add/mean fusion with one Transformer layer after DeepTCN.",
            "args": ["--calibration_encoder_mode", "deep_tcn_transformer", "--transformer_layers", "1"],
        },
        {
            "name": "cbr_baseline_add_deeptcn_transformer_l2_seed42",
            "notes": "Original add/mean fusion with two Transformer layers after DeepTCN.",
            "args": ["--calibration_encoder_mode", "deep_tcn_transformer", "--transformer_layers", "2"],
        },
        {
            "name": "cbr_deeptcn_transformer_attn_l1_lr3e4_drop005_seed42",
            "notes": "Best revised calibration family so far with lower LR/dropout.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
                "--calibration_encoder_mode",
                "deep_tcn_transformer",
                "--transformer_layers",
                "1",
                "--learning_rate",
                "0.0003",
                "--dropout",
                "0.05",
            ],
        },
        {
            "name": "cbr_deeptcn_transformer_attn_l1_drop015_seed42",
            "notes": "Best revised calibration family so far with stronger dropout.",
            "args": [
                "--calibration_fusion_mode",
                "mean_last_attention_summary",
                "--calibration_fusion_output_dim",
                "192",
                "--calibration_fusion_hidden_dim",
                "384",
                "--calibration_encoder_mode",
                "deep_tcn_transformer",
                "--transformer_layers",
                "1",
                "--dropout",
                "0.15",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_seed42",
            "notes": "Combine the two best single-axis changes: higher LR and weaker ordinal pressure.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_drop005_seed42",
            "notes": "Higher LR + weaker ordinal pressure + lower dropout.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--dropout",
                "0.05",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_no_fds_seed42",
            "notes": "Higher LR with FDS disabled.",
            "args": ["--learning_rate", "0.0006", "--no_fds_enabled"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_no_fds_seed42",
            "notes": "Weaker ordinal pressure with FDS disabled.",
            "args": [
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--no_fds_enabled",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_fds05_seed42",
            "notes": "Higher LR + weaker ordinal pressure + weaker FDS blend.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--fds_blend",
                "0.5",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_lds03_seed42",
            "notes": "Higher LR + weaker ordinal pressure + weaker LDS weighting.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--lds_gamma",
                "0.3",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_risk050_seed42",
            "notes": "Higher LR + weaker ordinal pressure + stronger risk auxiliary.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--risk_loss_weight",
                "0.50",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_lr6e4_ord010_seed123",
            "notes": "Seed robustness check for the best combined baseline candidate.",
            "args": [
                "--learning_rate",
                "0.0006",
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--seed",
                "123",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_ord000_seed42",
            "notes": "Remove ordinal prediction blending and auxiliary ordinal pressure entirely.",
            "args": ["--fms_combine_weight_ordinal", "0.0", "--ordinal_loss_weight", "0.0"],
        },
        {
            "name": "cbr_baseline_add_mean_ord005_seed42",
            "notes": "Very weak ordinal blending and auxiliary pressure.",
            "args": ["--fms_combine_weight_ordinal", "0.05", "--ordinal_loss_weight", "0.025"],
        },
        {
            "name": "cbr_baseline_add_mean_ord0075_seed42",
            "notes": "Intermediate ordinal pressure between 0.05 and the current best 0.10.",
            "args": ["--fms_combine_weight_ordinal", "0.075", "--ordinal_loss_weight", "0.035"],
        },
        {
            "name": "cbr_baseline_add_mean_ord0125_seed42",
            "notes": "Slightly stronger ordinal pressure than the current best 0.10.",
            "args": ["--fms_combine_weight_ordinal", "0.125", "--ordinal_loss_weight", "0.075"],
        },
        {
            "name": "cbr_baseline_add_mean_lr4e4_ord010_seed42",
            "notes": "Weak ordinal pressure with learning rate just below the baseline.",
            "args": ["--learning_rate", "0.0004", "--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05"],
        },
        {
            "name": "cbr_baseline_add_mean_lr5e4_ord010_seed42",
            "notes": "Weak ordinal pressure with learning rate between baseline and 0.0006.",
            "args": ["--learning_rate", "0.0005", "--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_wd1e4_seed42",
            "notes": "Weak ordinal pressure with lighter weight decay.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--weight_decay", "0.0001"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_wd5e4_seed42",
            "notes": "Weak ordinal pressure with stronger weight decay.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--weight_decay", "0.0005"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_bs32_seed42",
            "notes": "Weak ordinal pressure with smaller batch size.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--batch_size", "32"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_bs64_seed42",
            "notes": "Weak ordinal pressure with larger batch size.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--batch_size", "64"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_seed7",
            "notes": "Seed robustness check for the current best weak-ordinal candidate.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--seed", "7"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_seed123",
            "notes": "Second seed robustness check for the current best weak-ordinal candidate.",
            "args": ["--fms_combine_weight_ordinal", "0.10", "--ordinal_loss_weight", "0.05", "--seed", "123"],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_wd1e4_seed7",
            "notes": "Seed robustness check for the best weak-ordinal plus lighter weight-decay candidate.",
            "args": [
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--weight_decay",
                "0.0001",
                "--seed",
                "7",
            ],
        },
        {
            "name": "cbr_baseline_add_mean_ord010_wd1e4_seed123",
            "notes": "Second seed robustness check for the best weak-ordinal plus lighter weight-decay candidate.",
            "args": [
                "--fms_combine_weight_ordinal",
                "0.10",
                "--ordinal_loss_weight",
                "0.05",
                "--weight_decay",
                "0.0001",
                "--seed",
                "123",
            ],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low002_seed42",
            "notes": "Best validation family plus very weak low-FMS overprediction penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.02"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low005_seed42",
            "notes": "Best validation family plus weak low-FMS overprediction penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.05"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low010_seed42",
            "notes": "Best validation family plus moderate low-FMS overprediction penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.10"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low020_seed42",
            "notes": "Best validation family plus stronger low-FMS overprediction penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.20"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high002_seed42",
            "notes": "Best validation family plus very weak high-FMS underprediction penalty.",
            "args": [*best_base, "--high_underprediction_weight", "0.02"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high005_seed42",
            "notes": "Best validation family plus weak high-FMS underprediction penalty.",
            "args": [*best_base, "--high_underprediction_weight", "0.05"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high010_seed42",
            "notes": "Best validation family plus moderate high-FMS underprediction penalty.",
            "args": [*best_base, "--high_underprediction_weight", "0.10"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high020_seed42",
            "notes": "Best validation family plus stronger high-FMS underprediction penalty.",
            "args": [*best_base, "--high_underprediction_weight", "0.20"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_both002_seed42",
            "notes": "Best validation family plus very weak symmetric low/high signed-bias penalties.",
            "args": [*best_base, "--low_overprediction_weight", "0.02", "--high_underprediction_weight", "0.02"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_both005_seed42",
            "notes": "Best validation family plus weak symmetric low/high signed-bias penalties.",
            "args": [*best_base, "--low_overprediction_weight", "0.05", "--high_underprediction_weight", "0.05"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_both010_seed42",
            "notes": "Best validation family plus moderate symmetric low/high signed-bias penalties.",
            "args": [*best_base, "--low_overprediction_weight", "0.10", "--high_underprediction_weight", "0.10"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_both020_seed42",
            "notes": "Best validation family plus stronger symmetric low/high signed-bias penalties.",
            "args": [*best_base, "--low_overprediction_weight", "0.20", "--high_underprediction_weight", "0.20"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low010_high005_seed42",
            "notes": "Best validation family plus low-heavy asymmetric signed-bias penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.10", "--high_underprediction_weight", "0.05"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low005_high010_seed42",
            "notes": "Best validation family plus high-heavy asymmetric signed-bias penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.05", "--high_underprediction_weight", "0.10"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low020_high010_seed42",
            "notes": "Best validation family plus stronger low-heavy signed-bias penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.20", "--high_underprediction_weight", "0.10"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low010_high020_seed42",
            "notes": "Best validation family plus stronger high-heavy signed-bias penalty.",
            "args": [*best_base, "--low_overprediction_weight", "0.10", "--high_underprediction_weight", "0.20"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low010_thr3_seed42",
            "notes": "Low overprediction penalty widened to true FMS <= 3.",
            "args": [*best_base, "--low_overprediction_weight", "0.10", "--low_overprediction_threshold", "3.0"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_low010_thr5_seed42",
            "notes": "Low overprediction penalty widened to true FMS <= 5.",
            "args": [*best_base, "--low_overprediction_weight", "0.10", "--low_overprediction_threshold", "5.0"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high005_thr12_seed42",
            "notes": "High underprediction penalty expanded to true FMS >= 12.",
            "args": [*best_base, "--high_underprediction_weight", "0.05", "--high_underprediction_threshold", "12.0"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_high010_thr12_seed42",
            "notes": "Moderate high underprediction penalty expanded to true FMS >= 12.",
            "args": [*best_base, "--high_underprediction_weight", "0.10", "--high_underprediction_threshold", "12.0"],
        },
        {
            "name": "cbr_biaspen_ord010_wd1e4_both005_thr5_12_seed42",
            "notes": "Weak signed-bias penalties over wider low/high thresholds.",
            "args": [
                *best_base,
                "--low_overprediction_weight",
                "0.05",
                "--low_overprediction_threshold",
                "5.0",
                "--high_underprediction_weight",
                "0.05",
                "--high_underprediction_threshold",
                "12.0",
            ],
        },
        {
            "name": "cbr_bestbase_no_lds_seed42",
            "notes": "Best validation family without LDS target reweighting.",
            "args": [*best_base, "--no_lds_weighting"],
        },
        {
            "name": "cbr_bestbase_lds02_seed42",
            "notes": "Best validation family with weaker LDS gamma 0.2.",
            "args": [*best_base, "--lds_gamma", "0.2"],
        },
        {
            "name": "cbr_bestbase_lds04_seed42",
            "notes": "Best validation family with slightly weaker LDS gamma 0.4.",
            "args": [*best_base, "--lds_gamma", "0.4"],
        },
        {
            "name": "cbr_bestbase_lds06_seed42",
            "notes": "Best validation family with slightly stronger LDS gamma 0.6.",
            "args": [*best_base, "--lds_gamma", "0.6"],
        },
        {
            "name": "cbr_bestbase_lds08_seed42",
            "notes": "Best validation family with stronger LDS gamma 0.8.",
            "args": [*best_base, "--lds_gamma", "0.8"],
        },
        {
            "name": "cbr_bestbase_lds_weightmax2_seed42",
            "notes": "Best validation family with lower LDS max sample weight.",
            "args": [*best_base, "--lds_weight_max", "2.0"],
        },
        {
            "name": "cbr_bestbase_lds_weightmax4_seed42",
            "notes": "Best validation family with higher LDS max sample weight.",
            "args": [*best_base, "--lds_weight_max", "4.0"],
        },
        {
            "name": "cbr_bestbase_no_fds_seed42",
            "notes": "Best validation family with FDS disabled.",
            "args": [*best_base, "--no_fds_enabled"],
        },
        {
            "name": "cbr_bestbase_fds025_seed42",
            "notes": "Best validation family with weak FDS blend 0.25.",
            "args": [*best_base, "--fds_blend", "0.25"],
        },
        {
            "name": "cbr_bestbase_fds050_seed42",
            "notes": "Best validation family with moderate FDS blend 0.50.",
            "args": [*best_base, "--fds_blend", "0.50"],
        },
        {
            "name": "cbr_bestbase_fds090_seed42",
            "notes": "Best validation family with strong FDS blend 0.90.",
            "args": [*best_base, "--fds_blend", "0.90"],
        },
        {
            "name": "cbr_bestbase_lr35e5_seed42",
            "notes": "Best validation family with lower learning rate 3.5e-4.",
            "args": [*best_base, "--learning_rate", "0.00035"],
        },
        {
            "name": "cbr_bestbase_lr4e4_seed42",
            "notes": "Best validation family with lower learning rate 4e-4.",
            "args": [*best_base, "--learning_rate", "0.0004"],
        },
        {
            "name": "cbr_bestbase_lr5e4_seed42",
            "notes": "Best validation family with learning rate 5e-4.",
            "args": [*best_base, "--learning_rate", "0.0005"],
        },
        {
            "name": "cbr_bestbase_lr55e5_seed42",
            "notes": "Best validation family with learning rate 5.5e-4.",
            "args": [*best_base, "--learning_rate", "0.00055"],
        },
        {
            "name": "cbr_bestbase_lr65e5_seed42",
            "notes": "Best validation family with higher learning rate 6.5e-4.",
            "args": [*best_base, "--learning_rate", "0.00065"],
        },
        {
            "name": "cbr_bestbase_drop000_seed42",
            "notes": "Best validation family with dropout removed.",
            "args": [*best_base, "--dropout", "0.0"],
        },
        {
            "name": "cbr_bestbase_drop005_seed42",
            "notes": "Best validation family with lower dropout 0.05.",
            "args": [*best_base, "--dropout", "0.05"],
        },
        {
            "name": "cbr_bestbase_drop008_seed42",
            "notes": "Best validation family with slightly lower dropout 0.08.",
            "args": [*best_base, "--dropout", "0.08"],
        },
        {
            "name": "cbr_bestbase_drop012_seed42",
            "notes": "Best validation family with slightly higher dropout 0.12.",
            "args": [*best_base, "--dropout", "0.12"],
        },
        {
            "name": "cbr_bestbase_drop015_seed42",
            "notes": "Best validation family with higher dropout 0.15.",
            "args": [*best_base, "--dropout", "0.15"],
        },
        {
            "name": "cbr_bestbase_risk010_seed42",
            "notes": "Best validation family with much weaker rapid-rise auxiliary.",
            "args": [*best_base, "--risk_loss_weight", "0.10"],
        },
        {
            "name": "cbr_bestbase_risk025_seed42",
            "notes": "Best validation family with weaker rapid-rise auxiliary.",
            "args": [*best_base, "--risk_loss_weight", "0.25"],
        },
        {
            "name": "cbr_bestbase_risk045_seed42",
            "notes": "Best validation family with slightly stronger rapid-rise auxiliary.",
            "args": [*best_base, "--risk_loss_weight", "0.45"],
        },
        {
            "name": "cbr_bestbase_risk060_seed42",
            "notes": "Best validation family with stronger rapid-rise auxiliary.",
            "args": [*best_base, "--risk_loss_weight", "0.60"],
        },
        {
            "name": "cbr_bestbase_fall005_seed42",
            "notes": "Best validation family plus weak rapid-drop auxiliary head.",
            "args": [*best_base, "--fall_risk_head_enabled", "--fall_loss_weight", "0.05"],
        },
        {
            "name": "cbr_bestbase_fall010_seed42",
            "notes": "Best validation family plus moderate rapid-drop auxiliary head.",
            "args": [*best_base, "--fall_risk_head_enabled", "--fall_loss_weight", "0.10"],
        },
        {
            "name": "cbr_bestbase_fall020_seed42",
            "notes": "Best validation family plus stronger rapid-drop auxiliary head.",
            "args": [*best_base, "--fall_risk_head_enabled", "--fall_loss_weight", "0.20"],
        },
        {
            "name": "cbr_bestbase_curreg000_seed42",
            "notes": "Best validation family without current regression auxiliary head loss.",
            "args": [*best_base, "--current_reg_aux_weight", "0.0"],
        },
        {
            "name": "cbr_bestbase_curreg015_seed42",
            "notes": "Best validation family with weaker current regression auxiliary head loss.",
            "args": [*best_base, "--current_reg_aux_weight", "0.15"],
        },
        {
            "name": "cbr_bestbase_curreg050_seed42",
            "notes": "Best validation family with stronger current regression auxiliary head loss.",
            "args": [*best_base, "--current_reg_aux_weight", "0.50"],
        },
        {
            "name": "cbr_bestbase_smooth005_seed42",
            "notes": "Best validation family with very weak shape/trend auxiliary loss.",
            "args": [*best_base, "--smoothness_weight", "0.005"],
        },
        {
            "name": "cbr_bestbase_smooth010_seed42",
            "notes": "Best validation family with weak shape/trend auxiliary loss.",
            "args": [*best_base, "--smoothness_weight", "0.010"],
        },
        {
            "name": "cbr_bestbase_smooth020_seed42",
            "notes": "Best validation family with moderate shape/trend auxiliary loss.",
            "args": [*best_base, "--smoothness_weight", "0.020"],
        },
        {
            "name": "cbr_bestbase_loss_smoothl1_seed42",
            "notes": "Best validation family with SmoothL1 as the main level loss.",
            "args": [*best_base, "--loss_type", "smooth_l1"],
        },
        {
            "name": "cbr_bestbase_loss_mse_seed42",
            "notes": "Best validation family with MSE as the main level loss.",
            "args": [*best_base, "--loss_type", "mse"],
        },
        {
            "name": "cbr_bestbase_bs24_seed42",
            "notes": "Best validation family with smaller batch size 24.",
            "args": [*best_base, "--batch_size", "24"],
        },
        {
            "name": "cbr_bestbase_bs32_seed42",
            "notes": "Best validation family with smaller batch size 32.",
            "args": [*best_base, "--batch_size", "32"],
        },
        {
            "name": "cbr_bestbase_bs64_seed42",
            "notes": "Best validation family with larger batch size 64.",
            "args": [*best_base, "--batch_size", "64"],
        },
        {
            "name": "cbr_bestbase_hid256_seed42",
            "notes": "Best validation family with wider latent hidden state 256.",
            "args": [*best_base, "--hidden_dim", "256"],
        },
        {
            "name": "cbr_bestbase_d128_h256_seed42",
            "notes": "Best validation family with wider d_model 128 and hidden state 256.",
            "args": [*best_base, "--d_model", "128", "--hidden_dim", "256"],
        },
        {
            "name": "cbr_bestbase_d64_h128_seed42",
            "notes": "Best validation family with smaller d_model 64 and hidden state 128.",
            "args": [*best_base, "--d_model", "64", "--hidden_dim", "128"],
        },
        {
            "name": "cbr_bestbase_stream_dil1248_seed42",
            "notes": "Best validation family with shorter stream DeepTCN dilation stack.",
            "args": [*best_base, "--deep_tcn_dilations", "1", "2", "4", "8"],
        },
        {
            "name": "cbr_bestbase_stream_dil12481632_seed42",
            "notes": "Best validation family with longer stream DeepTCN dilation stack.",
            "args": [*best_base, "--deep_tcn_dilations", "1", "2", "4", "8", "16", "32"],
        },
        {
            "name": "cbr_bestbase_calib_dil124816_seed42",
            "notes": "Best validation family with shorter calibration DeepTCN dilation stack.",
            "args": [*best_base, "--calib_dilations", "1", "2", "4", "8", "16"],
        },
        {
            "name": "cbr_bestbase_calib_dil1248163264_seed42",
            "notes": "Best validation family with longer calibration DeepTCN dilation stack.",
            "args": [*best_base, "--calib_dilations", "1", "2", "4", "8", "16", "32", "64"],
        },
        {
            "name": "cbr_bestbase_scenario_seed42",
            "notes": "Best validation family with deployment-visible scenario static one-hot prior.",
            "args": [*best_base, "--static_features", "age", "mssq", "gender", "scenario"],
        },
        {
            "name": "cbr_bestbase_scenario_bias005_seed42",
            "notes": "Scenario static prior plus weak symmetric signed-bias penalties.",
            "args": [
                *best_base,
                "--static_features",
                "age",
                "mssq",
                "gender",
                "scenario",
                "--low_overprediction_weight",
                "0.05",
                "--high_underprediction_weight",
                "0.05",
            ],
        },
        {
            "name": "cbr_bestbase_scenario_no_fds_seed42",
            "notes": "Scenario static prior with FDS disabled.",
            "args": [*best_base, "--static_features", "age", "mssq", "gender", "scenario", "--no_fds_enabled"],
        },
        {
            "name": "cbr_bestbase_scenario_lr4e4_seed42",
            "notes": "Scenario static prior with lower learning rate.",
            "args": [
                *best_base,
                "--static_features",
                "age",
                "mssq",
                "gender",
                "scenario",
                "--learning_rate",
                "0.0004",
            ],
        },
        {
            "name": "cbr_bestbase_staticdrop005_seed42",
            "notes": "Best validation family with lower static encoder dropout.",
            "args": [*best_base, "--static_dropout", "0.05"],
        },
        {
            "name": "cbr_bestbase_staticdrop020_seed42",
            "notes": "Best validation family with stronger static encoder dropout.",
            "args": [*best_base, "--static_dropout", "0.20"],
        },
        {
            "name": "cbr_bestbase_calibfmsdrop005_seed42",
            "notes": "Best validation family with weak dropout on calibration FMS history.",
            "args": [*best_base, "--calib_fms_dropout", "0.05"],
        },
        {
            "name": "cbr_bestbase_calibfmsdrop010_seed42",
            "notes": "Best validation family with stronger dropout on calibration FMS history.",
            "args": [*best_base, "--calib_fms_dropout", "0.10"],
        },
        {
            "name": "cbr_bestbase_recent5_seed42",
            "notes": "Best validation family with shorter recent motion window.",
            "args": [*best_base, "--recent_window_seconds", "5.0"],
        },
        {
            "name": "cbr_bestbase_recent20_seed42",
            "notes": "Best validation family with longer recent motion window.",
            "args": [*best_base, "--recent_window_seconds", "20.0"],
        },
        {
            "name": "cbr_bestbase_recent30_seed42",
            "notes": "Best validation family with 30-second recent motion window.",
            "args": [*best_base, "--recent_window_seconds", "30.0"],
        },
        {
            "name": "cbr_bestbase_coarse005_seed42",
            "notes": "Best validation family plus weak coarse FMS band auxiliary head.",
            "args": [*best_base, "--coarse_band_bins", "2", "5", "10", "15", "--coarse_band_loss_weight", "0.05"],
        },
        {
            "name": "cbr_bestbase_coarse010_seed42",
            "notes": "Best validation family plus moderate coarse FMS band auxiliary head.",
            "args": [*best_base, "--coarse_band_bins", "2", "5", "10", "15", "--coarse_band_loss_weight", "0.10"],
        },
        {
            "name": "cbr_bestbase_regime005_seed42",
            "notes": "Best validation family plus weak rise/fall/plateau regime auxiliary head.",
            "args": [*best_base, "--regime_head_enabled", "--regime_loss_weight", "0.05"],
        },
        {
            "name": "cbr_bestbase_regime010_seed42",
            "notes": "Best validation family plus moderate rise/fall/plateau regime auxiliary head.",
            "args": [*best_base, "--regime_head_enabled", "--regime_loss_weight", "0.10"],
        },
        {
            "name": "cbr_bestbase_risk040_seed42",
            "notes": "Refine around the new risk-weight optimum with risk loss 0.40.",
            "args": [*best_base, "--risk_loss_weight", "0.40"],
        },
        {
            "name": "cbr_bestbase_risk050_seed42",
            "notes": "Refine around the new risk-weight optimum with risk loss 0.50.",
            "args": [*best_base, "--risk_loss_weight", "0.50"],
        },
        {
            "name": "cbr_bestbase_risk055_seed42",
            "notes": "Refine around the new risk-weight optimum with risk loss 0.55.",
            "args": [*best_base, "--risk_loss_weight", "0.55"],
        },
        {
            "name": "cbr_bestbase_risk045_low002_seed42",
            "notes": "New risk 0.45 validation winner plus very weak low-FMS overprediction penalty.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--low_overprediction_weight", "0.02"],
        },
        {
            "name": "cbr_bestbase_risk045_high002_seed42",
            "notes": "New risk 0.45 validation winner plus very weak high-FMS underprediction penalty.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--high_underprediction_weight", "0.02"],
        },
        {
            "name": "cbr_bestbase_risk045_both002_seed42",
            "notes": "New risk 0.45 validation winner plus very weak symmetric signed-bias penalties.",
            "args": [
                *best_base,
                "--risk_loss_weight",
                "0.45",
                "--low_overprediction_weight",
                "0.02",
                "--high_underprediction_weight",
                "0.02",
            ],
        },
        {
            "name": "cbr_bestbase_risk045_lr4e4_seed42",
            "notes": "New risk 0.45 validation winner with lower learning rate.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--learning_rate", "0.0004"],
        },
        {
            "name": "cbr_bestbase_risk045_lr5e4_seed42",
            "notes": "New risk 0.45 validation winner with learning rate 5e-4.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--learning_rate", "0.0005"],
        },
        {
            "name": "cbr_bestbase_risk045_no_fds_seed42",
            "notes": "New risk 0.45 validation winner with FDS disabled.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--no_fds_enabled"],
        },
        {
            "name": "cbr_bestbase_risk045_fds05_seed42",
            "notes": "New risk 0.45 validation winner with weaker FDS blend.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--fds_blend", "0.50"],
        },
        {
            "name": "cbr_bestbase_risk045_drop008_seed42",
            "notes": "New risk 0.45 validation winner with slightly lower dropout.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--dropout", "0.08"],
        },
        {
            "name": "cbr_bestbase_risk045_drop012_seed42",
            "notes": "New risk 0.45 validation winner with slightly higher dropout.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--dropout", "0.12"],
        },
        {
            "name": "cbr_bestbase_risk045_smooth005_seed42",
            "notes": "Combine the best risk-weight signal with the best weak smoothness signal.",
            "args": [*best_base, "--risk_loss_weight", "0.45", "--smoothness_weight", "0.005"],
        },
        {
            "name": "cbr_bestbase_risk040_smooth005_seed42",
            "notes": "Risk-weight refinement at 0.40 plus weak smoothness.",
            "args": [*best_base, "--risk_loss_weight", "0.40", "--smoothness_weight", "0.005"],
        },
        {
            "name": "cbr_bestbase_risk050_smooth005_seed42",
            "notes": "Risk-weight refinement at 0.50 plus weak smoothness.",
            "args": [*best_base, "--risk_loss_weight", "0.50", "--smoothness_weight", "0.005"],
        },
    ]


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("metrics", payload)
        best = metrics.get("best_val_metrics", {})
        return {
            "status": "completed",
            "best_epoch": metrics.get("best_epoch"),
            "val_mae": best.get("mae", best.get("current_fms_mae")),
            "val_rmse": best.get("rmse", best.get("current_fms_rmse")),
            "val_r2": best.get("r2", best.get("current_fms_r2")),
            "selection_value": metrics.get("best_selection_value"),
        }
    curves_path = run_dir / "training_curves.csv"
    if curves_path.exists():
        with curves_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            best = min(rows, key=lambda row: float(row["val_mae"]))
            return {
                "status": "partial",
                "best_epoch": int(best["epoch"]),
                "val_mae": float(best["val_mae"]),
                "val_rmse": float(best["val_rmse"]),
                "val_r2": None,
                "selection_value": float(best["val_mae"]),
            }
    return {"status": "missing"}


def append_report(report_path: Path, candidate: Dict[str, Any], summary: Dict[str, Any], returncode: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    val_mae = summary.get("val_mae")
    delta = None if val_mae is None else float(val_mae) - BASELINE_VAL_MAE
    line = [
        "",
        f"### {datetime.now().isoformat(timespec='seconds')} `{candidate['name']}`",
        "",
        f"- Notes: {candidate['notes']}",
        f"- Args: `{' '.join(candidate['args'])}`",
        f"- Return code: {returncode}",
        f"- Status: {summary.get('status')}",
        f"- Best epoch: {summary.get('best_epoch')}",
        f"- Validation MAE/RMSE/R2: {summary.get('val_mae')} / {summary.get('val_rmse')} / {summary.get('val_r2')}",
        f"- Delta vs baseline val MAE 1.7402: {delta}",
        "- Selection rule: validation MAE only; test not evaluated.",
    ]
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(line) + "\n")


def build_command(args: argparse.Namespace, candidate: Dict[str, Any]) -> List[str]:
    return [
        args.python,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        args.config,
        "--model",
        "online_fms_risk_tracker",
        "--device",
        args.device,
        "--runs_dir",
        args.runs_dir,
        "--run_name",
        candidate["name"],
        "--split_file",
        args.split_file,
        "--no_test_eval",
        *candidate["args"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration branch revision search candidates.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--config", default=BASE_CONFIG)
    parser.add_argument("--split_file", default=BASE_SPLIT)
    parser.add_argument("--runs_dir", default=RUNS_DIR)
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument(
        "--only_indices",
        default=None,
        help="Comma-separated global candidate indices to run instead of a contiguous range.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parsed = parser.parse_args()

    all_candidates = candidates()
    if parsed.only_indices:
        run_indices = [int(value.strip()) for value in parsed.only_indices.split(",") if value.strip()]
        selected_pairs = [(idx, all_candidates[idx]) for idx in run_indices]
    else:
        selected = all_candidates[int(parsed.start_index) :]
        if parsed.max_runs is not None:
            selected = selected[: int(parsed.max_runs)]
        selected_pairs = list(enumerate(selected, start=int(parsed.start_index)))
    if parsed.only_indices and parsed.max_runs is not None:
        selected_pairs = selected_pairs[: int(parsed.max_runs)]
    report_path = Path(parsed.report)
    for idx, candidate in selected_pairs:
        run_dir = Path(parsed.runs_dir) / candidate["name"]
        if (run_dir / "metrics.json").exists():
            summary = load_metrics(run_dir)
            append_report(report_path, candidate, summary, returncode=0)
            print(f"SKIP completed {idx}: {candidate['name']} val_mae={summary.get('val_mae')}")
            continue
        cmd = build_command(parsed, candidate)
        print(f"RUN {idx}: {' '.join(cmd)}", flush=True)
        if parsed.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        summary = load_metrics(run_dir)
        append_report(report_path, candidate, summary, completed.returncode)
        print(
            f"DONE {idx}: {candidate['name']} returncode={completed.returncode} "
            f"status={summary.get('status')} val_mae={summary.get('val_mae')}",
            flush=True,
        )
        if completed.returncode != 0:
            break


if __name__ == "__main__":
    main()
