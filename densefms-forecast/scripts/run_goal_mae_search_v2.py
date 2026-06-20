#!/usr/bin/env python
"""Long-running corrected start_only MAE search for the 0505 goal.

The normal search path is validation-only. Test evaluation is reachable only
after a validation-based final selection lock has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "runs/goal_mae_search_v2"
PRIMARY_HORIZONS = (5.0, 10.0, 15.0)
DIAGNOSTIC_HORIZONS = (2.5, 1.0)
V1_BASELINE_PRIMARY = 2.4943
BLOCKING_STATUSES = {"failed", "blocked", "interrupted", "running"}

MANIFEST_FIELDS = [
    "run_name",
    "status",
    "architecture_family",
    "command",
    "config_path",
    "checkpoint_path",
    "metrics_path",
    "prediction_csv_path",
    "train_prediction_csv_path",
    "plot_dir",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "max_epochs_planned",
    "epochs_completed",
    "best_epoch",
    "best_val_mae",
    "failure_reason",
    "interrupt_reason",
    "resume_action",
]

EXPERIMENT_FIELDS = [
    "run_name",
    "status",
    "architecture_family",
    "architecture_hypothesis",
    "model_type",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "static_feature_set",
    "forbidden_identity_features",
    "recent_start_observed",
    "sparse_observed",
    "predict_delta_from_anchor",
    "calibration_seconds",
    "recent_window_seconds",
    "horizon_seconds",
    "multi_horizon",
    "horizon_set",
    "loss_type",
    "loss_mode",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_model",
    "hidden_dim",
    "pooling",
    "seed",
    "max_epochs_planned",
    "epochs_completed",
    "best_epoch",
    "val_mae",
    "val_rmse",
    "val_n",
    "h5_val_mae",
    "h10_val_mae",
    "h15_val_mae",
    "h2p5_val_mae",
    "h1_diagnostic_val_mae",
    "primary_mean_val_mae",
    "selection_reason",
    "failure_reason",
    "sanity_status",
    "leakage_status",
    "checkpoint_path",
    "metrics_path",
    "prediction_csv_path",
    "train_prediction_csv_path",
    "plot_dir",
]

LEADERBOARD_FIELDS = [
    "rank",
    "status",
    "architecture_family",
    "architecture_hypothesis",
    "family_index",
    "family_count",
    "model_type",
    "fms_context_mode",
    "anchor_mode",
    "anchor_interval_seconds",
    "use_static",
    "static_feature_set",
    "predict_delta_from_anchor",
    "calibration_seconds",
    "recent_window_seconds",
    "loss_type",
    "loss_mode",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_model",
    "hidden_dim",
    "pooling",
    "max_epochs_planned",
    "epochs_completed",
    "h5_val_mae",
    "h10_val_mae",
    "h15_val_mae",
    "mean_val_mae_h5_h10_h15",
    "h2p5_val_mae",
    "h1_diagnostic_val_mae",
    "improvement_vs_v1_baseline_percent",
    "member_runs",
    "branch",
    "selection_reason",
    "failure_reason",
    "sanity_status",
    "leakage_status",
    "checkpoint_paths",
    "prediction_csv_paths",
    "plot_dirs",
    "final_selection_eligible",
]

FINAL_TEST_FIELDS = [
    "architecture_family",
    "run_name",
    "status",
    "horizon_seconds",
    "validation_primary_mean_mae",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_n",
    "h5_test_mae",
    "h10_test_mae",
    "h15_test_mae",
    "metrics_path",
    "prediction_csv_path",
]


@dataclass(frozen=True)
class Spec:
    architecture_family: str
    branch: str
    model: str
    horizon_seconds: float = 5.0
    multi_horizon: bool = False
    horizon_set: Sequence[float] = field(default_factory=tuple)
    use_static: bool = False
    hypothesis: str = ""
    selection_reason: str = ""
    epochs: int = 24
    patience: int = 8
    seed: int = 42
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 64
    hidden_dim: int = 128
    loss_type: str = "smooth_l1"
    loss_mode: str = "level_only"
    high_target_weight: float = 0.0
    high_target_threshold: float = 0.5
    change_weight: float = 0.0
    trend_weight: float = 0.0
    calibration_seconds: float = 90.0
    recent_window_seconds: float = 30.0
    fms_context_mode: str = "start_only"
    anchor_mode: str = "none"
    anchor_interval_seconds: float = 0.0
    predict_delta_from_anchor: bool = False
    per_horizon_heads: bool = False
    pooling: str = "mean"
    kernel_size: int = 3
    calib_dilations: Sequence[int] = field(default_factory=tuple)
    recent_dilations: str = "auto"
    transformer_layers: int = 1
    transformer_heads: int = 4
    transformer_ff_dim: int = 128
    mlp_layers: Sequence[int] = field(default_factory=tuple)
    gru_layers: int = 1
    branch_dropout: float = 0.0
    anchor_dropout: float = 0.0
    delta_scale: float = 0.5
    recent_encoder: str = "tcn"
    recent_attn_layers: int = 1
    recent_attn_heads: int = 4
    recent_attn_dropout: float = 0.1
    final_selection_eligible: bool = True

    @property
    def run_name(self) -> str:
        h = "mh" if self.multi_horizon else f"h{tag(self.horizon_seconds)}"
        static = "static" if self.use_static else "nostatic"
        bits = [
            "v2",
            safe(self.architecture_family),
            safe(self.branch),
            h,
            static,
            f"lr{tag(self.learning_rate)}",
            f"wd{tag(self.weight_decay)}",
            f"drop{tag(self.dropout)}",
            f"e{self.epochs}",
            f"s{self.seed}",
        ]
        return "_".join(bits)


def adaptive_variant_dicts() -> List[Dict[str, Any]]:
    return [
        {"branch": "adaptive_seed7", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed123", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed202", "seed": 202, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed314", "seed": 314, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed555", "seed": 555, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed777", "seed": 777, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed999", "seed": 999, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_lr2e4", "seed": 42, "learning_rate": 2e-4, "dropout": 0.03, "d_model": 128},
        {"branch": "adaptive_lr4e4", "seed": 42, "learning_rate": 4e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_lr2e4_d96", "seed": 42, "learning_rate": 2e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_lr1e4_d128", "seed": 42, "learning_rate": 1e-4, "dropout": 0.03, "d_model": 128},
        {"branch": "adaptive_seed7_lr25e5", "seed": 7, "learning_rate": 2.5e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed7_lr35e5", "seed": 7, "learning_rate": 3.5e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_drop0", "seed": 42, "learning_rate": 3e-4, "dropout": 0.0, "d_model": 96},
        {"branch": "adaptive_drop08", "seed": 42, "learning_rate": 3e-4, "dropout": 0.08, "d_model": 96},
        {"branch": "adaptive_seed7_drop03", "seed": 7, "learning_rate": 3e-4, "dropout": 0.03, "d_model": 96},
        {"branch": "adaptive_seed7_drop04", "seed": 7, "learning_rate": 3e-4, "dropout": 0.04, "d_model": 96},
        {"branch": "adaptive_seed7_drop06", "seed": 7, "learning_rate": 3e-4, "dropout": 0.06, "d_model": 96},
        {"branch": "adaptive_d128_drop05", "seed": 42, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_seed7_d128_drop05", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_seed123_d128_drop05", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_seed202_d128_drop05", "seed": 202, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_seed7_d112_drop05", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 112},
        {"branch": "adaptive_seed123_d112_drop05", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 112},
        {"branch": "adaptive_seed7_d128_lr25e5", "seed": 7, "learning_rate": 2.5e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_seed7_d128_lr35e5", "seed": 7, "learning_rate": 3.5e-4, "dropout": 0.05, "d_model": 128},
        {"branch": "adaptive_wd3e4", "seed": 42, "learning_rate": 3e-4, "weight_decay": 3e-4, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_wd3e5", "seed": 42, "learning_rate": 3e-4, "weight_decay": 3e-5, "dropout": 0.05, "d_model": 96},
        {"branch": "adaptive_seed7_calib120", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "calibration_seconds": 120.0},
        {"branch": "adaptive_seed202_calib120", "seed": 202, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "calibration_seconds": 120.0},
        {"branch": "adaptive_seed7_calib120_d128", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128, "calibration_seconds": 120.0},
        {"branch": "adaptive_seed7_recent45", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "recent_window_seconds": 45.0},
        {"branch": "adaptive_seed7_layers2", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_layers": 2},
        {"branch": "adaptive_seed7_ff192", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed123_ff192", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed202_ff192", "seed": 202, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed314_ff192", "seed": 314, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed777_ff192", "seed": 777, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff160", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 160},
        {"branch": "adaptive_seed7_ff224", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 224},
        {"branch": "adaptive_seed7_ff256", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 256},
        {"branch": "adaptive_seed7_ff192_lr25e5", "seed": 7, "learning_rate": 2.5e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_lr35e5", "seed": 7, "learning_rate": 3.5e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_lr4e4", "seed": 7, "learning_rate": 4e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed777_ff192_lr35e5", "seed": 777, "learning_rate": 3.5e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed123_ff192_lr35e5", "seed": 123, "learning_rate": 3.5e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_drop03", "seed": 7, "learning_rate": 3e-4, "dropout": 0.03, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_drop04", "seed": 7, "learning_rate": 3e-4, "dropout": 0.04, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_drop06", "seed": 7, "learning_rate": 3e-4, "dropout": 0.06, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_wd3e5", "seed": 7, "learning_rate": 3e-4, "weight_decay": 3e-5, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_wd3e4", "seed": 7, "learning_rate": 3e-4, "weight_decay": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_ff192_calib120", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192, "calibration_seconds": 120.0},
        {"branch": "adaptive_seed7_ff192_recent20", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192, "recent_window_seconds": 20.0},
        {"branch": "adaptive_seed7_ff192_recent25", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192, "recent_window_seconds": 25.0},
        {"branch": "adaptive_seed7_ff192_recent35", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 192, "recent_window_seconds": 35.0},
        {"branch": "adaptive_seed777_ff224", "seed": 777, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 224},
        {"branch": "adaptive_seed123_ff224", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "transformer_ff_dim": 224},
        {"branch": "adaptive_seed7_d112_ff192", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 112, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed123_d128_ff192", "seed": 123, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128, "transformer_ff_dim": 192},
        {"branch": "adaptive_seed7_d128_ff192", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 128, "transformer_ff_dim": 192},
        {"branch": "adaptive_l1", "seed": 42, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "loss_type": "l1"},
        {"branch": "adaptive_l1_seed7", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "loss_type": "l1"},
        {"branch": "adaptive_highfms", "seed": 42, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "high_target_weight": 0.4, "high_target_threshold": 0.55},
        {"branch": "adaptive_highfms_seed7", "seed": 7, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "high_target_weight": 0.4, "high_target_threshold": 0.55},
        {"branch": "adaptive_highfms_light", "seed": 42, "learning_rate": 3e-4, "dropout": 0.05, "d_model": 96, "high_target_weight": 0.2, "high_target_threshold": 0.55},
    ]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def tag(value: Any) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], limit: Optional[int] = None) -> str:
    shown = list(rows[:limit] if limit is not None else rows)
    if not shown:
        return "| empty |\n|---|\n"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def first_metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return ""


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build_specs() -> List[Spec]:
    specs: List[Spec] = []

    def mh(
        family: str,
        branch: str,
        model: str,
        hypothesis: str,
        reason: str,
        *,
        epochs: int = 24,
        final_selection_eligible: bool = True,
        **kwargs: Any,
    ) -> None:
        specs.append(
            Spec(
                architecture_family=family,
                branch=branch,
                model=model,
                multi_horizon=True,
                horizon_set=PRIMARY_HORIZONS,
                horizon_seconds=5.0,
                hypothesis=hypothesis,
                selection_reason=reason,
                epochs=epochs,
                final_selection_eligible=final_selection_eligible,
                **kwargs,
            )
        )

    def singles(
        family: str,
        branch: str,
        model: str,
        hypothesis: str,
        reason: str,
        *,
        horizons: Sequence[float] = PRIMARY_HORIZONS,
        epochs: int = 24,
        final_selection_eligible: bool = True,
        **kwargs: Any,
    ) -> None:
        for horizon in horizons:
            specs.append(
                Spec(
                    architecture_family=family,
                    branch=branch,
                    model=model,
                    horizon_seconds=float(horizon),
                    hypothesis=hypothesis,
                    selection_reason=reason,
                    epochs=epochs,
                    final_selection_eligible=final_selection_eligible,
                    **kwargs,
                )
            )

    # Representative architecture coverage. These run before final selection can lock.
    mh(
        "lcsa_shared",
        "representative",
        "lc_sa_tcnformer",
        "Shared h=5/10/15 LC-SA TCNFormer tests whether multi-horizon supervision reduces long-horizon collapse under corrected start_only inputs.",
        "Baseline reproduction and primary metric aligned representative.",
        learning_rate=5e-4,
        dropout=0.05,
        d_model=64,
    )
    mh(
        "lcsa_per_horizon_heads",
        "representative",
        "lc_sa_tcnformer",
        "Separate horizon heads may reduce interference between h=5, h=10, and h=15 while using the same legal start_only inputs.",
        "Architecture-first branch for horizon-specific error.",
        learning_rate=5e-4,
        dropout=0.05,
        d_model=64,
        per_horizon_heads=True,
    )
    mh(
        "anchor_delta_mlp",
        "representative",
        "anchor_delta_mlp",
        "A compact delta-from-start-FMS MLP may counter prediction shrinkage by anchoring the level at the legal window-start FMS.",
        "Different fusion family using only calibration, recent motion summary, and start FMS.",
        predict_delta_from_anchor=True,
        learning_rate=8e-4,
        hidden_dim=160,
        mlp_layers=(160, 96),
        dropout=0.05,
        delta_scale=0.75,
    )
    mh(
        "anchor_delta_gru",
        "representative",
        "anchor_delta_gru",
        "A recurrent recent-motion summary may better track motion dynamics before the forecast horizon than the MLP branch.",
        "Architecture coverage for recurrent temporal summarization with legal start FMS.",
        predict_delta_from_anchor=True,
        learning_rate=8e-4,
        hidden_dim=160,
        gru_layers=2,
        dropout=0.05,
        delta_scale=0.75,
    )
    mh(
        "recent_tcn_summary_calib",
        "representative",
        "recent_tcn_summary_calib",
        "Recent TCN summary fused with calibration statistics may improve high-FMS and long-horizon cases without dense recent FMS.",
        "Temporal modeling branch with explicit calibration summary.",
        predict_delta_from_anchor=True,
        learning_rate=7e-4,
        d_model=80,
        hidden_dim=160,
        pooling="attention",
        dropout=0.08,
        delta_scale=0.75,
    )
    mh(
        "gated_fusion",
        "representative",
        "gated_fusion",
        "A gated fusion branch may avoid over-trusting start FMS when recent motion indicates later drift.",
        "Fusion architecture branch targeting mean-regression failure.",
        predict_delta_from_anchor=True,
        learning_rate=7e-4,
        d_model=80,
        hidden_dim=160,
        pooling="attention",
        dropout=0.08,
        branch_dropout=0.05,
        delta_scale=0.75,
    )
    singles(
        "coff_lstm_tcn",
        "representative",
        "coff_lstm",
        "COFF-LSTM with TCN recent encoder checks whether the older recurrent decoder is more robust than direct fusion under start_only.",
        "Legacy temporal architecture branch; evaluated as one h=5/10/15 family.",
        learning_rate=6e-4,
        dropout=0.1,
        recent_encoder="tcn",
    )
    singles(
        "coff_lstm_transformer",
        "representative",
        "coff_lstm",
        "Replacing the COFF recent encoder with a Transformer tests longer-range recent-window interactions.",
        "Additional temporal architecture family after the minimum coverage target.",
        learning_rate=6e-4,
        dropout=0.1,
        recent_encoder="transformer",
        recent_attn_layers=1,
        recent_attn_heads=4,
    )
    mh(
        "lcsa_static",
        "representative",
        "lc_sa_tcnformer",
        "Age/Gender/MSSQ may explain susceptibility differences without participant identity leakage.",
        "Allowed static covariate branch, recorded separately from no-static.",
        learning_rate=5e-4,
        dropout=0.08,
        d_model=64,
        use_static=True,
    )
    mh(
        "lcsa_cross_attn",
        "representative",
        "lcsa_cross_attn",
        "Cross-attention from each forecast query to calibration tokens tests whether the LC-SA idea improves when calibration is not compressed to a single pooled vector.",
        "New architecture inspired by the current best LC-SA family; still corrected start_only and no test use.",
        epochs=60,
        learning_rate=3e-4,
        dropout=0.05,
        d_model=96,
        transformer_ff_dim=192,
        per_horizon_heads=True,
    )
    mh(
        "lcsa_cross_attn",
        "seed7_lr35e5",
        "lcsa_cross_attn",
        "The best existing LC-SA candidate benefited from seed 7 and lr=3.5e-4; this checks whether the cross-attention variant shares that optimization sweet spot.",
        "New architecture refinement around the current best optimizer setting.",
        epochs=80,
        seed=7,
        learning_rate=3.5e-4,
        dropout=0.05,
        d_model=96,
        transformer_ff_dim=192,
        per_horizon_heads=True,
    )
    mh(
        "lcsa_cross_attn",
        "ff256_seed7",
        "lcsa_cross_attn",
        "A wider calibration-query feed-forward block may improve the calibration-token matching that helped the ff192 LC-SA run.",
        "New architecture capacity check around the cross-attention branch.",
        epochs=80,
        seed=7,
        learning_rate=3e-4,
        dropout=0.05,
        d_model=96,
        transformer_ff_dim=256,
        per_horizon_heads=True,
    )
    mh(
        "gru_state_mixer",
        "representative",
        "gru_state_mixer",
        "A GRU-window forecaster tests a different recurrent inductive bias: bidirectional calibration encoding plus causal recent-window state mixing.",
        "New architecture not based on the LC-SA TCN/Transformer stack.",
        epochs=60,
        seed=7,
        learning_rate=4e-4,
        dropout=0.05,
        d_model=96,
        hidden_dim=128,
        gru_layers=1,
        per_horizon_heads=True,
    )
    mh(
        "gru_state_mixer",
        "d128_seed123",
        "gru_state_mixer",
        "If the recurrent mixer underfits, a larger hidden state and a strong prior seed checks whether capacity rather than architecture is limiting it.",
        "New recurrent architecture capacity/seed check.",
        epochs=60,
        seed=123,
        learning_rate=3e-4,
        dropout=0.05,
        d_model=128,
        hidden_dim=160,
        gru_layers=1,
        per_horizon_heads=True,
    )
    mh(
        "gru_state_mixer",
        "chunked_d64_seed7",
        "gru_state_mixer",
        "The first GRU mixer hit GPU memory pressure, so this smaller chunked-window variant retests the recurrent idea without changing the allowed input contract.",
        "New recurrent architecture retry after OOM; same start_only/no-extra-input policy.",
        epochs=60,
        seed=7,
        learning_rate=4e-4,
        dropout=0.05,
        d_model=64,
        hidden_dim=96,
        gru_layers=1,
        per_horizon_heads=True,
    )
    mh(
        "motion_conv_mixer",
        "representative",
        "motion_conv_mixer",
        "A compact causal ConvMixer uses motion-first depthwise/pointwise mixing plus calibration summary, deliberately avoiding the existing TCNFormer design.",
        "New architecture from a simpler motion-mixer baseline.",
        epochs=60,
        seed=7,
        learning_rate=5e-4,
        dropout=0.05,
        d_model=96,
        hidden_dim=128,
        kernel_size=5,
        transformer_layers=3,
        per_horizon_heads=True,
    )
    mh(
        "motion_conv_mixer",
        "deeper_k7",
        "motion_conv_mixer",
        "A deeper/wider causal ConvMixer checks whether the simple motion-first structure benefits from a larger temporal receptive field.",
        "New motion-mixer capacity/receptive-field check.",
        epochs=60,
        seed=7,
        learning_rate=4e-4,
        dropout=0.05,
        d_model=96,
        hidden_dim=160,
        kernel_size=7,
        transformer_layers=4,
        per_horizon_heads=True,
    )
    singles(
        "calib_only",
        "representative",
        "calib_only",
        "Calibration-only forecasting quantifies how much of the primary MAE is explained by early FMS and time context alone.",
        "Lower-complexity calibration usage baseline.",
        learning_rate=8e-4,
        dropout=0.1,
        final_selection_eligible=False,
    )
    singles(
        "recent_motion_tcn",
        "representative",
        "recent10_tcn",
        "Motion-only recent TCN separates motion signal value from calibration/start-FMS dependence.",
        "Motion-only allowed-input baseline; not the preferred final deployment model.",
        learning_rate=8e-4,
        dropout=0.1,
        final_selection_eligible=False,
    )

    # Planned refinements. The main loop runs these only after representative coverage progresses.
    for family, model, kwargs in [
        ("lcsa_shared", "lc_sa_tcnformer", {"d_model": 96, "dropout": 0.03, "learning_rate": 3e-4}),
        ("lcsa_per_horizon_heads", "lc_sa_tcnformer", {"d_model": 96, "dropout": 0.03, "learning_rate": 3e-4, "per_horizon_heads": True}),
        ("gated_fusion", "gated_fusion", {"d_model": 96, "hidden_dim": 192, "dropout": 0.05, "branch_dropout": 0.1, "pooling": "attention", "predict_delta_from_anchor": True, "delta_scale": 1.0, "learning_rate": 5e-4}),
        ("recent_tcn_summary_calib", "recent_tcn_summary_calib", {"d_model": 96, "hidden_dim": 192, "dropout": 0.05, "pooling": "attention", "predict_delta_from_anchor": True, "delta_scale": 1.0, "learning_rate": 5e-4}),
        ("anchor_delta_gru", "anchor_delta_gru", {"hidden_dim": 192, "gru_layers": 2, "dropout": 0.03, "predict_delta_from_anchor": True, "delta_scale": 1.0, "learning_rate": 5e-4}),
        ("anchor_delta_mlp", "anchor_delta_mlp", {"hidden_dim": 192, "mlp_layers": (192, 128, 64), "dropout": 0.03, "predict_delta_from_anchor": True, "delta_scale": 1.0, "learning_rate": 5e-4}),
    ]:
        mh(
            family,
            "refine_capacity",
            model,
            "After representative coverage, a larger but still regularized variant tests whether the earlier failures are underfit rather than input-limited.",
            "Promising-family capacity/regularization refinement pool.",
            epochs=60,
            **kwargs,
        )

    mh(
        "lcsa_static",
        "refine_static_regularized",
        "lc_sa_tcnformer",
        "If static helps susceptibility level, stronger regularization checks whether gains survive without identity-like overfit.",
        "Static branch refinement, still limited to Age/Gender/MSSQ.",
        epochs=60,
        use_static=True,
        learning_rate=3e-4,
        dropout=0.12,
        d_model=96,
        per_horizon_heads=True,
    )
    mh(
        "gated_fusion",
        "high_fms_weight",
        "gated_fusion",
        "High-FMS underprediction is a known failure; moderate target weighting tests whether it improves severe-sickness windows without leakage.",
        "Loss branch directed at high-FMS collapse.",
        epochs=60,
        predict_delta_from_anchor=True,
        learning_rate=5e-4,
        d_model=96,
        hidden_dim=192,
        dropout=0.05,
        pooling="attention",
        high_target_weight=0.4,
        high_target_threshold=0.55,
        delta_scale=1.0,
    )
    mh(
        "lcsa_per_horizon_heads",
        "recent45",
        "lc_sa_tcnformer",
        "A longer recent motion window may help h=10/15 if the representative run fails mainly at longer horizons.",
        "Window-length branch for long-horizon degradation.",
        epochs=60,
        learning_rate=3e-4,
        dropout=0.05,
        d_model=96,
        per_horizon_heads=True,
        recent_window_seconds=45.0,
    )
    mh(
        "lcsa_per_horizon_heads",
        "calib120",
        "lc_sa_tcnformer",
        "Longer calibration may improve susceptibility context if the baseline underuses early FMS trajectory.",
        "Calibration-length branch with the same corrected start_only policy.",
        epochs=60,
        learning_rate=3e-4,
        dropout=0.05,
        d_model=96,
        per_horizon_heads=True,
        calibration_seconds=120.0,
    )
    mh(
        "recent_tcn_summary_calib",
        "l1_delta",
        "recent_tcn_summary_calib",
        "L1 loss with delta anchoring tests whether mean-regression is worsened by smooth L1 around large errors.",
        "Optimization/loss branch after temporal-family coverage.",
        epochs=60,
        predict_delta_from_anchor=True,
        learning_rate=5e-4,
        d_model=96,
        hidden_dim=192,
        pooling="attention",
        loss_type="l1",
        dropout=0.05,
        delta_scale=1.0,
    )

    # Diagnostics are never primary-selection drivers.
    singles(
        "diagnostic_h2p5",
        "best_shape_placeholder",
        "lc_sa_tcnformer",
        "h=2.5 is a valid short-horizon forecasting diagnostic and is excluded from primary model selection.",
        "Required auxiliary horizon recording.",
        horizons=(2.5,),
        epochs=24,
        learning_rate=5e-4,
        dropout=0.05,
        final_selection_eligible=False,
    )
    singles(
        "diagnostic_h1",
        "best_shape_placeholder",
        "lc_sa_tcnformer",
        "h=1 is a lower-bound/sanity diagnostic and is excluded from primary model selection.",
        "Required diagnostic horizon recording.",
        horizons=(1.0,),
        epochs=24,
        learning_rate=5e-4,
        dropout=0.05,
        final_selection_eligible=False,
    )
    for raw_variant in adaptive_variant_dicts():
        variant = dict(raw_variant)
        branch = variant.pop("branch")
        mh(
            "lcsa_per_horizon_heads",
            branch,
            "lc_sa_tcnformer",
            "Adaptive refinement of the current best validation family to test seed/optimization/loss robustness before final lock.",
            "Generated because the user requested a long adaptive search and this is the best validation architecture family.",
            epochs=80,
            per_horizon_heads=True,
            **variant,
        )
    return specs


def train_command(args: argparse.Namespace, spec: Spec) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        args.config,
        "--runs_dir",
        str(args.output_dir),
        "--model",
        spec.model,
        "--run_name",
        spec.run_name,
        "--split_file",
        args.split_file,
        "--seed",
        str(spec.seed),
        "--epochs",
        str(spec.epochs),
        "--patience",
        str(spec.patience),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--learning_rate",
        f"{spec.learning_rate:g}",
        "--weight_decay",
        f"{spec.weight_decay:g}",
        "--loss_type",
        spec.loss_type,
        "--loss_mode",
        spec.loss_mode,
        "--calibration_seconds",
        f"{spec.calibration_seconds:g}",
        "--recent_window_seconds",
        f"{spec.recent_window_seconds:g}",
        "--horizon_seconds",
        f"{spec.horizon_seconds:g}",
        "--anchor_mode",
        spec.anchor_mode,
        "--anchor_interval_seconds",
        f"{spec.anchor_interval_seconds:g}",
        "--fms_context_mode",
        spec.fms_context_mode,
        "--d_model",
        str(spec.d_model),
        "--hidden_dim",
        str(spec.hidden_dim),
        "--kernel_size",
        str(spec.kernel_size),
        "--dropout",
        f"{spec.dropout:g}",
        "--transformer_layers",
        str(spec.transformer_layers),
        "--transformer_heads",
        str(spec.transformer_heads),
        "--transformer_ff_dim",
        str(spec.transformer_ff_dim),
        "--pooling",
        spec.pooling,
        "--gru_layers",
        str(spec.gru_layers),
        "--branch_dropout",
        f"{spec.branch_dropout:g}",
        "--anchor_dropout",
        f"{spec.anchor_dropout:g}",
        "--delta_scale",
        f"{spec.delta_scale:g}",
        "--recent_encoder",
        spec.recent_encoder,
        "--recent_attn_heads",
        str(spec.recent_attn_heads),
        "--recent_attn_layers",
        str(spec.recent_attn_layers),
        "--recent_attn_dropout",
        f"{spec.recent_attn_dropout:g}",
        "--trend_weight",
        f"{spec.trend_weight:g}",
        "--change_weight",
        f"{spec.change_weight:g}",
        "--high_target_weight",
        f"{spec.high_target_weight:g}",
        "--high_target_threshold",
        f"{spec.high_target_threshold:g}",
        "--high_fms_threshold",
        "10.0",
        "--no_test_eval",
        "--skip_existing",
    ]
    if spec.calib_dilations:
        cmd.append("--calib_dilations")
        cmd.extend(str(int(v)) for v in spec.calib_dilations)
    if spec.recent_dilations != "auto":
        cmd.extend(["--recent_dilations", spec.recent_dilations])
    if spec.mlp_layers:
        cmd.append("--mlp_layers")
        cmd.extend(str(int(v)) for v in spec.mlp_layers)
    if spec.predict_delta_from_anchor:
        cmd.append("--predict_delta_from_anchor")
    if spec.multi_horizon:
        cmd.append("--multi_horizon")
        cmd.append("--horizon_set")
        cmd.extend(f"{float(v):g}" for v in spec.horizon_set)
    if spec.per_horizon_heads:
        cmd.append("--per_horizon_heads")
    if spec.use_static:
        cmd.extend(["--use_static", "--static_features", "age", "gender", "mssq"])
    else:
        cmd.append("--no_static")
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def eval_command(args: argparse.Namespace, spec: Spec, split: str) -> List[str]:
    run_dir = Path(args.output_dir) / spec.run_name
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        split,
        "--split_file",
        args.split_file,
        "--batch_size",
        str(args.batch_size),
        "--calibration_seconds",
        f"{spec.calibration_seconds:g}",
        "--recent_window_seconds",
        f"{spec.recent_window_seconds:g}",
        "--horizon_seconds",
        f"{spec.horizon_seconds:g}",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def completed(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists()


def candidate_prefix(spec: Spec) -> str:
    return f"v2_{safe(spec.architecture_family)}_{safe(spec.branch)}_"


def spec_blocked_by_manifest(args: argparse.Namespace, spec: Spec) -> Optional[str]:
    for row in read_csv(Path(args.output_dir) / "resume_manifest.csv"):
        status = row.get("status", "")
        run_name = row.get("run_name", "")
        family = row.get("architecture_family", "")
        if status not in BLOCKING_STATUSES:
            continue
        if run_name == spec.run_name or run_name.startswith(candidate_prefix(spec)):
            return f"{status}:{run_name}:{row.get('failure_reason', '')}"
        if spec.model == "coff_lstm" and family.startswith("coff_lstm"):
            return f"{status}:{run_name}:coff_lstm branch skipped after runtime blocker"
    return None


def run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    return read_json(path)


def extract_epochs_completed(metrics: Mapping[str, Any]) -> int:
    history = metrics.get("metrics", {}).get("history", [])
    return len(history) if isinstance(history, list) else 0


def metric_for_run(run_dir: Path, spec: Spec) -> Dict[str, Any]:
    payload = run_metrics(run_dir)
    if not payload:
        return {}
    metrics = payload.get("metrics", {})
    val = metrics.get("best_val_metrics", {})
    out: Dict[str, Any] = {
        "best_epoch": metrics.get("best_epoch", ""),
        "epochs_completed": extract_epochs_completed(payload),
        "val_rmse": first_metric(val, "rmse"),
        "val_n": first_metric(val, "n"),
    }
    h_scores: Dict[float, float] = {}
    if spec.multi_horizon:
        by_h = val.get("by_horizon", {})
        if isinstance(by_h, Mapping):
            for key, item in by_h.items():
                if isinstance(item, Mapping):
                    h_scores[float(key)] = as_float(item.get("mae"))
    else:
        h_scores[float(spec.horizon_seconds)] = as_float(val.get("mae"))
    vals = [h_scores.get(h, math.nan) for h in PRIMARY_HORIZONS]
    primary = sum(vals) / len(vals) if all(math.isfinite(v) for v in vals) else math.nan
    out.update(
        {
            "by_horizon": h_scores,
            "val_mae": primary if math.isfinite(primary) else first_metric(val, "mae"),
            "primary_mean": primary,
            "h5": h_scores.get(5.0, ""),
            "h10": h_scores.get(10.0, ""),
            "h15": h_scores.get(15.0, ""),
            "h2p5": h_scores.get(2.5, ""),
            "h1": h_scores.get(1.0, ""),
        }
    )
    return out


def update_manifest(args: argparse.Namespace, spec: Spec, status: str, **extra: Any) -> None:
    out = Path(args.output_dir)
    rows = read_csv(out / "resume_manifest.csv")
    existing = {row.get("run_name", ""): row for row in rows if row.get("run_name")}
    run_dir = out / spec.run_name
    row = existing.get(spec.run_name, {})
    row.update(
        {
            "run_name": spec.run_name,
            "status": status,
            "architecture_family": spec.architecture_family,
            "command": " ".join(train_command(args, spec)),
            "config_path": args.config,
            "checkpoint_path": rel(run_dir / "best.pt"),
            "metrics_path": rel(run_dir / "metrics.json"),
            "prediction_csv_path": rel(run_dir / "val_predictions.csv"),
            "train_prediction_csv_path": rel(run_dir / "eval_train/train_predictions.csv"),
            "plot_dir": rel(run_dir / "plots"),
            "max_epochs_planned": spec.epochs,
            "resume_action": "skip_completed" if status == "completed" else row.get("resume_action", ""),
        }
    )
    row.update({key: value for key, value in extra.items() if value is not None})
    existing[spec.run_name] = row
    write_csv(out / "resume_manifest.csv", list(existing.values()), MANIFEST_FIELDS)


def init_docs(args: argparse.Namespace, specs: Sequence[Spec]) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_doc = ROOT / "docs/codex/test.md"
    test_doc_state = "present" if test_doc.exists() else "absent"
    families = sorted({spec.architecture_family for spec in specs if not spec.architecture_family.startswith("diagnostic_")})
    write_text(
        out / "input_contract.md",
        "\n".join(
            [
                "# Input Contract",
                "",
                f"- Policy file: `{rel(args.policy_file)}`",
                f"- docs/codex/test.md: {test_doc_state}",
                "- FULL_TRAINING_ALLOWED: true",
                "- Output directory: `runs/goal_mae_search_v2/`",
                "- Main track: `fms_context_mode=start_only`, `anchor_mode=none`, `anchor_interval_seconds=0`.",
                "- Allowed post-calibration FMS: exactly the recent motion window start FMS and its missing-value fallback.",
                "- CSV naming: `start_fms_index/start_fms_time/start_fms_value`; `anchor_*` is only a backward-compatible alias.",
                "- Forbidden FMS inputs: target/current/future FMS, recent dense FMS sequence, sparse_observed, recent_start_observed, calibration_end anchor, sparse_anchor.",
                "- Allowed static covariates: Age, Gender, MSSQ only.",
                "- Forbidden identity features: participant_id/session_id/condition_id/trial_id/experiment_id/file-derived identity.",
                "- Selection metric: validation MAE mean over h=5, h=10, h=15.",
                "- h=2.5 is auxiliary; h=1 is diagnostic only.",
                "- Adaptive search outputs train/val metrics and train/val predictions only. Test is final-only after `FINAL_SELECTION_LOCK.md`.",
                "- Sampling interval: 0.5s; default calibration=90s; default recent window=30s.",
                "- New architecture trials (`lcsa_cross_attn`, `gru_state_mixer`, `motion_conv_mixer`) use the same dataloader batch contract as prior runs.",
                "- New architecture trials do not receive participant/session/condition/trial/experiment/file identity, target/current/future FMS, dense recent FMS, or any additional metadata beyond the existing allowed inputs.",
                "- New architecture trials may transform allowed inputs internally, but may not add external information or change evaluation code.",
                "",
                "## Architecture Families Planned",
                "",
                *[f"- {family}" for family in families],
                "",
                "## Code Inventory",
                "",
                "- Dataset/windowing: `src/densefms_forecast/data.py`",
                "- Models: `src/densefms_forecast/model.py`",
                "- Training/metrics/prediction CSV: `src/densefms_forecast/train.py`",
                "- Evaluation: `src/densefms_forecast/evaluate.py`",
                "- Losses: `src/densefms_forecast/losses.py`",
                "- Sanity tests: `scripts/run_densefms_sanity_tests.py`",
                "- Search runner: `scripts/run_goal_mae_search_v2.py`",
            ]
        )
        + "\n",
    )
    write_text(
        out / "leakage_audit.md",
        "\n".join(
            [
                "# Leakage Audit",
                "",
                "Status: initialized before v2 search.",
                "",
                "- Train/validation/test split is loaded from `artifacts/densefms_split_seed42.json` when available.",
                "- Static scaler is fit on train sessions only by `fit_static_scaler`; val/test use transform only.",
                "- Target shift is implemented by `future_sequence_targets`: target index = current index + horizon_steps.",
                "- Calibration input is limited to first calibration_steps by `calibration_context_fms`.",
                "- Recent motion windows end at current index t and do not include motion after t.",
                "- start_only FMS is gathered at `t - recent_window_steps + 1` with latest finite fallback at or before nominal start.",
                "- `anchor_mode=none`, `anchor_interval_seconds=0`, and `fms_context_mode=start_only` are forced in all main-track train commands.",
                "- `--no_test_eval` is forced in all adaptive train commands.",
                "- Normal adaptive runner calls `evaluate.py` only for `--split train`; it never calls `--split test` before final lock.",
                "- Architecture forward sanity results are recorded in `architecture_sanity.csv`.",
                "- Lightweight repository sanity results are recorded in `sanity_tests.log`.",
            ]
        )
        + "\n",
    )
    write_text(
        out / "baseline_summary.md",
        "\n".join(
            [
                "# Baseline Summary",
                "",
                f"- v1 validation baseline primary mean (allowed as validation context only): {V1_BASELINE_PRIMARY:.4f}",
                "- v1 test files/results are not opened or used for v2 selection.",
                "- v2 representative baseline: `lcsa_shared/representative`, multi-horizon h=5/10/15, no static.",
                "- v2 static baseline is recorded separately as `lcsa_static/representative`.",
            ]
        )
        + "\n",
    )
    for path, fields in [
        (out / "resume_manifest.csv", MANIFEST_FIELDS),
        (out / "experiment_log.csv", EXPERIMENT_FIELDS),
        (out / "leaderboard.csv", LEADERBOARD_FIELDS),
    ]:
        if not path.exists():
            write_csv(path, [], fields)
    for path, text in [
        (out / "experiment_log.md", "# Experiment Log\n\nNo completed runs yet.\n"),
        (out / "leaderboard.md", "# Leaderboard\n\nNo completed primary family yet.\n"),
        (out / "best_model_summary.md", "# Best Model Summary\n\nNo validation-selected candidate yet.\n"),
        (out / "RUN_STATE.md", "# RUN_STATE\n\nStatus: initialized.\n"),
    ]:
        if not path.exists():
            write_text(path, text)


def run_subprocess(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=out_f, stderr=err_f, text=True)
    return int(proc.returncode)


def run_architecture_sanity(args: argparse.Namespace, specs: Sequence[Spec]) -> None:
    out = Path(args.output_dir)
    rows: List[Dict[str, Any]] = []
    family_reps: Dict[str, Spec] = {}
    for spec in specs:
        family_reps.setdefault(spec.architecture_family, spec)
    sys.path.insert(0, str(ROOT))
    import torch
    from src.densefms_forecast.model import build_model

    for family, spec in family_reps.items():
        try:
            horizon_steps = max(1, int(round(spec.horizon_seconds / 0.5)))
            model = build_model(
                spec.model,
                head_dim=6,
                calibration_steps=int(round(spec.calibration_seconds / 0.5)),
                horizon_steps=horizon_steps,
                recent_steps=int(round(spec.recent_window_seconds / 0.5)),
                sampling_interval=0.5,
                horizon_seconds=spec.horizon_seconds,
                d_model=spec.d_model,
                hidden_dim=spec.hidden_dim,
                kernel_size=spec.kernel_size,
                dropout=spec.dropout,
                transformer_layers=spec.transformer_layers,
                transformer_heads=spec.transformer_heads,
                transformer_ff_dim=spec.transformer_ff_dim,
                pooling=spec.pooling,
                anchor_mode=spec.anchor_mode,
                anchor_interval_seconds=spec.anchor_interval_seconds,
                fms_context_mode=spec.fms_context_mode,
                predict_delta_from_anchor=spec.predict_delta_from_anchor,
                multi_horizon=spec.multi_horizon,
                horizon_set=list(spec.horizon_set) if spec.horizon_set else None,
                per_horizon_heads=spec.per_horizon_heads,
                use_static=spec.use_static,
                static_dim=5 if spec.use_static else 4,
                mlp_layers=list(spec.mlp_layers) if spec.mlp_layers else None,
                gru_layers=spec.gru_layers,
                branch_dropout=spec.branch_dropout,
                anchor_dropout=spec.anchor_dropout,
                delta_scale=spec.delta_scale,
                recent_encoder=spec.recent_encoder,
                recent_attn_layers=spec.recent_attn_layers,
                recent_attn_heads=spec.recent_attn_heads,
                recent_attn_dropout=spec.recent_attn_dropout,
            )
            model.eval()
            steps = max(int(round(spec.calibration_seconds / 0.5)) + int(round(spec.recent_window_seconds / 0.5)) + 80, 260)
            head = torch.randn(2, steps, 6)
            fms = torch.rand(2, steps)
            lengths = torch.tensor([steps, steps - 3], dtype=torch.long)
            static = torch.randn(2, 5) if spec.use_static else None
            with torch.no_grad():
                output = model(head, fms, lengths, static=static)
            future = output.get("future")
            if future is None:
                raise AssertionError("model output has no future tensor")
            expected_last_dim = len(spec.horizon_set) if spec.multi_horizon else None
            if future.shape[0] != 2 or future.shape[1] <= 0:
                raise AssertionError(f"invalid future shape {tuple(future.shape)}")
            if spec.multi_horizon and future.ndim != 3:
                raise AssertionError(f"expected multi-horizon [B,T,H], got {tuple(future.shape)}")
            if spec.multi_horizon and expected_last_dim and future.shape[-1] != expected_last_dim:
                raise AssertionError(f"expected {expected_last_dim} horizons, got {future.shape[-1]}")
            if spec.fms_context_mode != "start_only" or spec.anchor_mode != "none":
                raise AssertionError("main-track policy flags not set")
            rows.append(
                {
                    "architecture_family": family,
                    "model_type": spec.model,
                    "status": "pass",
                    "future_shape": tuple(future.shape),
                    "fms_context_mode": spec.fms_context_mode,
                    "anchor_mode": spec.anchor_mode,
                    "note": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "architecture_family": family,
                    "model_type": spec.model,
                    "status": "fail",
                    "future_shape": "",
                    "fms_context_mode": spec.fms_context_mode,
                    "anchor_mode": spec.anchor_mode,
                    "note": repr(exc),
                }
            )
    fields = ["architecture_family", "model_type", "status", "future_shape", "fms_context_mode", "anchor_mode", "note"]
    write_csv(out / "architecture_sanity.csv", rows, fields)
    write_text(out / "architecture_sanity.md", "# Architecture Sanity\n\n" + markdown_table(rows, fields))


def run_repo_sanity(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    log = out / "sanity_tests.log"
    err = out / "sanity_tests.stderr.log"
    if log.exists() and "All sanity tests passed" in log.read_text(encoding="utf-8", errors="ignore"):
        return
    cmd = [sys.executable, "-u", "scripts/run_densefms_sanity_tests.py"]
    code = run_subprocess(cmd, ROOT, log, err)
    status = "pass" if code == 0 else f"fail_{code}"
    write_text(out / "sanity_tests_status.md", f"# Sanity Test Status\n\n- command: `{' '.join(cmd)}`\n- status: {status}\n")


def guard_no_test_outputs(args: argparse.Namespace) -> Optional[str]:
    out = Path(args.output_dir)
    if (out / "FINAL_SELECTION_LOCK.md").exists():
        return None
    forbidden: List[str] = []
    for path in out.rglob("*"):
        low = path.as_posix().lower()
        if "eval_test" in low or "test_predictions" in low or "test_metrics" in low:
            forbidden.append(rel(path))
    if forbidden:
        return "Forbidden pre-lock test outputs found: " + "; ".join(forbidden[:20])
    return None


def run_one(args: argparse.Namespace, spec: Spec, goal_start: float) -> None:
    out = Path(args.output_dir)
    run_dir = out / spec.run_name
    if completed(run_dir):
        m = metric_for_run(run_dir, spec)
        update_manifest(
            args,
            spec,
            "completed",
            best_epoch=m.get("best_epoch"),
            epochs_completed=m.get("epochs_completed"),
            best_val_mae=m.get("val_mae"),
            resume_action="skip_completed",
        )
        return
    blocked = spec_blocked_by_manifest(args, spec)
    if blocked:
        print(f"[{now_iso()}] SKIP blocked candidate {spec.run_name}: {blocked}", flush=True)
        return
    issue = guard_no_test_outputs(args)
    if issue:
        update_manifest(args, spec, "blocked", failure_reason=issue, resume_action="remove_forbidden_test_outputs_or_start_new_output_dir")
        return

    run_start = time.time()
    print(f"[{now_iso()}] START {spec.run_name} ({spec.architecture_family}/{spec.branch}, epochs={spec.epochs})", flush=True)
    update_manifest(args, spec, "started", start_time=now_iso(), elapsed_seconds=f"{time.time() - goal_start:.1f}")
    update_manifest(args, spec, "running", elapsed_seconds=f"{time.time() - goal_start:.1f}")
    write_run_state(args, build_specs(), goal_start, active=spec)
    logs = out / "logs"
    train_code = run_subprocess(
        train_command(args, spec),
        ROOT,
        logs / f"{spec.run_name}.stdout.log",
        logs / f"{spec.run_name}.stderr.log",
    )
    if train_code != 0:
        update_manifest(
            args,
            spec,
            "failed",
            end_time=now_iso(),
            elapsed_seconds=f"{time.time() - goal_start:.1f}",
            failure_reason=f"train_exit_{train_code}",
            resume_action="inspect_logs",
        )
        print(f"[{now_iso()}] FAIL train {spec.run_name} exit={train_code}", flush=True)
        return
    train_eval_code = run_subprocess(
        eval_command(args, spec, "train"),
        ROOT,
        logs / f"{spec.run_name}.eval_train.stdout.log",
        logs / f"{spec.run_name}.eval_train.stderr.log",
    )
    if train_eval_code != 0:
        update_manifest(
            args,
            spec,
            "failed",
            end_time=now_iso(),
            elapsed_seconds=f"{time.time() - goal_start:.1f}",
            failure_reason=f"train_eval_exit_{train_eval_code}",
            resume_action="inspect_logs",
        )
        print(f"[{now_iso()}] FAIL eval_train {spec.run_name} exit={train_eval_code}", flush=True)
        return
    m = metric_for_run(run_dir, spec)
    update_manifest(
        args,
        spec,
        "completed",
        end_time=now_iso(),
        elapsed_seconds=f"{time.time() - goal_start:.1f}",
        epochs_completed=m.get("epochs_completed"),
        best_epoch=m.get("best_epoch"),
        best_val_mae=m.get("val_mae"),
        resume_action="skip_completed",
    )
    print(
        f"[{now_iso()}] DONE {spec.run_name} primary={m.get('primary_mean', '')} "
        f"val_mae={m.get('val_mae', '')} seconds={time.time() - run_start:.1f}",
        flush=True,
    )


def family_rows(args: argparse.Namespace, specs: Sequence[Spec]) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Spec]] = {}
    by_candidate: Dict[tuple[str, str], List[Spec]] = {}
    for spec in specs:
        by_family.setdefault(spec.architecture_family, []).append(spec)
        by_candidate.setdefault((spec.architecture_family, spec.branch), []).append(spec)
    sanity = {row.get("architecture_family"): row.get("status") for row in read_csv(Path(args.output_dir) / "architecture_sanity.csv")}
    manifest = {row.get("run_name"): row for row in read_csv(Path(args.output_dir) / "resume_manifest.csv")}
    families = [family for family in by_family if not family.startswith("diagnostic_")]
    family_index = {family: idx + 1 for idx, family in enumerate(families)}
    rows: List[Dict[str, Any]] = []
    for (family, branch), items in by_candidate.items():
        representative = items[0]
        h_scores: Dict[float, float] = {}
        member_runs: List[str] = []
        ckpts: List[str] = []
        preds: List[str] = []
        plots: List[str] = []
        epochs_completed = 0
        failures: List[str] = []
        complete_count = 0
        for spec in items:
            run_dir = Path(args.output_dir) / spec.run_name
            row_status = manifest.get(spec.run_name, {}).get("status", "pending")
            if row_status in BLOCKING_STATUSES:
                failures.append(f"{spec.run_name}:{manifest.get(spec.run_name, {}).get('failure_reason', '')}")
            metrics = metric_for_run(run_dir, spec)
            if metrics:
                complete_count += 1
                member_runs.append(spec.run_name)
                h_scores.update(metrics.get("by_horizon", {}))
                epochs_completed += int(metrics.get("epochs_completed") or 0)
                if (run_dir / "best.pt").exists():
                    ckpts.append(rel(run_dir / "best.pt"))
                if (run_dir / "val_predictions.csv").exists():
                    preds.append(rel(run_dir / "val_predictions.csv"))
                if (run_dir / "plots").exists():
                    plots.append(rel(run_dir / "plots"))
        primary_vals = [h_scores.get(h, math.nan) for h in PRIMARY_HORIZONS]
        primary_mean = sum(primary_vals) / len(primary_vals) if all(math.isfinite(v) for v in primary_vals) else math.nan
        if all(math.isfinite(v) for v in primary_vals):
            status = "completed"
        elif family.startswith("diagnostic_") and h_scores:
            status = "diagnostic_completed"
        elif complete_count:
            status = f"partial_{complete_count}/{len(items)}"
        elif failures:
            status = "failed"
        else:
            status = "pending"
        improvement = ""
        if math.isfinite(primary_mean):
            improvement = 100.0 * (V1_BASELINE_PRIMARY - primary_mean) / V1_BASELINE_PRIMARY
        rows.append(
            {
                "status": status,
                "architecture_family": family,
                "architecture_hypothesis": representative.hypothesis,
                "family_index": family_index.get(family, ""),
                "family_count": len(families),
                "model_type": representative.model,
                "fms_context_mode": representative.fms_context_mode,
                "anchor_mode": representative.anchor_mode,
                "anchor_interval_seconds": representative.anchor_interval_seconds,
                "use_static": representative.use_static,
                "static_feature_set": "age+gender+mssq" if representative.use_static else "none",
                "predict_delta_from_anchor": representative.predict_delta_from_anchor,
                "calibration_seconds": representative.calibration_seconds,
                "recent_window_seconds": representative.recent_window_seconds,
                "loss_type": representative.loss_type,
                "loss_mode": representative.loss_mode,
                "optimizer": "AdamW",
                "learning_rate": representative.learning_rate,
                "weight_decay": representative.weight_decay,
                "dropout": representative.dropout,
                "d_model": representative.d_model,
                "hidden_dim": representative.hidden_dim,
                "pooling": representative.pooling,
                "max_epochs_planned": sum(s.epochs for s in items),
                "epochs_completed": epochs_completed,
                "h5_val_mae": h_scores.get(5.0, ""),
                "h10_val_mae": h_scores.get(10.0, ""),
                "h15_val_mae": h_scores.get(15.0, ""),
                "mean_val_mae_h5_h10_h15": primary_mean if math.isfinite(primary_mean) else "",
                "h2p5_val_mae": h_scores.get(2.5, ""),
                "h1_diagnostic_val_mae": h_scores.get(1.0, ""),
                "improvement_vs_v1_baseline_percent": improvement,
                "member_runs": ";".join(member_runs),
                "branch": branch,
                "selection_reason": representative.selection_reason,
                "failure_reason": "; ".join(failures),
                "sanity_status": sanity.get(family, ""),
                "leakage_status": "policy_guarded_start_only_no_test_prelock",
                "checkpoint_paths": ";".join(ckpts),
                "prediction_csv_paths": ";".join(preds),
                "plot_dirs": ";".join(plots),
                "final_selection_eligible": all(s.final_selection_eligible for s in items),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("final_selection_eligible")),
            as_float(row.get("mean_val_mae_h5_h10_h15"), math.inf),
        ),
    )
    rank = 1
    for row in ranked:
        if math.isfinite(as_float(row.get("mean_val_mae_h5_h10_h15"), math.inf)) and row.get("final_selection_eligible"):
            row["rank"] = rank
            rank += 1
        else:
            row["rank"] = ""
    return ranked


def experiment_rows(args: argparse.Namespace, specs: Sequence[Spec]) -> List[Dict[str, Any]]:
    manifest = {row.get("run_name"): row for row in read_csv(Path(args.output_dir) / "resume_manifest.csv")}
    sanity = {row.get("architecture_family"): row.get("status") for row in read_csv(Path(args.output_dir) / "architecture_sanity.csv")}
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        run_dir = Path(args.output_dir) / spec.run_name
        metrics = metric_for_run(run_dir, spec)
        row_status = manifest.get(spec.run_name, {}).get("status", "pending")
        rows.append(
            {
                "run_name": spec.run_name,
                "status": row_status,
                "architecture_family": spec.architecture_family,
                "architecture_hypothesis": spec.hypothesis,
                "model_type": spec.model,
                "fms_context_mode": spec.fms_context_mode,
                "anchor_mode": spec.anchor_mode,
                "anchor_interval_seconds": spec.anchor_interval_seconds,
                "use_static": spec.use_static,
                "static_feature_set": "age+gender+mssq" if spec.use_static else "none",
                "forbidden_identity_features": "none",
                "recent_start_observed": False,
                "sparse_observed": False,
                "predict_delta_from_anchor": spec.predict_delta_from_anchor,
                "calibration_seconds": spec.calibration_seconds,
                "recent_window_seconds": spec.recent_window_seconds,
                "horizon_seconds": "multi" if spec.multi_horizon else spec.horizon_seconds,
                "multi_horizon": spec.multi_horizon,
                "horizon_set": ";".join(f"{v:g}" for v in spec.horizon_set),
                "loss_type": spec.loss_type,
                "loss_mode": spec.loss_mode,
                "optimizer": "AdamW",
                "learning_rate": spec.learning_rate,
                "weight_decay": spec.weight_decay,
                "dropout": spec.dropout,
                "d_model": spec.d_model,
                "hidden_dim": spec.hidden_dim,
                "pooling": spec.pooling,
                "seed": spec.seed,
                "max_epochs_planned": spec.epochs,
                "epochs_completed": metrics.get("epochs_completed", manifest.get(spec.run_name, {}).get("epochs_completed", "")),
                "best_epoch": metrics.get("best_epoch", manifest.get(spec.run_name, {}).get("best_epoch", "")),
                "val_mae": metrics.get("val_mae", manifest.get(spec.run_name, {}).get("best_val_mae", "")),
                "val_rmse": metrics.get("val_rmse", ""),
                "val_n": metrics.get("val_n", ""),
                "h5_val_mae": metrics.get("h5", ""),
                "h10_val_mae": metrics.get("h10", ""),
                "h15_val_mae": metrics.get("h15", ""),
                "h2p5_val_mae": metrics.get("h2p5", ""),
                "h1_diagnostic_val_mae": metrics.get("h1", ""),
                "primary_mean_val_mae": metrics.get("primary_mean", ""),
                "selection_reason": spec.selection_reason,
                "failure_reason": manifest.get(spec.run_name, {}).get("failure_reason", ""),
                "sanity_status": sanity.get(spec.architecture_family, ""),
                "leakage_status": "start_only_no_test_prelock",
                "checkpoint_path": rel(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
                "metrics_path": rel(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else "",
                "prediction_csv_path": rel(run_dir / "val_predictions.csv") if (run_dir / "val_predictions.csv").exists() else "",
                "train_prediction_csv_path": rel(run_dir / "eval_train/train_predictions.csv") if (run_dir / "eval_train/train_predictions.csv").exists() else "",
                "plot_dir": rel(run_dir / "plots") if (run_dir / "plots").exists() else "",
            }
        )
    return rows


def best_family(args: argparse.Namespace, specs: Sequence[Spec]) -> Optional[Dict[str, Any]]:
    rows = family_rows(args, specs)
    return next((row for row in rows if row.get("rank") == 1), None)


def completed_primary_family_count(args: argparse.Namespace, specs: Sequence[Spec]) -> int:
    rows = family_rows(args, specs)
    completed_families = {
        str(row.get("architecture_family"))
        for row in rows
        if row.get("status") == "completed" and not str(row.get("architecture_family")).startswith("diagnostic_")
    }
    return len(completed_families)


def write_summaries(args: argparse.Namespace, specs: Sequence[Spec], goal_start: float) -> None:
    out = Path(args.output_dir)
    exp = experiment_rows(args, specs)
    board = family_rows(args, specs)
    write_csv(out / "experiment_log.csv", exp, EXPERIMENT_FIELDS)
    write_csv(out / "leaderboard.csv", board, LEADERBOARD_FIELDS)
    write_text(out / "experiment_log.md", "# Experiment Log\n\n" + markdown_table(exp, EXPERIMENT_FIELDS))
    write_text(out / "leaderboard.md", "# Leaderboard\n\n" + markdown_table(board, LEADERBOARD_FIELDS))
    best = best_family(args, specs)
    lines = ["# Best Model Summary", "", "- Selection basis: validation only.", "- h=1 diagnostic and final test are excluded from model selection."]
    if best:
        lines.extend(
            [
                f"- Primary validation family: {best.get('architecture_family')}",
                f"- Mean validation MAE h=5/10/15: {best.get('mean_val_mae_h5_h10_h15')}",
                f"- h=5 validation MAE: {best.get('h5_val_mae')}",
                f"- h=10 validation MAE: {best.get('h10_val_mae')}",
                f"- h=15 validation MAE: {best.get('h15_val_mae')}",
                f"- Improvement vs v1 validation baseline: {best.get('improvement_vs_v1_baseline_percent')}%",
                f"- Static usage: {best.get('use_static')} ({best.get('static_feature_set')})",
                f"- Member runs: {best.get('member_runs')}",
                f"- Checkpoints: {best.get('checkpoint_paths')}",
            ]
        )
    else:
        lines.append("- No complete final-eligible h=5/10/15 family yet.")
    h2 = next((row for row in board if row.get("architecture_family") == "diagnostic_h2p5"), None)
    h1 = next((row for row in board if row.get("architecture_family") == "diagnostic_h1"), None)
    if h2 or h1:
        lines.append("")
        lines.append("## Diagnostic Horizons")
        if h2:
            lines.append(f"- h=2.5 validation MAE: {h2.get('h2p5_val_mae')} ({h2.get('member_runs')})")
        if h1:
            lines.append(f"- h=1 diagnostic validation MAE: {h1.get('h1_diagnostic_val_mae')} ({h1.get('member_runs')})")
    final_rows = read_csv(out / "final_test_metrics.csv")
    if final_rows:
        lines.append("")
        lines.append("## Final Test")
        lines.append("- Final test was generated only after `FINAL_SELECTION_LOCK.md`.")
        for row in final_rows:
            lines.append(
                f"- {row.get('run_name')}: MAE={row.get('test_mae')}, RMSE={row.get('test_rmse')}, R2={row.get('test_r2')}, n={row.get('test_n')}"
            )
    write_text(out / "best_model_summary.md", "\n".join(lines) + "\n")
    write_run_state(args, specs, goal_start)


def write_run_state(args: argparse.Namespace, specs: Sequence[Spec], goal_start: float, active: Optional[Spec] = None) -> None:
    out = Path(args.output_dir)
    manifest = read_csv(out / "resume_manifest.csv")
    best = best_family(args, specs)
    elapsed = time.time() - goal_start
    remaining = max(0.0, args.max_wall_clock_hours * 3600.0 - elapsed)
    completed_count = sum(1 for row in manifest if row.get("status") == "completed")
    failed_count = sum(1 for row in manifest if row.get("status") == "failed")
    interrupted_count = sum(1 for row in manifest if row.get("status") == "interrupted")
    family_count = completed_primary_family_count(args, specs)
    test_doc = ROOT / "docs/codex/test.md"
    final_locked = (out / "FINAL_SELECTION_LOCK.md").exists() or bool(read_csv(out / "final_test_metrics.csv"))
    pending = [
        spec
        for spec in specs
        if not completed(Path(args.output_dir) / spec.run_name) and not spec_blocked_by_manifest(args, spec)
    ]
    issue = guard_no_test_outputs(args)
    lines = [
        "# RUN_STATE",
        "",
        f"- Updated: {now_iso()}",
        f"- Goal start timestamp: {datetime.fromtimestamp(goal_start).isoformat(timespec='seconds')}",
        f"- Elapsed hours: {elapsed / 3600.0:.3f}",
        f"- Remaining hours: {remaining / 3600.0:.3f}",
        f"- MAX_WALL_CLOCK_HOURS: {args.max_wall_clock_hours}",
        f"- FULL_TRAINING_ALLOWED: true",
        f"- docs/codex/test.md: {'present' if test_doc.exists() else 'absent'}",
        f"- Active run: {active.run_name if active else 'none'}",
        f"- Completed runs: {completed_count}",
        f"- Failed runs: {failed_count}",
        f"- Interrupted runs: {interrupted_count}",
        f"- Completed primary architecture families: {family_count}",
        f"- Best validation family: {best.get('architecture_family') if best else 'pending'}",
        f"- Best primary validation MAE: {best.get('mean_val_mae_h5_h10_h15') if best else 'pending'}",
        f"- Pre-lock test-output guard: {issue or 'pass'}",
        "",
        "## Next Candidate",
    ]
    if final_locked:
        lines.append("- Final selection is locked; no further candidates may be run in this goal.")
    elif pending:
        lines.append(f"- {pending[0].run_name}: {pending[0].hypothesis}")
    else:
        lines.append("- No pending planned run; extended adaptive candidates may be generated if minimum search time is not satisfied.")
    lines.extend(
        [
            "",
            "## Resume Rules",
            "- Completed runs are skipped by `--skip_existing` and the runner manifest.",
            "- Failed runs remain visible in `resume_manifest.csv`; inspect logs before rerun or use a new run_name.",
            "- Normal adaptive runs use `--no_test_eval`; test outputs are final-lock only.",
        ]
    )
    if completed_count > 60:
        lines.extend(
            [
                "",
                "## Soft Run-Count Guard",
                f"- Completed run count is {completed_count}, above 60.",
                "- Additional runs must be justified by expected learning value, not micro-run filling.",
            ]
        )
    write_text(out / "RUN_STATE.md", "\n".join(lines) + "\n")


def representative_specs(specs: Sequence[Spec]) -> List[Spec]:
    reps: List[Spec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.architecture_family.startswith("diagnostic_"):
            continue
        if spec.branch == "representative" and spec.architecture_family not in seen:
            seen.add(spec.architecture_family)
            reps.append(spec)
    return reps


def should_prioritize_representative(args: argparse.Namespace, spec: Spec, specs: Sequence[Spec]) -> bool:
    if spec.branch == "representative":
        return True
    return completed_primary_family_count(args, specs) < 7


def next_pending_spec(args: argparse.Namespace, specs: Sequence[Spec], goal_start: float) -> Optional[Spec]:
    for spec in specs:
        if spec_blocked_by_manifest(args, spec):
            continue
        if completed(Path(args.output_dir) / spec.run_name):
            continue
        if should_prioritize_representative(args, spec, specs):
            return spec
    for spec in specs:
        if spec_blocked_by_manifest(args, spec):
            continue
        if not completed(Path(args.output_dir) / spec.run_name):
            return spec
    return None


def extended_spec(args: argparse.Namespace, specs: Sequence[Spec]) -> Optional[Spec]:
    board_best = best_family(args, specs)
    if not board_best:
        return None
    family = str(board_best.get("architecture_family"))
    model = str(board_best.get("model_type"))
    if family.startswith("diagnostic_") or model in {"calib_only", "recent10_tcn"}:
        family = "lcsa_per_horizon_heads"
        model = "lc_sa_tcnformer"
    completed_names = {row.get("run_name") for row in read_csv(Path(args.output_dir) / "resume_manifest.csv")}
    planned_names = {spec.run_name for spec in specs}
    for raw_variant in adaptive_variant_dicts():
        variant = dict(raw_variant)
        spec = Spec(
            architecture_family=family,
            branch=variant.pop("branch"),
            model=model,
            multi_horizon=True,
            horizon_set=PRIMARY_HORIZONS,
            horizon_seconds=5.0,
            epochs=80,
            patience=8,
            hypothesis="Adaptive refinement of the current best validation family to test seed/optimization/loss robustness before final lock.",
            selection_reason="Generated because the minimum wall-clock policy was not yet satisfied and improvement remained below the 10% threshold.",
            predict_delta_from_anchor=model in {"anchor_delta_mlp", "anchor_delta_gru", "recent_tcn_summary_calib", "gated_fusion"},
            per_horizon_heads=family == "lcsa_per_horizon_heads",
            pooling="attention" if model in {"recent_tcn_summary_calib", "gated_fusion"} else "mean",
            **variant,
        )
        if spec.run_name in planned_names:
            continue
        if spec.run_name not in completed_names and not completed(Path(args.output_dir) / spec.run_name):
            specs.append(spec)  # type: ignore[arg-type]
            return spec
    return None


def enough_for_final_lock(args: argparse.Namespace, specs: Sequence[Spec], goal_start: float) -> tuple[bool, str]:
    elapsed_hours = (time.time() - goal_start) / 3600.0
    if elapsed_hours < 0.5:
        return False, "policy forbids goal completion under 30 minutes"
    if elapsed_hours < args.min_search_hours:
        return False, f"user-requested minimum search time not reached ({elapsed_hours:.2f}/{args.min_search_hours:.2f}h)"
    family_count = completed_primary_family_count(args, specs)
    if family_count < 7:
        return False, f"architecture coverage incomplete ({family_count}/7)"
    best = best_family(args, specs)
    if not best:
        return False, "no complete final-eligible validation family"
    improvement = as_float(best.get("improvement_vs_v1_baseline_percent"), -math.inf)
    min_hours = max(4.0, args.max_wall_clock_hours * 0.7)
    if improvement < 10.0 and elapsed_hours < min_hours:
        return False, f"primary improvement {improvement:.3f}% < 10%; minimum search time is {min_hours:.2f}h"
    return True, "validation lock criteria satisfied"


def parse_final_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = payload.get("metrics", {})
    out = {
        "test_mae": metrics.get("mae", ""),
        "test_rmse": metrics.get("rmse", ""),
        "test_r2": metrics.get("r2", ""),
        "test_n": metrics.get("n", ""),
    }
    by_h = metrics.get("by_horizon", {})
    if isinstance(by_h, Mapping):
        for h in PRIMARY_HORIZONS:
            item = by_h.get(str(h)) or by_h.get(f"{h:g}")
            if isinstance(item, Mapping):
                out[f"h{tag(h)}_test_mae"] = item.get("mae", "")
    return out


def write_final_lock(args: argparse.Namespace, specs: Sequence[Spec]) -> Optional[Dict[str, Any]]:
    best = best_family(args, specs)
    if not best:
        return None
    spec_by_name = {spec.run_name: spec for spec in specs}
    members = [name for name in str(best.get("member_runs", "")).split(";") if name]
    commands = []
    for name in members:
        spec = spec_by_name.get(name)
        if spec:
            commands.append(" ".join(eval_command(args, spec, "test")))
    text = [
        "# FINAL_SELECTION_LOCK",
        "",
        "- This file freezes the final candidate using validation metrics only.",
        f"- Selected architecture family: {best.get('architecture_family')}",
        f"- Selected member runs: {best.get('member_runs')}",
        f"- Checkpoint paths: {best.get('checkpoint_paths')}",
        f"- Primary validation mean MAE h=5/10/15: {best.get('mean_val_mae_h5_h10_h15')}",
        f"- h=5 validation MAE: {best.get('h5_val_mae')}",
        f"- h=10 validation MAE: {best.get('h10_val_mae')}",
        f"- h=15 validation MAE: {best.get('h15_val_mae')}",
        f"- Static usage: {best.get('use_static')} ({best.get('static_feature_set')})",
        f"- Selection reason: {best.get('selection_reason')}",
        "- h=1 diagnostic and final test results are excluded from selection.",
        "- No architecture, hyperparameter, window, preprocessing, loss, or static-usage tuning will occur after final test.",
        "",
        "## Final Test Command(s)",
        "",
        *[f"- `{cmd}`" for cmd in commands],
    ]
    write_text(Path(args.output_dir) / "FINAL_SELECTION_LOCK.md", "\n".join(text) + "\n")
    return best


def run_final_test(args: argparse.Namespace, specs: Sequence[Spec]) -> None:
    out = Path(args.output_dir)
    if (out / "final_test_metrics.csv").exists():
        print(f"[{now_iso()}] final_test_metrics.csv already exists; not rerunning final test.", flush=True)
        return
    best = write_final_lock(args, specs)
    if not best:
        return
    spec_by_name = {spec.run_name: spec for spec in specs}
    rows: List[Dict[str, Any]] = []
    logs = out / "logs"
    for name in [n for n in str(best.get("member_runs", "")).split(";") if n]:
        spec = spec_by_name.get(name)
        if not spec:
            continue
        print(f"[{now_iso()}] FINAL TEST {name}", flush=True)
        code = run_subprocess(
            eval_command(args, spec, "test"),
            ROOT,
            logs / f"{name}.final_test.stdout.log",
            logs / f"{name}.final_test.stderr.log",
        )
        metrics_path = out / name / "eval_test" / "metrics.json"
        parsed = parse_final_metrics(metrics_path)
        rows.append(
            {
                "architecture_family": best.get("architecture_family"),
                "run_name": name,
                "status": "completed" if code == 0 else f"failed_{code}",
                "horizon_seconds": "multi" if spec.multi_horizon else spec.horizon_seconds,
                "validation_primary_mean_mae": best.get("mean_val_mae_h5_h10_h15"),
                "test_mae": parsed.get("test_mae", ""),
                "test_rmse": parsed.get("test_rmse", ""),
                "test_r2": parsed.get("test_r2", ""),
                "test_n": parsed.get("test_n", ""),
                "h5_test_mae": parsed.get("h5_test_mae", ""),
                "h10_test_mae": parsed.get("h10_test_mae", ""),
                "h15_test_mae": parsed.get("h15_test_mae", ""),
                "metrics_path": rel(metrics_path),
                "prediction_csv_path": rel(out / name / "eval_test" / "test_predictions.csv"),
            }
        )
    write_csv(out / "final_test_metrics.csv", rows, FINAL_TEST_FIELDS)
    write_text(
        out / "final_test_audit.md",
        "# Final Test Audit\n\n"
        "- Final test was run only after `FINAL_SELECTION_LOCK.md` was written.\n"
        "- No validation/test-driven tuning is allowed after this point.\n\n"
        + markdown_table(rows, FINAL_TEST_FIELDS),
    )


def write_final_report(args: argparse.Namespace, specs: Sequence[Spec], goal_start: float) -> None:
    out = Path(args.output_dir)
    board = family_rows(args, specs)
    best = best_family(args, specs)
    final_rows = read_csv(out / "final_test_metrics.csv")
    git_status = subprocess.run(["git", "status", "--short"], cwd=str(ROOT), text=True, capture_output=True)
    elapsed_hours = (time.time() - goal_start) / 3600.0
    lines = [
        "# Final Report",
        "",
        "## 요약",
        "",
        f"- 실제 wall-clock 사용 시간: {elapsed_hours:.2f}시간",
        f"- 완료된 primary architecture family 수: {completed_primary_family_count(args, specs)}",
        "- adaptive search 중 test metric/prediction/plot은 생성하지 않았고, validation lock 이후 final test만 수행했다.",
        "",
        "## Validation 기준 선택",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- 선택 family: {best.get('architecture_family')}",
                f"- primary validation MAE mean(h=5/10/15): {best.get('mean_val_mae_h5_h10_h15')}",
                f"- h=5 validation MAE: {best.get('h5_val_mae')}",
                f"- h=10 validation MAE: {best.get('h10_val_mae')}",
                f"- h=15 validation MAE: {best.get('h15_val_mae')}",
                f"- static 사용: {best.get('use_static')} ({best.get('static_feature_set')})",
                f"- member runs: {best.get('member_runs')}",
            ]
        )
    else:
        lines.append("- complete validation-selected model 없음.")
    h2 = next((row for row in board if row.get("architecture_family") == "diagnostic_h2p5"), None)
    h1 = next((row for row in board if row.get("architecture_family") == "diagnostic_h1"), None)
    lines.extend(["", "## Horizon별 기록", ""])
    for h, key in [(5, "h5_val_mae"), (10, "h10_val_mae"), (15, "h15_val_mae")]:
        candidates = [row for row in board if row.get(key) not in ("", None)]
        if candidates:
            best_h = min(candidates, key=lambda row: as_float(row.get(key), math.inf))
            lines.append(f"- h={h} best validation MAE: {best_h.get(key)} ({best_h.get('architecture_family')})")
    if h2:
        lines.append(f"- h=2.5 auxiliary validation MAE: {h2.get('h2p5_val_mae')} ({h2.get('member_runs')})")
    if h1:
        lines.append(f"- h=1 diagnostic validation MAE: {h1.get('h1_diagnostic_val_mae')} ({h1.get('member_runs')})")
    new_arch_descriptions = {
        "lcsa_cross_attn": "기존 LC-SA/TcnFormer를 참고한 새 구조. forecast query가 calibration token 전체에 cross-attention한다.",
        "gru_state_mixer": "기존 TCNFormer를 덜 참고한 recurrent 구조. bidirectional calibration GRU와 causal recent-window GRU를 horizon별 head로 fusion한다.",
        "motion_conv_mixer": "기존 구조를 거의 쓰지 않는 motion-first 구조. causal ConvMixer block과 calibration summary만으로 start_only 조건을 맞춘다.",
    }
    new_rows = [
        row
        for row in board
        if row.get("architecture_family") in new_arch_descriptions
        and row.get("mean_val_mae_h5_h10_h15") not in ("", None)
        and math.isfinite(as_float(row.get("mean_val_mae_h5_h10_h15")))
    ]
    pending_new = [
        spec
        for spec in specs
        if spec.architecture_family in new_arch_descriptions
        and not completed(out / spec.run_name)
        and not spec_blocked_by_manifest(args, spec)
    ]
    failed_new_names = {
        row.get("run_name")
        for row in read_csv(out / "resume_manifest.csv")
        if row.get("status") in {"failed", "interrupted", "blocked", "running"}
    }
    failed_new = [spec for spec in specs if spec.architecture_family in new_arch_descriptions and spec.run_name in failed_new_names]
    lines.extend(["", "## 새 모델 구조 탐색", ""])
    for family, description in new_arch_descriptions.items():
        lines.append(f"- `{family}`: {description}")
    if new_rows:
        lines.append("")
        lines.append("완료된 새 구조 validation 결과:")
        lines.append(
            markdown_table(
                new_rows,
                [
                    "rank",
                    "architecture_family",
                    "branch",
                    "mean_val_mae_h5_h10_h15",
                    "h5_val_mae",
                    "h10_val_mae",
                    "h15_val_mae",
                    "member_runs",
                ],
                limit=20,
            )
        )
    else:
        lines.append("")
        lines.append("- 아직 완료된 새 구조 validation 결과 없음.")
    if pending_new and not final_rows:
        lines.append("")
        lines.append("시간 부족 또는 interrupt 시 이어서 실행할 새 구조 후보:")
        lines.append(
            markdown_table(
                [
                    {
                        "run_name": spec.run_name,
                        "architecture_family": spec.architecture_family,
                        "branch": spec.branch,
                        "model_type": spec.model,
                        "hypothesis": spec.hypothesis,
                    }
                    for spec in pending_new
                ],
                ["run_name", "architecture_family", "branch", "model_type", "hypothesis"],
                limit=30,
            )
        )
    elif pending_new and final_rows:
        lines.append("")
        lines.append("- final test가 이미 완료되었으므로, 현재 goal에서는 남은 새 구조 후보를 추가 실행하지 않는다.")
        lines.append("- 남은 후보는 별도 새 goal에서만 validation-only 조건으로 재개할 수 있다.")
    if failed_new:
        lines.append("")
        lines.append("실패/중단된 새 구조 후보:")
        lines.append(
            markdown_table(
                [
                    {
                        "run_name": spec.run_name,
                        "architecture_family": spec.architecture_family,
                        "branch": spec.branch,
                    }
                    for spec in failed_new
                ],
                ["run_name", "architecture_family", "branch"],
                limit=20,
            )
        )
    lines.extend(["", "## Final Test 1회 결과", ""])
    if final_rows:
        lines.append(markdown_table(final_rows, FINAL_TEST_FIELDS))
    else:
        lines.append("- final test 미수행. 사유는 `FINAL_SELECTION_LOCK.md` 또는 `final_test_audit.md` 확인.")
    failed = [row for row in read_csv(out / "resume_manifest.csv") if row.get("status") in {"failed", "interrupted", "blocked", "running"}]
    lines.extend(["", "## Failed/Interrupted/Resume", ""])
    if failed:
        lines.append(markdown_table(failed, ["run_name", "status", "failure_reason", "resume_action"], limit=30))
    else:
        lines.append("- failed/interrupted/blocked run 없음.")
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            "- `runs/goal_mae_search_v2/experiment_log.csv`",
            "- `runs/goal_mae_search_v2/leaderboard.csv`",
            "- `runs/goal_mae_search_v2/best_model_summary.md`",
            "- `runs/goal_mae_search_v2/final_test_audit.md`",
            "",
            "## Git Status",
            "",
            "```text",
            git_status.stdout.strip(),
            "```",
        ]
    )
    write_text(out / "final_report.md", "\n".join(lines) + "\n")


def load_or_create_goal_start(args: argparse.Namespace) -> float:
    path = Path(args.output_dir) / "goal_runtime.json"
    if path.exists():
        try:
            payload = read_json(path)
            started = float(payload.get("goal_start_epoch", 0.0))
            if started > 0:
                return started
        except Exception:
            pass
    started = time.time()
    write_text(
        path,
        json.dumps(
            {
                "goal_start_epoch": started,
                "goal_start_iso": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                "max_wall_clock_hours": args.max_wall_clock_hours,
            },
            indent=2,
        )
        + "\n",
    )
    return started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DenseFMS corrected start_only MAE search v2.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--config", default="configs/lc_sa_tcnformer.yaml")
    parser.add_argument("--split_file", default="./artifacts/densefms_split_seed42.json")
    parser.add_argument("--policy_file", default="docs/codex/goal_mae_search_policy_0505.md")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_wall_clock_hours", type=float, default=8.0)
    parser.add_argument("--min_search_hours", type=float, default=5.0)
    parser.add_argument("--final_reserve_minutes", type=float, default=40.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--no_final_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.policy_file = Path(args.policy_file)
    specs = build_specs()
    init_docs(args, specs)
    goal_start = load_or_create_goal_start(args)
    print(f"[{now_iso()}] DenseFMS MAE search v2 initialized at {args.output_dir}", flush=True)
    run_architecture_sanity(args, specs)
    run_repo_sanity(args)
    write_summaries(args, specs, goal_start)

    if args.dry_run:
        rows = [
            {
                "run_name": spec.run_name,
                "architecture_family": spec.architecture_family,
                "command": " ".join(train_command(args, spec)),
                "hypothesis": spec.hypothesis,
            }
            for spec in specs
        ]
        write_csv(args.output_dir / "dry_run_commands.csv", rows, ["run_name", "architecture_family", "command", "hypothesis"])
        print(f"[{now_iso()}] dry run commands written.", flush=True)
        return

    if args.summary_only:
        write_summaries(args, specs, goal_start)
        print(f"[{now_iso()}] summary files refreshed.", flush=True)
        return

    while True:
        write_summaries(args, specs, goal_start)
        elapsed = time.time() - goal_start
        remaining = args.max_wall_clock_hours * 3600.0 - elapsed
        if remaining <= args.final_reserve_minutes * 60.0:
            print(f"[{now_iso()}] stopping search for final reserve; remaining={remaining / 60.0:.1f}m", flush=True)
            break
        lock_ok, reason = enough_for_final_lock(args, specs, goal_start)
        if lock_ok:
            print(f"[{now_iso()}] final-lock criteria met: {reason}", flush=True)
            break
        spec = next_pending_spec(args, specs, goal_start)
        if spec is None:
            spec = extended_spec(args, specs)
        if spec is None:
            print(f"[{now_iso()}] no candidate available; final lock status: {reason}", flush=True)
            break
        print(f"[{now_iso()}] final lock not allowed yet: {reason}", flush=True)
        run_one(args, spec, goal_start)

    write_summaries(args, specs, goal_start)
    lock_ok, reason = enough_for_final_lock(args, specs, goal_start)
    if not lock_ok:
        write_text(args.output_dir / "FINAL_SELECTION_LOCK_BLOCKED.md", f"# Final Selection Lock Blocked\n\n- Reason: {reason}\n")
        print(f"[{now_iso()}] FINAL SELECTION BLOCKED: {reason}", flush=True)
    else:
        blocked = args.output_dir / "FINAL_SELECTION_LOCK_BLOCKED.md"
        if blocked.exists():
            blocked.unlink()
        write_final_lock(args, specs)
        if not args.no_final_test:
            run_final_test(args, specs)
        write_summaries(args, specs, goal_start)
    write_final_report(args, specs, goal_start)
    print(f"[{now_iso()}] v2 search runner finished.", flush=True)


if __name__ == "__main__":
    main()
