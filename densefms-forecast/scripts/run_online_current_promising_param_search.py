"""Narrow parameter search around promising online-current candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


BASE_CONFIG = "configs/online_current/selected_fds_static4.yaml"
BASELINE_RUN = (
    "runs/online_fms_current_tracking_0508/"
    "deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42"
)
BASELINE_SPLIT = f"{BASELINE_RUN}/split.json"
PREVIOUS_BEST_RUN = "runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42"
PERSON_PRIOR_RUN = "runs/online_fms_current_tracking_0509_remaining/remaining_p5_person_prior_seed42"
EXPLICIT_STATE_RUN = "runs/online_fms_current_tracking_0509_remaining/remaining_p8_explicit_state_shared_aux_seed42"


PARAM_CANDIDATES: List[Dict[str, Any]] = [
    {
        "run_name": "psearch_causal_dyn_risk020_ord015_seed42",
        "family": "causal_dynamics_v1",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1", "--risk_loss_weight", "0.20", "--fms_combine_weight_ordinal", "0.15"],
    },
    {
        "run_name": "psearch_causal_dyn_risk030_ord015_seed42",
        "family": "causal_dynamics_v1",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1", "--risk_loss_weight", "0.30", "--fms_combine_weight_ordinal", "0.15"],
    },
    {
        "run_name": "psearch_causal_dyn_fds075_ord015_seed42",
        "family": "causal_dynamics_v1",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1", "--fds_blend", "0.75", "--fms_combine_weight_ordinal", "0.15"],
    },
    {
        "run_name": "psearch_causal_dyn_lr035_seed42",
        "family": "causal_dynamics_v1",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1", "--learning_rate", "0.00035"],
    },
    {
        "run_name": "psearch_causal_dyn_dropout015_seed42",
        "family": "causal_dynamics_v1",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1", "--dropout", "0.15"],
    },
    {
        "run_name": "psearch_causal_dyn_event_delta_light_seed42",
        "family": "causal_dynamics_v1_aux",
        "extra_args": [
            "--motion_feature_mode",
            "causal_dynamics_v1",
            "--future_aux_horizon_seconds",
            "5.0",
            "10.0",
            "15.0",
            "--future_aux_loss_weight",
            "0.03",
            "--delta_aux_loss_weight",
            "0.05",
            "--event_aux_loss_weight",
            "0.02",
            "--event_delta_threshold",
            "1.0",
        ],
    },
    {
        "run_name": "psearch_causal_dyn_trajectory_w002_seed42",
        "family": "causal_dynamics_v1_aux",
        "extra_args": [
            "--motion_feature_mode",
            "causal_dynamics_v1",
            "--trajectory_loss_weight",
            "0.02",
            "--trajectory_delta_seconds",
            "5.0",
            "--trajectory_delta_weight",
            "1.0",
            "--trajectory_centered_weight",
            "0.3",
            "--trajectory_range_weight",
            "0.1",
            "--trajectory_loss_type",
            "mae",
        ],
    },
    {
        "run_name": "psearch_person_prior_ord010_seed42",
        "family": "person_prior",
        "extra_args": ["--current_head_mode", "person_prior", "--fms_combine_weight_ordinal", "0.10"],
    },
    {
        "run_name": "psearch_person_prior_risk020_seed42",
        "family": "person_prior",
        "extra_args": ["--current_head_mode", "person_prior", "--risk_loss_weight", "0.20"],
    },
    {
        "run_name": "psearch_explicit_state_aux_light_seed42",
        "family": "explicit_state_shared_aux",
        "extra_args": [
            "--state_feedback_mode",
            "predicted_current",
            "--future_aux_horizon_seconds",
            "5.0",
            "10.0",
            "15.0",
            "--future_aux_loss_weight",
            "0.015",
            "--delta_aux_loss_weight",
            "0.025",
            "--event_aux_loss_weight",
            "0.01",
            "--regime_head_enabled",
            "--regime_loss_weight",
            "0.01",
        ],
    },
]


def _completed(run_dir: Path) -> bool:
    return (run_dir / "best.pt").exists() and (run_dir / "metrics.json").exists() and (run_dir / "val_predictions.csv").exists()


def _command(args: argparse.Namespace, candidate: Dict[str, Any]) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        args.base_config,
        "--model",
        "online_fms_risk_tracker",
        "--run_name",
        str(candidate["run_name"]),
        "--runs_dir",
        args.runs_dir,
        "--split_file",
        args.split_file,
        "--no_test_eval",
        "--skip_existing",
        "--save_predictions",
        "--save_plots",
    ]
    cmd.extend(str(value) for value in candidate.get("extra_args", []))
    return cmd


def _run(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return int(proc.wait())


def _analysis_runs(args: argparse.Namespace, completed: Sequence[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    pairs = [
        ("fds_static4", Path(BASELINE_RUN)),
        ("integrated_p4_causal_dynamics_v1_seed42", Path(PREVIOUS_BEST_RUN)),
        ("remaining_p5_person_prior_seed42", Path(PERSON_PRIOR_RUN)),
        ("remaining_p8_explicit_state_shared_aux_seed42", Path(EXPLICIT_STATE_RUN)),
    ]
    pairs.extend((str(item["run_name"]), Path(args.runs_dir) / str(item["run_name"])) for item in completed)
    existing = [(label, run_dir) for label, run_dir in pairs if _completed(run_dir)]
    return [str(run_dir) for _, run_dir in existing], [label for label, _ in existing]


def _run_analysis(args: argparse.Namespace, completed: Sequence[Dict[str, Any]]) -> int:
    run_dirs, labels = _analysis_runs(args, completed)
    if not run_dirs:
        return 0
    cmd = [
        args.python,
        "scripts/analyze_online_current_tracking.py",
        "--run_dirs",
        *run_dirs,
        "--labels",
        *labels,
        "--out_dir",
        args.analysis_dir,
        "--split",
        "val",
        "--primary_label",
        "fds_static4",
        "--trajectory_count",
        str(args.trajectory_count),
    ]
    return _run(cmd, Path(args.analysis_dir) / "logs" / "analyze_latest.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run narrow parameter search for promising online-current candidates.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--base_config", default=BASE_CONFIG)
    parser.add_argument("--split_file", default=BASELINE_SPLIT)
    parser.add_argument("--runs_dir", default="runs/online_fms_current_tracking_0509_param_search")
    parser.add_argument("--analysis_dir", default="runs/online_fms_current_tracking_0509_param_search/analysis")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--family", nargs="+", default=["all"])
    parser.add_argument("--trajectory_count", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    selected = [
        item for item in PARAM_CANDIDATES if "all" in args.family or str(item["family"]) in set(str(v) for v in args.family)
    ]
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    commands = [_command(args, item) for item in selected]
    (analysis_dir / "dry_run_commands.txt").write_text("\n".join(" ".join(cmd) for cmd in commands) + "\n", encoding="utf-8")
    (analysis_dir / "candidate_manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    if not args.execute:
        for cmd in commands:
            print("$ " + " ".join(cmd))
        return

    completed: List[Dict[str, Any]] = []
    failed: List[str] = []
    for candidate, cmd in zip(selected, commands):
        run_name = str(candidate["run_name"])
        if _completed(Path(args.runs_dir) / run_name):
            completed.append(candidate)
            continue
        code = _run(cmd, analysis_dir / "logs" / f"{run_name}.log")
        if code != 0 or not _completed(Path(args.runs_dir) / run_name):
            failed.append(run_name)
            break
        completed.append(candidate)
        _run_analysis(args, completed)
    if completed:
        _run_analysis(args, completed)
    print(f"Completed {len(completed)} candidates, failed {len(failed)} candidates.")


if __name__ == "__main__":
    main()
