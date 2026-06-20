"""Generate or execute the 2026-05-09 online current-FMS improvement plan.

Default mode is dry-run command generation. This keeps test data untouched and
avoids accidental full training when the active goal has not explicitly allowed
it. Pass ``--execute`` only after full-training permission is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


BASE_CONFIG = "configs/online_current/selected_fds_static4.yaml"
BASELINE_RUN = (
    "runs/online_fms_current_tracking_0508/"
    "deep_tcn_latent_gru_420_large_calib240_lds_gamma05_state_decoder_static4_fds_seed42"
)
BASELINE_LABEL = "fds_static4"
BASELINE_SPLIT = f"{BASELINE_RUN}/split.json"

PLAN_CANDIDATES: List[Dict[str, Any]] = [
    {
        "phase": "phase1",
        "run_name": "integrated_p1_risk015_seed42",
        "factor": "risk_loss_weight",
        "purpose": "weaker rapid-rise auxiliary regularization around the selected baseline",
        "extra_args": ["--risk_loss_weight", "0.15"],
    },
    {
        "phase": "phase1",
        "run_name": "integrated_p1_risk035_seed42",
        "factor": "risk_loss_weight",
        "purpose": "stronger rapid-rise auxiliary regularization around the selected baseline",
        "extra_args": ["--risk_loss_weight", "0.35"],
    },
    {
        "phase": "phase1",
        "run_name": "integrated_p1_ordblend015_seed42",
        "factor": "fms_combine_weight_ordinal",
        "purpose": "lower ordinal blend to preserve continuous amplitude",
        "extra_args": ["--fms_combine_weight_ordinal", "0.15"],
    },
    {
        "phase": "phase1",
        "run_name": "integrated_p1_ordblend025_seed42",
        "factor": "fms_combine_weight_ordinal",
        "purpose": "higher ordinal blend to stabilize severity ordering",
        "extra_args": ["--fms_combine_weight_ordinal", "0.25"],
    },
    {
        "phase": "phase1",
        "run_name": "integrated_p1_fdsblend075_seed42",
        "factor": "fds_blend",
        "purpose": "weaken FDS pull toward average trajectories",
        "extra_args": ["--fds_blend", "0.75"],
    },
    {
        "phase": "phase2",
        "run_name": "integrated_p2_future_delta_event_light_seed42",
        "factor": "future_delta_event_aux",
        "purpose": "near-future FMS, delta, and rise/drop/plateau auxiliary supervision",
        "extra_args": [
            "--future_aux_horizon_seconds",
            "5.0",
            "10.0",
            "15.0",
            "--future_aux_loss_weight",
            "0.05",
            "--delta_aux_loss_weight",
            "0.10",
            "--event_aux_loss_weight",
            "0.03",
            "--event_delta_threshold",
            "1.0",
        ],
    },
    {
        "phase": "phase2",
        "run_name": "integrated_p2_delta_only_light_seed42",
        "factor": "delta_aux",
        "purpose": "isolate future-delta supervision without future-level or event losses",
        "extra_args": [
            "--future_aux_horizon_seconds",
            "5.0",
            "10.0",
            "15.0",
            "--future_aux_loss_weight",
            "0.0",
            "--delta_aux_loss_weight",
            "0.05",
            "--event_aux_loss_weight",
            "0.0",
        ],
    },
    {
        "phase": "phase2",
        "run_name": "integrated_p2_trajectory_w003_d5_seed42",
        "factor": "trajectory_loss",
        "purpose": "weak trajectory-shape auxiliary with 5s deltas",
        "extra_args": [
            "--trajectory_loss_weight",
            "0.03",
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
        "phase": "phase4",
        "run_name": "integrated_p4_causal_dynamics_v1_seed42",
        "factor": "motion_feature_mode",
        "purpose": "append causal derivative, energy, sign-change, and complexity proxies",
        "extra_args": ["--motion_feature_mode", "causal_dynamics_v1"],
    },
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _candidate_selected(candidate: Dict[str, Any], phases: Sequence[str]) -> bool:
    return "all" in phases or str(candidate["phase"]) in phases


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
        candidate["run_name"],
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


def _completed(run_dir: Path) -> bool:
    return (run_dir / "best.pt").exists() and (run_dir / "metrics.json").exists() and (run_dir / "val_predictions.csv").exists()


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run_analysis(args: argparse.Namespace, selected_completed: Sequence[Dict[str, Any]]) -> int:
    if not selected_completed:
        return 0
    run_dirs = [BASELINE_RUN] + [str(Path(args.runs_dir) / str(item["run_name"])) for item in selected_completed]
    labels = [BASELINE_LABEL] + [str(item["run_name"]) for item in selected_completed]
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
        BASELINE_LABEL,
        "--trajectory_count",
        str(args.trajectory_count),
    ]
    return _run(cmd, Path(args.analysis_dir) / "logs" / "analyze_latest.log")


def _row_for_label(rows: Sequence[Dict[str, str]], label: str) -> Dict[str, str]:
    return next((row for row in rows if row.get("label") == label), {})


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _decision(mae: float, plot_bad: float, plot_good: float, plot_medium: float) -> str:
    if not (mae == mae):
        return "rerun-needed"
    if mae > 2.20:
        return "reject"
    if plot_bad == plot_bad:
        non_bad = plot_good + plot_medium
        if plot_bad <= 5 and mae <= 1.975:
            return "keep"
        if mae <= 1.915 and plot_bad < 8:
            return "keep"
        if plot_bad <= 4 and non_bad >= 8 and mae <= 2.10:
            return "promote-later"
        if plot_bad <= 3 and non_bad >= 9 and mae <= 2.20:
            return "promote-later"
        if plot_bad >= 8:
            return "reject"
    if mae <= 1.915:
        return "keep"
    if mae <= 1.975:
        return "rerun-needed"
    return "reject"


def _append_run_detail_report(
    report_path: Path,
    *,
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    analysis_code: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_name = str(candidate["run_name"])
    run_dir = Path(args.runs_dir) / run_name
    leaderboard_path = Path(args.analysis_dir) / "online_current_validation_leaderboard.csv"
    plot_summary_path = Path(args.analysis_dir) / "plot_judgment_summary.csv"
    leaderboard = _read_csv(leaderboard_path)
    plot_summary = _read_csv(plot_summary_path)
    row = _row_for_label(leaderboard, run_name)
    baseline = _row_for_label(leaderboard, BASELINE_LABEL)
    plot_row = _row_for_label(plot_summary, run_name)
    curves = _read_csv(run_dir / "training_curves.csv")
    last_epoch = int(_float_or_nan(curves[-1].get("epoch"))) if curves else 0
    best_epoch = int(_float_or_nan(row.get("best_epoch"))) if row else 0
    total_seconds = sum(_float_or_nan(item.get("seconds")) for item in curves)
    mae = _float_or_nan(row.get("mae"))
    baseline_mae = _float_or_nan(baseline.get("mae"))
    plot_good = _float_or_nan(plot_row.get("plot_good"))
    plot_medium = _float_or_nan(plot_row.get("plot_medium"))
    plot_bad = _float_or_nan(plot_row.get("plot_bad"))
    decision = _decision(mae, plot_bad, plot_good, plot_medium)
    lines = [
        f"## Completed Run - {run_name} - {_now()}",
        "",
        f"- config: `{args.base_config}`",
        f"- CLI override: `{' '.join(str(v) for v in candidate.get('extra_args', [])) or '(none)'}`",
        f"- purpose: {candidate['purpose']}",
        f"- changed factor family: `{candidate['factor']}`",
        "- sanity: repository lightweight sanity suite was run before training; per-run command uses fixed split and `--no_test_eval`",
        f"- training budget: epochs_completed={last_epoch}, best_epoch={best_epoch}, elapsed_seconds={total_seconds:.1f}",
        f"- analysis exit_code: {analysis_code}",
        f"- validation MAE/RMSE: {row.get('mae', 'NA')} / {row.get('rmse', 'NA')}",
        f"- validation session Pearson: {row.get('pearson_session_mean', 'NA')}",
        f"- validation centered MAE: {row.get('centered_mae_session_mean', 'NA')}",
        f"- validation delta corr 5s: {row.get('delta_corr_5s', 'NA')}",
        f"- validation direction acc 5s: {row.get('direction_acc_5s', 'NA')}",
        f"- validation flat rate: {row.get('flat_range_lt25pct_session_rate', 'NA')}",
        f"- PLOT proxy judgment: good={plot_row.get('plot_good', 'NA')}, medium={plot_row.get('plot_medium', 'NA')}, bad={plot_row.get('plot_bad', 'NA')} on fixed baseline-selected validation set",
        f"- baseline MAE delta: {mae - baseline_mae:.6f}" if mae == mae and baseline_mae == baseline_mae else "- baseline MAE delta: NA",
        f"- decision: `{decision}`",
        f"- outputs: checkpoint=`{run_dir / 'best.pt'}`, metrics=`{run_dir / 'metrics.json'}`, predictions=`{run_dir / 'val_predictions.csv'}`, plots=`{run_dir / 'plots'}`",
        f"- leaderboard: `{leaderboard_path}`",
        f"- warnings: PLOT judgment is metric-derived proxy, not human visual inspection.",
        "",
    ]
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_live_report(
    report_path: Path,
    *,
    selected: Sequence[Dict[str, Any]],
    commands: Sequence[Sequence[str]],
    args: argparse.Namespace,
    completed: Sequence[str],
    failed: Sequence[str],
    dry_run: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## Integrated Improvement Plan Update - {_now()}",
        "",
        f"- mode: {'dry-run only' if dry_run else 'execute validation-only training'}",
        f"- base config: `{args.base_config}`",
        f"- fixed split: `{args.split_file}`",
        f"- runs dir: `{args.runs_dir}`",
        "- test evaluation: skipped by `--no_test_eval`",
        f"- selected candidates: {len(selected)}",
        f"- completed in this invocation: {len(completed)}",
        f"- failed in this invocation: {len(failed)}",
        "",
        "| phase | run | isolated factor | purpose | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    completed_set = set(completed)
    failed_set = set(failed)
    for candidate in selected:
        run_name = str(candidate["run_name"])
        if run_name in completed_set:
            status = "completed"
        elif run_name in failed_set:
            status = "failed"
        elif dry_run:
            status = "pending-dry-run"
        else:
            status = "pending-or-skipped"
        lines.append(
            f"| {candidate['phase']} | `{run_name}` | `{candidate['factor']}` | {candidate['purpose']} | {status} |"
        )
    lines.extend(["", "### Commands", ""])
    if commands:
        for cmd in commands:
            lines.append("```bash")
            lines.append(" ".join(cmd))
            lines.append("```")
    else:
        lines.append("- no commands generated")
    lines.append("")
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run the 2026-05-09 integrated online-current plan.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--base_config", default=BASE_CONFIG)
    parser.add_argument("--split_file", default=BASELINE_SPLIT)
    parser.add_argument("--runs_dir", default="runs/online_fms_current_tracking_0509_integrated")
    parser.add_argument("--analysis_dir", default="runs/online_fms_current_tracking_0509_integrated/analysis")
    parser.add_argument("--live_report", default="docs/codex/online_current_improvement_live_report.md")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase", nargs="+", choices=["all", "phase1", "phase2", "phase4"], default=["all"])
    parser.add_argument("--trajectory_count", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Run validation-only training commands instead of dry-run output.")
    args = parser.parse_args()

    selected = [item for item in PLAN_CANDIDATES if _candidate_selected(item, args.phase)]
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    commands = [_command(args, item) for item in selected]
    command_path = analysis_dir / "dry_run_commands.txt"
    command_path.write_text("\n".join(" ".join(cmd) for cmd in commands) + ("\n" if commands else ""), encoding="utf-8")
    manifest_path = analysis_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    completed: List[str] = []
    completed_specs: List[Dict[str, Any]] = []
    failed: List[str] = []
    if not args.execute:
        for cmd in commands:
            print("$ " + " ".join(cmd))
        print(f"Dry-run command file: {command_path}")
        _write_live_report(
            Path(args.live_report),
            selected=selected,
            commands=commands,
            args=args,
            completed=completed,
            failed=failed,
            dry_run=True,
        )
        return

    for candidate, cmd in zip(selected, commands):
        run_name = str(candidate["run_name"])
        if _completed(Path(args.runs_dir) / run_name):
            completed.append(run_name)
            completed_specs.append(candidate)
            continue
        code = _run(cmd, analysis_dir / "logs" / f"{run_name}.log")
        if code != 0 or not _completed(Path(args.runs_dir) / run_name):
            failed.append(run_name)
            break
        completed.append(run_name)
        completed_specs.append(candidate)
        analysis_code = _run_analysis(args, completed_specs)
        _append_run_detail_report(
            Path(args.live_report),
            candidate=candidate,
            args=args,
            analysis_code=analysis_code,
        )
        _write_live_report(
            Path(args.live_report),
            selected=selected,
            commands=commands,
            args=args,
            completed=completed,
            failed=failed,
            dry_run=False,
        )
    if completed_specs:
        _run_analysis(args, completed_specs)
    _write_live_report(
        Path(args.live_report),
        selected=selected,
        commands=commands,
        args=args,
        completed=completed,
        failed=failed,
        dry_run=False,
    )
    print(f"Completed {len(completed)} candidates, failed {len(failed)} candidates.")


if __name__ == "__main__":
    main()
