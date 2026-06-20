"""Generate or execute remaining 2026-05-09 online-current experiments.

This runner covers the structural candidates that were not included in the
initial integrated sweep: multi-timescale response, person prior, residual
state update, coarse severity bands, scenario prior, regime/uncertainty heads,
and motion-only self-supervised pretraining.
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
BASELINE_SPLIT = f"{BASELINE_RUN}/split.json"
PREVIOUS_BEST_RUN = "runs/online_fms_current_tracking_0509_integrated/integrated_p4_causal_dynamics_v1_seed42"


REMAINING_CANDIDATES: List[Dict[str, Any]] = [
    {
        "phase": "phase5",
        "run_name": "remaining_p5_multitimescale_v1_seed42",
        "factor": "motion_feature_mode",
        "purpose": "5/15/30/60s causal motion-response summaries and complexity-drop proxies",
        "extra_args": ["--motion_feature_mode", "multi_timescale_v1"],
    },
    {
        "phase": "phase5",
        "run_name": "remaining_p5_stream_multiscale_seed42",
        "factor": "stream_context_mode",
        "purpose": "GRU+TCN multiscale state response path as a structural lag-response check",
        "extra_args": ["--stream_context_mode", "gru_tcn_multiscale", "--motion_feature_mode", "causal_dynamics_v1"],
    },
    {
        "phase": "phase5",
        "run_name": "remaining_p5_event_only_seed42",
        "factor": "event_aux_loss",
        "purpose": "rise/fall/plateau event auxiliary without future-level or delta losses",
        "extra_args": [
            "--future_aux_horizon_seconds",
            "5.0",
            "10.0",
            "15.0",
            "--future_aux_loss_weight",
            "0.0",
            "--delta_aux_loss_weight",
            "0.0",
            "--event_aux_loss_weight",
            "0.03",
            "--event_delta_threshold",
            "1.0",
        ],
    },
    {
        "phase": "phase5",
        "run_name": "remaining_p5_person_prior_seed42",
        "factor": "current_head_mode",
        "purpose": "calibration/static-conditioned bias, scale, and response-speed prior",
        "extra_args": ["--current_head_mode", "person_prior"],
    },
    {
        "phase": "phase7",
        "run_name": "remaining_p7_residual_update_seed42",
        "factor": "current_head_mode",
        "purpose": "predicted-state residual update head with bounded per-step deltas",
        "extra_args": ["--current_head_mode", "residual_update", "--current_delta_scale", "1.0"],
    },
    {
        "phase": "phase8",
        "run_name": "remaining_p8_explicit_state_shared_aux_seed42",
        "factor": "shared_latent_state_aux",
        "purpose": "latent-GRU state with predicted-current feedback plus future/event/regime supervision",
        "extra_args": [
            "--state_feedback_mode",
            "predicted_current",
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
            "--regime_head_enabled",
            "--regime_loss_weight",
            "0.02",
        ],
    },
    {
        "phase": "phase9",
        "run_name": "remaining_p9_coarse_band_aux_seed42",
        "factor": "coarse_band_aux",
        "purpose": "low/mid/high/very-high severity band auxiliary head",
        "extra_args": ["--coarse_band_bins", "5.0", "10.0", "15.0", "--coarse_band_loss_weight", "0.05"],
    },
    {
        "phase": "phase10",
        "run_name": "remaining_p10_motion_pretrained_seed42",
        "factor": "motion_pretraining",
        "purpose": "initialize causal-dynamics DeepTCN stream from motion-only future-energy pretraining",
        "extra_args": [
            "--motion_feature_mode",
            "causal_dynamics_v1",
            "--motion_pretrain_checkpoint",
            "{motion_pretrain_checkpoint}",
        ],
    },
    {
        "phase": "phase11",
        "run_name": "remaining_p11_scenario_prior_seed42",
        "factor": "static_features",
        "purpose": "deployment-visible scenario/content one-hot prior parsed from session filename",
        "extra_args": ["--static_features", "age", "mssq", "gender", "scenario"],
    },
    {
        "phase": "phase12",
        "run_name": "remaining_p12_regime_aux_seed42",
        "factor": "regime_head",
        "purpose": "stable/slow-rise/rapid-rise/high-plateau/recovery regime classifier auxiliary",
        "extra_args": [
            "--regime_head_enabled",
            "--regime_loss_weight",
            "0.03",
            "--regime_delta_slow_threshold",
            "0.5",
            "--regime_delta_rapid_threshold",
            "2.0",
            "--regime_high_threshold",
            "12.0",
        ],
    },
    {
        "phase": "phase12",
        "run_name": "remaining_p12_uncertainty_head_seed42",
        "factor": "uncertainty_head",
        "purpose": "heteroscedastic current-FMS uncertainty head with small NLL auxiliary weight",
        "extra_args": ["--uncertainty_head_enabled", "--uncertainty_loss_weight", "0.005"],
    },
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pretrain_checkpoint(args: argparse.Namespace) -> Path:
    return Path(args.pretrain_runs_dir) / args.pretrain_run_name / "best_motion_encoder.pt"


def _replace_tokens(values: Sequence[Any], args: argparse.Namespace) -> List[str]:
    checkpoint = str(_pretrain_checkpoint(args))
    return [str(value).replace("{motion_pretrain_checkpoint}", checkpoint) for value in values]


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
    cmd.extend(_replace_tokens(candidate.get("extra_args", []), args))
    return cmd


def _pretrain_command(args: argparse.Namespace) -> List[str]:
    return [
        args.python,
        "scripts/pretrain_online_current_motion_encoder.py",
        "--data_dir",
        args.data_dir,
        "--config",
        args.base_config,
        "--split_file",
        args.split_file,
        "--out_dir",
        args.pretrain_runs_dir,
        "--run_name",
        args.pretrain_run_name,
        "--motion_feature_mode",
        "causal_dynamics_v1",
        "--hidden_dim",
        "192",
        "--deep_tcn_dilations",
        "1",
        "2",
        "4",
        "8",
        "16",
        "--kernel_size",
        "3",
        "--dropout",
        "0.10",
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.pretrain_epochs),
        "--patience",
        str(args.pretrain_patience),
    ]


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


def _analysis_runs(args: argparse.Namespace, selected_completed: Sequence[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    pairs = [
        ("fds_static4", Path(BASELINE_RUN)),
        ("integrated_p4_causal_dynamics_v1_seed42", Path(PREVIOUS_BEST_RUN)),
    ]
    pairs.extend((str(item["run_name"]), Path(args.runs_dir) / str(item["run_name"])) for item in selected_completed)
    existing = [(label, run_dir) for label, run_dir in pairs if _completed(run_dir)]
    return [str(run_dir) for _, run_dir in existing], [label for label, _ in existing]


def _run_analysis(args: argparse.Namespace, selected_completed: Sequence[Dict[str, Any]]) -> int:
    run_dirs, labels = _analysis_runs(args, selected_completed)
    if len(run_dirs) < 1:
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
        labels[0],
        "--trajectory_count",
        str(args.trajectory_count),
    ]
    return _run(cmd, Path(args.analysis_dir) / "logs" / "analyze_latest.log")


def _append_run_detail_report(report_path: Path, args: argparse.Namespace, candidate: Dict[str, Any], analysis_code: int) -> None:
    run_name = str(candidate["run_name"])
    run_dir = Path(args.runs_dir) / run_name
    leaderboard_path = Path(args.analysis_dir) / "online_current_validation_leaderboard.csv"
    rows = _read_csv(leaderboard_path)
    row = next((item for item in rows if item.get("label") == run_name), {})
    lines = [
        f"## Completed Remaining Run - {run_name} - {_now()}",
        "",
        f"- phase: `{candidate['phase']}`",
        f"- changed factor: `{candidate['factor']}`",
        f"- purpose: {candidate['purpose']}",
        f"- CLI override: `{' '.join(_replace_tokens(candidate.get('extra_args', []), args)) or '(none)'}`",
        "- test evaluation: skipped by `--no_test_eval`",
        f"- analysis exit_code: {analysis_code}",
        f"- validation MAE/RMSE: {row.get('mae', 'NA')} / {row.get('rmse', 'NA')}",
        f"- validation session Pearson: {row.get('pearson_session_mean', 'NA')}",
        f"- validation centered MAE: {row.get('centered_mae_session_mean', 'NA')}",
        f"- validation delta corr 5s: {row.get('delta_corr_5s', 'NA')}",
        f"- outputs: checkpoint=`{run_dir / 'best.pt'}`, metrics=`{run_dir / 'metrics.json'}`, predictions=`{run_dir / 'val_predictions.csv'}`",
        f"- leaderboard: `{leaderboard_path}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_live_report(
    report_path: Path,
    *,
    selected: Sequence[Dict[str, Any]],
    commands: Sequence[Sequence[str]],
    pretrain_cmd: Sequence[str],
    args: argparse.Namespace,
    completed: Sequence[str],
    failed: Sequence[str],
    dry_run: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    completed_set = set(completed)
    failed_set = set(failed)
    lines = [
        f"## Remaining Experiment Update - {_now()}",
        "",
        f"- mode: {'dry-run only' if dry_run else 'execute validation-only training'}",
        f"- base config: `{args.base_config}`",
        f"- fixed split: `{args.split_file}`",
        f"- runs dir: `{args.runs_dir}`",
        f"- motion pretrain checkpoint: `{_pretrain_checkpoint(args)}`",
        "- test evaluation: skipped by `--no_test_eval`",
        "",
        "| phase | run | isolated factor | status |",
        "| --- | --- | --- | --- |",
    ]
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
        lines.append(f"| {candidate['phase']} | `{run_name}` | `{candidate['factor']}` | {status} |")
    lines.extend(["", "### Pretrain Command", "", "```bash", " ".join(pretrain_cmd), "```", "", "### Training Commands", ""])
    for cmd in commands:
        lines.extend(["```bash", " ".join(cmd), "```"])
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run remaining online-current experiments.")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--base_config", default=BASE_CONFIG)
    parser.add_argument("--split_file", default=BASELINE_SPLIT)
    parser.add_argument("--runs_dir", default="runs/online_fms_current_tracking_0509_remaining")
    parser.add_argument("--analysis_dir", default="runs/online_fms_current_tracking_0509_remaining/analysis")
    parser.add_argument("--live_report", default="docs/codex/online_current_remaining_live_report_0509.md")
    parser.add_argument("--pretrain_runs_dir", default="runs/online_fms_current_tracking_0509_remaining/motion_pretrain")
    parser.add_argument("--pretrain_run_name", default="motion_energy_causal_dynamics_v1_seed42")
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["all", "phase5", "phase7", "phase8", "phase9", "phase10", "phase11", "phase12"],
        default=["all"],
    )
    parser.add_argument("--trajectory_count", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    args = parser.parse_args()

    selected = [item for item in REMAINING_CANDIDATES if _candidate_selected(item, args.phase)]
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pretrain_cmd = _pretrain_command(args)
    commands = [_command(args, item) for item in selected]
    (analysis_dir / "dry_run_commands.txt").write_text("\n".join(" ".join(cmd) for cmd in commands) + "\n", encoding="utf-8")
    (analysis_dir / "pretrain_command.txt").write_text(" ".join(pretrain_cmd) + "\n", encoding="utf-8")
    (analysis_dir / "candidate_manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    completed: List[str] = []
    completed_specs: List[Dict[str, Any]] = []
    failed: List[str] = []
    if not args.execute:
        print("$ " + " ".join(pretrain_cmd))
        for cmd in commands:
            print("$ " + " ".join(cmd))
        _write_live_report(
            Path(args.live_report),
            selected=selected,
            commands=commands,
            pretrain_cmd=pretrain_cmd,
            args=args,
            completed=completed,
            failed=failed,
            dry_run=True,
        )
        return

    needs_pretrain = any("{motion_pretrain_checkpoint}" in " ".join(map(str, item.get("extra_args", []))) for item in selected)
    checkpoint = _pretrain_checkpoint(args)
    if needs_pretrain and not checkpoint.exists() and not args.skip_pretrain:
        code = _run(pretrain_cmd, analysis_dir / "logs" / f"pretrain_{args.pretrain_run_name}.log")
        if code != 0 or not checkpoint.exists():
            failed.append(args.pretrain_run_name)
            _write_live_report(
                Path(args.live_report),
                selected=selected,
                commands=commands,
                pretrain_cmd=pretrain_cmd,
                args=args,
                completed=completed,
                failed=failed,
                dry_run=False,
            )
            raise SystemExit(code or 1)
    if needs_pretrain and not checkpoint.exists():
        raise FileNotFoundError(f"Missing motion pretrain checkpoint: {checkpoint}")

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
        _append_run_detail_report(Path(args.live_report), args, candidate, analysis_code)
        _write_live_report(
            Path(args.live_report),
            selected=selected,
            commands=commands,
            pretrain_cmd=pretrain_cmd,
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
        pretrain_cmd=pretrain_cmd,
        args=args,
        completed=completed,
        failed=failed,
        dry_run=False,
    )
    print(f"Completed {len(completed)} candidates, failed {len(failed)} candidates.")


if __name__ == "__main__":
    main()
