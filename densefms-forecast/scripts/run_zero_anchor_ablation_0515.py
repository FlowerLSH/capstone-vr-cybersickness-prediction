"""Run or dry-run zero-anchor ablation experiments and summarize validation metrics.

Default behavior is dry-run command generation plus summary collection from any
completed runs. Use --run only when full training is explicitly allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "configs/online_current/selected_deeptcn_risk035_static4.yaml"
BASE_SPLIT = "runs/online_fms_current_tracking_0509_deeptcn_improve/deeptcn_imp_risk035_seed42/split.json"
RUNS_DIR = "runs/zero_anchor_ablation_0515"
SUMMARY_DIR = "save/zero_anchor_ablation_0515"
REUSE_RUNS_DIR = Path("runs/calibration_branch_revision_0513")
REUSED_120_BASE = REUSE_RUNS_DIR / "cbr_bestbase_risk045_smooth005_seed42"
REUSED_120_ZERO = REUSE_RUNS_DIR / "cbr_zero_anchor_highgate_t12_w030_pos4_delta2_seed42"


BASE_RECIPE_ARGS = [
    "--fms_combine_weight_ordinal",
    "0.10",
    "--ordinal_loss_weight",
    "0.05",
    "--weight_decay",
    "0.0001",
    "--risk_loss_weight",
    "0.45",
    "--smoothness_weight",
    "0.005",
    "--max_session_points",
    "420",
    "--recent_window_seconds",
    "10.0",
    "--selection_metric",
    "mae",
    "--selection_mode",
    "min",
]

ZERO_ANCHOR_ARGS = [
    "--current_head_mode",
    "zero_anchor_mixture",
    "--current_delta_scale",
    "2.0",
    "--anchor_gate_loss_weight",
    "0.3",
    "--anchor_gate_threshold",
    "12.0",
    "--anchor_gate_pos_weight",
    "4.0",
    "--learning_rate",
    "0.001",
    "--weight_decay",
    "0.0001",
    "--freeze_loaded_parameters",
    "--trainable_parameter_patterns",
    "current_anchor",
]


def path_text(path: str | Path) -> str:
    return Path(path).as_posix()


@dataclass(frozen=True)
class AblationSpec:
    label: str
    group: str
    calibration_seconds: float
    head_channel_mode: str = "all"
    use_static: bool = True
    zero_anchor: bool = True
    notes: str = ""
    reuse_base_run_dir: Optional[Path] = None
    reuse_zero_run_dir: Optional[Path] = None

    @property
    def suffix(self) -> str:
        if self.label == "calib0_no_calib_no_anchor":
            return "calib000_no_calib_no_anchor"
        if self.group == "calibration_length":
            return f"calib{int(self.calibration_seconds):03d}"
        return self.label

    @property
    def base_run_name(self) -> str:
        return f"za_ablate_{self.suffix}_base_risk045_smooth005_seed42"

    @property
    def zero_run_name(self) -> Optional[str]:
        if not self.zero_anchor:
            return None
        return f"za_ablate_{self.suffix}_zero_anchor_d2_seed42"


def build_specs(reuse_existing_120: bool = True) -> List[AblationSpec]:
    specs: List[AblationSpec] = [
        AblationSpec(
            label="calib0_no_calib_no_anchor",
            group="calibration_length",
            calibration_seconds=0.0,
            zero_anchor=False,
            notes="0s no-calibration/no-anchor motion-only baseline.",
        ),
        AblationSpec("calib30_zero_anchor", "calibration_length", 30.0),
        AblationSpec("calib60_zero_anchor", "calibration_length", 60.0),
        AblationSpec("calib120_zero_anchor", "calibration_length", 120.0),
        AblationSpec("calib180_zero_anchor", "calibration_length", 180.0),
        AblationSpec("component_all", "component_120s", 120.0, notes="120s all 6D head channels plus static."),
        AblationSpec("component_linear_only", "component_120s", 120.0, head_channel_mode="linear_only"),
        AblationSpec("component_angular_only", "component_120s", 120.0, head_channel_mode="angular_only"),
        AblationSpec("component_no_static", "component_120s", 120.0, use_static=False),
    ]
    if not reuse_existing_120:
        return specs

    replaced: List[AblationSpec] = []
    for spec in specs:
        if spec.calibration_seconds == 120.0 and spec.head_channel_mode == "all" and spec.use_static:
            replaced.append(
                AblationSpec(
                    label=spec.label,
                    group=spec.group,
                    calibration_seconds=spec.calibration_seconds,
                    head_channel_mode=spec.head_channel_mode,
                    use_static=spec.use_static,
                    zero_anchor=spec.zero_anchor,
                    notes=(spec.notes + " Reuses existing 120s risk045/zero-anchor runs.").strip(),
                    reuse_base_run_dir=REUSED_120_BASE,
                    reuse_zero_run_dir=REUSED_120_ZERO if spec.zero_anchor else None,
                )
            )
        else:
            replaced.append(spec)
    return replaced


def _common_train_command(args: argparse.Namespace, run_name: str, extra_args: Sequence[str]) -> List[str]:
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
        run_name,
        "--split_file",
        args.split_file,
        "--no_test_eval",
        "--skip_existing",
        *extra_args,
    ]


def _spec_args(spec: AblationSpec) -> List[str]:
    out = [
        "--calibration_seconds",
        f"{float(spec.calibration_seconds):g}",
        *BASE_RECIPE_ARGS,
    ]
    if float(spec.calibration_seconds) == 0.0:
        out.extend(["--fms_context_mode", "none"])
    if spec.head_channel_mode != "all":
        out.extend(["--head_channel_mode", spec.head_channel_mode])
    if not spec.use_static:
        out.append("--no_static")
    return out


def build_commands(args: argparse.Namespace, spec: AblationSpec) -> List[Dict[str, Any]]:
    if spec.reuse_zero_run_dir is not None or spec.reuse_base_run_dir is not None:
        return []
    base_args = _spec_args(spec)
    base_cmd = _common_train_command(args, spec.base_run_name, base_args)
    commands = [
        {
            "label": spec.label,
            "stage": "risk045_base",
            "run_name": spec.base_run_name,
            "command": base_cmd,
        }
    ]
    if spec.zero_anchor:
        assert spec.zero_run_name is not None
        init_checkpoint = path_text(Path(args.runs_dir) / spec.base_run_name / "best.pt")
        zero_args = [
            *_spec_args(spec),
            "--init_checkpoint",
            init_checkpoint,
            *ZERO_ANCHOR_ARGS,
        ]
        commands.append(
            {
                "label": spec.label,
                "stage": "zero_anchor_finetune",
                "run_name": spec.zero_run_name,
                "command": _common_train_command(args, spec.zero_run_name, zero_args),
            }
        )
    return commands


def _metric(metrics: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float("nan")


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {
            "status": "missing",
            "run_dir": path_text(run_dir),
            "metrics_json": False,
            "best_pt": (run_dir / "best.pt").exists(),
            "val_predictions": (run_dir / "val_predictions.csv").exists(),
            "config_snapshot": (run_dir / "config_snapshot.json").exists(),
        }
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics_root = payload.get("metrics", {})
    val_metrics = metrics_root.get("val_metrics", {})
    best_metrics = metrics_root.get("best_val_metrics", payload.get("best_val_metrics", {}))
    source = val_metrics or best_metrics
    task = payload.get("task", {})
    return {
        "status": "completed",
        "run_dir": path_text(run_dir),
        "metrics_json": True,
        "best_pt": (run_dir / "best.pt").exists(),
        "val_predictions": (run_dir / "val_predictions.csv").exists(),
        "config_snapshot": (run_dir / "config_snapshot.json").exists(),
        "best_epoch": metrics_root.get("best_epoch", payload.get("best_epoch")),
        "selection_metric": metrics_root.get("selection_metric", payload.get("selection_metric")),
        "selection_mode": metrics_root.get("selection_mode", payload.get("selection_mode")),
        "selection_value": metrics_root.get("best_selection_value", payload.get("best_selection_value")),
        "val_mae": _metric(source, "mae", "current_fms_mae"),
        "val_rmse": _metric(source, "rmse", "current_fms_rmse"),
        "val_r2": _metric(source, "current_fms_r2", "r2"),
        "test_eval_skipped": bool(task.get("test_eval_skipped", "test_metrics" not in metrics_root)),
        "head_channel_mode": payload.get("data_info", {}).get("head_channel_mode"),
    }


def selected_run_dir(args: argparse.Namespace, spec: AblationSpec) -> Path:
    if spec.zero_anchor:
        if spec.reuse_zero_run_dir is not None:
            return spec.reuse_zero_run_dir
        assert spec.zero_run_name is not None
        return Path(args.runs_dir) / spec.zero_run_name
    if spec.reuse_base_run_dir is not None:
        return spec.reuse_base_run_dir
    return Path(args.runs_dir) / spec.base_run_name


def build_summary_rows(args: argparse.Namespace, specs: Sequence[AblationSpec]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        run_dir = selected_run_dir(args, spec)
        summary = load_run_summary(run_dir)
        summary["head_channel_mode"] = summary.get("head_channel_mode") or spec.head_channel_mode
        row = {
            "group": spec.group,
            "label": spec.label,
            "calibration_seconds": float(spec.calibration_seconds),
            "head_channel_mode": spec.head_channel_mode,
            "use_static": bool(spec.use_static),
            "zero_anchor": bool(spec.zero_anchor),
            "base_run_name": spec.base_run_name,
            "zero_run_name": spec.zero_run_name or "",
            "selected_run_dir": path_text(run_dir),
            "reused_existing_run": bool(spec.reuse_base_run_dir or spec.reuse_zero_run_dir),
            "notes": spec.notes,
            **summary,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [
        "group",
        "label",
        "calibration_seconds",
        "head_channel_mode",
        "use_static",
        "status",
        "val_mae",
        "val_rmse",
        "val_r2",
        "selected_run_dir",
    ]
    lines = [
        "# Zero-Anchor Ablation Validation Summary",
        "",
        "Validation metrics only. Test metrics are not used for ablation selection.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key, "")) for key in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sort_key_val_mae(row: Mapping[str, Any]) -> float:
    try:
        value = float(row.get("val_mae", float("nan")))
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")


def write_leaderboard(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    completed = sorted(completed, key=_sort_key_val_mae)
    headers = [
        "rank",
        "label",
        "group",
        "calibration_seconds",
        "head_channel_mode",
        "use_static",
        "val_mae",
        "val_rmse",
        "val_r2",
        "selected_run_dir",
    ]
    lines = [
        "# Zero-Anchor Ablation Validation Leaderboard",
        "",
        "Sorted by validation MAE. Test metrics are not used for ranking.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rank, row in enumerate(completed, start=1):
        enriched = {"rank": rank, **row}
        lines.append("| " + " | ".join(_fmt(enriched.get(key, "")) for key in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dry_run_outputs(path: Path, commands: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(shlex.quote(str(part)) for part in item["command"]) for item in commands]
    (path / "dry_run_commands.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (path / "dry_run_commands.json").write_text(json.dumps(list(commands), indent=2), encoding="utf-8")


def _finite_rows(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out = []
    for row in rows:
        try:
            value = float(row.get("val_mae", float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if value == value:
            out.append(row)
    return out


def write_plots(summary_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        payload = {"plots_written": [], "skipped": f"matplotlib unavailable: {exc}"}
        (summary_dir / "plot_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    plot_paths: List[str] = []
    for group, filename, label_key in [
        ("calibration_length", "calibration_length_validation_metrics.png", "calibration_seconds"),
        ("component_120s", "component_ablation_validation_metrics.png", "label"),
    ]:
        group_rows = _finite_rows(row for row in rows if row.get("group") == group)
        if not group_rows:
            continue
        labels = [str(row.get(label_key)) for row in group_rows]
        metrics = [
            ("val_mae", "Validation MAE"),
            ("val_rmse", "Validation RMSE"),
            ("val_r2", "Validation R2"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (key, title) in zip(axes, metrics):
            values = [float(row.get(key, float("nan"))) for row in group_rows]
            ax.bar(labels, values, color="#4C78A8")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
        out_path = summary_dir / filename
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        plot_paths.append(path_text(out_path))
    payload = {"plots_written": plot_paths, "skipped": None}
    (summary_dir / "plot_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def filter_specs(specs: Sequence[AblationSpec], only: Optional[str]) -> List[AblationSpec]:
    if not only:
        return list(specs)
    requested = {value.strip() for value in only.split(",") if value.strip()}
    return [
        spec
        for spec in specs
        if spec.label in requested
        or spec.base_run_name in requested
        or (spec.zero_run_name is not None and spec.zero_run_name in requested)
    ]


def run_commands(commands: Sequence[Mapping[str, Any]]) -> int:
    for item in commands:
        cmd = [str(part) for part in item["command"]]
        print(f"RUN {item['stage']} {item['run_name']}: {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--config", default=BASE_CONFIG)
    parser.add_argument("--split_file", default=BASE_SPLIT)
    parser.add_argument("--runs_dir", default=RUNS_DIR)
    parser.add_argument("--summary_dir", default=SUMMARY_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default=None, help="Comma-separated labels or run names to include.")
    parser.add_argument("--max_commands", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Execute training commands. Requires explicit full-training permission.")
    parser.add_argument("--no_reuse_existing_120", action="store_true")
    args = parser.parse_args()

    specs = filter_specs(build_specs(reuse_existing_120=not args.no_reuse_existing_120), args.only)
    commands: List[Dict[str, Any]] = []
    for spec in specs:
        commands.extend(build_commands(args, spec))
    if args.max_commands is not None:
        commands = commands[: int(args.max_commands)]

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_dry_run_outputs(summary_dir, commands)
    rows = build_summary_rows(args, specs)
    write_csv(summary_dir / "ablation_validation_summary.csv", rows)
    write_markdown(summary_dir / "ablation_validation_summary.md", rows)
    leaderboard_rows = sorted([row for row in rows if row.get("status") == "completed"], key=_sort_key_val_mae)
    write_csv(summary_dir / "ablation_validation_leaderboard.csv", leaderboard_rows)
    write_leaderboard(summary_dir / "ablation_validation_leaderboard.md", rows)
    plot_status = write_plots(summary_dir, rows)

    print(
        json.dumps(
            {
                "mode": "run" if args.run else "dry_run",
                "spec_count": len(specs),
                "command_count": len(commands),
                "summary_csv": path_text(summary_dir / "ablation_validation_summary.csv"),
                "summary_md": path_text(summary_dir / "ablation_validation_summary.md"),
                "leaderboard_csv": path_text(summary_dir / "ablation_validation_leaderboard.csv"),
                "leaderboard_md": path_text(summary_dir / "ablation_validation_leaderboard.md"),
                "plot_status": plot_status,
            },
            indent=2,
        )
    )
    if args.run:
        raise SystemExit(run_commands(commands))


if __name__ == "__main__":
    main()
