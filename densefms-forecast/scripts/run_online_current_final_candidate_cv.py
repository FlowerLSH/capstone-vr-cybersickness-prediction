"""Run participant-level 5-fold CV for selected online-current model recipes.

The script creates fixed participant folds, trains validation-selected models
inside each fold, evaluates each fold's held-out test split once, and aggregates
the resulting metrics.  It is intentionally recipe-based: checkpoints from the
original v1/v2 split are not evaluated directly on CV folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from densefms_forecast.data import load_raw_sessions
from densefms_forecast.utils import load_json, save_json


DEFAULT_PYTHON = sys.executable
DEFAULT_DATA_DIR = "DenseFMS/Dataset"
DEFAULT_RUNS_DIR = "runs/online_current_5fold_cv_0514"
DEFAULT_SPLIT_DIR = "reports/online_current_5fold_cv_0514/splits"
DEFAULT_REPORT_DIR = "reports/online_current_5fold_cv_0514"


RECIPES: Dict[str, Dict[str, Any]] = {
    "selected_risk035": {
        "kind": "full",
        "config": "configs/online_current/selected_deeptcn_risk035_static4.yaml",
        "description": "Original selected DeepTCN risk035 static4 recipe.",
    },
    "risk045_smooth005": {
        "kind": "full",
        "config": "runs/calibration_branch_revision_0513/cbr_bestbase_risk045_smooth005_seed42/config_snapshot.json",
        "description": "Risk0.45 + smooth0.005 base recipe.",
    },
    "zero_anchor_highgate_delta2": {
        "kind": "head",
        "base_recipe": "risk045_smooth005",
        "config": "runs/calibration_branch_revision_0513/cbr_zero_anchor_highgate_t12_w030_pos4_delta2_seed42/config_snapshot.json",
        "description": "Zero-anchor high-gate delta2 head fine-tune on fold-local risk045 base.",
    },
    "range_scaled_delta2": {
        "kind": "head",
        "base_recipe": "risk045_smooth005",
        "config": "runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/config_snapshot.json",
        "description": "Range-scaled delta2 head fine-tune on fold-local risk045 base.",
    },
}


def _as_path(path: str | Path) -> Path:
    return Path(path)


def _run_name(recipe: str, fold: int) -> str:
    return f"fold{fold:02d}_{recipe}_seed42"


def _run_dir(args: argparse.Namespace, recipe: str, fold: int) -> Path:
    return Path(args.runs_dir) / _run_name(recipe, fold)


def _split_path(args: argparse.Namespace, fold: int) -> Path:
    return Path(args.split_dir) / f"fold{fold:02d}_split.json"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _metric_get(metrics: Mapping[str, Any], path: Sequence[str], default: float = float("nan")) -> float:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.std(finite, ddof=0)) if finite.size else float("nan")


def _format(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.4f}"


def _participant_session_counts(sessions: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for session in sessions:
        if session.participant_id is None:
            raise ValueError("participant_id is required for participant-level CV.")
        pid = str(session.participant_id)
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def _balanced_participant_folds(pid_counts: Mapping[str, int], n_folds: int) -> List[List[str]]:
    folds: List[List[str]] = [[] for _ in range(int(n_folds))]
    fold_counts = [0 for _ in range(int(n_folds))]
    # Deterministic bin-packing by session count keeps fold session counts close.
    for pid, count in sorted(pid_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        target = min(range(int(n_folds)), key=lambda idx: (fold_counts[idx], idx))
        folds[target].append(pid)
        fold_counts[target] += int(count)
    return [sorted(fold) for fold in folds]


def create_splits(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_sessions, _mapping, data_info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(args.calibration_seconds),
        horizon_seconds=float(args.horizon_seconds),
        default_sampling_interval=float(args.sampling_interval),
        max_session_points=int(args.max_session_points) if args.max_session_points is not None else None,
    )
    pid_counts = _participant_session_counts(raw_sessions)
    folds = _balanced_participant_folds(pid_counts, int(args.n_folds))
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    descriptors: List[Dict[str, Any]] = []
    all_pids = set(pid_counts)
    for fold_idx in range(int(args.n_folds)):
        test_groups = set(folds[fold_idx])
        val_groups = set(folds[(fold_idx + 1) % int(args.n_folds)])
        train_groups = all_pids - test_groups - val_groups
        groups = {
            "train": sorted(train_groups),
            "val": sorted(val_groups),
            "test": sorted(test_groups),
        }
        counts = {
            split: int(sum(pid_counts[pid] for pid in split_groups))
            for split, split_groups in groups.items()
        }
        split_info = {
            "group_key": "participant_id",
            "groups": groups,
            "counts": counts,
            "cv": {
                "method": "deterministic_participant_group_5fold",
                "fold": int(fold_idx),
                "n_folds": int(args.n_folds),
                "val_fold": int((fold_idx + 1) % int(args.n_folds)),
                "test_fold": int(fold_idx),
                "train_folds": [
                    int(idx)
                    for idx in range(int(args.n_folds))
                    if idx not in {fold_idx, (fold_idx + 1) % int(args.n_folds)}
                ],
                "participant_fold_groups": folds,
                "sampling_interval": float(args.sampling_interval),
                "calibration_seconds": float(args.calibration_seconds),
                "recent_window_seconds": float(args.recent_window_seconds),
                "horizon_seconds": float(args.horizon_seconds),
                "max_session_points": int(args.max_session_points) if args.max_session_points is not None else None,
            },
        }
        path = _split_path(args, fold_idx)
        save_json(path, split_info)
        descriptors.append(
            {
                "fold": fold_idx,
                "split_file": str(path),
                "counts": counts,
                "participants": {split: len(split_groups) for split, split_groups in groups.items()},
            }
        )
    save_json(
        split_dir / "fold_manifest.json",
        {
            "data_info": data_info,
            "n_folds": int(args.n_folds),
            "folds": descriptors,
            "participant_session_counts": dict(sorted(pid_counts.items())),
        },
    )
    return descriptors


def _train_command(args: argparse.Namespace, recipe: str, fold: int) -> List[str]:
    spec = RECIPES[recipe]
    command = [
        args.python,
        "-m",
        "src.densefms_forecast.train",
        "--data_dir",
        args.data_dir,
        "--config",
        str(spec["config"]),
        "--model",
        "online_fms_risk_tracker",
        "--device",
        args.device,
        "--runs_dir",
        args.runs_dir,
        "--run_name",
        _run_name(recipe, fold),
        "--split_file",
        str(_split_path(args, fold)),
        "--no_test_eval",
        "--skip_existing",
    ]
    if args.epochs is not None:
        command.extend(["--epochs", str(int(args.epochs))])
    if args.max_train_batches is not None:
        command.extend(["--max_train_batches", str(int(args.max_train_batches))])
    if args.max_eval_batches is not None:
        command.extend(["--max_eval_batches", str(int(args.max_eval_batches))])
    if spec["kind"] == "head":
        base_recipe = str(spec["base_recipe"])
        base_ckpt = _run_dir(args, base_recipe, fold) / "best.pt"
        command.extend(["--init_checkpoint", str(base_ckpt)])
    return command


def _eval_command(args: argparse.Namespace, recipe: str, fold: int) -> List[str]:
    run_dir = _run_dir(args, recipe, fold)
    return [
        args.python,
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(run_dir / "best.pt"),
        "--data_dir",
        args.data_dir,
        "--split",
        "test",
        "--split_file",
        str(_split_path(args, fold)),
        "--device",
        args.device,
        "--batch_size",
        str(int(args.batch_size)),
        "--save_predictions",
    ]


def _run_subprocess(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(list(command), check=True)


def _recipe_order(recipes: Sequence[str]) -> List[Tuple[str, str]]:
    selected = [recipe for recipe in recipes if recipe in RECIPES]
    missing = [recipe for recipe in recipes if recipe not in RECIPES]
    if missing:
        raise ValueError(f"Unknown recipe(s): {missing}; available={sorted(RECIPES)}")
    # Full recipes first, then head recipes that depend on risk045.
    return [(recipe, str(RECIPES[recipe]["kind"])) for recipe in selected]


def run_cv(args: argparse.Namespace) -> None:
    create_splits(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    recipes = [item[0] for item in _recipe_order(args.recipes)]
    command_rows: List[Dict[str, Any]] = []
    for fold in range(int(args.n_folds)):
        for recipe in recipes:
            train_command = _train_command(args, recipe, fold)
            eval_command = _eval_command(args, recipe, fold)
            command_rows.append({"fold": fold, "recipe": recipe, "phase": "train", "command": " ".join(train_command)})
            command_rows.append({"fold": fold, "recipe": recipe, "phase": "eval_test", "command": " ".join(eval_command)})
            if args.dry_run:
                continue
            spec = RECIPES[recipe]
            if spec["kind"] == "head":
                base_ckpt = _run_dir(args, str(spec["base_recipe"]), fold) / "best.pt"
                if not base_ckpt.exists():
                    raise FileNotFoundError(f"Head recipe {recipe} requires fold-local base checkpoint {base_ckpt}.")
            run_dir = _run_dir(args, recipe, fold)
            if not ((run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists()):
                _run_subprocess(train_command)
            else:
                print(f"Skipping completed train run: {run_dir}", flush=True)
            eval_metrics = run_dir / "eval_test" / "metrics.json"
            if not eval_metrics.exists():
                _run_subprocess(eval_command)
            else:
                print(f"Skipping completed test eval: {eval_metrics}", flush=True)
            aggregate_results(args)
    _write_csv(report_dir / "cv_commands.csv", command_rows, ["fold", "recipe", "phase", "command"])
    if args.dry_run:
        print(json.dumps({"dry_run": True, "commands": command_rows[:12], "command_count": len(command_rows)}, indent=2))


def _load_val_metrics(run_dir: Path) -> Tuple[int | None, Dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None, {}
    payload = load_json(metrics_path)
    metrics = payload.get("metrics", {})
    best_epoch = metrics.get("best_epoch")
    best_val = metrics.get("best_val_metrics") or metrics.get("val_metrics") or {}
    return (int(best_epoch) if best_epoch is not None else None), best_val


def _load_test_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "eval_test" / "metrics.json"
    if not metrics_path.exists():
        return {}
    return load_json(metrics_path).get("metrics", {})


def _metric_row(args: argparse.Namespace, recipe: str, fold: int) -> Dict[str, Any]:
    run_dir = _run_dir(args, recipe, fold)
    best_epoch, val = _load_val_metrics(run_dir)
    test = _load_test_metrics(run_dir)
    return {
        "fold": int(fold),
        "recipe": recipe,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "val_mae": _metric_get(val, ["mae"]),
        "val_rmse": _metric_get(val, ["rmse"]),
        "val_r2": _metric_get(val, ["current_fms_r2"]),
        "val_range_ratio": _metric_get(val, ["trajectory", "pred_true_range_ratio_mean"]),
        "val_pearson": _metric_get(val, ["trajectory", "pearson_session_mean"]),
        "val_dir10": _metric_get(val, ["trajectory", "direction_acc_10s"]),
        "val_drop10": _metric_get(val, ["trajectory", "direction_acc_drop_10s"]),
        "test_mae": _metric_get(test, ["mae"]),
        "test_rmse": _metric_get(test, ["rmse"]),
        "test_r2": _metric_get(test, ["current_fms_r2"]),
        "test_range_ratio": _metric_get(test, ["trajectory", "pred_true_range_ratio_mean"]),
        "test_pearson": _metric_get(test, ["trajectory", "pearson_session_mean"]),
        "test_dir10": _metric_get(test, ["trajectory", "direction_acc_10s"]),
        "test_rise10": _metric_get(test, ["trajectory", "direction_acc_rise_10s"]),
        "test_drop10": _metric_get(test, ["trajectory", "direction_acc_drop_10s"]),
        "test_low_warning_fpr": _metric_get(test, ["warning_high_fms_false_positive_rate"]),
        "test_warning_fnr": _metric_get(test, ["warning_high_fms_false_negative_rate"]),
        "complete": bool(val and test),
    }


def aggregate_results(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    recipes = [recipe for recipe in args.recipes if recipe in RECIPES]
    fold_rows: List[Dict[str, Any]] = []
    for recipe in recipes:
        for fold in range(int(args.n_folds)):
            fold_rows.append(_metric_row(args, recipe, fold))
    if fold_rows:
        _write_csv(report_dir / "cv_fold_metrics.csv", fold_rows, list(fold_rows[0].keys()))
    metric_names = [
        "val_mae",
        "val_rmse",
        "val_r2",
        "val_range_ratio",
        "val_pearson",
        "val_dir10",
        "val_drop10",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_range_ratio",
        "test_pearson",
        "test_dir10",
        "test_rise10",
        "test_drop10",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for recipe in recipes:
        rows = [row for row in fold_rows if row["recipe"] == recipe and bool(row.get("complete"))]
        summary: Dict[str, Any] = {
            "recipe": recipe,
            "completed_folds": len(rows),
            "description": RECIPES[recipe]["description"],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
        summary_rows.append(summary)
    if summary_rows:
        _write_csv(report_dir / "cv_summary.csv", summary_rows, list(summary_rows[0].keys()))
    payload = {
        "runs_dir": args.runs_dir,
        "split_dir": args.split_dir,
        "recipes": {recipe: RECIPES[recipe] for recipe in recipes},
        "fold_rows": fold_rows,
        "summary": summary_rows,
    }
    save_json(report_dir / "cv_summary.json", payload)
    _write_markdown_report(report_dir, summary_rows, fold_rows)
    return payload


def _write_markdown_report(report_dir: Path, summary_rows: Sequence[Mapping[str, Any]], fold_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Online Current Final Candidate 5-Fold CV",
        "",
        "## Protocol",
        "",
        "- Grouping: participant-level deterministic 5-fold split.",
        "- For fold `i`, test is participant fold `i`, validation is fold `i+1`, train is the remaining three folds.",
        "- Model selection is validation MAE inside each fold.",
        "- Test is evaluated once per completed fold after validation selection.",
        "- Head fine-tune recipes initialize only from the fold-local `risk045_smooth005` checkpoint.",
        "",
        "## Summary",
        "",
        "| recipe | folds | test MAE | test R2 | test range ratio | test Pearson | test dir10 | test drop10 | val MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['recipe']} | {int(row['completed_folds'])} | "
            f"{_format(float(row.get('test_mae_mean', float('nan'))))} +/- {_format(float(row.get('test_mae_std', float('nan'))))} | "
            f"{_format(float(row.get('test_r2_mean', float('nan'))))} +/- {_format(float(row.get('test_r2_std', float('nan'))))} | "
            f"{_format(float(row.get('test_range_ratio_mean', float('nan'))))} +/- {_format(float(row.get('test_range_ratio_std', float('nan'))))} | "
            f"{_format(float(row.get('test_pearson_mean', float('nan'))))} +/- {_format(float(row.get('test_pearson_std', float('nan'))))} | "
            f"{_format(float(row.get('test_dir10_mean', float('nan'))))} +/- {_format(float(row.get('test_dir10_std', float('nan'))))} | "
            f"{_format(float(row.get('test_drop10_mean', float('nan'))))} +/- {_format(float(row.get('test_drop10_std', float('nan'))))} | "
            f"{_format(float(row.get('val_mae_mean', float('nan'))))} +/- {_format(float(row.get('val_mae_std', float('nan'))))} |"
        )
    lines.extend(["", "## Fold Metrics", ""])
    lines.extend(
        [
            "| fold | recipe | val MAE | test MAE | test R2 | range ratio | Pearson | dir10 | complete |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fold_rows:
        lines.append(
            "| "
            f"{int(row['fold'])} | {row['recipe']} | {_format(float(row['val_mae']))} | "
            f"{_format(float(row['test_mae']))} | {_format(float(row['test_r2']))} | "
            f"{_format(float(row['test_range_ratio']))} | {_format(float(row['test_pearson']))} | "
            f"{_format(float(row['test_dir10']))} | {bool(row['complete'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are cross-validation robustness metrics, not a new hyperparameter search over test folds.",
            "- If fewer than five folds are completed, the summary is interim.",
            "",
        ]
    )
    (report_dir / "cv_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs_dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--recent_window_seconds", type=float, default=10.0)
    parser.add_argument("--horizon_seconds", type=float, default=10.0)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--recipes", nargs="+", default=list(RECIPES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--create_splits_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.create_splits_only:
        descriptors = create_splits(args)
        print(json.dumps({"split_dir": args.split_dir, "folds": descriptors}, indent=2))
        return
    if args.aggregate_only:
        payload = aggregate_results(args)
        print(json.dumps({"summary": payload["summary"]}, indent=2))
        return
    run_cv(args)


if __name__ == "__main__":
    main()
