"""Run the five integer-oriented online-current DenseFMS variants.

The script keeps test evaluation out of training. It trains each variant with
validation-based checkpoint selection, then evaluates the selected checkpoint on
the test split and writes a compact leaderboard/report artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "online_current" / "selected_deeptcn_risk035_static4.yaml"
DATA_DIR = REPO_ROOT / "DenseFMS" / "Dataset"
RUNS_DIR = REPO_ROOT / "runs" / "online_current_integer_top5_0520"
REPORT_DIR = REPO_ROOT / "reports" / "online_current_integer_top5_0520"
GENERATED_CONFIG_DIR = RUNS_DIR / "generated_configs"


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "01_oce_ts_cdf_bce",
        "name": "OCE-TS style CDF-BCE classifier",
        "paper_hint": "Ordinal cross-entropy / CDF-aware time-series classification",
        "model": {
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 1.0,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.10,
            "ordinal_loss_weight": 0.60,
            "ordinal_loss_mode": "oce",
            "coarse_band_loss_weight": 0.0,
        },
    },
    {
        "id": "02_hca_coarse_to_fine",
        "name": "HCA coarse-to-fine hierarchical classifier",
        "paper_hint": "Hierarchical Classification Auxiliary network",
        "model": {
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.90,
            "coarse_band_bins": [2.0, 5.0, 8.0, 12.0, 16.0],
        },
        "loss": {
            "current_reg_aux_weight": 0.10,
            "ordinal_loss_weight": 0.45,
            "ordinal_loss_mode": "ce",
            "coarse_band_loss_weight": 0.35,
        },
    },
    {
        "id": "03_r2c_softmax_ce",
        "name": "Regression-as-classification 21-bin CE",
        "paper_hint": "Regression as classification with full integer bins",
        "model": {
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 1.0,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.10,
            "ordinal_loss_weight": 0.60,
            "ordinal_loss_mode": "ce",
            "coarse_band_loss_weight": 0.0,
        },
    },
    {
        "id": "04_softlabel_slace",
        "name": "Soft ordinal labels / SLACE-style classifier",
        "paper_hint": "Soft ordinal labels with local unimodal smoothing",
        "model": {
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 1.0,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.10,
            "ordinal_loss_weight": 0.60,
            "ordinal_loss_mode": "soft_ce",
            "ordinal_soft_label_sigma": 1.20,
            "ordinal_soft_label_kernel": "gaussian",
            "coarse_band_loss_weight": 0.0,
        },
    },
    {
        "id": "05_ord2seq_cumulative",
        "name": "Ord2Seq-style cumulative binary sequence",
        "paper_hint": "Ordinal label sequence with K-1 cumulative binary targets",
        "model": {
            "ordinal_head_mode": "cumulative",
            "fms_combine_weight_ordinal": 1.0,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.10,
            "ordinal_loss_weight": 0.60,
            "ordinal_loss_mode": "cumulative_bce",
            "coarse_band_loss_weight": 0.0,
        },
    },
]


METRIC_FIELDS = [
    ("val_mae", ("metrics", "best_val_metrics", "mae")),
    ("val_rmse", ("metrics", "best_val_metrics", "rmse")),
    ("val_r2", ("metrics", "best_val_metrics", "current_fms_r2")),
    ("val_integer_exact", ("metrics", "best_val_metrics", "integer_exact_accuracy")),
    ("val_integer_pm1", ("metrics", "best_val_metrics", "integer_off_by_one_accuracy")),
    ("val_integer_hard_mae", ("metrics", "best_val_metrics", "integer_hard_mae")),
    ("val_ordinal_acc", ("metrics", "best_val_metrics", "ordinal_accuracy")),
    ("val_ordinal_pm1", ("metrics", "best_val_metrics", "ordinal_off_by_one_accuracy")),
    ("val_high8_f1", ("metrics", "best_val_metrics", "caution_high_fms_f1")),
    ("val_high12_f1", ("metrics", "best_val_metrics", "warning_high_fms_f1")),
    ("test_mae", ("test_metrics", "mae")),
    ("test_rmse", ("test_metrics", "rmse")),
    ("test_r2", ("test_metrics", "current_fms_r2")),
    ("test_integer_exact", ("test_metrics", "integer_exact_accuracy")),
    ("test_integer_pm1", ("test_metrics", "integer_off_by_one_accuracy")),
    ("test_integer_hard_mae", ("test_metrics", "integer_hard_mae")),
    ("test_ordinal_acc", ("test_metrics", "ordinal_accuracy")),
    ("test_ordinal_pm1", ("test_metrics", "ordinal_off_by_one_accuracy")),
    ("test_high8_f1", ("test_metrics", "caution_high_fms_f1")),
    ("test_high12_f1", ("test_metrics", "warning_high_fms_f1")),
]


def _deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _metric(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return ""
        value = value[key]
    return value


def _format(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _run(command: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def _make_config(base: Mapping[str, Any], variant: Mapping[str, Any], args: argparse.Namespace) -> Path:
    config = deepcopy(base)
    config["runs_dir"] = str(RUNS_DIR.relative_to(REPO_ROOT))
    config.setdefault("evaluation", {})["no_test_eval"] = True
    config.setdefault("training", {})["epochs"] = int(args.epochs)
    config.setdefault("training", {})["patience"] = int(args.patience)
    config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    config.setdefault("training", {})["seed"] = int(args.seed)
    config.setdefault("training", {})["selection_metric"] = "mae"
    config.setdefault("training", {})["selection_mode"] = "min"
    config.setdefault("data", {})["max_session_points"] = 420
    config.setdefault("model", {})["ordinal_bins"] = list(range(21))
    _deep_update(config.setdefault("model", {}), variant.get("model", {}))
    _deep_update(config.setdefault("loss", {}), variant.get("loss", {}))
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = GENERATED_CONFIG_DIR / f"{variant['id']}.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=False)
    return config_path


def _collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        run_dir = RUNS_DIR / variant["id"]
        metrics_path = run_dir / "metrics.json"
        test_metrics_path = run_dir / "eval_test" / "metrics.json"
        if not metrics_path.exists() or not test_metrics_path.exists():
            continue
        train_metrics = _read_json(metrics_path)
        test_metrics = _read_json(test_metrics_path).get("metrics", {})
        merged = dict(train_metrics)
        merged["test_metrics"] = test_metrics
        row: dict[str, Any] = {
            "id": variant["id"],
            "name": variant["name"],
            "best_epoch": _metric(train_metrics, ("metrics", "best_epoch")),
            "paper_hint": variant["paper_hint"],
        }
        for field, path in METRIC_FIELDS:
            row[field] = _metric(merged, path)
        rows.append(row)
    rows.sort(key=lambda item: float(item["val_mae"]) if item.get("val_mae") != "" else float("inf"))
    return rows


def _write_leaderboard(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "integer_top5_leaderboard.csv"
    fieldnames = ["id", "name", "best_epoch", "paper_hint"] + [name for name, _ in METRIC_FIELDS]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = REPORT_DIR / "integer_top5_report.md"
    lines = [
        "# Online Current Integer Top-5 Experiment Report",
        "",
        f"- Base config: `{BASE_CONFIG.relative_to(REPO_ROOT)}`",
        f"- Runs dir: `{RUNS_DIR.relative_to(REPO_ROOT)}`",
        f"- Budget: 5 variants, seed={args.seed}, max_epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}",
        "- Selection: validation MAE only; test split was evaluated after the checkpoint was selected.",
        "- Integer output: `predicted_fms_ordinal_hard` = argmax ordinal bin on the 0-20 FMS scale.",
        "",
        "## Validation leaderboard",
        "",
        "| rank | id | best_epoch | val_mae | val_int_exact | val_int_pm1 | val_high8_f1 | val_high12_f1 | test_mae | test_int_exact | test_int_pm1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["id"]),
                    _format(row.get("best_epoch", "")),
                    _format(row.get("val_mae", "")),
                    _format(row.get("val_integer_exact", "")),
                    _format(row.get("val_integer_pm1", "")),
                    _format(row.get("val_high8_f1", "")),
                    _format(row.get("val_high12_f1", "")),
                    _format(row.get("test_mae", "")),
                    _format(row.get("test_integer_exact", "")),
                    _format(row.get("test_integer_pm1", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Variant mapping",
            "",
            "| id | method mapping |",
            "|---|---|",
        ]
    )
    for variant in VARIANTS:
        lines.append(f"| {variant['id']} | {variant['name']} ({variant['paper_hint']}) |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    for variant in VARIANTS:
        config_path = _make_config(base_config, variant, args)
        if args.prepare_only:
            print(f"Prepared {config_path.relative_to(REPO_ROOT)}")
            continue
        run_dir = RUNS_DIR / variant["id"]
        best_ckpt = run_dir / "best.pt"
        if not args.skip_train:
            if args.skip_existing and best_ckpt.exists():
                print(f"Skipping existing training run: {variant['id']}")
            else:
                command = [
                    sys.executable,
                    "-m",
                    "src.densefms_forecast.train",
                    "--data_dir",
                    str(DATA_DIR),
                    "--config",
                    str(config_path),
                    "--model",
                    "online_fms_risk_tracker",
                    "--run_name",
                    variant["id"],
                ]
                if args.device:
                    command += ["--device", args.device]
                if args.max_train_batches is not None:
                    command += ["--max_train_batches", str(args.max_train_batches)]
                if args.max_eval_batches is not None:
                    command += ["--max_eval_batches", str(args.max_eval_batches)]
                _run(command, cwd=REPO_ROOT)
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Missing selected checkpoint: {best_ckpt}")
        if not args.skip_test:
            command = [
                sys.executable,
                "-m",
                "src.densefms_forecast.evaluate",
                "--checkpoint",
                str(best_ckpt),
                "--data_dir",
                str(DATA_DIR),
                "--split",
                "test",
                "--batch_size",
                str(args.batch_size),
            ]
            if args.device:
                command += ["--device", args.device]
            _run(command, cwd=REPO_ROOT)

    if args.prepare_only:
        return

    rows = _collect_rows()
    if not rows:
        raise RuntimeError("No completed rows found for the integer top-5 report.")
    _write_leaderboard(rows, args)


if __name__ == "__main__":
    main()
