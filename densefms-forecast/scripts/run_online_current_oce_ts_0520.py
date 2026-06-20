"""Run OCE-TS-inspired online-current DenseFMS variants.

This runner follows the OCE-TS recipe as closely as the DenseFMS forecasting
stack allows: a softmax ordinal distribution head, target-to-probability (TPT)
labels from a truncated Gaussian over ordered FMS bins, cumulative
ordinal-cross-entropy loss, and expected-value decoding back to FMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "online_current" / "selected_deeptcn_risk035_static4.yaml"
DATA_DIR = REPO_ROOT / "DenseFMS" / "Dataset"
RUNS_DIR = REPO_ROOT / "runs" / "online_current_oce_ts_0520"
REPORT_DIR = REPO_ROOT / "reports" / "online_current_oce_ts_0520"
GENERATED_CONFIG_DIR = RUNS_DIR / "generated_configs"
REFERENCE_BEST_TEST_METRICS = (
    REPO_ROOT
    / "runs"
    / "calibration_branch_revision_0513"
    / "cbr_zero_anchor_highgate_t12_w030_pos4_delta2_seed42"
    / "eval_test"
    / "metrics.json"
)


def _linspace_bins(start: float, stop: float, steps: int) -> list[float]:
    if steps <= 1:
        return [float(start)]
    delta = (float(stop) - float(start)) / float(steps - 1)
    return [float(start) + float(i) * delta for i in range(steps)]


VARIANTS: list[dict[str, Any]] = [
    {
        "id": "01_oce_ts_k21_sigma050",
        "name": "OCE-TS integer FMS bins, sigma=0.50",
        "paper_mapping": "TPT truncated Gaussian labels + cumulative OCE + expected-value decode",
        "model": {
            "ordinal_bins": list(range(21)),
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.90,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.15,
            "ordinal_loss_weight": 0.70,
            "ordinal_loss_mode": "oce_ts",
            "ordinal_soft_label_sigma": 0.50,
            "ordinal_ev_loss_weight": 0.20,
        },
    },
    {
        "id": "02_oce_ts_k41_sigma030",
        "name": "OCE-TS half-FMS bins, sigma=0.30",
        "paper_mapping": "Finer ordered intervals to approximate continuous TPT bins",
        "model": {
            "ordinal_bins": _linspace_bins(0.0, 20.0, 41),
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.90,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.15,
            "ordinal_loss_weight": 0.55,
            "ordinal_loss_mode": "oce_ts",
            "ordinal_soft_label_sigma": 0.30,
            "ordinal_ev_loss_weight": 0.20,
        },
    },
    {
        "id": "03_oce_ts_k81_sigma020",
        "name": "OCE-TS quarter-FMS bins, sigma=0.20",
        "paper_mapping": "High-resolution ordered intervals with smaller TPT bandwidth",
        "model": {
            "ordinal_bins": _linspace_bins(0.0, 20.0, 81),
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.85,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.15,
            "ordinal_loss_weight": 0.35,
            "ordinal_loss_mode": "oce_ts",
            "ordinal_soft_label_sigma": 0.20,
            "ordinal_ev_loss_weight": 0.20,
        },
    },
    {
        "id": "04_oce_ts_k21_strong_head",
        "name": "OCE-TS integer bins with stronger classifier decoding",
        "paper_mapping": "Classifier-dominant point forecast, closest to replacing regression with OCE",
        "model": {
            "ordinal_bins": list(range(21)),
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.97,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.05,
            "ordinal_loss_weight": 1.00,
            "ordinal_loss_mode": "oce_ts",
            "ordinal_soft_label_sigma": 0.50,
            "ordinal_ev_loss_weight": 0.10,
        },
    },
    {
        "id": "05_oce_ts_k21_low_weighted",
        "name": "OCE-TS integer bins with low-FMS cost weighting",
        "paper_mapping": "OCE-TS plus project-specific weighting for low-range false-alarm bias",
        "model": {
            "ordinal_bins": list(range(21)),
            "ordinal_head_mode": "softmax",
            "fms_combine_weight_ordinal": 0.90,
            "coarse_band_bins": [],
        },
        "loss": {
            "current_reg_aux_weight": 0.15,
            "ordinal_loss_weight": 0.70,
            "ordinal_loss_mode": "oce_ts",
            "ordinal_soft_label_sigma": 0.50,
            "ordinal_ev_loss_weight": 0.20,
            "ordinal_low_weight": 3.00,
            "ordinal_low_threshold": 2.00,
        },
    },
]


METRIC_FIELDS = [
    ("val_mae", ("metrics", "best_val_metrics", "mae")),
    ("val_rmse", ("metrics", "best_val_metrics", "rmse")),
    ("val_r2", ("metrics", "best_val_metrics", "current_fms_r2")),
    ("val_integer_exact", ("metrics", "best_val_metrics", "integer_exact_accuracy")),
    ("val_integer_pm1", ("metrics", "best_val_metrics", "integer_off_by_one_accuracy")),
    ("val_high8_f1", ("metrics", "best_val_metrics", "caution_high_fms_f1")),
    ("val_high12_f1", ("metrics", "best_val_metrics", "warning_high_fms_f1")),
    ("test_mae", ("test_metrics", "mae")),
    ("test_rmse", ("test_metrics", "rmse")),
    ("test_r2", ("test_metrics", "current_fms_r2")),
    ("test_integer_exact", ("test_metrics", "integer_exact_accuracy")),
    ("test_integer_pm1", ("test_metrics", "integer_off_by_one_accuracy")),
    ("test_high8_f1", ("test_metrics", "caution_high_fms_f1")),
    ("test_high12_f1", ("test_metrics", "warning_high_fms_f1")),
    ("test_low_0_2_bias", ("test_metrics", "low_fms", "0_2", "signed_bias")),
]


REFERENCE_FIELDS = [
    ("test_mae", ("metrics", "mae")),
    ("test_rmse", ("metrics", "rmse")),
    ("test_r2", ("metrics", "current_fms_r2")),
    ("test_high8_f1", ("metrics", "caution_high_fms_f1")),
    ("test_high12_f1", ("metrics", "warning_high_fms_f1")),
    ("test_low_0_2_bias", ("metrics", "low_fms", "0_2", "signed_bias")),
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


def _repo_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _selected_variants(ids: Sequence[str] | None) -> list[dict[str, Any]]:
    if not ids:
        return VARIANTS
    wanted = set(ids)
    selected = [variant for variant in VARIANTS if variant["id"] in wanted]
    missing = wanted - {variant["id"] for variant in selected}
    if missing:
        raise ValueError(f"Unknown variant id(s): {', '.join(sorted(missing))}")
    return selected


def _make_config(base: Mapping[str, Any], variant: Mapping[str, Any], args: argparse.Namespace) -> Path:
    config = deepcopy(base)
    config["runs_dir"] = _display_path(RUNS_DIR)
    config.setdefault("evaluation", {})["no_test_eval"] = True
    config.setdefault("training", {})["epochs"] = int(args.epochs)
    config.setdefault("training", {})["patience"] = int(args.patience)
    config.setdefault("training", {})["batch_size"] = int(args.batch_size)
    config.setdefault("training", {})["seed"] = int(args.seed)
    config.setdefault("training", {})["selection_metric"] = "mae"
    config.setdefault("training", {})["selection_mode"] = "min"
    config.setdefault("data", {})["max_session_points"] = 420
    _deep_update(config.setdefault("model", {}), variant.get("model", {}))
    _deep_update(config.setdefault("loss", {}), variant.get("loss", {}))
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = GENERATED_CONFIG_DIR / f"{variant['id']}.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=False)
    return config_path


def _collect_rows(variants: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        run_dir = RUNS_DIR / str(variant["id"])
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
            "paper_mapping": variant["paper_mapping"],
        }
        for field, path in METRIC_FIELDS:
            row[field] = _metric(merged, path)
        rows.append(row)
    rows.sort(key=lambda item: float(item["val_mae"]) if item.get("val_mae") != "" else float("inf"))
    return rows


def _reference_row() -> dict[str, Any] | None:
    if not REFERENCE_BEST_TEST_METRICS.exists():
        return None
    payload = _read_json(REFERENCE_BEST_TEST_METRICS)
    row: dict[str, Any] = {
        "id": "reference_best_existing",
        "name": "Existing best: calibration-branch zero-anchor highgate t12 w030 pos4 delta2",
        "best_epoch": "",
        "paper_mapping": "Reference test-only comparison; not part of OCE-TS validation search",
    }
    for field, path in REFERENCE_FIELDS:
        row[field] = _metric(payload, path)
    return row


def _write_leaderboard(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "oce_ts_leaderboard.csv"
    fieldnames = ["id", "name", "best_epoch", "paper_mapping"] + [name for name, _ in METRIC_FIELDS]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    comparison_path = REPORT_DIR / "oce_ts_test_comparison.csv"
    comparison_fields = [
        "id",
        "name",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_high8_f1",
        "test_high12_f1",
        "test_low_0_2_bias",
    ]
    comparison_rows = [{field: row.get(field, "") for field in comparison_fields} for row in rows]
    ref = _reference_row()
    if ref is not None:
        comparison_rows.append({field: ref.get(field, "") for field in comparison_fields})
    with comparison_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    report_path = REPORT_DIR / "oce_ts_report.md"
    lines = [
        "# OCE-TS DenseFMS Experiment Report",
        "",
        f"- Base config: `{_display_path(BASE_CONFIG)}`",
        f"- Runs dir: `{_display_path(RUNS_DIR)}`",
        f"- Budget: {len(rows)} completed variants, seed={args.seed}, max_epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}",
        "- Selection: validation MAE only; test split is evaluated after checkpoint selection.",
        "- OCE-TS mapping: TPT truncated Gaussian targets, cumulative ordinal cross-entropy, expected-value decoding.",
        "",
        "## Validation Leaderboard",
        "",
        "| rank | id | best_epoch | val_mae | val_r2 | val_high8_f1 | val_high12_f1 | test_mae | test_high8_f1 | test_high12_f1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    _format(row.get("val_r2", "")),
                    _format(row.get("val_high8_f1", "")),
                    _format(row.get("val_high12_f1", "")),
                    _format(row.get("test_mae", "")),
                    _format(row.get("test_high8_f1", "")),
                    _format(row.get("test_high12_f1", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Existing Best Test Reference", ""])
    if ref is None:
        lines.append(f"- Reference metrics missing: `{_display_path(REFERENCE_BEST_TEST_METRICS)}`")
    else:
        lines.append(
            "- Existing best test: "
            f"MAE={_format(ref.get('test_mae', ''))}, "
            f"RMSE={_format(ref.get('test_rmse', ''))}, "
            f"R2={_format(ref.get('test_r2', ''))}, "
            f"high8_F1={_format(ref.get('test_high8_f1', ''))}, "
            f"high12_F1={_format(ref.get('test_high12_f1', ''))}, "
            f"low_0_2_bias={_format(ref.get('test_low_0_2_bias', ''))}."
        )
    lines.extend(["", "## Variant Mapping", "", "| id | method mapping |", "|---|---|"])
    for variant in VARIANTS:
        lines.append(f"| {variant['id']} | {variant['name']} ({variant['paper_mapping']}) |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {comparison_path}")
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
    parser.add_argument("--variant_ids", nargs="*", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--report_dir", default=None)
    args = parser.parse_args()

    global RUNS_DIR, REPORT_DIR, GENERATED_CONFIG_DIR
    RUNS_DIR = _repo_path(args.runs_dir, RUNS_DIR)
    REPORT_DIR = _repo_path(args.report_dir, REPORT_DIR)
    GENERATED_CONFIG_DIR = RUNS_DIR / "generated_configs"
    variants = _selected_variants(args.variant_ids)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    with BASE_CONFIG.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    for variant in variants:
        config_path = _make_config(base_config, variant, args)
        if args.prepare_only:
            print(f"Prepared {_display_path(config_path)}")
            continue
        run_dir = RUNS_DIR / str(variant["id"])
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
                    str(variant["id"]),
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
            ]
            if args.device:
                command += ["--device", args.device]
            _run(command, cwd=REPO_ROOT)

    if args.prepare_only or args.skip_test:
        return
    rows = _collect_rows(variants)
    if not rows:
        raise RuntimeError("No completed rows found for the OCE-TS report.")
    _write_leaderboard(rows, args)


if __name__ == "__main__":
    main()
