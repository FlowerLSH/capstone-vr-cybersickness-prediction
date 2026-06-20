"""Evaluate validation gates for calibration-summary early-fusion candidates.

This script reads validation prediction CSVs only. It does not run training and
does not evaluate the original test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_online_current_goal_metrics import _parse_input, _row


DEFAULT_INPUTS = [
    "range_original=runs/head_redesign_ablation_0513/range_scaled_delta2_120_seed42/val_predictions.csv",
    (
        "earlyfusion_add="
        "runs/overnight_current_fms_goal_0514_120s/"
        "range_calsummary_earlyfusion_add_goalcomp_seed42/val_predictions.csv"
    ),
    (
        "earlyfusion_add_low002="
        "runs/overnight_current_fms_goal_0514_120s/"
        "range_calsummary_earlyfusion_add_low002_goalcomp_seed42/val_predictions.csv"
    ),
]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _safe_row(label: str, path: Path, pred_column: str, thresholds: Sequence[float]) -> Dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "status": "missing",
            "gate_pass_count": 0,
            "test_candidate": False,
        }
    row = _row(label, path, pred_column=pred_column, low_bin_max=2.0, thresholds=thresholds)
    row["status"] = "ok"
    return row


def _apply_gates(
    row: Dict[str, Any],
    *,
    mae_max: float,
    r2_min: float,
    strict_low_bias_max: float,
    high12_f1_min: float,
    required_pass_count: int,
    require_strict_low_gate: bool,
) -> Dict[str, Any]:
    if row.get("status") != "ok":
        row.update(
            {
                "gate_mae": False,
                "gate_r2": False,
                "gate_strict_low_bias": False,
                "gate_high12_f1": False,
                "gate_pass_count": 0,
                "test_candidate": False,
            }
        )
        return row
    high12_f1 = row.get("high12_f1", float("nan"))
    gates = {
        "gate_mae": float(row["mae"]) <= float(mae_max),
        "gate_r2": float(row["r2"]) >= float(r2_min),
        "gate_strict_low_bias": float(row["strict_low_signed_bias"]) <= float(strict_low_bias_max),
        "gate_high12_f1": float(high12_f1) >= float(high12_f1_min),
    }
    pass_count = int(sum(bool(value) for value in gates.values()))
    row.update(gates)
    row["gate_pass_count"] = pass_count
    row["test_candidate"] = pass_count >= int(required_pass_count) and (
        bool(gates["gate_strict_low_bias"]) or not bool(require_strict_low_gate)
    )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Calibration-Summary Early-Fusion Validation Gate Report",
        "",
        "Validation prediction CSV만 평가한다. Original test set은 사용하지 않는다.",
        "",
        "## Gates",
        "",
        f"- MAE <= `{args.mae_max}`",
        f"- R2 >= `{args.r2_min}`",
        f"- strict `0<=FMS<2` signed bias <= `{args.strict_low_bias_max}`",
        f"- high12 F1 >= `{args.high12_f1_min}`",
        f"- test candidate if pass count >= `{args.required_pass_count}`",
        f"- require strict low gate: `{bool(args.require_strict_low_gate)}`",
        "",
        "## Results",
        "",
        "| label | status | MAE | R2 | strict low bias | high8 F1 | high12 F1 | pass count | test candidate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("label", "")),
                    str(row.get("status", "")),
                    _fmt(row.get("mae", float("nan"))),
                    _fmt(row.get("r2", float("nan"))),
                    _fmt(row.get("strict_low_signed_bias", float("nan"))),
                    _fmt(row.get("high8_f1", float("nan"))),
                    _fmt(row.get("high12_f1", float("nan"))),
                    str(row.get("gate_pass_count", 0)),
                    str(bool(row.get("test_candidate", False))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `missing`은 아직 해당 validation run이 끝나지 않았거나 prediction CSV가 생성되지 않았다는 뜻이다.",
            "- `test_candidate=True`는 original test 평가 후보라는 뜻이지, test를 이미 사용했다는 뜻이 아니다.",
            "- C3 판단은 helper `target<=2`가 아니라 `strict_low_signed_bias`만 사용한다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in [_parse_input(spec) for spec in args.inputs]:
        row = _safe_row(label, path, pred_column=args.pred_column, thresholds=args.thresholds)
        rows.append(
            _apply_gates(
                row,
                mae_max=args.mae_max,
                r2_min=args.r2_min,
                strict_low_bias_max=args.strict_low_bias_max,
                high12_f1_min=args.high12_f1_min,
                required_pass_count=args.required_pass_count,
                require_strict_low_gate=bool(args.require_strict_low_gate),
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_dir", default="reports/overnight_current_fms_goal_0514_120s/calsummary_earlyfusion_gate_eval")
    parser.add_argument("--pred_column", default="predicted_fms_now")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--mae_max", type=float, default=1.70)
    parser.add_argument("--r2_min", type=float, default=0.70)
    parser.add_argument("--strict_low_bias_max", type=float, default=2.80)
    parser.add_argument("--high12_f1_min", type=float, default=0.7771)
    parser.add_argument("--required_pass_count", type=int, default=2)
    parser.add_argument("--require_strict_low_gate", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = evaluate(args)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "validation_gate_metrics.csv", rows)
    _write_markdown(out_dir / "validation_gate_report.md", rows, args)
    (out_dir / "validation_gate_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
