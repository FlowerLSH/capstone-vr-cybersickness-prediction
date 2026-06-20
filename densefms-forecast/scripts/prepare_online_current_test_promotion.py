"""Prepare final-test evaluation commands from validation gate results.

This script is intentionally dry-run only. It reads validation gate metrics,
selects candidates marked as test_candidate=True, and writes reproducible
final-test evaluation commands without executing them.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


DEFAULT_GATE_CSV = (
    "reports/overnight_current_fms_goal_0514_120s/"
    "calsummary_earlyfusion_gate_eval/validation_gate_metrics.csv"
)
DEFAULT_OUT_DIR = "reports/overnight_current_fms_goal_0514_120s/test_promotion_commands"
DEFAULT_FEATURES = "reports/overnight_current_fms_goal_0514_120s/calibration_summary_features_train_val.json"


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z %z")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _finite_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def select_candidates(rows: Sequence[Mapping[str, object]], *, max_candidates: int = 1) -> List[Mapping[str, object]]:
    candidates = [
        row
        for row in rows
        if str(row.get("status", "")).lower() == "ok" and _truthy(row.get("test_candidate", False))
    ]
    candidates.sort(
        key=lambda row: (
            -int(_finite_float(row.get("gate_pass_count"), 0.0)),
            _finite_float(row.get("strict_low_signed_bias"), float("inf")),
            _finite_float(row.get("mae"), float("inf")),
            -_finite_float(row.get("r2"), float("-inf")),
            -_finite_float(row.get("high12_f1"), float("-inf")),
        )
    )
    return list(candidates[: max(0, int(max_candidates))])


def build_eval_command(args: argparse.Namespace, row: Mapping[str, object]) -> List[str]:
    pred_path = _normalize_path(row.get("path", ""))
    run_dir = pred_path.parent
    checkpoint = run_dir / "best.pt"
    split_file = run_dir / "split.json"
    command = [
        args.python,
        "-m",
        "src.densefms_forecast.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--data_dir",
        args.data_dir,
        "--split",
        args.split,
        "--split_file",
        str(split_file),
        "--batch_size",
        str(int(args.batch_size)),
        "--max_session_points",
        str(int(args.max_session_points)),
        "--save_predictions",
    ]
    if args.calibration_residual_features_path:
        command.extend(["--calibration_residual_features_path", args.calibration_residual_features_path])
    return command


def _read_gate_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "source_val_predictions", "checkpoint", "split_file", "command"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_report(
    path: Path,
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, object]],
    command_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Final-Test Promotion Commands",
        "",
        f"작성일: {_now()}",
        "",
        "Validation gate 결과만 읽어서 final-test evaluation command를 생성한다. 이 스크립트는 test를 실행하지 않는다.",
        "",
        "## Inputs",
        "",
        f"- gate csv: `{args.gate_csv}`",
        f"- split: `{args.split}`",
        f"- max_session_points: `{args.max_session_points}`",
        "",
        "## Selected Candidates",
        "",
    ]
    if not selected:
        lines.extend(
            [
                "No candidate passed the validation gate. Do not run original test.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| label | MAE | R2 | strict low bias | high12 F1 | gate pass count |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                "| {label} | {mae} | {r2} | {bias} | {h12} | {passes} |".format(
                    label=row.get("label", ""),
                    mae=row.get("mae", ""),
                    r2=row.get("r2", ""),
                    bias=row.get("strict_low_signed_bias", ""),
                    h12=row.get("high12_f1", ""),
                    passes=row.get("gate_pass_count", ""),
                )
            )
        lines.append("")
        lines.append("## Commands")
        lines.append("")
        for row in command_rows:
            lines.extend(
                [
                    f"### {row['label']}",
                    "",
                    "```bash",
                    str(row["command"]),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "## Rule",
                "",
                "Run at most one final-test command after validation selection. Do not compare multiple test outputs for selection.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_csv", default=DEFAULT_GATE_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", default="/mnt/c/Users/rio/AppData/Local/Programs/Python/Python310/python.exe")
    parser.add_argument("--data_dir", default="DenseFMS/Dataset")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--calibration_residual_features_path", default=DEFAULT_FEATURES)
    parser.add_argument("--max_candidates", type=int, default=1)
    parser.add_argument("--require_candidate", action="store_true")
    args = parser.parse_args()

    gate_rows = _read_gate_rows(Path(args.gate_csv))
    selected = select_candidates(gate_rows, max_candidates=args.max_candidates)
    if args.require_candidate and not selected:
        raise SystemExit("No validation-gated candidate is eligible for final test.")

    command_rows: List[Dict[str, object]] = []
    for row in selected:
        command = build_eval_command(args, row)
        pred_path = _normalize_path(row.get("path", ""))
        command_rows.append(
            {
                "label": row.get("label", ""),
                "source_val_predictions": str(pred_path),
                "checkpoint": str(pred_path.parent / "best.pt"),
                "split_file": str(pred_path.parent / "split.json"),
                "command": shlex.join(command),
            }
        )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "test_promotion_commands.csv", command_rows)
    _write_report(out_dir / "test_promotion_commands.md", args, selected, command_rows)
    print(f"wrote {len(command_rows)} final-test promotion command(s) to {out_dir}")


if __name__ == "__main__":
    main()
