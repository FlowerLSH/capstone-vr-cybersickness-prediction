"""Analyze signed prediction bias by target-FMS bins for online current-FMS CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _parse_input(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip() or Path(path).stem, Path(path)
    path = Path(spec)
    return path.stem, path


def _bin_label(edges: Sequence[float], idx: int) -> str:
    return f"{edges[idx]:g}_{edges[idx + 1]:g}"


def _bin_index(value: float, edges: Sequence[float]) -> int | None:
    if not math.isfinite(value):
        return None
    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == len(edges) - 2:
            if left <= value <= right:
                return idx
        elif left <= value < right:
            return idx
    return None


def _metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "target_mean": float("nan"),
            "pred_mean": float("nan"),
            "bias_mean": float("nan"),
            "bias_median": float("nan"),
            "bias_abs_mean": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "over_rate": float("nan"),
            "under_rate": float("nan"),
            "target_min": float("nan"),
            "target_max": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
        }
    targets = sorted(float(row["target"]) for row in rows)
    preds = sorted(float(row["pred"]) for row in rows)
    biases = [float(row["bias"]) for row in rows]
    abs_errors = [abs(value) for value in biases]
    n = len(rows)
    midpoint = n // 2
    if n % 2:
        median = sorted(biases)[midpoint]
    else:
        ordered_bias = sorted(biases)
        median = 0.5 * (ordered_bias[midpoint - 1] + ordered_bias[midpoint])
    return {
        "n": n,
        "target_mean": sum(targets) / n,
        "pred_mean": sum(preds) / n,
        "bias_mean": sum(biases) / n,
        "bias_median": median,
        "bias_abs_mean": sum(abs_errors) / n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(value * value for value in biases) / n),
        "over_rate": sum(1 for value in biases if value > 0.0) / n,
        "under_rate": sum(1 for value in biases if value < 0.0) / n,
        "target_min": targets[0],
        "target_max": targets[-1],
        "pred_min": preds[0],
        "pred_max": preds[-1],
    }


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def analyze(
    inputs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    bins: Sequence[float],
    target_column: str,
    pred_column: str,
) -> Dict[str, Any]:
    if len(bins) < 2 or any(float(bins[idx]) >= float(bins[idx + 1]) for idx in range(len(bins) - 1)):
        raise ValueError("--bins must be a strictly increasing list.")

    by_bin: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    by_session: Dict[Tuple[str, str, str, str], List[Dict[str, float]]] = defaultdict(list)
    by_input: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    label_order: List[str] = []
    loaded_counts: Dict[str, int] = defaultdict(int)
    loaded_inputs: List[Dict[str, Any]] = []

    for label, path in inputs:
        if label not in label_order:
            label_order.append(label)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            count = 0
            for item in reader:
                target = _float(item.get(target_column))
                pred = _float(item.get(pred_column))
                if not (math.isfinite(target) and math.isfinite(pred)):
                    continue
                bin_idx = _bin_index(target, bins)
                if bin_idx is None:
                    continue
                row = {
                    "target": target,
                    "pred": pred,
                    "bias": pred - target,
                }
                bin_name = _bin_label(bins, bin_idx)
                by_bin[(label, bin_name)].append(row)
                by_input[label].append(row)
                participant_id = str(item.get("participant_id") or "")
                session_id = str(item.get("session_id") or "")
                source_file = str(item.get("source_file") or "")
                by_session[(label, participant_id, session_id, source_file)].append(row)
                count += 1
            loaded_counts[label] += count
            loaded_inputs.append({"label": label, "path": str(path), "valid_rows": count})

    bin_rows: List[Dict[str, Any]] = []
    for label in label_order:
        overall = _metrics(by_input[label])
        bin_rows.append({"run_label": label, "fms_bin": "overall", **overall})
        for idx in range(len(bins) - 1):
            bin_name = _bin_label(bins, idx)
            bin_rows.append({"run_label": label, "fms_bin": bin_name, **_metrics(by_bin[(label, bin_name)])})

    session_rows: List[Dict[str, Any]] = []
    for (label, participant_id, session_id, source_file), rows in sorted(by_session.items()):
        metric = _metrics(rows)
        session_rows.append(
            {
                "run_label": label,
                "participant_id": participant_id,
                "session_id": session_id,
                "source_file": source_file,
                "target_range": metric["target_max"] - metric["target_min"],
                "pred_range": metric["pred_max"] - metric["pred_min"],
                **metric,
            }
        )
    worst_mae_rows = sorted(session_rows, key=lambda row: float(row["mae"]), reverse=True)
    worst_bias_rows = sorted(session_rows, key=lambda row: abs(float(row["bias_mean"])), reverse=True)

    metric_fields = [
        "run_label",
        "fms_bin",
        "n",
        "target_mean",
        "pred_mean",
        "bias_mean",
        "bias_median",
        "bias_abs_mean",
        "mae",
        "rmse",
        "over_rate",
        "under_rate",
        "target_min",
        "target_max",
        "pred_min",
        "pred_max",
    ]
    session_fields = [
        "run_label",
        "participant_id",
        "session_id",
        "source_file",
        "n",
        "target_mean",
        "pred_mean",
        "bias_mean",
        "bias_median",
        "mae",
        "rmse",
        "over_rate",
        "under_rate",
        "target_range",
        "pred_range",
        "target_min",
        "target_max",
        "pred_min",
        "pred_max",
    ]
    _write_csv(out_dir / "signed_bias_by_fms_bin.csv", bin_rows, metric_fields)
    _write_csv(out_dir / "session_signed_bias.csv", session_rows, session_fields)
    _write_csv(out_dir / "worst_sessions_by_mae.csv", worst_mae_rows[:50], session_fields)
    _write_csv(out_dir / "worst_sessions_by_abs_bias.csv", worst_bias_rows[:50], session_fields)

    summary = {
        "inputs": loaded_inputs,
        "label_totals": [{"label": label, "valid_rows": loaded_counts.get(label, 0)} for label in label_order],
        "bins": list(map(float, bins)),
        "target_column": target_column,
        "pred_column": pred_column,
        "bin_rows": bin_rows,
        "worst_sessions_by_mae_top10": worst_mae_rows[:10],
        "worst_sessions_by_abs_bias_top10": worst_bias_rows[:10],
    }
    (out_dir / "signed_bias_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Online Current FMS Signed Bias Analysis",
        "",
        f"- target column: `{target_column}`",
        f"- prediction column: `{pred_column}`",
        f"- bins: `{', '.join(f'{value:g}' for value in bins)}`",
        "",
        "## Bias By Target FMS Bin",
        "",
        "| run | target FMS bin | n | target mean | pred mean | signed bias | MAE | RMSE | over-rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bin_rows:
        lines.append(
            "| "
            f"{row['run_label']} | {row['fms_bin']} | {int(row['n'])} | "
            f"{_format_float(float(row['target_mean']))} | {_format_float(float(row['pred_mean']))} | "
            f"{_format_float(float(row['bias_mean']))} | {_format_float(float(row['mae']))} | "
            f"{_format_float(float(row['rmse']))} | {_format_float(float(row['over_rate']))} |"
        )
    lines.extend(
        [
            "",
            "## Worst Sessions By MAE",
            "",
            "| run | participant | session | n | signed bias | MAE | target range | pred range |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst_mae_rows[:20]:
        lines.append(
            "| "
            f"{row['run_label']} | {row['participant_id']} | {row['session_id']} | {int(row['n'])} | "
            f"{_format_float(float(row['bias_mean']))} | {_format_float(float(row['mae']))} | "
            f"{_format_float(float(row['target_range']))} | {_format_float(float(row['pred_range']))} |"
        )
    lines.append("")
    (out_dir / "signed_bias_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="CSV inputs as label=path or path.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bins", nargs="+", type=float, default=[0.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--target_column", default="target_fms_now")
    parser.add_argument("--pred_column", default="predicted_fms_now")
    args = parser.parse_args()

    inputs = [_parse_input(spec) for spec in args.inputs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(inputs, out_dir, args.bins, args.target_column, args.pred_column)
    print(json.dumps({"out_dir": str(out_dir), "inputs": summary["inputs"]}, indent=2))


if __name__ == "__main__":
    main()
