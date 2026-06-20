"""Cluster analysis for low-FMS overprediction failures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_inputs(items: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).parent.name
        parsed.append((label.strip(), Path(path)))
    return parsed


def _bin_numeric(values: pd.Series, edges: Sequence[float], labels: Sequence[str], missing: str = "missing") -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(numeric, bins=list(edges), labels=list(labels), include_lowest=True, right=False)
    return binned.astype("object").where(numeric.notna(), missing).astype(str)


def _scenario_from_source(source: object) -> str:
    text = str(source or "").lower()
    if "backward_texture" in text:
        return "backward_texture"
    if "forward_texture" in text:
        return "forward_texture"
    if "high_density" in text:
        return "high_density"
    if "low_density" in text:
        return "low_density"
    if "forward_whiteline" in text:
        return "forward_whiteline"
    if "reverse_optical_flow" in text:
        return "reverse_optical_flow"
    return "unknown"


def _cluster_rows(df: pd.DataFrame, group_col: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    optional_mean_cols = [
        "predicted_fms_pre_low_suppressor",
        "current_low_suppressor_correction",
        "current_low_suppressor_gate",
    ]
    for (run, split, group), g in df.groupby(["run_label", "split", group_col], dropna=False):
        diff = g["predicted_fms_now"].astype(float) - g["target_fms_now"].astype(float)
        row = {
            "run_label": run,
            "split": split,
            "group_by": group_col,
            "group": group,
            "n": int(len(g)),
            "target_mean": float(g["target_fms_now"].mean()),
            "pred_mean": float(g["predicted_fms_now"].mean()),
            "bias_mean": float(diff.mean()),
            "bias_median": float(diff.median()),
            "mae": float(diff.abs().mean()),
            "over_rate": float((diff > 0).mean()),
            "anchor_mean": float(g["anchor_fms"].mean()) if "anchor_fms" in g else float("nan"),
            "elapsed_mean": float(g["elapsed_seconds"].mean()) if "elapsed_seconds" in g else float("nan"),
        }
        for col in optional_mean_cols:
            if col in g.columns:
                values = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
                row[f"{col}_median"] = float(values.median()) if values.notna().any() else float("nan")
        rows.append(row)
    rows.sort(key=lambda row: (str(row["group_by"]), str(row["run_label"]), str(row["split"]), -int(row["n"])))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _write_report(path: Path, cluster_rows: Sequence[Mapping[str, object]], session_rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Low-FMS Failure Cluster Report",
        "",
        "True FMS가 `[0, 2)`인 row만 대상으로 `predicted_fms_now - target_fms_now` signed bias를 집계했다.",
        "",
        "## Anchor Bin Summary",
        "",
        "| run | split | anchor bin | n | target mean | pred mean | bias | MAE | over rate | pre-low | low corr | low gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cluster_rows:
        if row.get("group_by") != "anchor_bin":
            continue
        lines.append(
            "| {run} | {split} | {group} | {n} | {target} | {pred} | {bias} | {mae} | {over} | {pre_low} | {corr} | {gate} |".format(
                run=row["run_label"],
                split=row["split"],
                group=row["group"],
                n=int(row["n"]),
                target=_fmt(row["target_mean"]),
                pred=_fmt(row["pred_mean"]),
                bias=_fmt(row["bias_mean"]),
                mae=_fmt(row["mae"]),
                over=_fmt(row["over_rate"]),
                pre_low=_fmt(row.get("predicted_fms_pre_low_suppressor_mean", float("nan"))),
                corr=_fmt(row.get("current_low_suppressor_correction_mean", float("nan"))),
                gate=_fmt(row.get("current_low_suppressor_gate_mean", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Worst Sessions",
            "",
            "| run | split | session | n | target mean | pred mean | bias | MAE | anchor mean | elapsed mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in session_rows[:40]:
        lines.append(
            "| {run} | {split} | {session} | {n} | {target} | {pred} | {bias} | {mae} | {anchor} | {elapsed} |".format(
                run=row["run_label"],
                split=row["split"],
                session=row["group"],
                n=int(row["n"]),
                target=_fmt(row["target_mean"]),
                pred=_fmt(row["pred_mean"]),
                bias=_fmt(row["bias_mean"]),
                mae=_fmt(row["mae"]),
                anchor=_fmt(row["anchor_mean"]),
                elapsed=_fmt(row["elapsed_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(inputs: Sequence[Tuple[str, Path]], out_dir: Path, target_col: str, pred_col: str) -> None:
    frames: List[pd.DataFrame] = []
    for label, path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if target_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"{path} must contain {target_col} and {pred_col}.")
        df = df.copy()
        df["run_label"] = label
        if "split" not in df.columns:
            df["split"] = path.parent.name
        if pred_col != "predicted_fms_now":
            df["predicted_fms_now"] = pd.to_numeric(df[pred_col], errors="coerce")
        if target_col != "target_fms_now":
            df["target_fms_now"] = pd.to_numeric(df[target_col], errors="coerce")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged["target_fms_now"] = pd.to_numeric(merged["target_fms_now"], errors="coerce")
    merged["predicted_fms_now"] = pd.to_numeric(merged["predicted_fms_now"], errors="coerce")
    low = merged[
        np.isfinite(merged["target_fms_now"])
        & np.isfinite(merged["predicted_fms_now"])
        & (merged["target_fms_now"] >= 0.0)
        & (merged["target_fms_now"] < 2.0)
    ].copy()
    if low.empty:
        raise ValueError("No rows found with 0 <= target_fms_now < 2.")
    if "anchor_fms" not in low.columns:
        low["anchor_fms"] = np.nan
    if "current_index" in low.columns and "calibration_steps" in low.columns and "sampling_interval" in low.columns:
        low["elapsed_seconds"] = (
            pd.to_numeric(low["current_index"], errors="coerce") - pd.to_numeric(low["calibration_steps"], errors="coerce")
        ) * pd.to_numeric(low["sampling_interval"], errors="coerce")
    elif "current_time" in low.columns and "calibration_seconds" in low.columns:
        low["elapsed_seconds"] = pd.to_numeric(low["current_time"], errors="coerce") - pd.to_numeric(
            low["calibration_seconds"], errors="coerce"
        )
    else:
        low["elapsed_seconds"] = np.nan
    low["anchor_bin"] = _bin_numeric(
        low["anchor_fms"],
        edges=[-np.inf, 1.0, 2.0, 5.0, 8.0, 12.0, np.inf],
        labels=["<=1", "1_2", "2_5", "5_8", "8_12", ">=12"],
    )
    low["elapsed_bin"] = _bin_numeric(
        low["elapsed_seconds"],
        edges=[-np.inf, 30.0, 60.0, 90.0, 120.0, np.inf],
        labels=["0_30s", "30_60s", "60_90s", "90_120s", ">=120s"],
    )
    low["mssq_bin"] = _bin_numeric(
        low["mssq"] if "mssq" in low.columns else pd.Series(np.nan, index=low.index),
        edges=[-np.inf, 10.0, 20.0, 30.0, np.inf],
        labels=["<10", "10_20", "20_30", ">=30"],
    )
    low["age_bin"] = _bin_numeric(
        low["age"] if "age" in low.columns else pd.Series(np.nan, index=low.index),
        edges=[-np.inf, 22.0, 26.0, 30.0, np.inf],
        labels=["<22", "22_26", "26_30", ">=30"],
    )
    low["gender_group"] = low["gender"].astype(str) if "gender" in low.columns else "missing"
    low["scenario_group"] = low["source_file"].map(_scenario_from_source) if "source_file" in low.columns else "missing"

    cluster_rows: List[Dict[str, object]] = []
    for group_col in ["anchor_bin", "elapsed_bin", "mssq_bin", "age_bin", "gender_group", "scenario_group"]:
        cluster_rows.extend(_cluster_rows(low, group_col))

    session_rows = _cluster_rows(low, "session_id")
    session_rows.sort(key=lambda row: (str(row["split"]), str(row["run_label"]), -float(row["bias_mean"]), -int(row["n"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "low_failure_clusters.csv", cluster_rows)
    _write_csv(out_dir / "low_failure_worst_sessions.csv", session_rows)
    low.to_csv(out_dir / "low_failure_rows.csv", index=False)
    _write_report(out_dir / "low_failure_cluster_report.md", cluster_rows, session_rows)
    print({"out_dir": str(out_dir), "low_rows": int(len(low)), "inputs": [label for label, _ in inputs]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low-FMS overprediction clusters from online-current prediction CSVs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Items formatted as label=path or path.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_column", default="target_fms_now")
    parser.add_argument("--pred_column", default="predicted_fms_now")
    args = parser.parse_args()
    analyze(
        _parse_inputs(args.inputs),
        Path(args.out_dir),
        target_col=str(args.target_column),
        pred_col=str(args.pred_column),
    )


if __name__ == "__main__":
    main()
