"""Summarize calibration-prior gate/cap behavior from prediction CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


FMS_BINS: Sequence[Tuple[float, float, str]] = (
    (0.0, 2.0, "0_2"),
    (2.0, 5.0, "2_5"),
    (5.0, 10.0, "5_10"),
    (10.0, 15.0, "10_15"),
    (15.0, 20.000001, "15_20"),
)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_one(label: str, path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    required = {
        "target_fms_now",
        "predicted_fms_now",
        "current_calib_prior_gate",
        "current_calib_prior_cap",
        "current_calib_prior_capped_value",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    rows: List[Dict[str, object]] = []
    for lo, hi, name in FMS_BINS:
        g = df[(df["target_fms_now"] >= lo) & (df["target_fms_now"] < hi)]
        rows.append(
            {
                "run_label": label,
                "fms_bin": name,
                "n": int(len(g)),
                "target_mean": float(g["target_fms_now"].mean()),
                "pred_mean": float(g["predicted_fms_now"].mean()),
                "bias": float((g["predicted_fms_now"] - g["target_fms_now"]).mean()),
                "gate_mean": float(g["current_calib_prior_gate"].mean()),
                "gate_p90": float(g["current_calib_prior_gate"].quantile(0.9)),
                "gate_gt_0.5_rate": float((g["current_calib_prior_gate"] > 0.5).mean()),
                "cap_mean": float(g["current_calib_prior_cap"].mean()),
                "capped_mean": float(g["current_calib_prior_capped_value"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibration-prior gate/cap columns by FMS bin.")
    parser.add_argument("--inputs", nargs="+", required=True, help="label=prediction_csv")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for spec in args.inputs:
        if "=" not in spec:
            raise ValueError(f"Input must be label=path, got {spec!r}")
        label, path_text = spec.split("=", 1)
        rows.extend(_summarize_one(label, Path(path_text)))
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "gate_by_fms_bin.csv", rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
