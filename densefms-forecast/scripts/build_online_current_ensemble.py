"""Build prediction-level ensembles for online current-FMS candidates.

The script averages already-saved prediction CSVs. It does not load
checkpoints, train models, or evaluate a split by itself, so it is safe for
validation-only model selection as long as the input CSVs are validation CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


KEY_COLUMNS = ["session_id", "current_index", "current_time", "target_fms_now"]
DEFAULT_ORDINAL_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20]


def _parse_member(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path)
    path = Path(value)
    return path.parent.name, path


def _prediction_path(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    direct = path / f"{split}_predictions.csv"
    if direct.exists():
        return direct
    nested = path / f"eval_{split}" / f"{split}_predictions.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Missing {split} prediction CSV under {path}")


def _load_member(label: str, path: Path, split: str) -> pd.DataFrame:
    csv_path = _prediction_path(path, split)
    frame = pd.read_csv(csv_path)
    missing = set(KEY_COLUMNS + ["predicted_fms_now"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    frame = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
    frame[f"predicted_fms_now__{label}"] = frame["predicted_fms_now"].astype(float)
    return frame


def _validate_alignment(base: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    if len(base) != len(other):
        raise ValueError(f"Member {label} has {len(other)} rows; expected {len(base)}.")
    for column in KEY_COLUMNS:
        left = base[column].to_numpy()
        right = other[column].to_numpy()
        if np.issubdtype(base[column].dtype, np.number) and np.issubdtype(other[column].dtype, np.number):
            aligned = np.allclose(left.astype(np.float64), right.astype(np.float64), rtol=0.0, atol=1e-7, equal_nan=True)
        else:
            aligned = np.array_equal(left, right)
        if not aligned:
            raise ValueError(f"Member {label} is not aligned on column {column}.")


def _normalise_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}.")
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("Weights must be non-negative.")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    return arr / total


def _ordinal_bins(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    edges = np.asarray(list(bins), dtype=np.float64)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    target = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(target)),
        "n": float(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Average online current-FMS prediction CSVs.")
    parser.add_argument("--members", nargs="+", required=True, help="Member specs as label=run_dir_or_csv.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--ordinal_bins", nargs="*", type=float, default=DEFAULT_ORDINAL_BINS)
    args = parser.parse_args()

    members = [_parse_member(value) for value in args.members]
    if len(members) < 2:
        raise ValueError("At least two members are required for an ensemble.")
    weights = _normalise_weights(args.weights, len(members))

    labels: List[str] = []
    frames: List[pd.DataFrame] = []
    for label, path in members:
        labels.append(label)
        frames.append(_load_member(label, path, args.split))

    base = frames[0].copy()
    for label, frame in zip(labels[1:], frames[1:]):
        _validate_alignment(base, frame, label)

    member_predictions = np.vstack(
        [frame[f"predicted_fms_now__{label}"].to_numpy(dtype=np.float64) for label, frame in zip(labels, frames)]
    )
    prediction = weights @ member_predictions

    out = base.drop(columns=[f"predicted_fms_now__{labels[0]}"])
    out["run_name"] = args.label or Path(args.out_dir).name
    out["model_name"] = "online_current_prediction_ensemble"
    out["split"] = args.split
    out["predicted_fms_now"] = prediction
    out["fms_absolute_error"] = np.abs(out["predicted_fms_now"].to_numpy(dtype=np.float64) - out["target_fms_now"].to_numpy(dtype=np.float64))
    if "ordinal_bin_pred" in out.columns:
        out["ordinal_bin_pred"] = _ordinal_bins(out["predicted_fms_now"].to_numpy(dtype=np.float64), args.ordinal_bins)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out["alarm_warning_high_fms"] | rapid
    for label, frame in zip(labels, frames):
        out[f"member_pred_{label}"] = frame[f"predicted_fms_now__{label}"].to_numpy(dtype=np.float64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)

    payload = {
        "task": {"ensemble": True, "test_eval_skipped": args.split != "test"},
        "split": args.split,
        "members": [{"label": label, "path": str(path), "weight": float(weight)} for (label, path), weight in zip(members, weights)],
        "metrics": _metrics(out),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"Saved ensemble predictions to {out_dir}")


if __name__ == "__main__":
    main()
