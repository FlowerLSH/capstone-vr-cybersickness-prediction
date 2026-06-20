"""Search causal prediction filters for online current-FMS outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


KEY_COLUMNS = ["session_id", "current_index", "current_time", "target_fms_now"]
DEFAULT_ORDINAL_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20]


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


def _apply_causal_filter(
    frame: pd.DataFrame,
    alpha: float,
    trend_beta: float,
    raw_weight: float,
    scale: float,
    bias: float,
) -> pd.DataFrame:
    alpha = float(alpha)
    trend_beta = float(trend_beta)
    raw_weight = float(raw_weight)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if not (0.0 <= raw_weight <= 1.0):
        raise ValueError("raw_weight must be in [0, 1].")

    out = frame.sort_values(["session_id", "current_index", "current_time"]).copy()
    filtered = np.zeros(len(out), dtype=np.float64)
    raw = out["predicted_fms_now"].to_numpy(dtype=np.float64)
    row_positions = np.arange(len(out))
    for _session_id, group in out.groupby("session_id", sort=False):
        positions = row_positions[group.index.to_numpy()]
        prev_ema = raw[positions[0]]
        prev_prev_ema = prev_ema
        for pos in positions:
            ema = alpha * raw[pos] + (1.0 - alpha) * prev_ema
            trend = ema - prev_prev_ema
            boosted = ema + trend_beta * trend
            filtered[pos] = raw_weight * raw[pos] + (1.0 - raw_weight) * boosted
            prev_prev_ema = prev_ema
            prev_ema = ema
    out["base_predicted_fms_now"] = out["predicted_fms_now"].astype(float)
    corrected = float(scale) * filtered + float(bias)
    out["predicted_fms_now"] = np.clip(corrected, 0.0, 20.0)
    out["fms_absolute_error"] = np.abs(
        out["predicted_fms_now"].to_numpy(dtype=np.float64) - out["target_fms_now"].to_numpy(dtype=np.float64)
    )
    if "ordinal_bin_pred" in out.columns:
        bins = np.asarray(DEFAULT_ORDINAL_BINS, dtype=np.float64)
        out["ordinal_bin_pred"] = np.digitize(out["predicted_fms_now"], bins[1:-1], right=False).astype(int)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns and "alarm_warning_high_fms" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out["alarm_warning_high_fms"] | rapid
    return out


def _float_grid(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search causal EMA/trend filters on saved validation predictions.")
    parser.add_argument("--input", required=True, help="Run directory or prediction CSV.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", default="causal_filter")
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--trend_betas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--raw_weights", nargs="+", type=float, default=[0.0])
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--biases", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()

    input_path = _prediction_path(Path(args.input), args.split)
    base = pd.read_csv(input_path)
    missing = set(KEY_COLUMNS + ["predicted_fms_now"]) - set(base.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")

    rows: List[Dict[str, float]] = []
    best_frame: pd.DataFrame | None = None
    best_payload: Dict[str, object] | None = None
    for alpha in _float_grid(args.alphas):
        for trend_beta in _float_grid(args.trend_betas):
            for raw_weight in _float_grid(args.raw_weights):
                for scale in _float_grid(args.scales):
                    for bias in _float_grid(args.biases):
                        candidate = _apply_causal_filter(
                            base,
                            alpha=alpha,
                            trend_beta=trend_beta,
                            raw_weight=raw_weight,
                            scale=scale,
                            bias=bias,
                        )
                        metrics = _metrics(candidate)
                        row = {
                            "alpha": float(alpha),
                            "trend_beta": float(trend_beta),
                            "raw_weight": float(raw_weight),
                            "scale": float(scale),
                            "bias": float(bias),
                            **metrics,
                        }
                        rows.append(row)
                        if best_payload is None or float(metrics["mae"]) < float(best_payload["metrics"]["mae"]):
                            best_frame = candidate
                            best_payload = {
                                "task": {
                                    "causal_prediction_filter": True,
                                    "test_eval_skipped": args.split != "test",
                                },
                                "split": args.split,
                                "input": str(args.input),
                                "alpha": float(alpha),
                                "trend_beta": float(trend_beta),
                                "raw_weight": float(raw_weight),
                                "scale": float(scale),
                                "bias": float(bias),
                                "metrics": metrics,
                            }

    if best_frame is None or best_payload is None:
        raise RuntimeError("No filter candidates were evaluated.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["mae", "rmse"]).to_csv(out_dir / "causal_filter_search.csv", index=False)
    best_frame = best_frame.copy()
    best_frame["run_name"] = args.label
    best_frame["model_name"] = "online_current_causal_prediction_filter"
    best_frame["split"] = args.split
    best_frame.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
