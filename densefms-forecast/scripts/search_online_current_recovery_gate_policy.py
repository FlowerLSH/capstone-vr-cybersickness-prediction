"""Search validation-only recovery-gate policies for online-current predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


KEY_COLUMNS = ["participant_id", "session_id", "current_index"]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(float)
    y_pred = y_pred[mask].astype(float)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = 1.0 - float(np.sum(err * err)) / denom if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(y_true.size)}


def _high_fms_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(float)
    y_pred = y_pred[mask].astype(float)
    true_high = y_true >= float(threshold)
    pred_high = y_pred >= float(threshold)
    tp = float(np.sum(true_high & pred_high))
    fp = float(np.sum(~true_high & pred_high))
    fn = float(np.sum(true_high & ~pred_high))
    tn = float(np.sum(~true_high & ~pred_high))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        f"high{threshold:g}_precision": precision,
        f"high{threshold:g}_recall": recall,
        f"high{threshold:g}_f1": f1,
        f"high{threshold:g}_false_positive_rate": fp / (fp + tn) if fp + tn > 0 else 0.0,
        f"high{threshold:g}_false_negative_rate": fn / (tp + fn) if tp + fn > 0 else 0.0,
    }


def _goal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    row = _regression_metrics(y_true, y_pred)
    low = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true >= 0.0) & (y_true < 2.0)
    if np.any(low):
        low_bias = float(np.mean(y_pred[low] - y_true[low]))
        low_mae = float(np.mean(np.abs(y_pred[low] - y_true[low])))
        low_n = int(np.sum(low))
    else:
        low_bias = float("nan")
        low_mae = float("nan")
        low_n = 0
    row.update({"original_low_0_2_n": low_n, "original_low_0_2_bias": low_bias, "original_low_0_2_mae": low_mae})
    row.update(_high_fms_metrics(y_true, y_pred, 8.0))
    row.update(_high_fms_metrics(y_true, y_pred, 12.0))
    row["goal_composite_strict120"] = (
        float(row["mae"])
        + 0.25 * max(0.0, float(low_bias) - 2.5)
        + 2.0 * max(0.0, 0.70 - float(row["r2"]))
        + 0.5 * max(0.0, 0.76 - float(row["high12_f1"]))
    )
    return row


def _policy_grid() -> Iterable[Dict[str, object]]:
    yield {"method": "base", "anchor_min": float("inf"), "gate_threshold": 1.1, "strength": 0.0}
    yield {"method": "min_gate_pred", "anchor_min": 0.0, "gate_threshold": 0.0, "strength": 1.0}
    for anchor_min in [2.0, 3.0, 5.0, 8.0]:
        for gate_threshold in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60]:
            yield {
                "method": "min_gate_pred",
                "anchor_min": anchor_min,
                "gate_threshold": gate_threshold,
                "strength": 1.0,
            }
            for strength in [0.25, 0.50, 1.0, 1.5, 2.0]:
                yield {
                    "method": "subtract_correction",
                    "anchor_min": anchor_min,
                    "gate_threshold": gate_threshold,
                    "strength": strength,
                }


def _load_joined(base_path: Path, gate_path: Path) -> pd.DataFrame:
    base = pd.read_csv(base_path)
    gate_cols = KEY_COLUMNS + [
        "predicted_fms_now",
        "predicted_fms_pre_low_suppressor",
        "current_low_suppressor_correction",
        "current_low_suppressor_gate",
    ]
    gate = pd.read_csv(gate_path)
    missing = [col for col in KEY_COLUMNS if col not in base.columns or col not in gate.columns]
    if missing:
        raise ValueError(f"Both CSVs must contain key columns. Missing: {missing}")
    gate = gate[[col for col in gate_cols if col in gate.columns]].copy()
    rename = {
        "predicted_fms_now": "gate_predicted_fms_now",
        "predicted_fms_pre_low_suppressor": "gate_predicted_fms_pre_low_suppressor",
        "current_low_suppressor_correction": "gate_current_low_suppressor_correction",
        "current_low_suppressor_gate": "gate_current_low_suppressor_gate",
    }
    gate = gate.rename(columns=rename)
    joined = base.merge(gate, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(joined) != len(base):
        raise ValueError(f"Join lost rows: base={len(base)} joined={len(joined)}")
    return joined


def _apply_policy(df: pd.DataFrame, policy: Mapping[str, object]) -> np.ndarray:
    base_pred = pd.to_numeric(df["predicted_fms_now"], errors="coerce").to_numpy(dtype=float)
    anchor = pd.to_numeric(df.get("anchor_fms", np.nan), errors="coerce").to_numpy(dtype=float)
    gate = pd.to_numeric(df.get("gate_current_low_suppressor_gate", np.nan), errors="coerce").to_numpy(dtype=float)
    gate_pred = pd.to_numeric(df.get("gate_predicted_fms_now", np.nan), errors="coerce").to_numpy(dtype=float)
    correction = pd.to_numeric(df.get("gate_current_low_suppressor_correction", np.nan), errors="coerce").to_numpy(dtype=float)
    pred = base_pred.copy()
    method = str(policy["method"])
    active = np.isfinite(anchor) & np.isfinite(gate) & (anchor >= float(policy["anchor_min"])) & (
        gate >= float(policy["gate_threshold"])
    )
    if method == "base":
        return pred
    if method == "min_gate_pred":
        usable = active & np.isfinite(gate_pred)
        pred[usable] = np.minimum(pred[usable], gate_pred[usable])
    elif method == "subtract_correction":
        usable = active & np.isfinite(correction)
        pred[usable] = pred[usable] - float(policy["strength"]) * correction[usable]
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.clip(pred, 0.0, 20.0)


def _evaluate_policy(df: pd.DataFrame, policy: Mapping[str, object]) -> Dict[str, object]:
    y_true = pd.to_numeric(df["target_fms_now"], errors="coerce").to_numpy(dtype=float)
    y_pred = _apply_policy(df, policy)
    row: Dict[str, object] = dict(policy)
    row.update(_goal_metrics(y_true, y_pred))
    changed = np.isfinite(y_pred) & np.isfinite(pd.to_numeric(df["predicted_fms_now"], errors="coerce").to_numpy(dtype=float))
    base_pred = pd.to_numeric(df["predicted_fms_now"], errors="coerce").to_numpy(dtype=float)
    row["changed_rate"] = float(np.mean(np.abs(y_pred[changed] - base_pred[changed]) > 1e-8)) if np.any(changed) else float("nan")
    row["mean_delta_vs_base"] = float(np.nanmean(y_pred - base_pred))
    return row


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


def _select(rows: Sequence[Mapping[str, object]], mode: str) -> Dict[str, object]:
    candidates = pd.DataFrame(rows)
    mode = str(mode)
    if mode == "low_target_then_mae":
        low_ok = candidates[candidates["original_low_0_2_bias"] <= 2.5].copy()
        if not low_ok.empty:
            return dict(
                low_ok.sort_values(
                    ["mae", "r2", "high12_f1", "goal_composite_strict120"],
                    ascending=[True, False, False, True],
                ).iloc[0]
            )
    elif mode != "composite":
        raise ValueError("selection_mode must be one of: composite, low_target_then_mae.")
    return dict(
        candidates.sort_values(
            ["goal_composite_strict120", "mae", "original_low_0_2_bias", "high12_f1"],
            ascending=[True, True, True, False],
        ).iloc[0]
    )


def _write_predictions(path: Path, df: pd.DataFrame, policy: Mapping[str, object], run_name: str) -> None:
    out = df.copy()
    pred = _apply_policy(out, policy)
    out["base_predicted_fms_now"] = out["predicted_fms_now"]
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pd.to_numeric(out["predicted_fms_now"], errors="coerce") - pd.to_numeric(out["target_fms_now"], errors="coerce"))
    out["run_name"] = run_name
    out["recovery_gate_policy_method"] = str(policy["method"])
    out["recovery_gate_policy_anchor_min"] = float(policy["anchor_min"])
    out["recovery_gate_policy_gate_threshold"] = float(policy["gate_threshold"])
    out["recovery_gate_policy_strength"] = float(policy["strength"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _write_report(path: Path, selected_val: Mapping[str, object], selected_test: Optional[Mapping[str, object]]) -> None:
    lines = [
        "# Recovery Gate Policy Search",
        "",
        "Validation prediction에서만 policy를 선택하고, 선택된 policy를 test prediction에 적용했다.",
        "",
        "## Selected Validation Policy",
        "",
        "| method | anchor min | gate threshold | strength | val MAE | val R2 | val original 0_2 bias | val high8 F1 | val high12 F1 | composite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {method} | {anchor:.2f} | {gate:.2f} | {strength:.2f} | {mae:.4f} | {r2:.4f} | {bias:+.4f} | {h8:.4f} | {h12:.4f} | {comp:.4f} |".format(
            method=selected_val["method"],
            anchor=float(selected_val["anchor_min"]),
            gate=float(selected_val["gate_threshold"]),
            strength=float(selected_val["strength"]),
            mae=float(selected_val["mae"]),
            r2=float(selected_val["r2"]),
            bias=float(selected_val["original_low_0_2_bias"]),
            h8=float(selected_val["high8_f1"]),
            h12=float(selected_val["high12_f1"]),
            comp=float(selected_val["goal_composite_strict120"]),
        ),
    ]
    if selected_test is not None:
        lines.extend(
            [
                "",
                "## Applied Test Metrics",
                "",
                "| test MAE | test R2 | test original 0_2 bias | test high8 F1 | test high12 F1 | composite |",
                "|---:|---:|---:|---:|---:|---:|",
                "| {mae:.4f} | {r2:.4f} | {bias:+.4f} | {h8:.4f} | {h12:.4f} | {comp:.4f} |".format(
                    mae=float(selected_test["mae"]),
                    r2=float(selected_test["r2"]),
                    bias=float(selected_test["original_low_0_2_bias"]),
                    h8=float(selected_test["high8_f1"]),
                    h12=float(selected_test["high12_f1"]),
                    comp=float(selected_test["goal_composite_strict120"]),
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search recovery-gate policies on validation predictions.")
    parser.add_argument("--base_val_csv", required=True)
    parser.add_argument("--gate_val_csv", required=True)
    parser.add_argument("--base_test_csv", default=None)
    parser.add_argument("--gate_test_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", default="recovery_gate_policy")
    parser.add_argument("--selection_mode", choices=["composite", "low_target_then_mae"], default="composite")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    val = _load_joined(Path(args.base_val_csv), Path(args.gate_val_csv))
    val_rows = [_evaluate_policy(val, policy) for policy in _policy_grid()]
    selected_val = _select(val_rows, mode=args.selection_mode)
    selected_val["selection_mode"] = str(args.selection_mode)
    _write_csv(out_dir / "recovery_gate_policy_val_grid.csv", val_rows)
    _write_csv(out_dir / "recovery_gate_policy_selected_val.csv", [selected_val])
    _write_predictions(out_dir / "val_predictions.csv", val, selected_val, f"{args.run_name}_val")

    selected_test: Optional[Dict[str, object]] = None
    if args.base_test_csv and args.gate_test_csv:
        test = _load_joined(Path(args.base_test_csv), Path(args.gate_test_csv))
        selected_test = _evaluate_policy(test, selected_val)
        _write_csv(out_dir / "recovery_gate_policy_selected_test.csv", [selected_test])
        _write_predictions(out_dir / "test_predictions.csv", test, selected_val, f"{args.run_name}_test")
    _write_report(out_dir / "recovery_gate_policy_report.md", selected_val, selected_test)
    print({"out_dir": str(out_dir), "selected": selected_val, "test": selected_test})


if __name__ == "__main__":
    main()
