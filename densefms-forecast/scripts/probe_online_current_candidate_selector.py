"""Train-only candidate-selector probe for four online-current predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MEMBERS = [
    "member_pred_selected_risk035",
    "member_pred_risk045",
    "member_pred_zero_anchor",
    "member_pred_range_scaled",
]


def _add_features(frame: pd.DataFrame, members: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    out = frame.copy()
    matrix = out.loc[:, list(members)].astype(float).to_numpy(dtype=np.float64)
    out["member_mean"] = np.mean(matrix, axis=1)
    out["member_std"] = np.std(matrix, axis=1)
    out["member_min"] = np.min(matrix, axis=1)
    out["member_max"] = np.max(matrix, axis=1)
    out["member_range"] = out["member_max"] - out["member_min"]
    if "anchor_fms" in out.columns:
        anchor = pd.to_numeric(out["anchor_fms"], errors="coerce").astype(float)
        for col in members:
            out[f"{col}_minus_anchor"] = pd.to_numeric(out[col], errors="coerce").astype(float) - anchor
    base = ["predicted_fms_now", "anchor_fms", "current_time", "current_index", "age", "mssq"]
    features = [
        col
        for col in out.columns
        if col in base or col in members or col.startswith("member_") or col.endswith("_minus_anchor")
    ]
    values = out.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    return values.to_numpy(dtype=np.float64), features


def _high_f1(y: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    true = y >= float(threshold)
    hit = pred >= float(threshold)
    tp = float(np.sum(true & hit))
    fp = float(np.sum(~true & hit))
    fn = float(np.sum(true & ~hit))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _metrics(label: str, y: np.ndarray, pred: np.ndarray, closest_label: np.ndarray | None = None, selected: np.ndarray | None = None) -> Dict[str, object]:
    err = pred - y
    low = (y >= 0.0) & (y < 2.0)
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    row: Dict[str, object] = {
        "label": label,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "r2": 1.0 - float(np.sum(err * err)) / ss_tot if ss_tot > 1e-12 else float("nan"),
        "original_low_0_2_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "high8_f1": _high_f1(y, pred, 8.0),
        "high12_f1": _high_f1(y, pred, 12.0),
    }
    if closest_label is not None and selected is not None:
        row["closest_label_accuracy"] = float(np.mean(selected == closest_label))
    return row


def _models(seed: int):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "hgb": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=64,
            random_state=int(seed),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=64,
            max_features=0.8,
            random_state=int(seed),
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--member_cols", nargs="*", default=DEFAULT_MEMBERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)
    x_train, features = _add_features(train, args.member_cols)
    x_val, _ = _add_features(val, args.member_cols)
    y_train = train["target_fms_now"].to_numpy(dtype=np.float64)
    y_val = val["target_fms_now"].to_numpy(dtype=np.float64)
    m_train = train.loc[:, args.member_cols].to_numpy(dtype=np.float64)
    m_val = val.loc[:, args.member_cols].to_numpy(dtype=np.float64)
    target_label = np.argmin(np.abs(m_train - y_train[:, None]), axis=1)
    closest_val = np.argmin(np.abs(m_val - y_val[:, None]), axis=1)

    rows: List[Dict[str, object]] = [_metrics("equal4", y_val, np.mean(m_val, axis=1))]
    for name, model in _models(args.seed).items():
        model.fit(x_train, target_label)
        hard = np.asarray(model.predict(x_val), dtype=np.int64)
        hard_pred = m_val[np.arange(len(m_val)), hard]
        rows.append(_metrics(f"{name}_hard", y_val, hard_pred, closest_val, hard))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_val)
            classes = np.asarray(model.classes_, dtype=np.int64)
            soft_pred = np.sum(proba * m_val[:, classes], axis=1)
            rows.append(_metrics(f"{name}_soft", y_val, soft_pred))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "candidate_selector_probe_metrics.csv", rows)
    (out_dir / "candidate_selector_probe_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
