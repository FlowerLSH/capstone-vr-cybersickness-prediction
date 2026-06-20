"""Tune deployment alert policies for online-current high-risk predictions.

The model emits per-timestep probabilities. This script tunes only the
deployment policy on validation predictions, then applies the selected policy
to test predictions without using test metrics for selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertPolicy:
    on_threshold: float
    off_threshold: float
    min_duration_steps: int
    cooldown_steps: int


def _episodes(flags: np.ndarray) -> List[Tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool)
    episodes: List[Tuple[int, int]] = []
    active = False
    start = 0
    for idx, flag in enumerate(flags):
        if flag and not active:
            start = idx
            active = True
        if active and ((not flag) or idx == len(flags) - 1):
            end = idx - 1 if not flag else idx
            episodes.append((start, end))
            active = False
    return episodes


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _score_col(df: pd.DataFrame, horizon_seconds: float, fms_threshold: float) -> str:
    candidates = [
        f"p_high_risk_{horizon_seconds:g}s_thr{fms_threshold:g}",
        f"p_high_risk_{int(horizon_seconds)}s_thr{fms_threshold:g}",
        f"p_high_risk_{horizon_seconds:g}s_thr{int(fms_threshold)}",
        f"p_high_risk_{int(horizon_seconds)}s_thr{int(fms_threshold)}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find high-risk score column among {candidates}")


def _label_cols(df: pd.DataFrame, horizon_seconds: float, fms_threshold: float) -> Tuple[Optional[str], Optional[str]]:
    label_candidates = [
        f"high_risk_label_{horizon_seconds:g}s_thr{fms_threshold:g}",
        f"high_risk_label_{int(horizon_seconds)}s_thr{fms_threshold:g}",
        f"high_risk_label_{horizon_seconds:g}s_thr{int(fms_threshold)}",
        f"high_risk_label_{int(horizon_seconds)}s_thr{int(fms_threshold)}",
    ]
    valid_candidates = [
        f"high_risk_valid_{horizon_seconds:g}s_thr{fms_threshold:g}",
        f"high_risk_valid_{int(horizon_seconds)}s_thr{fms_threshold:g}",
        f"high_risk_valid_{horizon_seconds:g}s_thr{int(fms_threshold)}",
        f"high_risk_valid_{int(horizon_seconds)}s_thr{int(fms_threshold)}",
    ]
    label_col = next((col for col in label_candidates if col in df.columns), None)
    valid_col = next((col for col in valid_candidates if col in df.columns), None)
    return label_col, valid_col


def _as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).to_numpy() > 0.5
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy(dtype=bool)


def apply_policy(scores: np.ndarray, policy: AlertPolicy) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    desired = np.zeros(len(scores), dtype=bool)
    state = False
    for idx, score in enumerate(scores):
        if not np.isfinite(score):
            state = False
        elif state:
            if score < policy.off_threshold:
                state = False
        elif score >= policy.on_threshold:
            state = True
        desired[idx] = state

    min_steps = max(1, int(policy.min_duration_steps))
    if min_steps > 1:
        sustained = np.zeros_like(desired)
        run_len = 0
        for idx, flag in enumerate(desired):
            run_len = run_len + 1 if flag else 0
            sustained[idx] = run_len >= min_steps
        desired = sustained

    cooldown = max(0, int(policy.cooldown_steps))
    if cooldown <= 0:
        return desired

    output = np.zeros_like(desired)
    active = False
    cooldown_until = -1
    for idx, flag in enumerate(desired):
        if active:
            if flag:
                output[idx] = True
            else:
                active = False
                cooldown_until = idx + cooldown
            continue
        if idx < cooldown_until:
            continue
        if flag:
            active = True
            output[idx] = True
    return output


def _pointwise_metrics(df: pd.DataFrame, alerts_by_row: np.ndarray, horizon_seconds: float, fms_threshold: float) -> Dict[str, float]:
    label_col, valid_col = _label_cols(df, horizon_seconds, fms_threshold)
    if label_col is None or valid_col is None:
        return {
            "point_n": float("nan"),
            "point_positive_rate": float("nan"),
            "point_precision": float("nan"),
            "point_recall": float("nan"),
            "point_f1": float("nan"),
            "point_pred_rate": float("nan"),
        }
    labels = _as_bool(df[label_col])
    valid = _as_bool(df[valid_col])
    pred = alerts_by_row.astype(bool)
    labels = labels[valid]
    pred = pred[valid]
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "point_n": int(labels.size),
        "point_positive_rate": float(np.mean(labels)) if labels.size else float("nan"),
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": _f1(precision, recall),
        "point_pred_rate": float(np.mean(pred)) if pred.size else float("nan"),
        "point_tp": tp,
        "point_fp": fp,
        "point_fn": fn,
        "point_tn": tn,
    }


def _event_metrics_for_alerts(
    df: pd.DataFrame,
    alerts_by_row: np.ndarray,
    fms_threshold: float,
    horizon_steps: int,
    sampling_interval: float,
) -> Dict[str, float]:
    event_total = 0
    event_tp = 0
    lead_times: List[float] = []
    alert_total = 0
    false_alerts = 0
    alert_true_episodes = 0

    tmp = df[["session_id", "current_index", "target_fms_now"]].copy()
    tmp["_alert"] = alerts_by_row.astype(bool)
    for _sid, g in tmp.groupby("session_id", sort=False):
        g = g.sort_values("current_index")
        fms = g["target_fms_now"].astype(float).to_numpy()
        alerts = g["_alert"].to_numpy(dtype=bool)
        high = np.isfinite(fms) & (fms >= fms_threshold)
        high_eps = _episodes(high)
        protected = np.zeros(len(alerts), dtype=bool)
        for start, _end in high_eps:
            pre_start = max(0, start - horizon_steps)
            pre_end = start - 1
            if pre_end < pre_start:
                # Already high at the first evaluated timestep. There is no
                # observable pre-onset warning window inside this prediction CSV.
                continue
            event_total += 1
            protected[pre_start : start + 1] = True
            hits = np.where(alerts[pre_start : pre_end + 1])[0]
            if hits.size:
                first = int(pre_start + hits[0])
                event_tp += 1
                lead_times.append(float(start - first) * float(sampling_interval))

        for start, end in _episodes(alerts):
            alert_total += 1
            if np.any(protected[start : end + 1]):
                alert_true_episodes += 1
            else:
                false_alerts += 1

    event_precision = alert_true_episodes / alert_total if alert_total > 0 else 0.0
    event_recall = event_tp / event_total if event_total > 0 else 0.0
    return {
        "event_count": float(event_total),
        "event_tp": float(event_tp),
        "alert_episode_count": float(alert_total),
        "true_alert_episode_count": float(alert_true_episodes),
        "false_alert_episode_count": float(false_alerts),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": _f1(event_precision, event_recall),
        "episode_precision": event_precision,
        "mean_lead_time_seconds": float(np.mean(lead_times)) if lead_times else float("nan"),
        "median_lead_time_seconds": float(np.median(lead_times)) if lead_times else float("nan"),
    }


def evaluate_policy(
    df: pd.DataFrame,
    fms_threshold: float,
    horizon_seconds: float,
    sampling_interval: float,
    policy: AlertPolicy,
) -> Dict[str, float]:
    score_col = _score_col(df, horizon_seconds, fms_threshold)
    alerts = np.zeros(len(df), dtype=bool)
    for _sid, indices in df.groupby("session_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        order = np.argsort(df.loc[idx, "current_index"].astype(float).to_numpy(), kind="mergesort")
        sorted_idx = idx[order]
        scores = df.loc[sorted_idx, score_col].astype(float).to_numpy()
        alerts[sorted_idx] = apply_policy(scores, policy)

    horizon_steps = int(round(float(horizon_seconds) / float(sampling_interval)))
    row: Dict[str, float] = {
        "fms_threshold": float(fms_threshold),
        "on_threshold": float(policy.on_threshold),
        "off_threshold": float(policy.off_threshold),
        "min_duration_steps": int(policy.min_duration_steps),
        "min_duration_seconds": float(policy.min_duration_steps) * float(sampling_interval),
        "cooldown_steps": int(policy.cooldown_steps),
        "cooldown_seconds": float(policy.cooldown_steps) * float(sampling_interval),
    }
    row.update(_pointwise_metrics(df, alerts, horizon_seconds, fms_threshold))
    row.update(_event_metrics_for_alerts(df, alerts, fms_threshold, horizon_steps, sampling_interval))
    return row


def _policy_grid(
    on_thresholds: Sequence[float],
    hysteresis_margins: Sequence[float],
    min_duration_seconds: Sequence[float],
    cooldown_seconds: Sequence[float],
    sampling_interval: float,
) -> Iterable[AlertPolicy]:
    seen = set()
    for on in on_thresholds:
        for margin in hysteresis_margins:
            off = max(0.0, min(float(on), float(on) - float(margin)))
            for min_s in min_duration_seconds:
                min_steps = max(1, int(round(float(min_s) / float(sampling_interval))))
                for cooldown_s in cooldown_seconds:
                    cooldown_steps = max(0, int(round(float(cooldown_s) / float(sampling_interval))))
                    key = (round(float(on), 6), round(float(off), 6), min_steps, cooldown_steps)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield AlertPolicy(
                        on_threshold=float(on),
                        off_threshold=float(off),
                        min_duration_steps=min_steps,
                        cooldown_steps=cooldown_steps,
                    )


def _select_policy(rows: pd.DataFrame, mode: str) -> pd.Series:
    if rows.empty:
        raise ValueError("Cannot select from an empty policy grid.")
    data = rows.copy()
    if mode == "best_event_f1":
        sort_cols = ["event_f1", "event_precision", "event_recall", "mean_lead_time_seconds"]
        return data.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]
    if mode == "precision_at_recall90":
        candidates = data[data["event_recall"] >= 0.90]
        if candidates.empty:
            candidates = data
        sort_cols = ["event_precision", "event_f1", "event_recall", "mean_lead_time_seconds"]
        return candidates.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]
    if mode == "precision_at_recall80":
        candidates = data[data["event_recall"] >= 0.80]
        if candidates.empty:
            candidates = data
        sort_cols = ["event_precision", "event_f1", "event_recall", "mean_lead_time_seconds"]
        return candidates.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]
    if mode == "point_f1":
        sort_cols = ["point_f1", "point_precision", "point_recall", "event_f1"]
        return data.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]
    raise ValueError(f"Unknown selection mode: {mode}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy_from_row(row: Mapping[str, object], sampling_interval: float) -> AlertPolicy:
    return AlertPolicy(
        on_threshold=float(row["on_threshold"]),
        off_threshold=float(row["off_threshold"]),
        min_duration_steps=int(row["min_duration_steps"]),
        cooldown_steps=int(row["cooldown_steps"]),
    )


def _fmt(value: object, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def _write_summary(
    path: Path,
    run_name: str,
    val_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# Alert Policy Tuning Report",
        "",
        f"Run: `{run_name}`",
        "",
        "Validation prediction만으로 alert policy를 선택하고, 선택된 policy를 test prediction에 적용했다.",
        "",
        "## Selected Policies",
        "",
        "| threshold | selection | on | off | min s | cooldown s | val event P | val event R | val event F1 | test event P | test event R | test event F1 | test false alerts | test lead s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test_lookup = {
        (float(row["fms_threshold"]), str(row["selection_mode"])): row for row in test_rows
    }
    for row in val_rows:
        key = (float(row["fms_threshold"]), str(row["selection_mode"]))
        test = test_lookup.get(key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["fms_threshold"], 0),
                    str(row["selection_mode"]),
                    _fmt(row["on_threshold"], 2),
                    _fmt(row["off_threshold"], 2),
                    _fmt(row["min_duration_seconds"], 1),
                    _fmt(row["cooldown_seconds"], 1),
                    _fmt(row["event_precision"]),
                    _fmt(row["event_recall"]),
                    _fmt(row["event_f1"]),
                    _fmt(test.get("event_precision", float("nan"))),
                    _fmt(test.get("event_recall", float("nan"))),
                    _fmt(test.get("event_f1", float("nan"))),
                    _fmt(test.get("false_alert_episode_count", float("nan")), 0),
                    _fmt(test.get("mean_lead_time_seconds", float("nan")), 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `best_event_f1`은 validation event F1을 직접 최대화한다.",
            "- `precision_at_recall90`은 validation event recall 0.90 이상을 유지하면서 precision을 최대화한다.",
            "- `precision_at_recall80`은 validation event recall 0.80 이상을 유지하면서 더 공격적으로 false alert를 줄인다.",
            "- Test 행은 선택 이후 적용 결과이며, policy 선택에는 사용하지 않았다.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune alert threshold/hysteresis/min-duration/cooldown on validation predictions.")
    parser.add_argument("--val_prediction_csv", required=True)
    parser.add_argument("--test_prediction_csv", default=None)
    parser.add_argument("--out_dir", default="reports/online_current_alert_policy_0514")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--fms_thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--horizon_seconds", type=float, default=20.0)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--on_thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--hysteresis_margins", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--min_duration_seconds", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--cooldown_seconds", nargs="+", type=float, default=[0.0, 5.0, 10.0, 20.0, 30.0])
    parser.add_argument("--selection_modes", nargs="+", default=["best_event_f1", "precision_at_recall90", "precision_at_recall80", "point_f1"])
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_prediction_csv)
    test_df = pd.read_csv(args.test_prediction_csv) if args.test_prediction_csv else None
    run_name = args.run_name or str(val_df["run_name"].iloc[0] if "run_name" in val_df.columns and len(val_df) else Path(args.val_prediction_csv).parent.name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_grid_rows: List[Dict[str, object]] = []
    selected_val_rows: List[Dict[str, object]] = []
    selected_test_rows: List[Dict[str, object]] = []

    grid = list(
        _policy_grid(
            on_thresholds=args.on_thresholds,
            hysteresis_margins=args.hysteresis_margins,
            min_duration_seconds=args.min_duration_seconds,
            cooldown_seconds=args.cooldown_seconds,
            sampling_interval=args.sampling_interval,
        )
    )

    for fms_threshold in args.fms_thresholds:
        threshold_rows: List[Dict[str, object]] = []
        for policy in grid:
            row = evaluate_policy(
                val_df,
                fms_threshold=fms_threshold,
                horizon_seconds=args.horizon_seconds,
                sampling_interval=args.sampling_interval,
                policy=policy,
            )
            row["run_name"] = run_name
            row["split"] = "val"
            threshold_rows.append(row)
            all_grid_rows.append(row)
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_label = f"thr{fms_threshold:g}"
        threshold_df.to_csv(out_dir / f"val_policy_grid_{threshold_label}.csv", index=False)

        for mode in args.selection_modes:
            selected = _select_policy(threshold_df, mode)
            policy = _policy_from_row(selected, args.sampling_interval)
            selected_val = dict(selected)
            selected_val["selection_mode"] = mode
            selected_val_rows.append(selected_val)
            if test_df is not None:
                test_row = evaluate_policy(
                    test_df,
                    fms_threshold=fms_threshold,
                    horizon_seconds=args.horizon_seconds,
                    sampling_interval=args.sampling_interval,
                    policy=policy,
                )
                test_row["run_name"] = run_name
                test_row["split"] = "test"
                test_row["selection_mode"] = mode
                selected_test_rows.append(test_row)

    _write_csv(out_dir / "selected_policy_validation_metrics.csv", selected_val_rows)
    if selected_test_rows:
        _write_csv(out_dir / "selected_policy_test_metrics.csv", selected_test_rows)
    _write_summary(out_dir / "alert_policy_tuning_summary.md", run_name, selected_val_rows, selected_test_rows)
    print(f"wrote alert policy tuning outputs to {out_dir}")


if __name__ == "__main__":
    main()
