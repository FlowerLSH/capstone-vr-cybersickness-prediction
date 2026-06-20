"""Strict onset-warning evaluation for online-current high-risk predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).to_numpy() > 0.5
    return values.astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).to_numpy(dtype=bool)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        starts = np.cumsum(np.r_[0, counts[:-1]])
        ranks = (starts + (counts - 1) / 2.0)[inverse]
    return ranks + 1.0


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    if labels.size == 0 or int(labels.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(labels.sum()), 1.0)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _prf(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    pred = scores >= float(threshold)
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": float(np.mean(pred)) if pred.size else float("nan"),
        "false_alarm_rate": fp / (fp + tn) if fp + tn > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _pointwise_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask].astype(float)
    row = _prf(labels, scores, threshold)
    row.update(
        {
            "n": int(labels.size),
            "positive_rate": float(np.mean(labels)) if labels.size else float("nan"),
            "auprc": _auprc(labels, scores),
            "auroc": _auroc(labels, scores),
        }
    )
    return row


def _future_high_labels(fms: np.ndarray, threshold: float, horizon_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(fms), dtype=bool)
    valid = np.zeros(len(fms), dtype=bool)
    for idx in range(len(fms)):
        if idx + horizon_steps >= len(fms) or not np.isfinite(fms[idx]):
            continue
        future = fms[idx + 1 : idx + horizon_steps + 1]
        if not np.isfinite(future).any():
            continue
        labels[idx] = bool(np.nanmax(future) >= threshold)
        valid[idx] = True
    return labels, valid


def _past_low_mask(fms: np.ndarray, threshold: float, past_steps: int) -> np.ndarray:
    mask = np.zeros(len(fms), dtype=bool)
    for idx in range(len(fms)):
        if not np.isfinite(fms[idx]) or fms[idx] >= threshold:
            continue
        start = max(0, idx - int(past_steps))
        hist = fms[start : idx + 1]
        if np.isfinite(hist).all() and np.nanmax(hist) < threshold:
            mask[idx] = True
    return mask


def _episodes(fms: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    high = np.isfinite(fms) & (fms >= threshold)
    episodes: List[Tuple[int, int]] = []
    in_ep = False
    start = 0
    for idx, flag in enumerate(high):
        if flag and not in_ep:
            start = idx
            in_ep = True
        if in_ep and ((not flag) or idx == len(high) - 1):
            end = idx - 1 if not flag else idx
            episodes.append((start, end))
            in_ep = False
    return episodes


def _event_level_metrics(
    df: pd.DataFrame,
    threshold_value: float,
    score_col: str,
    probability_threshold: float,
    horizon_steps: int,
    cooldown_steps: int,
    sampling_interval: float,
) -> Dict[str, float]:
    event_total = 0
    event_tp = 0
    lead_times: List[float] = []
    false_alerts = 0
    alert_total = 0
    for _sid, g in df.groupby("session_id"):
        g = g.sort_values("current_index")
        fms = g["target_fms_now"].astype(float).to_numpy()
        scores = g[score_col].astype(float).to_numpy()
        alerts = np.isfinite(scores) & (scores >= probability_threshold)
        eps = _episodes(fms, threshold_value)
        matched_alert = np.zeros(len(alerts), dtype=bool)
        for start, _end in eps:
            pre_start = max(0, start - horizon_steps)
            pre_end = start - 1
            if pre_end < pre_start:
                continue
            window_alerts = np.where(alerts[pre_start : pre_end + 1])[0]
            event_total += 1
            if window_alerts.size:
                first = int(pre_start + window_alerts[0])
                event_tp += 1
                lead_times.append(float(start - first) * sampling_interval)
                matched_alert[pre_start : pre_end + 1] |= alerts[pre_start : pre_end + 1]
        # Count alert episodes that are not within any pre-onset warning window.
        alert_eps = _episodes(alerts.astype(float), 0.5)
        protected = np.zeros(len(alerts), dtype=bool)
        for start, _end in eps:
            protected[max(0, start - horizon_steps) : start + 1] = True
        last_counted = -10**9
        for start, _end in alert_eps:
            if start - last_counted < cooldown_steps:
                continue
            alert_total += 1
            if not protected[start]:
                false_alerts += 1
            last_counted = start
    precision = event_tp / alert_total if alert_total > 0 else 0.0
    recall = event_tp / event_total if event_total > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "event_count": float(event_total),
        "event_tp": float(event_tp),
        "alert_episode_count": float(alert_total),
        "false_alert_episode_count": float(false_alerts),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "mean_lead_time_seconds": float(np.mean(lead_times)) if lead_times else float("nan"),
        "median_lead_time_seconds": float(np.median(lead_times)) if lead_times else float("nan"),
    }


def _threshold_from_file(path: Optional[Path], run_name: str, event: str, fallback: float) -> float:
    if path is None or not path.exists():
        return float(fallback)
    df = pd.read_csv(path)
    rows = df[(df.get("run", "") == run_name) & (df.get("event", "") == event) & (df.get("mode", "") == "best_f1")]
    if rows.empty:
        return float(fallback)
    return float(rows.iloc[0]["threshold"])


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate high-risk predictions with stricter onset-warning metrics.")
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--out_dir", default="reports/online_current_onset_warning_0514")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--threshold_tuning_csv", default=None)
    parser.add_argument("--fms_thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--horizon_seconds", type=float, default=20.0)
    parser.add_argument("--past_low_seconds", type=float, default=10.0)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--probability_threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_path = Path(args.prediction_csv)
    df = pd.read_csv(pred_path)
    run_name = args.run_name or str(df["run_name"].iloc[0] if "run_name" in df.columns and len(df) else pred_path.parent.name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizon_steps = int(round(float(args.horizon_seconds) / float(args.sampling_interval)))
    past_steps = int(round(float(args.past_low_seconds) / float(args.sampling_interval)))
    tuning_path = Path(args.threshold_tuning_csv) if args.threshold_tuning_csv else None

    point_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    for threshold in args.fms_thresholds:
        score_col = f"p_high_risk_{args.horizon_seconds:g}s_thr{threshold:g}"
        # Existing CSV names use 20s instead of 20.0s.
        if score_col not in df.columns:
            score_col = f"p_high_risk_{int(args.horizon_seconds)}s_thr{threshold:g}"
        if score_col not in df.columns:
            continue
        event_name = f"future_high{threshold:g}_{int(args.horizon_seconds)}s"
        prob_thr = _threshold_from_file(tuning_path, run_name, event_name, float(args.probability_threshold))
        for session_id, g in df.groupby("session_id"):
            g = g.sort_values("current_index")
            fms = g["target_fms_now"].astype(float).to_numpy()
            scores = g[score_col].astype(float).to_numpy()
            labels, valid = _future_high_labels(fms, threshold, horizon_steps)
            current_below = np.isfinite(fms) & (fms < threshold)
            onset_mask = _past_low_mask(fms, threshold, past_steps)
            for scope, scope_mask in (
                ("window_any", valid),
                ("current_below", valid & current_below),
                (f"onset_past{int(args.past_low_seconds)}s_low", valid & onset_mask),
            ):
                m = scope_mask & np.isfinite(scores)
                if m.any():
                    metrics = _pointwise_metrics(labels[m], scores[m], prob_thr)
                else:
                    metrics = _pointwise_metrics(np.array([], dtype=bool), np.array([], dtype=float), prob_thr)
                point_rows.append(
                    {
                        "run": run_name,
                        "session_id": session_id,
                        "fms_threshold": threshold,
                        "scope": scope,
                        **metrics,
                    }
                )
        for scope in ("window_any", "current_below", f"onset_past{int(args.past_low_seconds)}s_low"):
            sub = [row for row in point_rows if row["fms_threshold"] == threshold and row["scope"] == scope]
            labels: List[bool] = []
            scores: List[float] = []
            for _sid, g in df.groupby("session_id"):
                g = g.sort_values("current_index")
                fms = g["target_fms_now"].astype(float).to_numpy()
                sc = g[score_col].astype(float).to_numpy()
                lab, valid = _future_high_labels(fms, threshold, horizon_steps)
                current_below = np.isfinite(fms) & (fms < threshold)
                onset_mask = _past_low_mask(fms, threshold, past_steps)
                if scope == "window_any":
                    m = valid
                elif scope == "current_below":
                    m = valid & current_below
                else:
                    m = valid & onset_mask
                m = m & np.isfinite(sc)
                labels.extend(lab[m].tolist())
                scores.extend(sc[m].tolist())
            metrics = _pointwise_metrics(np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float), prob_thr)
            point_rows.append({"run": run_name, "session_id": "__pooled__", "fms_threshold": threshold, "scope": scope, **metrics})
        event_metrics = _event_level_metrics(
            df,
            threshold,
            score_col,
            prob_thr,
            horizon_steps,
            cooldown_steps=horizon_steps,
            sampling_interval=float(args.sampling_interval),
        )
        event_rows.append(
            {
                "run": run_name,
                "fms_threshold": threshold,
                "probability_threshold": prob_thr,
                "horizon_seconds": float(args.horizon_seconds),
                "past_low_seconds": float(args.past_low_seconds),
                **event_metrics,
            }
        )

    pooled_rows = [row for row in point_rows if row["session_id"] == "__pooled__"]
    _write_csv(out_dir / "onset_pointwise_metrics.csv", point_rows)
    _write_csv(out_dir / "onset_pointwise_pooled_metrics.csv", pooled_rows)
    _write_csv(out_dir / "onset_event_level_metrics.csv", event_rows)
    lines = ["# Onset Warning Evaluation", ""]
    lines.append("Prediction CSV 기준 사후 평가다. 학습/선택용 test 사용 여부는 입력 CSV의 split에 따른다.")
    lines.append("")
    lines.append("## Pooled Pointwise Metrics")
    lines.append("")
    lines.append("| threshold | scope | n | positive rate | AUPRC | AUROC | precision | recall | F1 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in pooled_rows:
        lines.append(
            "| {thr:g} | {scope} | {n} | {prev:.4f} | {auprc:.4f} | {auroc:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} |".format(
                thr=float(row["fms_threshold"]),
                scope=row["scope"],
                n=int(row["n"]),
                prev=float(row["positive_rate"]),
                auprc=float(row["auprc"]),
                auroc=float(row["auroc"]),
                p=float(row["precision"]),
                r=float(row["recall"]),
                f1=float(row["f1"]),
            )
        )
    lines.append("")
    lines.append("## Event-Level Metrics")
    lines.append("")
    lines.append("| threshold | events | alert episodes | event precision | event recall | event F1 | mean lead s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in event_rows:
        lines.append(
            "| {thr:g} | {events:.0f} | {alerts:.0f} | {p:.4f} | {r:.4f} | {f1:.4f} | {lead:.2f} |".format(
                thr=float(row["fms_threshold"]),
                events=float(row["event_count"]),
                alerts=float(row["alert_episode_count"]),
                p=float(row["event_precision"]),
                r=float(row["event_recall"]),
                f1=float(row["event_f1"]),
                lead=float(row["mean_lead_time_seconds"]),
            )
        )
    (out_dir / "onset_warning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote onset warning evaluation to {out_dir}")


if __name__ == "__main__":
    main()
