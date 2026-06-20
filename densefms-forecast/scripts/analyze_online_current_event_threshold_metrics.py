"""Collapse online-current high-FMS threshold predictions to episode/onset metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


Segment = Tuple[int, int]


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _segments(mask: Sequence[bool]) -> List[Segment]:
    spans: List[Segment] = []
    start: int | None = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if start is not None and ((not flag) or idx == len(mask) - 1):
            end = idx - 1 if not flag else idx
            spans.append((start, end))
            start = None
    return spans


def _overlaps(left: Segment, right: Segment) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _read_rows(path: Path, pred_column: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"session_id", "current_index", "target_fms_now", pred_column}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            rows.append(
                {
                    "session_id": str(row["session_id"]),
                    "current_index": int(float(row["current_index"])),
                    "target": _as_float(row["target_fms_now"]),
                    "score": _as_float(row[pred_column]),
                }
            )
    return rows


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _pointwise(target: Sequence[float], score: Sequence[float], threshold: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0.0
    for yv, pv in zip(target, score):
        if not (math.isfinite(yv) and math.isfinite(pv)):
            continue
        y = yv >= threshold
        p = pv >= threshold
        if y and p:
            tp += 1.0
        elif (not y) and p:
            fp += 1.0
        elif y and (not p):
            fn += 1.0
        else:
            tn += 1.0
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "point_tp": tp,
        "point_fp": fp,
        "point_fn": fn,
        "point_tn": tn,
        "point_precision": precision,
        "point_recall": recall,
        "point_f1": _f1(precision, recall),
        "point_false_positive_rate": _safe_div(fp, fp + tn),
        "point_false_negative_rate": _safe_div(fn, tp + fn),
    }


def _episode_overlap(target_spans: Sequence[Segment], pred_spans: Sequence[Segment]) -> Dict[str, float]:
    matched_targets = set()
    matched_preds = set()
    for ti, target_span in enumerate(target_spans):
        for pi, pred_span in enumerate(pred_spans):
            if _overlaps(target_span, pred_span):
                matched_targets.add(ti)
                matched_preds.add(pi)
    tp_events = float(len(matched_targets))
    precision = _safe_div(float(len(matched_preds)), float(len(pred_spans)))
    recall = _safe_div(tp_events, float(len(target_spans)))
    return {
        "target_episode_count": float(len(target_spans)),
        "pred_episode_count": float(len(pred_spans)),
        "overlap_target_tp": tp_events,
        "overlap_pred_tp": float(len(matched_preds)),
        "overlap_false_alert_episodes": float(len(pred_spans) - len(matched_preds)),
        "overlap_missed_target_episodes": float(len(target_spans) - len(matched_targets)),
        "overlap_precision": precision,
        "overlap_recall": recall,
        "overlap_f1": _f1(precision, recall),
    }


def _onset_window(
    target_spans: Sequence[Segment],
    pred_spans: Sequence[Segment],
    lead_steps: int,
    grace_steps: int,
    cooldown_steps: int,
) -> Dict[str, float]:
    event_tp = 0
    matched_pred_indices = set()
    lead_distances: List[int] = []
    for target_start, _target_end in target_spans:
        window = (max(0, target_start - lead_steps), target_start + grace_steps)
        candidates = [
            (pi, span)
            for pi, span in enumerate(pred_spans)
            if pi not in matched_pred_indices and _overlaps(span, window)
        ]
        if not candidates:
            continue
        pi, pred_span = min(candidates, key=lambda item: max(item[1][0], window[0]))
        matched_pred_indices.add(pi)
        event_tp += 1
        alert_time = min(max(pred_span[0], window[0]), window[1])
        lead_distances.append(target_start - alert_time)

    counted_pred_indices = set()
    last_counted_start = -10**9
    for pi, (pred_start, _pred_end) in enumerate(pred_spans):
        if pred_start - last_counted_start < cooldown_steps:
            continue
        counted_pred_indices.add(pi)
        last_counted_start = pred_start

    matched_counted = len(matched_pred_indices.intersection(counted_pred_indices))
    alert_total = len(counted_pred_indices)
    precision = _safe_div(float(matched_counted), float(alert_total))
    recall = _safe_div(float(event_tp), float(len(target_spans)))
    return {
        "onset_target_episode_count": float(len(target_spans)),
        "onset_alert_episode_count": float(alert_total),
        "onset_event_tp": float(event_tp),
        "onset_false_alert_episodes": float(alert_total - matched_counted),
        "onset_missed_target_episodes": float(len(target_spans) - event_tp),
        "onset_precision": precision,
        "onset_recall": recall,
        "onset_f1": _f1(precision, recall),
        "onset_mean_lead_steps": sum(lead_distances) / len(lead_distances) if lead_distances else float("nan"),
    }


def _session_metrics(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    lead_steps: int,
    grace_steps: int,
    cooldown_steps: int,
) -> Dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["current_index"]))
    target = [float(row["target"]) for row in ordered]
    score = [float(row["score"]) for row in ordered]
    target_mask = [math.isfinite(v) and v >= threshold for v in target]
    pred_mask = [math.isfinite(v) and v >= threshold for v in score]
    target_spans = _segments(target_mask)
    pred_spans = _segments(pred_mask)
    metrics: Dict[str, float] = {}
    metrics.update(_pointwise(target, score, threshold))
    metrics.update(_episode_overlap(target_spans, pred_spans))
    metrics.update(_onset_window(target_spans, pred_spans, lead_steps, grace_steps, cooldown_steps))
    return metrics


def _sum_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    weighted_lead_num = 0.0
    weighted_lead_den = 0.0
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {
                "threshold",
                "point_precision",
                "point_recall",
                "point_f1",
                "point_false_positive_rate",
                "point_false_negative_rate",
                "overlap_precision",
                "overlap_recall",
                "overlap_f1",
                "onset_precision",
                "onset_recall",
                "onset_f1",
                "onset_mean_lead_steps",
            }:
                sums[key] += float(value)
        lead = float(row.get("onset_mean_lead_steps", float("nan")))
        count = float(row.get("onset_event_tp", 0.0))
        if math.isfinite(lead) and count > 0:
            weighted_lead_num += lead * count
            weighted_lead_den += count
    point_precision = _safe_div(sums["point_tp"], sums["point_tp"] + sums["point_fp"])
    point_recall = _safe_div(sums["point_tp"], sums["point_tp"] + sums["point_fn"])
    overlap_precision = _safe_div(sums["overlap_pred_tp"], sums["pred_episode_count"])
    overlap_recall = _safe_div(sums["overlap_target_tp"], sums["target_episode_count"])
    onset_precision = _safe_div(
        sums["onset_alert_episode_count"] - sums["onset_false_alert_episodes"],
        sums["onset_alert_episode_count"],
    )
    onset_recall = _safe_div(sums["onset_event_tp"], sums["onset_target_episode_count"])
    sums.update(
        {
            "point_precision": point_precision,
            "point_recall": point_recall,
            "point_f1": _f1(point_precision, point_recall),
            "point_false_positive_rate": _safe_div(sums["point_fp"], sums["point_fp"] + sums["point_tn"]),
            "point_false_negative_rate": _safe_div(sums["point_fn"], sums["point_tp"] + sums["point_fn"]),
            "overlap_precision": overlap_precision,
            "overlap_recall": overlap_recall,
            "overlap_f1": _f1(overlap_precision, overlap_recall),
            "onset_precision": onset_precision,
            "onset_recall": onset_recall,
            "onset_f1": _f1(onset_precision, onset_recall),
            "onset_mean_lead_steps": _safe_div(weighted_lead_num, weighted_lead_den) if weighted_lead_den else float("nan"),
        }
    )
    return dict(sums)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "nan"
    if abs(f - round(f)) < 1e-9 and abs(f) >= 10:
        return str(int(round(f)))
    return f"{f:.4f}"


def _write_markdown(
    path: Path,
    label: str,
    prediction_csv: Path,
    pred_column: str,
    pooled_rows: Sequence[Mapping[str, Any]],
    sampling_interval: float,
    lead_seconds: float,
    grace_seconds: float,
    cooldown_seconds: float,
) -> None:
    lines = [
        "# Event-Collapsed High-FMS Threshold Metrics",
        "",
        f"- label: `{label}`",
        f"- prediction_csv: `{prediction_csv}`",
        f"- pred_column: `{pred_column}`",
        f"- sampling_interval: `{sampling_interval:g}` seconds",
        f"- fixed score thresholds: same as FMS thresholds",
        f"- onset window: target onset - `{lead_seconds:g}s` through target onset + `{grace_seconds:g}s`",
        f"- alert cooldown: `{cooldown_seconds:g}s`",
        "",
        "## Summary",
        "",
        "| FMS threshold | point F1 | point precision | point recall | target episodes | pred episodes | overlap F1 | overlap precision | overlap recall | onset F1 | onset precision | onset recall | onset alerts | onset missed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        lines.append(
            "| {thr} | {point_f1} | {point_p} | {point_r} | {target_eps} | {pred_eps} | {overlap_f1} | {overlap_p} | {overlap_r} | {onset_f1} | {onset_p} | {onset_r} | {onset_alerts} | {onset_missed} |".format(
                thr=_fmt(row["threshold"]),
                point_f1=_fmt(row["point_f1"]),
                point_p=_fmt(row["point_precision"]),
                point_r=_fmt(row["point_recall"]),
                target_eps=_fmt(row["target_episode_count"]),
                pred_eps=_fmt(row["pred_episode_count"]),
                overlap_f1=_fmt(row["overlap_f1"]),
                overlap_p=_fmt(row["overlap_precision"]),
                overlap_r=_fmt(row["overlap_recall"]),
                onset_f1=_fmt(row["onset_f1"]),
                onset_p=_fmt(row["onset_precision"]),
                onset_r=_fmt(row["onset_recall"]),
                onset_alerts=_fmt(row["onset_alert_episode_count"]),
                onset_missed=_fmt(row["onset_missed_target_episodes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `point F1` is the old per-timestep threshold metric and can reward long high-FMS plateaus repeatedly.",
            "- `overlap F1` collapses each target/predicted high-FMS run into one episode, so one long plateau contributes at most one detected event.",
            "- `onset F1` is stricter: a predicted high-FMS episode must occur near the target episode onset, after applying the cooldown.",
            "- This is a diagnostic test-set recalculation only. It does not tune thresholds on the test set and does not change model selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--pred_column", default="predicted_fms_now")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--onset_lead_seconds", type=float, default=0.0)
    parser.add_argument("--onset_grace_seconds", type=float, default=5.0)
    parser.add_argument("--cooldown_seconds", type=float, default=20.0)
    args = parser.parse_args()

    prediction_csv = Path(args.prediction_csv)
    out_dir = Path(args.out_dir)
    label = args.label or prediction_csv.parent.name
    rows = _read_rows(prediction_csv, args.pred_column)
    by_session: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session_id"])].append(row)

    lead_steps = int(round(float(args.onset_lead_seconds) / float(args.sampling_interval)))
    grace_steps = int(round(float(args.onset_grace_seconds) / float(args.sampling_interval)))
    cooldown_steps = int(round(float(args.cooldown_seconds) / float(args.sampling_interval)))

    session_rows: List[Dict[str, Any]] = []
    pooled_rows: List[Dict[str, Any]] = []
    for threshold in args.thresholds:
        per_threshold_rows: List[Dict[str, Any]] = []
        for session_id, session_values in sorted(by_session.items()):
            metrics = _session_metrics(session_values, threshold, lead_steps, grace_steps, cooldown_steps)
            row = {"label": label, "session_id": session_id, "threshold": threshold, **metrics}
            session_rows.append(row)
            per_threshold_rows.append(row)
        pooled_rows.append({"label": label, "session_id": "__pooled__", "threshold": threshold, **_sum_metrics(per_threshold_rows)})

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "event_threshold_session_metrics.csv", session_rows)
    _write_csv(out_dir / "event_threshold_pooled_metrics.csv", pooled_rows)
    (out_dir / "event_threshold_pooled_metrics.json").write_text(json.dumps(pooled_rows, indent=2), encoding="utf-8")
    _write_markdown(
        out_dir / "event_threshold_summary.md",
        label,
        prediction_csv,
        args.pred_column,
        pooled_rows,
        float(args.sampling_interval),
        float(args.onset_lead_seconds),
        float(args.onset_grace_seconds),
        float(args.cooldown_seconds),
    )
    print(json.dumps({"out_dir": str(out_dir), "pooled_rows": pooled_rows}, indent=2))


if __name__ == "__main__":
    main()
