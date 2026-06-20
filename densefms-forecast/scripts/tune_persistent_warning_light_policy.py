"""Tune persistent yellow/red warning-light policies on validation predictions.

This script evaluates the UI semantics where an alert light is useful if it
either appears before a high-FMS onset or remains active during a true high-FMS
state. Test predictions, when provided, are evaluated only with validation-
selected policies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertPolicy:
    on: float
    off: float
    min_steps: int
    cooldown_steps: int

    @property
    def label(self) -> str:
        min_s = self.min_steps * 0.5
        cool_s = self.cooldown_steps * 0.5
        return f"on={self.on:.2f}/off={self.off:.2f}/min={min_s:g}s/cool={cool_s:g}s"


def _score_col(frame: pd.DataFrame, horizon_seconds: float, threshold: float) -> str:
    candidates = [
        f"p_high_risk_{horizon_seconds:g}s_thr{threshold:g}",
        f"p_high_risk_{int(horizon_seconds)}s_thr{threshold:g}",
        f"p_high_risk_{horizon_seconds:g}s_thr{int(threshold)}",
        f"p_high_risk_{int(horizon_seconds)}s_thr{int(threshold)}",
    ]
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"No score column found for horizon={horizon_seconds:g}s threshold={threshold:g}.")


def _episodes(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    values = np.asarray(flags, dtype=bool)
    out: List[Tuple[int, int]] = []
    active = False
    start = 0
    for idx, flag in enumerate(values):
        if flag and not active:
            start = idx
            active = True
        if active and ((not flag) or idx == len(values) - 1):
            end = idx - 1 if not flag else idx
            out.append((start, end))
            active = False
    return out


def _apply_policy(scores: np.ndarray, policy: AlertPolicy) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    desired = np.zeros(len(scores), dtype=bool)
    state = False
    for idx, score in enumerate(scores):
        if not np.isfinite(score):
            state = False
        elif state:
            if score < policy.off:
                state = False
        elif score >= policy.on:
            state = True
        desired[idx] = state

    min_steps = max(1, int(policy.min_steps))
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


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0


def _evaluate(
    frame: pd.DataFrame,
    threshold: float,
    score_col: str,
    policy: AlertPolicy,
    horizon_steps: int,
    sampling_interval: float,
) -> Dict[str, float]:
    alert_eps = 0
    advance_alert_eps = 0
    high_state_alert_eps = 0
    ui_false_alert_eps = 0
    strict_pre_false_eps = 0
    observable_events = 0
    pre_warned_events = 0
    late_detected_events = 0
    missed_events = 0
    lead_times: List[float] = []
    late_delays: List[float] = []
    alert_seconds = 0.0
    false_alert_seconds = 0.0
    warning_zone_seconds = 0.0
    warning_zone_covered_seconds = 0.0
    high_seconds = 0.0
    high_covered_seconds = 0.0

    tmp = frame[["session_id", "current_index", "target_fms_now", score_col]].copy()
    for _sid, group in tmp.groupby("session_id", sort=False):
        group = group.sort_values("current_index")
        fms = group["target_fms_now"].astype(float).to_numpy()
        scores = group[score_col].astype(float).to_numpy()
        alerts = _apply_policy(scores, policy)
        high = np.isfinite(fms) & (fms >= float(threshold))
        high_eps = _episodes(high)
        pre_event_zone = np.zeros(len(alerts), dtype=bool)
        high_state_zone = high.copy()
        warning_zone = high.copy()
        observable_high_eps: List[Tuple[int, int]] = []

        for start, end in high_eps:
            pre_start = max(0, start - horizon_steps)
            pre_end = start - 1
            if pre_end >= pre_start:
                observable_events += 1
                observable_high_eps.append((start, end))
                pre_event_zone[pre_start:start] = True
                warning_zone[pre_start : end + 1] = True

        for start, end in observable_high_eps:
            pre_start = max(0, start - horizon_steps)
            pre_end = start - 1
            hits = np.where(alerts[pre_start : pre_end + 1])[0]
            if hits.size:
                first = pre_start + int(hits[0])
                pre_warned_events += 1
                lead_times.append(float(start - first) * float(sampling_interval))
            else:
                late_hits = np.where(alerts[start : end + 1])[0]
                if late_hits.size:
                    late_detected_events += 1
                    late_delays.append(float(int(late_hits[0])) * float(sampling_interval))
                else:
                    missed_events += 1

        for start, end in _episodes(alerts):
            alert_eps += 1
            overlaps_pre = bool(np.any(pre_event_zone[start : end + 1]))
            overlaps_high = bool(np.any(high_state_zone[start : end + 1]))
            if overlaps_pre:
                advance_alert_eps += 1
            elif overlaps_high:
                high_state_alert_eps += 1
            else:
                ui_false_alert_eps += 1
            if not overlaps_pre:
                strict_pre_false_eps += 1

        alert_seconds += float(np.sum(alerts)) * float(sampling_interval)
        false_alert_seconds += float(np.sum(alerts & ~warning_zone)) * float(sampling_interval)
        warning_zone_seconds += float(np.sum(warning_zone)) * float(sampling_interval)
        warning_zone_covered_seconds += float(np.sum(alerts & warning_zone)) * float(sampling_interval)
        high_seconds += float(np.sum(high_state_zone)) * float(sampling_interval)
        high_covered_seconds += float(np.sum(alerts & high_state_zone)) * float(sampling_interval)

    useful_alert_eps = advance_alert_eps + high_state_alert_eps
    ui_precision = useful_alert_eps / alert_eps if alert_eps else 0.0
    pre_recall = pre_warned_events / observable_events if observable_events else 0.0
    high_coverage = high_covered_seconds / high_seconds if high_seconds else 0.0
    duration_precision = (alert_seconds - false_alert_seconds) / alert_seconds if alert_seconds else 0.0
    warning_zone_recall = warning_zone_covered_seconds / warning_zone_seconds if warning_zone_seconds else 0.0
    return {
        "fms_threshold": float(threshold),
        "on_threshold": float(policy.on),
        "off_threshold": float(policy.off),
        "min_duration_steps": float(policy.min_steps),
        "min_duration_seconds": float(policy.min_steps) * float(sampling_interval),
        "cooldown_steps": float(policy.cooldown_steps),
        "cooldown_seconds": float(policy.cooldown_steps) * float(sampling_interval),
        "alert_episode_count": float(alert_eps),
        "advance_alert_episode_count": float(advance_alert_eps),
        "high_state_alert_episode_count": float(high_state_alert_eps),
        "ui_false_alert_episode_count": float(ui_false_alert_eps),
        "strict_pre_onset_false_alert_episode_count": float(strict_pre_false_eps),
        "ui_episode_precision": float(ui_precision),
        "pre_onset_event_count": float(observable_events),
        "pre_onset_event_tp": float(pre_warned_events),
        "pre_onset_event_recall": float(pre_recall),
        "late_detected_event_count": float(late_detected_events),
        "missed_event_count": float(missed_events),
        "ui_episode_f1_vs_pre_recall": float(_f1(ui_precision, pre_recall)),
        "alert_seconds": float(alert_seconds),
        "ui_false_alert_seconds": float(false_alert_seconds),
        "duration_precision_warning_zone": float(duration_precision),
        "warning_zone_duration_recall": float(warning_zone_recall),
        "duration_f1_warning_zone": float(_f1(duration_precision, warning_zone_recall)),
        "high_state_duration_coverage": float(high_coverage),
        "mean_lead_time_seconds": float(np.mean(lead_times)) if lead_times else float("nan"),
        "median_lead_time_seconds": float(np.median(lead_times)) if lead_times else float("nan"),
        "mean_late_delay_seconds": float(np.mean(late_delays)) if late_delays else float("nan"),
    }


def _policies(args: argparse.Namespace) -> Iterable[AlertPolicy]:
    for on in args.on_thresholds:
        for delta in args.off_deltas:
            off = max(0.0, float(on) - float(delta))
            if off > on:
                continue
            for min_s in args.min_duration_seconds:
                min_steps = max(1, int(round(float(min_s) / float(args.sampling_interval))))
                for cooldown_s in args.cooldown_seconds:
                    cooldown_steps = max(0, int(round(float(cooldown_s) / float(args.sampling_interval))))
                    yield AlertPolicy(float(on), float(off), min_steps, cooldown_steps)


def _select_rows(grid: pd.DataFrame) -> pd.DataFrame:
    selected: List[pd.Series] = []
    for threshold, group in grid.groupby("fms_threshold", sort=True):
        rules = [
            ("best_duration_f1", group),
            (
                "max_ui_precision_pre_recall80_highcov80",
                group[(group["pre_onset_event_recall"] >= 0.80) & (group["high_state_duration_coverage"] >= 0.80)],
            ),
            (
                "max_ui_precision_pre_recall90_highcov80",
                group[(group["pre_onset_event_recall"] >= 0.90) & (group["high_state_duration_coverage"] >= 0.80)],
            ),
        ]
        for mode, subset in rules:
            if subset.empty:
                subset = group
            if mode == "best_duration_f1":
                order_cols = ["duration_f1_warning_zone", "ui_episode_precision", "pre_onset_event_recall"]
            else:
                order_cols = ["ui_episode_precision", "duration_f1_warning_zone", "pre_onset_event_recall"]
            best = subset.sort_values(order_cols, ascending=[False, False, False]).iloc[0].copy()
            best["selection_mode"] = mode
            best["policy"] = AlertPolicy(
                float(best["on_threshold"]),
                float(best["off_threshold"]),
                int(best["min_duration_steps"]),
                int(best["cooldown_steps"]),
            ).label
            selected.append(best)
    return pd.DataFrame(selected)


def _write_summary(path: Path, selected_val: pd.DataFrame, selected_test: Optional[pd.DataFrame]) -> None:
    lines = ["# Persistent Warning-Light Policy Tuning", ""]
    lines.append("Validation-selected policies. Test rows are final-report-only when provided.")
    for name, frame in [("Validation", selected_val), ("Test", selected_test)]:
        if frame is None:
            continue
        lines.extend(["", f"## {name}", ""])
        lines.append(
            "| threshold | selection | policy | UI precision | pre-onset recall | high-state coverage | UI false eps | lead s |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for _, row in frame.iterrows():
            lines.append(
                "| {thr:g} | {mode} | {policy} | {p:.4f} | {r:.4f} | {cov:.4f} | {false:.0f} | {lead:.2f} |".format(
                    thr=float(row["fms_threshold"]),
                    mode=str(row["selection_mode"]),
                    policy=str(row["policy"]),
                    p=float(row["ui_episode_precision"]),
                    r=float(row["pre_onset_event_recall"]),
                    cov=float(row["high_state_duration_coverage"]),
                    false=float(row["ui_false_alert_episode_count"]),
                    lead=float(row["mean_lead_time_seconds"]) if pd.notna(row["mean_lead_time_seconds"]) else float("nan"),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 12.0])
    parser.add_argument("--horizon_seconds", type=float, default=20.0)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument(
        "--on_thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument("--off_deltas", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--min_duration_seconds", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--cooldown_seconds", nargs="+", type=float, default=[0.0, 5.0, 10.0, 20.0, 30.0])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(args.val_csv)
    horizon_steps = int(round(float(args.horizon_seconds) / float(args.sampling_interval)))
    grid_rows: List[Dict[str, float]] = []
    for threshold in args.thresholds:
        score_col = _score_col(val, args.horizon_seconds, threshold)
        for policy in _policies(args):
            row = _evaluate(val, threshold, score_col, policy, horizon_steps, args.sampling_interval)
            row["run_name"] = args.run_name
            row["split"] = "val"
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "val_persistent_policy_grid.csv", index=False)
    selected_val = _select_rows(grid)
    selected_val.to_csv(out_dir / "selected_persistent_policy_validation_metrics.csv", index=False)

    selected_test: Optional[pd.DataFrame] = None
    if args.test_csv:
        test = pd.read_csv(args.test_csv)
        test_rows: List[Dict[str, float]] = []
        for _, row in selected_val.iterrows():
            threshold = float(row["fms_threshold"])
            score_col = _score_col(test, args.horizon_seconds, threshold)
            policy = AlertPolicy(
                float(row["on_threshold"]),
                float(row["off_threshold"]),
                int(row["min_duration_steps"]),
                int(row["cooldown_steps"]),
            )
            metrics = _evaluate(test, threshold, score_col, policy, horizon_steps, args.sampling_interval)
            metrics["run_name"] = args.run_name
            metrics["split"] = "test"
            metrics["selection_mode"] = row["selection_mode"]
            metrics["policy"] = policy.label
            test_rows.append(metrics)
        selected_test = pd.DataFrame(test_rows)
        selected_test.to_csv(out_dir / "selected_persistent_policy_test_metrics.csv", index=False)

    _write_summary(out_dir / "persistent_warning_policy_summary.md", selected_val, selected_test)


if __name__ == "__main__":
    main()
