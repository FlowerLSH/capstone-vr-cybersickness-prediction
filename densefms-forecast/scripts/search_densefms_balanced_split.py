"""Search a participant-level DenseFMS split using distribution balance only.

This script intentionally does not train or evaluate a model.  It searches
random participant splits and selects the split whose train/validation/test
diagnostic distributions are closest by pre-model statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in [SRC, SCRIPTS]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from densefms_forecast.data import apply_saved_split, load_raw_sessions
from densefms_forecast.utils import load_json, save_json, seconds_to_steps

from analyze_densefms_split_diagnostics import (
    FMS_BINS,
    _aggregate_split,
    _bin_counts,
    _format,
    _markdown_report,
    _participant_rows,
    _safe_float,
    _session_row,
    _smd,
    _write_csv,
)


BALANCE_METRICS: Sequence[Tuple[str, float]] = [
    ("age", 0.35),
    ("mssq", 0.35),
    ("calibration_end_fms", 1.00),
    ("calibration_mean_fms", 1.00),
    ("calibration_range_fms", 0.80),
    ("prediction_mean_fms", 1.35),
    ("prediction_range_fms", 1.35),
    ("low_fraction", 1.20),
    ("warning_fraction", 1.00),
    ("high_fraction", 1.20),
    ("max_rise_10s", 1.00),
    ("max_drop_10s", 1.00),
    ("motion_delta_norm_mean", 0.25),
    ("calibration_motion_delta_norm_mean", 0.25),
]

BIN_LABELS = ["0_2", "2_5", "5_10", "10_15", "15_20"]


def _make_counts(
    groups: Mapping[str, Sequence[str]],
    pid_to_session_count: Mapping[str, int],
) -> Dict[str, int]:
    return {
        split: int(sum(int(pid_to_session_count[pid]) for pid in pids))
        for split, pids in groups.items()
    }


def _candidate_groups(
    participants: Sequence[str],
    rng: np.random.Generator,
    train_frac: float,
    val_frac: float,
) -> Dict[str, List[str]]:
    shuffled = np.asarray(list(participants), dtype=object)
    rng.shuffle(shuffled)
    n = int(len(shuffled))
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": sorted(str(x) for x in shuffled[:n_train].tolist()),
        "val": sorted(str(x) for x in shuffled[n_train : n_train + n_val].tolist()),
        "test": sorted(str(x) for x in shuffled[n_train + n_val :].tolist()),
    }


def _summaries_for_groups(
    groups: Mapping[str, Sequence[str]],
    pid_to_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    pid_to_bins: Mapping[str, Mapping[str, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Counter[str]]]:
    rows: List[Dict[str, Any]] = []
    split_bin_counts: Dict[str, Counter[str]] = {split: Counter() for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for pid in groups[split]:
            for base_row in pid_to_rows[pid]:
                row = dict(base_row)
                row["split"] = split
                rows.append(row)
            split_bin_counts[split].update(pid_to_bins[pid])

    split_summary = [
        _aggregate_split(
            split,
            [row for row in rows if row["split"] == split],
            dict(split_bin_counts[split]),
        )
        for split in ["train", "val", "test"]
    ]
    return rows, split_summary, split_bin_counts


def _score_split(
    split_summary: Sequence[Mapping[str, Any]],
    total_sessions: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Dict[str, Any]:
    summary_by_split = {row["split"]: row for row in split_summary}
    train = summary_by_split["train"]
    val = summary_by_split["val"]
    test = summary_by_split["test"]

    weighted_smd_sum = 0.0
    max_abs_smd = 0.0
    val_max_abs_smd = 0.0
    test_max_abs_smd = 0.0
    val_test_max_abs_smd = 0.0
    metric_rows: List[Dict[str, Any]] = []
    for metric, weight in BALANCE_METRICS:
        val_smd = abs(_smd(train, val, metric))
        test_smd = abs(_smd(train, test, metric))
        val_test_smd = abs(_smd(val, test, metric))
        val_smd = 0.0 if not math.isfinite(val_smd) else val_smd
        test_smd = 0.0 if not math.isfinite(test_smd) else test_smd
        val_test_smd = 0.0 if not math.isfinite(val_test_smd) else val_test_smd
        weighted_smd_sum += float(weight) * (val_smd + test_smd + 0.75 * val_test_smd)
        val_max_abs_smd = max(val_max_abs_smd, val_smd)
        test_max_abs_smd = max(test_max_abs_smd, test_smd)
        val_test_max_abs_smd = max(val_test_max_abs_smd, val_test_smd)
        max_abs_smd = max(max_abs_smd, val_smd, test_smd, val_test_smd)
        metric_rows.append(
            {
                "metric": metric,
                "weight": weight,
                "val_vs_train_smd": val_smd,
                "test_vs_train_smd": test_smd,
                "test_vs_val_smd": val_test_smd,
            }
        )

    bin_fraction_penalty = 0.0
    for label in BIN_LABELS:
        key = f"bin_{label}_fraction"
        train_frac_value = _safe_float(train.get(key))
        val_frac_value = _safe_float(val.get(key))
        test_frac_value = _safe_float(test.get(key))
        if not (math.isfinite(train_frac_value) and math.isfinite(val_frac_value) and math.isfinite(test_frac_value)):
            continue
        bin_fraction_penalty += 5.0 * (
            abs(val_frac_value - train_frac_value)
            + abs(test_frac_value - train_frac_value)
            + 0.75 * abs(test_frac_value - val_frac_value)
        )

    target_sessions = {
        "train": float(total_sessions) * float(train_frac),
        "val": float(total_sessions) * float(val_frac),
        "test": float(total_sessions) * float(test_frac),
    }
    session_count_penalty = 0.0
    for split in ["train", "val", "test"]:
        observed = float(summary_by_split[split]["sessions"])
        session_count_penalty += abs(observed - target_sessions[split]) / max(float(total_sessions), 1.0)
    session_count_penalty *= 4.0

    max_smd_penalty = 0.25 * max_abs_smd
    score = weighted_smd_sum + bin_fraction_penalty + session_count_penalty + max_smd_penalty
    return {
        "score": float(score),
        "weighted_smd_sum": float(weighted_smd_sum),
        "bin_fraction_penalty": float(bin_fraction_penalty),
        "session_count_penalty": float(session_count_penalty),
        "max_abs_smd": float(max_abs_smd),
        "val_max_abs_smd": float(val_max_abs_smd),
        "test_max_abs_smd": float(test_max_abs_smd),
        "val_test_max_abs_smd": float(val_test_max_abs_smd),
        "metric_rows": metric_rows,
    }


def _score_existing_split(
    split_info: Mapping[str, Any],
    pid_to_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    pid_to_bins: Mapping[str, Mapping[str, int]],
    total_sessions: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Dict[str, Any]:
    groups = {
        split: sorted(str(pid) for pid in split_info["groups"][split])
        for split in ["train", "val", "test"]
    }
    rows, split_summary, _ = _summaries_for_groups(groups, pid_to_rows, pid_to_bins)
    score = _score_split(split_summary, total_sessions, train_frac, val_frac, test_frac)
    return {
        "groups": groups,
        "rows": rows,
        "split_summary": split_summary,
        "score": score,
    }


def _candidate_row(
    rank: int,
    candidate_seed: int,
    score_info: Mapping[str, Any],
    split_summary: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    summary_by_split = {row["split"]: row for row in split_summary}
    row: Dict[str, Any] = {
        "rank": int(rank),
        "candidate_seed": int(candidate_seed),
        "score": score_info["score"],
        "weighted_smd_sum": score_info["weighted_smd_sum"],
        "bin_fraction_penalty": score_info["bin_fraction_penalty"],
        "session_count_penalty": score_info["session_count_penalty"],
        "max_abs_smd": score_info["max_abs_smd"],
        "val_max_abs_smd": score_info["val_max_abs_smd"],
        "test_max_abs_smd": score_info["test_max_abs_smd"],
        "val_test_max_abs_smd": score_info["val_test_max_abs_smd"],
    }
    for split in ["train", "val", "test"]:
        summary = summary_by_split[split]
        row[f"{split}_participants"] = len(groups[split])
        row[f"{split}_sessions"] = int(summary["sessions"])
        row[f"{split}_prediction_mean_fms"] = summary["prediction_mean_fms_mean"]
        row[f"{split}_prediction_range_fms"] = summary["prediction_range_fms_mean"]
        row[f"{split}_low_fraction"] = summary["low_fraction_mean"]
        row[f"{split}_high_fraction"] = summary["high_fraction_mean"]
        row[f"{split}_max_rise_10s"] = summary["max_rise_10s_mean"]
        row[f"{split}_max_drop_10s"] = summary["max_drop_10s_mean"]
    return row


def _standardized_difference_rows(split_summary: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_by_split = {row["split"]: row for row in split_summary}
    train_summary = summary_by_split["train"]
    val_summary = summary_by_split["val"]
    test_summary = summary_by_split["test"]
    smd_metrics = [metric for metric, _ in BALANCE_METRICS]
    smd_rows: List[Dict[str, Any]] = []
    for split, split_row in summary_by_split.items():
        for metric in smd_metrics:
            smd_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "smd": 0.0 if split == "train" else _smd(train_summary, split_row, metric),
                    "train_mean": train_summary.get(f"{metric}_mean"),
                    "split_mean": split_row.get(f"{metric}_mean"),
                }
            )
    val_test_rows: List[Dict[str, Any]] = []
    for metric in smd_metrics:
        val_test_rows.append(
            {
                "reference_split": "val",
                "compared_split": "test",
                "metric": metric,
                "smd": _smd(val_summary, test_summary, metric),
                "reference_mean": val_summary.get(f"{metric}_mean"),
                "compared_mean": test_summary.get(f"{metric}_mean"),
            }
        )
    return smd_rows, val_test_rows


def _write_search_report(
    out_dir: Path,
    args: argparse.Namespace,
    best_payload: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_payload: Mapping[str, Any] | None,
) -> None:
    best_summary = {row["split"]: row for row in best_payload["split_summary"]}
    lines = [
        "# DenseFMS Balanced Participant Split Search",
        "",
        "## Selection Rule",
        "",
        "- No model training, validation metric, test metric, checkpoint, prediction CSV, or plot was used.",
        "- Candidate splits were scored only by participant-level distribution balance diagnostics.",
        "- The saved split should be treated as a new fixed experimental split before model selection begins.",
        "",
        "## Search Budget",
        "",
        f"- candidate count: `{int(args.num_candidates)}`",
        f"- seed start: `{int(args.seed)}`",
        f"- train/val/test participant fractions: `{args.train_frac:.3f}` / `{args.val_frac:.3f}` / `{args.test_frac:.3f}`",
        "",
        "## Best Split",
        "",
        f"- candidate seed: `{int(best_payload['candidate_seed'])}`",
        f"- balance score: `{_format(float(best_payload['score']['score']))}`",
        f"- max abs SMD: `{_format(float(best_payload['score']['max_abs_smd']))}`",
        f"- val max abs SMD: `{_format(float(best_payload['score']['val_max_abs_smd']))}`",
        f"- test max abs SMD: `{_format(float(best_payload['score']['test_max_abs_smd']))}`",
        f"- test-vs-val max abs SMD: `{_format(float(best_payload['score']['val_test_max_abs_smd']))}`",
        "",
        "| split | participants | sessions | points | mean FMS | calib-end | range | low frac | high frac | rise | drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        row = best_summary[split]
        lines.append(
            "| "
            f"{split} | {int(row['participants'])} | {int(row['sessions'])} | {int(row['prediction_points'])} | "
            f"{_format(float(row['prediction_mean_fms_mean']))} | {_format(float(row['calibration_end_fms_mean']))} | "
            f"{_format(float(row['prediction_range_fms_mean']))} | {_format(float(row['low_fraction_mean']))} | "
            f"{_format(float(row['high_fraction_mean']))} | {_format(float(row['max_rise_10s_mean']))} | "
            f"{_format(float(row['max_drop_10s_mean']))} |"
        )
    lines.extend(["", "## FMS Bin Fractions", "", "| split | 0-2 | 2-5 | 5-10 | 10-15 | 15-20 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for split in ["train", "val", "test"]:
        row = best_summary[split]
        lines.append(
            "| "
            + split
            + " | "
            + " | ".join(_format(float(row.get(f"bin_{label}_fraction", float("nan")))) for label in BIN_LABELS)
            + " |"
        )
    if reference_payload is not None:
        reference_score = reference_payload["score"]
        lines.extend(
            [
                "",
                "## Reference Split Comparison",
                "",
                "| split file | score | max abs SMD | val max SMD | test max SMD | test-vs-val max SMD |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                "| reference | "
                f"{_format(float(reference_score['score']))} | {_format(float(reference_score['max_abs_smd']))} | "
                f"{_format(float(reference_score['val_max_abs_smd']))} | {_format(float(reference_score['test_max_abs_smd']))} | "
                f"{_format(float(reference_score['val_test_max_abs_smd']))} |",
                "| selected balanced v2 | "
                f"{_format(float(best_payload['score']['score']))} | {_format(float(best_payload['score']['max_abs_smd']))} | "
                f"{_format(float(best_payload['score']['val_max_abs_smd']))} | {_format(float(best_payload['score']['test_max_abs_smd']))} | "
                f"{_format(float(best_payload['score']['val_test_max_abs_smd']))} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| rank | seed | score | max SMD | train sess | val sess | test sess |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(candidate_rows)[:10]:
        lines.append(
            "| "
            f"{int(row['rank'])} | {int(row['candidate_seed'])} | {_format(float(row['score']))} | "
            f"{_format(float(row['max_abs_smd']))} | {int(row['train_sessions'])} | "
            f"{int(row['val_sessions'])} | {int(row['test_sessions'])} |"
        )
    lines.extend(
        [
            "",
            "## Warning",
            "",
            "This split is a distribution-balanced experimental split, not a performance-selected split. "
            "Once model training starts on this split, validation-only selection and final test-only reporting should be enforced.",
            "",
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "balanced_split_search_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if abs(float(args.train_frac) + float(args.val_frac) + float(args.test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    raw_sessions, mapping, data_info = load_raw_sessions(
        args.data_dir,
        calibration_seconds=float(args.calibration_seconds),
        horizon_seconds=float(args.horizon_seconds),
        default_sampling_interval=float(args.sampling_interval),
        max_session_points=int(args.max_session_points) if args.max_session_points is not None else None,
    )
    sampling_interval = float(data_info["sampling_interval"])
    calibration_steps = int(data_info["calibration_steps"])
    recent_steps = seconds_to_steps(float(args.recent_window_seconds), sampling_interval, name="recent_window_seconds")
    rise_drop_horizon_steps = seconds_to_steps(float(args.rise_drop_horizon_seconds), sampling_interval, name="rise_drop_horizon_seconds")

    pid_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pid_to_bins: Dict[str, Counter[str]] = defaultdict(Counter)
    pid_to_session_count: Dict[str, int] = defaultdict(int)
    for session in raw_sessions:
        if session.participant_id is None:
            raise ValueError("participant_id is required for participant-level balanced split search.")
        pid = str(session.participant_id)
        row = _session_row(
            "pool",
            session,
            calibration_steps,
            recent_steps,
            sampling_interval,
            rise_drop_horizon_steps,
            float(args.rise_drop_threshold),
            float(args.flat_range_threshold),
        )
        pid_to_rows[pid].append(row)
        pid_to_session_count[pid] += 1
        start = int(row["prediction_start_index"])
        fms = np.asarray(session.fms_raw if session.fms_raw is not None else session.fms, dtype=np.float64)
        pid_to_bins[pid].update(_bin_counts(fms[start : session.length], FMS_BINS))

    participants = sorted(pid_to_rows)
    total_sessions = int(sum(pid_to_session_count.values()))
    if len(participants) < 3:
        raise ValueError(f"Need at least 3 participants, got {len(participants)}.")

    best_payload: Dict[str, Any] | None = None
    all_candidate_rows: List[Dict[str, Any]] = []
    for index in range(int(args.num_candidates)):
        candidate_seed = int(args.seed) + index
        rng = np.random.default_rng(candidate_seed)
        groups = _candidate_groups(participants, rng, float(args.train_frac), float(args.val_frac))
        rows, split_summary, _ = _summaries_for_groups(groups, pid_to_rows, pid_to_bins)
        score = _score_split(split_summary, total_sessions, float(args.train_frac), float(args.val_frac), float(args.test_frac))
        candidate_payload = {
            "candidate_seed": candidate_seed,
            "groups": groups,
            "rows": rows,
            "split_summary": split_summary,
            "score": score,
        }
        if best_payload is None or float(score["score"]) < float(best_payload["score"]["score"]):
            best_payload = candidate_payload
        all_candidate_rows.append(_candidate_row(index + 1, candidate_seed, score, split_summary, groups))

    if best_payload is None:
        raise RuntimeError("No split candidates were generated.")
    all_candidate_rows = sorted(all_candidate_rows, key=lambda row: float(row["score"]))
    for rank, row in enumerate(all_candidate_rows, start=1):
        row["rank"] = rank

    best_groups = best_payload["groups"]
    split_info = {
        "group_key": "participant_id",
        "groups": {split: list(best_groups[split]) for split in ["train", "val", "test"]},
        "counts": _make_counts(best_groups, pid_to_session_count),
        "selection": {
            "method": "participant_distribution_balance_only",
            "no_model_metrics_used": True,
            "candidate_count": int(args.num_candidates),
            "seed_start": int(args.seed),
            "candidate_seed": int(best_payload["candidate_seed"]),
            "score": best_payload["score"],
            "balance_metrics": [{"metric": metric, "weight": weight} for metric, weight in BALANCE_METRICS],
            "bin_labels": BIN_LABELS,
            "calibration_seconds": float(args.calibration_seconds),
            "recent_window_seconds": float(args.recent_window_seconds),
            "horizon_seconds": float(args.horizon_seconds),
            "max_session_points": int(args.max_session_points) if args.max_session_points is not None else None,
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "split_participant_stratified_v2.json", split_info)

    rows = best_payload["rows"]
    split_summary = best_payload["split_summary"]
    smd_rows, val_test_smd_rows = _standardized_difference_rows(split_summary)
    participant_summary = _participant_rows(rows)
    _write_csv(out_dir / "candidate_scores.csv", all_candidate_rows, list(all_candidate_rows[0].keys()))
    _write_csv(out_dir / "best_session_split_diagnostics.csv", rows, list(rows[0].keys()))
    _write_csv(out_dir / "best_split_summary.csv", split_summary, list(split_summary[0].keys()))
    _write_csv(out_dir / "best_split_standardized_differences.csv", smd_rows, list(smd_rows[0].keys()))
    _write_csv(out_dir / "best_val_test_standardized_differences.csv", val_test_smd_rows, list(val_test_smd_rows[0].keys()))
    if participant_summary:
        _write_csv(out_dir / "best_participant_split_summary.csv", participant_summary, list(participant_summary[0].keys()))

    reference_payload = None
    if args.reference_split_file:
        reference_payload = _score_existing_split(
            load_json(args.reference_split_file),
            pid_to_rows,
            pid_to_bins,
            total_sessions,
            float(args.train_frac),
            float(args.val_frac),
            float(args.test_frac),
        )
        ref_smd, ref_val_test_smd = _standardized_difference_rows(reference_payload["split_summary"])
        _write_csv(out_dir / "reference_split_summary.csv", reference_payload["split_summary"], list(reference_payload["split_summary"][0].keys()))
        _write_csv(out_dir / "reference_split_standardized_differences.csv", ref_smd, list(ref_smd[0].keys()))
        _write_csv(out_dir / "reference_val_test_standardized_differences.csv", ref_val_test_smd, list(ref_val_test_smd[0].keys()))

    payload = {
        "args": vars(args),
        "data_info": data_info,
        "mapping": mapping,
        "split_info": split_info,
        "best": {
            "candidate_seed": best_payload["candidate_seed"],
            "score": best_payload["score"],
            "split_summary": split_summary,
        },
        "reference": None
        if reference_payload is None
        else {
            "score": reference_payload["score"],
            "split_summary": reference_payload["split_summary"],
        },
    }
    (out_dir / "balanced_split_search_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _markdown_report(out_dir / "best_diagnostic", data_info, split_summary, smd_rows, val_test_smd_rows, participant_summary, rows)
    _write_search_report(out_dir, args, best_payload, all_candidate_rows, reference_payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reference_split_file", default="")
    parser.add_argument("--num_candidates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--sampling_interval", type=float, default=0.5)
    parser.add_argument("--calibration_seconds", type=float, default=120.0)
    parser.add_argument("--recent_window_seconds", type=float, default=10.0)
    parser.add_argument("--horizon_seconds", type=float, default=10.0)
    parser.add_argument("--max_session_points", type=int, default=420)
    parser.add_argument("--rise_drop_horizon_seconds", type=float, default=10.0)
    parser.add_argument("--rise_drop_threshold", type=float, default=3.0)
    parser.add_argument("--flat_range_threshold", type=float, default=2.5)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"out_dir": args.out_dir, "best": payload["best"], "reference": payload["reference"]}, indent=2))


if __name__ == "__main__":
    main()
