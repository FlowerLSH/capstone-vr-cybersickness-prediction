"""Plot annotated online-current warning predictions on validation trajectories."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECIPES: Mapping[str, str] = {
    "selected_risk035": "selected_risk035_warnext_stage1_full_seed42",
    "risk045_smooth005": "risk045_smooth005_warnext_stage1_full_seed42",
    "zero_anchor_highgate_delta2": "zero_anchor_highgate_delta2_warnext_stage1_full_seed42",
    "range_scaled_delta2": "range_scaled_delta2_warnext_stage1_full_seed42",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _shade_regions(ax: plt.Axes, x: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> None:
    if len(x) == 0 or len(mask) == 0:
        return
    active = np.asarray(mask, dtype=bool)
    starts: List[int] = []
    ends: List[int] = []
    in_region = False
    start = 0
    for idx, flag in enumerate(active):
        if flag and not in_region:
            start = idx
            in_region = True
        if in_region and (not flag or idx == len(active) - 1):
            end = idx if not flag else idx + 1
            starts.append(start)
            ends.append(end)
            in_region = False
    for start, end in zip(starts, ends):
        left = float(x[start])
        right = float(x[min(end - 1, len(x) - 1)])
        if right <= left and len(x) > 1:
            step = float(np.nanmedian(np.diff(x)))
            right = left + max(step, 0.5)
        ax.axvspan(left, right, color=color, alpha=alpha, linewidth=0)


def _rug(ax: plt.Axes, x: np.ndarray, mask: np.ndarray, y: float, color: str, label: str) -> None:
    x_sel = x[np.asarray(mask, dtype=bool)]
    if len(x_sel):
        ax.scatter(x_sel, np.full_like(x_sel, y, dtype=float), s=10, marker="|", color=color, label=label, alpha=0.75)


def _plot_probability(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    marker: str,
    threshold: float = 0.5,
) -> np.ndarray:
    ax.plot(x, y, color=color, linewidth=1.2, alpha=0.85, label=label)
    active = np.isfinite(y) & (y >= threshold)
    if active.any():
        ax.scatter(x[active], y[active], s=18, marker=marker, color=color, edgecolor="white", linewidth=0.4)
    return active


def _session_movement_score(df: pd.DataFrame) -> pd.Series:
    return df.groupby("session_id")["target_fms_now"].agg(lambda s: float(np.nanmax(s) - np.nanmin(s))).sort_values(
        ascending=False
    )


def _plot_session(recipe: str, session_id: str, session: pd.DataFrame, out_path: Path) -> Dict[str, object]:
    g = session.sort_values("current_index").copy()
    x = g["current_time"].astype(float).to_numpy()
    y_true = g["target_fms_now"].astype(float).to_numpy()
    y_pred = g["predicted_fms_now"].astype(float).to_numpy()

    pred_ge8 = np.isfinite(y_pred) & (y_pred >= 8.0)
    pred_ge12 = np.isfinite(y_pred) & (y_pred >= 12.0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.2, 1.3]},
    )
    ax_fms, ax_high, ax_event = axes

    _shade_regions(ax_fms, x, pred_ge8, "#f2c94c", 0.12)
    _shade_regions(ax_fms, x, pred_ge12, "#eb5757", 0.18)
    ax_fms.plot(x, y_true, color="black", linewidth=1.8, label="true current FMS")
    ax_fms.plot(x, y_pred, color="#d95f02", linewidth=1.6, label="pred current FMS")
    if pred_ge8.any():
        ax_fms.scatter(x[pred_ge8 & ~pred_ge12], y_pred[pred_ge8 & ~pred_ge12], s=22, color="#f2c94c", edgecolor="#7a5b00", linewidth=0.4, label="pred >= 8")
    if pred_ge12.any():
        ax_fms.scatter(x[pred_ge12], y_pred[pred_ge12], s=24, color="#eb5757", edgecolor="#8f1d1d", linewidth=0.4, label="pred >= 12")
    ax_fms.axhline(8.0, color="#c49a00", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_fms.axhline(12.0, color="#c1272d", linestyle="--", linewidth=1.0, alpha=0.75)
    ax_fms.set_ylabel("FMS")
    ax_fms.set_ylim(-0.5, 20.8)
    ax_fms.grid(True, alpha=0.25)
    ax_fms.legend(loc="upper left", ncol=4, fontsize=8)

    p_high8 = g.get("p_high_risk_20s_thr8", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    p_high12 = g.get("p_high_risk_20s_thr12", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    active_high8 = _plot_probability(ax_high, x, p_high8, "#2ca25f", "P(FMS>=8 within 20s)", "o")
    active_high12 = _plot_probability(ax_high, x, p_high12, "#de2d26", "P(FMS>=12 within 20s)", "o")
    if "high_risk_label_20s_thr8" in g:
        _rug(ax_high, x, _as_bool(g["high_risk_label_20s_thr8"]).to_numpy(), -0.05, "#2ca25f", "true high8")
    if "high_risk_label_20s_thr12" in g:
        _rug(ax_high, x, _as_bool(g["high_risk_label_20s_thr12"]).to_numpy(), -0.10, "#de2d26", "true high12")
    ax_high.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_high.set_ylabel("high-risk prob")
    ax_high.set_ylim(-0.16, 1.05)
    ax_high.grid(True, alpha=0.25)
    ax_high.legend(loc="upper left", ncol=4, fontsize=8)

    p_rise10 = g.get("p_rapid_rise_10s", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    p_rise20 = g.get("p_rapid_rise_20s", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    p_drop10 = g.get("p_rapid_drop_10s", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    p_drop20 = g.get("p_rapid_drop_20s", pd.Series(np.nan, index=g.index)).astype(float).to_numpy()
    active_rise10 = _plot_probability(ax_event, x, p_rise10, "#3182bd", "P(rise 10s)", "^")
    active_rise20 = _plot_probability(ax_event, x, p_rise20, "#08519c", "P(rise 20s)", "^")
    active_drop10 = _plot_probability(ax_event, x, p_drop10, "#756bb1", "P(drop 10s)", "v")
    active_drop20 = _plot_probability(ax_event, x, p_drop20, "#54278f", "P(drop 20s)", "v")
    if "rapid_rise_label_10s" in g:
        _rug(ax_event, x, _as_bool(g["rapid_rise_label_10s"]).to_numpy(), -0.05, "#3182bd", "true rise10")
    if "rapid_drop_label_10s" in g:
        _rug(ax_event, x, _as_bool(g["rapid_drop_label_10s"]).to_numpy(), -0.10, "#756bb1", "true drop10")
    ax_event.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_event.set_ylabel("rise/drop prob")
    ax_event.set_xlabel("current time (s)")
    ax_event.set_ylim(-0.16, 1.05)
    ax_event.grid(True, alpha=0.25)
    ax_event.legend(loc="upper left", ncol=6, fontsize=8)

    title = f"{recipe} | {session_id}"
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "recipe": recipe,
        "session_id": session_id,
        "plot_path": str(out_path),
        "points": int(len(g)),
        "true_range": float(np.nanmax(y_true) - np.nanmin(y_true)),
        "pred_range": float(np.nanmax(y_pred) - np.nanmin(y_pred)),
        "pred_ge8_count": int(pred_ge8.sum()),
        "pred_ge12_count": int(pred_ge12.sum()),
        "high8_prob_ge05_count": int(active_high8.sum()),
        "high12_prob_ge05_count": int(active_high12.sum()),
        "rise10_prob_ge05_count": int(active_rise10.sum()),
        "rise20_prob_ge05_count": int(active_rise20.sum()),
        "drop10_prob_ge05_count": int(active_drop10.sum()),
        "drop20_prob_ge05_count": int(active_drop20.sum()),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: Sequence[Mapping[str, object]], selected_sessions: Sequence[str]) -> None:
    by_recipe: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_recipe.setdefault(str(row["recipe"]), []).append(row)
    lines = [
        "# Warning Annotation Plot Summary",
        "",
        "Validation prediction CSV 기준으로 만든 시각화다. Test set은 사용하지 않았다.",
        "",
        "## 표시 기준",
        "",
        "- `pred >= 8`: `predicted_fms_now >= 8`인 시점이다.",
        "- `pred >= 12`: `predicted_fms_now >= 12`인 시점이다.",
        "- `P(FMS>=8/12 within 20s)`: high-risk head 확률이며, 0.5 이상인 시점에 마커가 찍힌다.",
        "- `P(rise/drop 10s/20s)`: rapid rise/drop head 확률이며, 0.5 이상인 시점에 삼각형 마커가 찍힌다.",
        "- 아래쪽 rug mark의 `true ...`는 해당 label이 실제로 positive인 구간을 보기 위한 보조 표시다.",
        "",
        "## 폴더",
        "",
        "- 전체 validation session plot: `all_sessions/<model>/`",
        "- 변화폭이 큰 대표 session plot: `selected_high_movement/<model>/`",
        "- 세션별 event count CSV: `warning_annotation_plot_index.csv`",
        "",
        "## 대표 세션",
        "",
    ]
    for sid in selected_sessions:
        lines.append(f"- `{sid}`")
    lines.extend(["", "## 모델별 전체 count", ""])
    lines.append(
        "| model | plots | pred>=8 | pred>=12 | high8 prob>=.5 | high12 prob>=.5 | rise10>=.5 | rise20>=.5 | drop10>=.5 | drop20>=.5 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for recipe, recipe_rows in by_recipe.items():
        total = lambda key: sum(int(row[key]) for row in recipe_rows)
        lines.append(
            f"| {recipe} | {len(recipe_rows)} | {total('pred_ge8_count')} | {total('pred_ge12_count')} | "
            f"{total('high8_prob_ge05_count')} | {total('high12_prob_ge05_count')} | "
            f"{total('rise10_prob_ge05_count')} | {total('rise20_prob_ge05_count')} | "
            f"{total('drop10_prob_ge05_count')} | {total('drop20_prob_ge05_count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot warning annotations on online-current validation trajectories.")
    parser.add_argument("--runs_dir", default="runs/online_current_warning_extension_0514")
    parser.add_argument("--report_dir", default="reports/online_current_warning_extension_0514/warning_annotation_plots")
    parser.add_argument("--selected_sessions", type=int, default=12)
    parser.add_argument("--recipes", nargs="*", default=list(RECIPES.keys()))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    first_recipe = args.recipes[0]
    first_csv = runs_dir / RECIPES[first_recipe] / "val_predictions.csv"
    first_df = pd.read_csv(first_csv)
    selected_sessions = list(_session_movement_score(first_df).head(args.selected_sessions).index)

    rows: List[Dict[str, object]] = []
    for recipe in args.recipes:
        run_name = RECIPES[recipe]
        pred_csv = runs_dir / run_name / "val_predictions.csv"
        df = pd.read_csv(pred_csv)
        for session_id, session in df.groupby("session_id"):
            session_safe = _safe_name(str(session_id))
            out_all = report_dir / "all_sessions" / recipe / f"{session_safe}.png"
            rows.append(_plot_session(recipe, str(session_id), session, out_all))
            if session_id in selected_sessions:
                out_selected = report_dir / "selected_high_movement" / recipe / f"{session_safe}.png"
                _plot_session(recipe, str(session_id), session, out_selected)

    _write_csv(report_dir / "warning_annotation_plot_index.csv", rows)
    _write_summary(report_dir / "README.md", rows, selected_sessions)
    print(f"wrote {len(rows)} full-session plots under {report_dir}")


if __name__ == "__main__":
    main()
