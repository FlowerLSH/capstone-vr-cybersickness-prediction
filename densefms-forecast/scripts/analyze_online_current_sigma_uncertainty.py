"""Analyze validation uncertainty sigma from online-current warning-extension runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique_values
    if np.any(counts > 1):
        starts = np.cumsum(np.r_[0, counts[:-1]])
        avg = starts + (counts - 1) / 2.0
        ranks = avg[inverse]
    return ranks + 1.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask].astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _quantile_bin_rows(recipe: str, sigma: np.ndarray, abs_err: np.ndarray, bins: int) -> List[Dict[str, object]]:
    mask = np.isfinite(sigma) & np.isfinite(abs_err)
    sigma = sigma[mask]
    abs_err = abs_err[mask]
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(sigma, quantiles)
    rows: List[Dict[str, object]] = []
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            m = (sigma >= left) & (sigma <= right)
        else:
            m = (sigma >= left) & (sigma < right)
        if not m.any():
            continue
        s = sigma[m]
        e = abs_err[m]
        rows.append(
            {
                "recipe": recipe,
                "bin": idx + 1,
                "n": int(m.sum()),
                "sigma_min": float(np.min(s)),
                "sigma_max": float(np.max(s)),
                "sigma_mean": float(np.mean(s)),
                "abs_error_mean": float(np.mean(e)),
                "abs_error_median": float(np.median(e)),
                "coverage_1sigma": float(np.mean(e <= s)),
                "coverage_2sigma": float(np.mean(e <= 2.0 * s)),
                "coverage_3sigma": float(np.mean(e <= 3.0 * s)),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(f) else f"{f:.4f}"


def _plot_bins(out_dir: Path, bin_rows: Sequence[Mapping[str, object]]) -> None:
    df = pd.DataFrame(bin_rows)
    if df.empty:
        return
    for recipe, g in df.groupby("recipe"):
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        x = g["bin"].astype(int).to_numpy()
        sigma_mean = g["sigma_mean"].astype(float).to_numpy()
        err_mean = g["abs_error_mean"].astype(float).to_numpy()
        cov1 = g["coverage_1sigma"].astype(float).to_numpy()
        ax1.plot(x, sigma_mean, marker="o", color="#2b8cbe", label="mean sigma")
        ax1.plot(x, err_mean, marker="o", color="#d95f02", label="mean abs error")
        ax1.set_xlabel("sigma quantile bin")
        ax1.set_ylabel("FMS points")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(x, cov1, marker="s", color="#31a354", label="coverage |err|<=sigma")
        ax2.axhline(0.683, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("coverage")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        fig.suptitle(recipe)
        fig.tight_layout()
        fig.savefig(out_dir / f"{recipe}_sigma_bins.png", dpi=150)
        plt.close(fig)


def _plot_scatter(out_dir: Path, recipe: str, sigma: np.ndarray, abs_err: np.ndarray) -> None:
    mask = np.isfinite(sigma) & np.isfinite(abs_err)
    sigma = sigma[mask]
    abs_err = abs_err[mask]
    if sigma.size == 0:
        return
    if sigma.size > 3000:
        rng = np.random.default_rng(42)
        idx = rng.choice(sigma.size, size=3000, replace=False)
        sigma = sigma[idx]
        abs_err = abs_err[idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sigma, abs_err, s=8, alpha=0.25, color="#525252")
    max_v = float(np.nanmax([np.max(sigma), np.max(abs_err), 1.0]))
    ax.plot([0, max_v], [0, max_v], color="#31a354", linestyle="--", linewidth=1.0, label="|err| = sigma")
    ax.plot([0, max_v], [0, 2 * max_v], color="#756bb1", linestyle=":", linewidth=1.0, label="|err| = 2*sigma")
    ax.set_xlabel("predicted sigma (FMS points)")
    ax.set_ylabel("absolute error (FMS points)")
    ax.set_title(recipe)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{recipe}_sigma_vs_abs_error.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty sigma in validation predictions.")
    parser.add_argument("--runs_dir", default="runs/online_current_warning_extension_0514")
    parser.add_argument("--report_dir", default="reports/online_current_warning_extension_0514/sigma_uncertainty_analysis")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    for recipe, run_name in RECIPES.items():
        df = pd.read_csv(runs_dir / run_name / "val_predictions.csv")
        sigma = df["predicted_fms_sigma"].astype(float).to_numpy()
        abs_err = df["fms_absolute_error"].astype(float).to_numpy()
        mask = np.isfinite(sigma) & np.isfinite(abs_err) & (sigma > 0)
        sigma = sigma[mask]
        abs_err = abs_err[mask]
        rmse = float(np.sqrt(np.mean(abs_err**2)))
        rms_sigma = float(np.sqrt(np.mean(sigma**2)))
        high_err_3 = abs_err >= 3.0
        high_err_5 = abs_err >= 5.0
        top20_thr = float(np.quantile(sigma, 0.80))
        top20 = sigma >= top20_thr
        summary_rows.append(
            {
                "recipe": recipe,
                "n": int(sigma.size),
                "mae": float(np.mean(abs_err)),
                "rmse": rmse,
                "sigma_mean": float(np.mean(sigma)),
                "sigma_median": float(np.median(sigma)),
                "sigma_p10": float(np.quantile(sigma, 0.10)),
                "sigma_p90": float(np.quantile(sigma, 0.90)),
                "pearson_sigma_abs_error": _pearson(sigma, abs_err),
                "spearman_sigma_abs_error": _spearman(sigma, abs_err),
                "auc_sigma_detect_abs_error_ge3": _auc_score(sigma, high_err_3.astype(float)),
                "auc_sigma_detect_abs_error_ge5": _auc_score(sigma, high_err_5.astype(float)),
                "coverage_1sigma": float(np.mean(abs_err <= sigma)),
                "coverage_2sigma": float(np.mean(abs_err <= 2.0 * sigma)),
                "coverage_3sigma": float(np.mean(abs_err <= 3.0 * sigma)),
                "rmse_over_rms_sigma": rmse / max(rms_sigma, 1e-12),
                "mae_over_expected_gaussian_abs_error": float(np.mean(abs_err))
                / max(math.sqrt(2.0 / math.pi) * float(np.mean(sigma)), 1e-12),
                "top20_sigma_threshold": top20_thr,
                "top20_sigma_mae": float(np.mean(abs_err[top20])),
                "bottom80_sigma_mae": float(np.mean(abs_err[~top20])),
            }
        )
        bin_rows.extend(_quantile_bin_rows(recipe, sigma, abs_err, args.bins))
        _plot_scatter(out_dir, recipe, sigma, abs_err)

    _write_csv(out_dir / "sigma_uncertainty_summary.csv", summary_rows)
    _write_csv(out_dir / "sigma_uncertainty_bins.csv", bin_rows)
    _plot_bins(out_dir, bin_rows)

    lines = ["# Sigma Uncertainty Analysis", ""]
    lines.append("Validation prediction CSV 기준 분석이다. Test set은 사용하지 않았다.")
    lines.append("")
    lines.append("`predicted_fms_sigma`는 uncertainty head가 출력한 current FMS 표준편차 추정값이다. 값 단위는 FMS point다.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| recipe | MAE | sigma mean | sigma median | corr(sigma, abs err) | spearman | AUC err>=3 | cover 1sigma | cover 2sigma | scale gap RMSE/RMSsigma | top20 sigma MAE | bottom80 MAE |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {recipe} | {mae} | {sm} | {smed} | {corr} | {sp} | {auc3} | {cov1} | {cov2} | {gap} | {top} | {bot} |".format(
                recipe=row["recipe"],
                mae=_fmt(row["mae"]),
                sm=_fmt(row["sigma_mean"]),
                smed=_fmt(row["sigma_median"]),
                corr=_fmt(row["pearson_sigma_abs_error"]),
                sp=_fmt(row["spearman_sigma_abs_error"]),
                auc3=_fmt(row["auc_sigma_detect_abs_error_ge3"]),
                cov1=_fmt(row["coverage_1sigma"]),
                cov2=_fmt(row["coverage_2sigma"]),
                gap=_fmt(row["rmse_over_rms_sigma"]),
                top=_fmt(row["top20_sigma_mae"]),
                bot=_fmt(row["bottom80_sigma_mae"]),
            )
        )
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- 이상적인 Gaussian sigma라면 `|error| <= sigma` coverage가 약 0.683, `|error| <= 2*sigma` coverage가 약 0.954에 가까워야 한다.")
    lines.append("- 현재 coverage가 낮고 `RMSE/RMSsigma`가 1보다 훨씬 크면 sigma의 절대 스케일이 너무 작다는 뜻이다.")
    lines.append("- 하지만 `corr(sigma, abs err)`와 top20 sigma MAE가 bottom80보다 크면, 절대 calibration은 틀려도 ranking signal로는 쓸 수 있다.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `sigma_uncertainty_summary.csv`: 모델별 전체 요약")
    lines.append("- `sigma_uncertainty_bins.csv`: sigma quintile별 error/coverage")
    lines.append("- `*_sigma_bins.png`: sigma bin별 평균 error와 coverage")
    lines.append("- `*_sigma_vs_abs_error.png`: sigma와 absolute error scatter")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote sigma analysis to {out_dir}")


if __name__ == "__main__":
    main()
