"""Plot validation comparisons for online current-FMS tracker runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_")[:90] or "session"


def _read_run(run_dir: Path, label: str, split: str) -> Dict[str, object]:
    curves_path = run_dir / "training_curves.csv"
    preds_path = run_dir / f"{split}_predictions.csv"
    if not curves_path.exists():
        raise FileNotFoundError(f"Missing training curves: {curves_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {preds_path}")
    curves = pd.read_csv(curves_path)
    preds = pd.read_csv(preds_path)
    best_idx = int(curves["val_mae"].idxmin())
    return {
        "label": label,
        "run_dir": run_dir,
        "curves": curves,
        "preds": preds,
        "best_epoch": int(curves.loc[best_idx, "epoch"]),
        "best_val_mae": float(curves.loc[best_idx, "val_mae"]),
        "best_val_rmse": float(curves.loc[best_idx, "val_rmse"]),
    }


def _plot_mae_curves(runs: List[Dict[str, object]], out_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for run in runs:
        curves = run["curves"]
        label = str(run["label"])
        plt.plot(curves["epoch"], curves["val_mae"], linewidth=2, label=label)
        plt.scatter([run["best_epoch"]], [run["best_val_mae"]], s=45)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("Current FMS Validation MAE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_mae_curves.png", dpi=160)
    plt.close()


def _plot_scatter(runs: List[Dict[str, object]], out_dir: Path, max_points: int) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), sharex=True, sharey=True)
    if len(runs) == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, run in zip(axes, runs):
        preds = run["preds"]
        if len(preds) > max_points:
            preds = preds.iloc[rng.choice(len(preds), size=max_points, replace=False)]
        true = preds["target_fms_now"].to_numpy(dtype=float)
        pred = preds["predicted_fms_now"].to_numpy(dtype=float)
        ax.scatter(true, pred, s=8, alpha=0.22)
        ax.plot([0, 20], [0, 20], color="black", linewidth=1.2)
        ax.set_title(f"{run['label']}\nMAE {run['best_val_mae']:.3f}")
        ax.set_xlabel("True current FMS")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted current FMS")
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_scatter.png", dpi=160)
    plt.close(fig)


def _plot_error_distribution(runs: List[Dict[str, object]], out_dir: Path) -> None:
    errors = [run["preds"]["fms_absolute_error"].to_numpy(dtype=float) for run in runs]
    labels = [str(run["label"]) for run in runs]
    plt.figure(figsize=(9, 5))
    plt.boxplot(errors, tick_labels=labels, showfliers=False)
    plt.ylabel("Absolute Error")
    plt.title("Validation Error Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "error_distribution.png", dpi=160)
    plt.close()


def _select_sessions(primary_preds: pd.DataFrame, count: int) -> List[str]:
    grouped = primary_preds.groupby("session_id")["fms_absolute_error"].mean().sort_values()
    if grouped.empty:
        return []
    indices = np.linspace(0, len(grouped) - 1, num=min(count, len(grouped)), dtype=int)
    return [str(grouped.index[idx]) for idx in indices]


def _plot_trajectories(runs: List[Dict[str, object]], out_dir: Path, primary_label: str, count: int) -> None:
    primary = next((run for run in runs if str(run["label"]) == primary_label), runs[0])
    sessions = _select_sessions(primary["preds"], count)
    for idx, session_id in enumerate(sessions):
        plt.figure(figsize=(11, 4.8))
        true_plotted = False
        for run in runs:
            frame = run["preds"]
            session = frame[frame["session_id"] == session_id].sort_values("current_time")
            if session.empty:
                continue
            if not true_plotted:
                plt.plot(
                    session["current_time"],
                    session["target_fms_now"],
                    color="black",
                    linewidth=2.4,
                    label="true current FMS",
                )
                true_plotted = True
            plt.plot(
                session["current_time"],
                session["predicted_fms_now"],
                linewidth=1.8,
                alpha=0.9,
                label=str(run["label"]),
            )
        plt.xlabel("Time (s)")
        plt.ylabel("FMS")
        plt.title(session_id)
        plt.grid(alpha=0.22)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_{idx + 1:02d}_{_safe_name(session_id)}.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot online current-FMS validation comparisons.")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--primary_label", default=None)
    parser.add_argument("--trajectory_count", type=int, default=5)
    parser.add_argument("--max_scatter_points", type=int, default=8000)
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.labels):
        raise ValueError("--run_dirs and --labels must have the same length.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [_read_run(Path(run_dir), label, args.split) for run_dir, label in zip(args.run_dirs, args.labels)]
    leaderboard = pd.DataFrame(
        [
            {
                "label": run["label"],
                "run_dir": str(run["run_dir"]),
                "best_epoch": run["best_epoch"],
                "val_mae": run["best_val_mae"],
                "val_rmse": run["best_val_rmse"],
                "prediction_rows": len(run["preds"]),
            }
            for run in runs
        ]
    ).sort_values("val_mae")
    leaderboard.to_csv(out_dir / "current_fms_3run_leaderboard.csv", index=False)
    primary_label = args.primary_label or str(leaderboard.iloc[0]["label"])

    _plot_mae_curves(runs, out_dir)
    _plot_scatter(runs, out_dir, args.max_scatter_points)
    _plot_error_distribution(runs, out_dir)
    _plot_trajectories(runs, out_dir, primary_label, args.trajectory_count)
    print(f"Saved plots to {out_dir}")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
