"""Validation-only weight search for online current-FMS prediction ensembles."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize


KEY_COLUMNS = ["session_id", "current_index", "current_time", "target_fms_now"]
DEFAULT_ORDINAL_BINS = [0, 2, 4, 6, 8, 10, 12, 15, 20]


def _parse_member(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path)
    path = Path(value)
    return path.parent.name, path


def _runner_members(runs_dir: Path) -> List[Tuple[str, Path]]:
    try:
        from scripts.run_online_current_long_search import DEFAULT_CANDIDATES
    except ModuleNotFoundError:
        module_path = Path(__file__).with_name("run_online_current_long_search.py")
        spec = importlib.util.spec_from_file_location("run_online_current_long_search", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DEFAULT_CANDIDATES = module.DEFAULT_CANDIDATES

    members: List[Tuple[str, Path]] = []
    for spec in DEFAULT_CANDIDATES:
        run_name = str(spec["run_name"])
        label = run_name
        if label.startswith("current_fms_"):
            label = label[len("current_fms_") :]
        members.append((label, runs_dir / run_name))
    return members


def _prediction_path(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    direct = path / f"{split}_predictions.csv"
    if direct.exists():
        return direct
    nested = path / f"eval_{split}" / f"{split}_predictions.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Missing {split} prediction CSV under {path}")


def _load_prediction(label: str, path: Path, split: str) -> pd.DataFrame:
    csv_path = _prediction_path(path, split)
    frame = pd.read_csv(csv_path).sort_values(KEY_COLUMNS).reset_index(drop=True)
    missing = set(KEY_COLUMNS + ["predicted_fms_now"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    frame[f"pred__{label}"] = frame["predicted_fms_now"].astype(float)
    return frame


def _assert_aligned(base: pd.DataFrame, frame: pd.DataFrame, label: str) -> None:
    if len(base) != len(frame):
        raise ValueError(f"{label} row count mismatch: {len(frame)} != {len(base)}")
    for column in KEY_COLUMNS:
        left = base[column].to_numpy()
        right = frame[column].to_numpy()
        if np.issubdtype(base[column].dtype, np.number) and np.issubdtype(frame[column].dtype, np.number):
            aligned = np.allclose(left.astype(np.float64), right.astype(np.float64), rtol=0.0, atol=1e-7, equal_nan=True)
        else:
            aligned = np.array_equal(left, right)
        if not aligned:
            raise ValueError(f"{label} is not aligned on {column}")


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _optimise_weights(x: np.ndarray, y: np.ndarray, starts: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    n_members = int(x.shape[0])
    bounds = [(0.0, 1.0)] * n_members
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def objective(w: np.ndarray) -> float:
        return _mae(w @ x, y)

    best_w = np.full(n_members, 1.0 / n_members, dtype=np.float64)
    best_score = objective(best_w)
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )
        if result.success:
            weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
            total = float(np.sum(weights))
            if total > 0:
                weights = weights / total
                score = objective(weights)
                if score < best_score:
                    best_w = weights
                    best_score = score
    return best_w, float(best_score)


def _optimise_weights_linprog(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve the convex simplex-constrained MAE ensemble problem exactly."""

    n_members = int(x.shape[0])
    n_samples = int(x.shape[1])
    x_t = sparse.csr_matrix(np.asarray(x.T, dtype=np.float64))
    identity = sparse.eye(n_samples, format="csr", dtype=np.float64)
    a_ub = sparse.vstack(
        [
            sparse.hstack([x_t, -identity], format="csr"),
            sparse.hstack([-x_t, -identity], format="csr"),
        ],
        format="csr",
    )
    b_ub = np.concatenate([y, -y]).astype(np.float64)
    a_eq = sparse.hstack(
        [
            sparse.csr_matrix(np.ones((1, n_members), dtype=np.float64)),
            sparse.csr_matrix((1, n_samples), dtype=np.float64),
        ],
        format="csr",
    )
    b_eq = np.array([1.0], dtype=np.float64)
    c = np.concatenate([np.zeros(n_members, dtype=np.float64), np.ones(n_samples, dtype=np.float64) / max(n_samples, 1)])
    bounds = [(0.0, 1.0)] * n_members + [(0.0, None)] * n_samples
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")
    weights = np.clip(np.asarray(result.x[:n_members], dtype=np.float64), 0.0, 1.0)
    total = float(np.sum(weights))
    if total <= 0:
        raise RuntimeError("linprog returned zero ensemble weights")
    weights = weights / total
    return weights, _mae(weights @ x, y)


def _ordinal_bins(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    edges = np.asarray(list(bins), dtype=np.float64)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _metrics(frame: pd.DataFrame) -> Dict[str, float]:
    y = frame["target_fms_now"].to_numpy(dtype=np.float64)
    pred = frame["predicted_fms_now"].to_numpy(dtype=np.float64)
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(y)),
        "prediction_std": float(np.std(pred)),
        "target_std": float(np.std(y)),
        "n": float(len(frame)),
    }


def _write_ensemble(
    out_dir: Path,
    split: str,
    label: str,
    base: pd.DataFrame,
    labels: Sequence[str],
    paths: Sequence[Path],
    weights: np.ndarray,
    predictions: np.ndarray,
    ordinal_bins: Sequence[float],
) -> None:
    pred = weights @ predictions
    out = base.drop(columns=[column for column in base.columns if column.startswith("pred__")])
    out["run_name"] = label
    out["model_name"] = "online_current_weighted_prediction_ensemble"
    out["split"] = split
    out["predicted_fms_now"] = pred
    out["fms_absolute_error"] = np.abs(pred - out["target_fms_now"].to_numpy(dtype=np.float64))
    if "ordinal_bin_pred" in out.columns:
        out["ordinal_bin_pred"] = _ordinal_bins(pred, ordinal_bins)
    if "alarm_caution" in out.columns:
        out["alarm_caution"] = out["predicted_fms_now"] >= 8.0
    if "alarm_warning_high_fms" in out.columns:
        out["alarm_warning_high_fms"] = out["predicted_fms_now"] >= 12.0
    if "final_warning" in out.columns and "alarm_warning_high_fms" in out.columns:
        rapid = out["alarm_warning_rapid_rise"] if "alarm_warning_rapid_rise" in out.columns else False
        out["final_warning"] = out["alarm_warning_high_fms"] | rapid
    for member_label, member_pred in zip(labels, predictions):
        out[f"member_pred_{member_label}"] = member_pred
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    payload = {
        "task": {"ensemble": True, "weighted_ensemble": True, "test_eval_skipped": split != "test"},
        "split": split,
        "members": [
            {"label": member_label, "path": str(path), "weight": float(weight)}
            for member_label, path, weight in zip(labels, paths, weights)
        ],
        "metrics": _metrics(out),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search validation-only ensemble weights.")
    parser.add_argument("--members", nargs="*", default=[])
    parser.add_argument("--include_run_globs", nargs="*", default=[])
    parser.add_argument("--include_runner_candidates", action="store_true")
    parser.add_argument("--runs_dir", default="runs/online_fms_current_tracking_0507")
    parser.add_argument("--all_members_once", action="store_true")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--min_size", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=6)
    parser.add_argument("--max_optimised_combos", type=int, default=100)
    parser.add_argument("--random_starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build_out_dir", default=None)
    parser.add_argument("--label", default="weighted_ensemble")
    parser.add_argument("--optimizer", choices=["slsqp", "linprog"], default="slsqp")
    parser.add_argument("--ordinal_bins", nargs="*", type=float, default=DEFAULT_ORDINAL_BINS)
    args = parser.parse_args()

    members = [_parse_member(value) for value in args.members]
    for pattern in args.include_run_globs:
        for path_value in sorted(glob.glob(pattern)):
            path = Path(path_value)
            if path.is_dir() and (path / f"{args.split}_predictions.csv").exists():
                members.append((path.name, path))
    if args.include_runner_candidates:
        members.extend(_runner_members(Path(args.runs_dir)))
    seen_labels = set()
    for label, _ in members:
        if label in seen_labels:
            raise ValueError(f"Duplicate ensemble member label: {label}")
        seen_labels.add(label)
    labels = [label for label, _ in members]
    paths = [path for _, path in members]
    if len(labels) < 2:
        raise ValueError("At least two members are required.")

    frames = [_load_prediction(label, path, args.split) for label, path in members]
    base = frames[0]
    for label, frame in zip(labels[1:], frames[1:]):
        _assert_aligned(base, frame, label)
    target = base["target_fms_now"].to_numpy(dtype=np.float64)
    prediction_matrix = np.vstack([frame[f"pred__{label}"].to_numpy(dtype=np.float64) for label, frame in zip(labels, frames)])

    candidates: List[Dict[str, object]] = []
    if args.all_members_once:
        combo = tuple(range(len(labels)))
        equal_weights = np.full(len(combo), 1.0 / len(combo), dtype=np.float64)
        candidates.append(
            {
                "combo": combo,
                "labels": list(labels),
                "equal_mae": _mae(equal_weights @ prediction_matrix, target),
                "equal_rmse": _rmse(equal_weights @ prediction_matrix, target),
            }
        )
        selected = candidates
    else:
        max_size = min(int(args.max_size), len(labels))
        for size in range(max(2, int(args.min_size)), max_size + 1):
            for combo in itertools.combinations(range(len(labels)), size):
                subset = prediction_matrix[list(combo)]
                equal_weights = np.full(size, 1.0 / size, dtype=np.float64)
                candidates.append(
                    {
                        "combo": combo,
                        "labels": [labels[i] for i in combo],
                        "equal_mae": _mae(equal_weights @ subset, target),
                        "equal_rmse": _rmse(equal_weights @ subset, target),
                    }
                )
        candidates.sort(key=lambda row: float(row["equal_mae"]))
        selected = candidates[: max(1, int(args.max_optimised_combos))]
    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None
    for candidate in selected:
        combo = tuple(int(i) for i in candidate["combo"])
        subset = prediction_matrix[list(combo)]
        size = len(combo)
        if args.optimizer == "linprog":
            weights, opt_mae = _optimise_weights_linprog(subset, target)
        else:
            starts = [np.full(size, 1.0 / size, dtype=np.float64)]
            starts.extend(np.eye(size, dtype=np.float64))
            starts.extend(rng.dirichlet(np.ones(size), size=max(0, int(args.random_starts))))
            weights, opt_mae = _optimise_weights(subset, target, starts)
        opt_pred = weights @ subset
        row = {
            "labels": ",".join(str(labels[i]) for i in combo),
            "weights": ",".join(f"{float(w):.8f}" for w in weights),
            "equal_mae": float(candidate["equal_mae"]),
            "equal_rmse": float(candidate["equal_rmse"]),
            "opt_mae": opt_mae,
            "opt_rmse": _rmse(opt_pred, target),
            "size": size,
        }
        rows.append(row)
        if best is None or float(row["opt_mae"]) < float(best["opt_mae"]):
            best = {**row, "combo": combo, "weights_array": weights}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["opt_mae", "opt_rmse"]).to_csv(out_dir / "validation_ensemble_weight_search.csv", index=False)
    assert best is not None
    best_payload = {
        "split": args.split,
        "labels": str(best["labels"]).split(","),
        "weights": [float(w) for w in best["weights_array"]],
        "opt_mae": float(best["opt_mae"]),
        "opt_rmse": float(best["opt_rmse"]),
        "equal_mae": float(best["equal_mae"]),
        "equal_rmse": float(best["equal_rmse"]),
    }
    with open(out_dir / "best_weighted_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    if args.build_out_dir:
        combo = tuple(int(i) for i in best["combo"])
        _write_ensemble(
            Path(args.build_out_dir),
            args.split,
            args.label,
            base,
            [labels[i] for i in combo],
            [paths[i] for i in combo],
            np.asarray(best["weights_array"], dtype=np.float64),
            prediction_matrix[list(combo)],
            args.ordinal_bins,
        )

    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
