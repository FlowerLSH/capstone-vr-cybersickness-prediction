"""Build merged member-prediction CSVs from online-current 5-fold CV runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd


RECIPES: Mapping[str, str] = {
    "selected_risk035": "member_pred_selected_risk035",
    "risk045_smooth005": "member_pred_risk045",
    "zero_anchor_highgate_delta2": "member_pred_zero_anchor",
    "range_scaled_delta2": "member_pred_range_scaled",
}

MERGE_KEYS = ["participant_id", "session_id", "current_index"]


def _run_dir(runs_dir: Path, fold: int, recipe: str) -> Path:
    return runs_dir / f"fold{fold:02d}_{recipe}_seed42"


def _prediction_path(runs_dir: Path, fold: int, recipe: str, split: str) -> Path:
    run_dir = _run_dir(runs_dir, fold, recipe)
    if split == "val":
        return run_dir / "val_predictions.csv"
    if split == "test":
        return run_dir / "eval_test" / "test_predictions.csv"
    raise ValueError("split must be val or test")


def _load_member_frame(path: Path, member_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in MERGE_KEYS + ["predicted_fms_now"] if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame.loc[:, MERGE_KEYS + ["predicted_fms_now"]].rename(columns={"predicted_fms_now": member_col})


def _build_split(args: argparse.Namespace, split: str) -> pd.DataFrame:
    runs_dir = Path(args.runs_dir)
    rows: List[pd.DataFrame] = []
    for fold in range(int(args.n_folds)):
        base_recipe = str(args.base_recipe)
        base_path = _prediction_path(runs_dir, fold, base_recipe, split)
        base = pd.read_csv(base_path)
        base["cv_fold"] = int(fold)
        base["cv_prediction_split"] = split
        base = base.rename(columns={"predicted_fms_now": RECIPES[base_recipe]})
        for recipe, member_col in RECIPES.items():
            if recipe == base_recipe:
                continue
            member_path = _prediction_path(runs_dir, fold, recipe, split)
            member = _load_member_frame(member_path, member_col)
            base = base.merge(member, on=MERGE_KEYS, how="inner", validate="one_to_one")
        base["predicted_fms_now"] = base[RECIPES[base_recipe]]
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", default="runs/online_current_5fold_cv_0514")
    parser.add_argument("--out_dir", default="reports/online_current_5fold_cv_0514/member_ensemble_inputs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--base_recipe", choices=sorted(RECIPES), default="selected_risk035")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {"runs_dir": str(args.runs_dir), "n_folds": int(args.n_folds), "recipes": dict(RECIPES)}
    for split in ["val", "test"]:
        frame = _build_split(args, split)
        out_path = out_dir / f"cv_{split}_member_predictions.csv"
        frame.to_csv(out_path, index=False)
        manifest[f"{split}_rows"] = int(len(frame))
        manifest[f"{split}_path"] = str(out_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
