"""Data inspection, loading, splitting, and padding for DenseFMS.

The training pipeline deliberately separates model inputs from shifted targets:
calibration FMS is sliced once from the first C steps, and post-calibration FMS is
kept only in the batch target tensor used by the loss.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import check_disjoint, seconds_to_steps, save_json


CANONICAL_HEADERLESS_COLUMNS = [
    "timestamp",
    "fms",
    "acc_x",
    "acc_y",
    "acc_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
    "gender",
    "mssq",
    "age",
]

PARTICIPANT_PATTERNS = ("participant", "subject", "user", "pid", "sid", "id")
SESSION_PATTERNS = ("session", "trial", "experiment", "condition", "run")
AGE_PATTERNS = ("age", "participant_age", "subject_age", "user_age")
GENDER_PATTERNS = ("gender", "sex", "participant_gender", "subject_gender", "user_gender")
MSSQ_PATTERNS = (
    "mssq",
    "mssq_score",
    "mssq_total",
    "motion_sickness_susceptibility",
    "motion_sickness_susceptibility_score",
    "susceptibility",
    "susceptibility_score",
)
GENDER_CATEGORY_ORDER = ["male", "female", "unknown"]
GENDER_BINARY_ORDER = ["male", "female"]
SUPPORTED_GENDER_ENCODINGS = ("category3", "binary2")
FMS_SCALE_MIN = 0.0
FMS_SCALE_MAX = 20.0
SCENARIO_CATEGORY_ORDER = [
    "base",
    "reverse_optical_flow_general",
    "reverse_optical_flow_backward_texture",
    "reverse_optical_flow_forward_texture",
    "reverse_optical_flow_high_density",
    "reverse_optical_flow_low_density",
    "rof_forward_whiteline",
    "rof_original",
    "unknown",
]
SUPPORTED_STATIC_FEATURES = ("age", "gender", "mssq", "scenario")
HEAD_IMPUTATION_NEUTRAL_VALUE = 0.0
SUPPORTED_HEAD_CHANNEL_MODES = ("all", "linear_only", "angular_only")
LINEAR_HEAD_CHANNEL_SLICE = slice(0, 3)
ANGULAR_HEAD_CHANNEL_SLICE = slice(3, 6)


@dataclass
class DenseFMSSession:
    head: np.ndarray
    fms: np.ndarray
    time: np.ndarray
    participant_id: Optional[str]
    session_id: str
    source_file: str
    fms_raw: Optional[np.ndarray] = None
    head_raw: Optional[np.ndarray] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    mssq: Optional[float] = None
    static: Optional[np.ndarray] = None
    static_feature_names: Optional[List[str]] = None
    head_missing_mask: Optional[np.ndarray] = None
    head_imputation_report: Optional[Dict[str, Any]] = None
    original_length: Optional[int] = None
    max_session_points: Optional[int] = None
    calibration_residual_features: Optional[np.ndarray] = None
    calibration_residual_feature_names: Optional[List[str]] = None

    @property
    def length(self) -> int:
        return int(self.head.shape[0])


def session_identity_keys(session: DenseFMSSession) -> List[str]:
    """Return stable keys used to join per-session auxiliary artifacts."""
    keys: List[str] = []
    if session.source_file:
        source = str(session.source_file)
        keys.extend([source, str(Path(source)), Path(source).name, Path(source).stem])
    if session.session_id:
        keys.append(str(session.session_id))
    if session.participant_id and session.session_id:
        keys.append(f"{session.participant_id}::{session.session_id}")
    out: List[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def normalize_head_channel_mode(head_channel_mode: Optional[str] = None) -> str:
    mode = str(head_channel_mode or "all").strip().lower().replace("-", "_")
    aliases = {
        "full": "all",
        "both": "all",
        "linear": "linear_only",
        "acc": "linear_only",
        "acc_only": "linear_only",
        "acceleration_only": "linear_only",
        "angular": "angular_only",
        "gyro": "angular_only",
        "gyro_only": "angular_only",
        "angular_velocity_only": "angular_only",
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_HEAD_CHANNEL_MODES:
        raise ValueError(
            f"Unsupported head_channel_mode '{head_channel_mode}'. "
            f"Expected one of {list(SUPPORTED_HEAD_CHANNEL_MODES)}."
        )
    return mode


def apply_head_channel_mode(head: np.ndarray, head_channel_mode: Optional[str] = None) -> np.ndarray:
    """Mask excluded 6D head-motion channels while preserving input shape."""
    mode = normalize_head_channel_mode(head_channel_mode)
    out = np.asarray(head, dtype=np.float32).copy()
    if mode == "all":
        return out
    if out.ndim != 2 or out.shape[1] != 6:
        raise ValueError(f"head_channel_mode requires head shape [T, 6], got {out.shape}.")
    if mode == "linear_only":
        out[:, ANGULAR_HEAD_CHANNEL_SLICE] = 0.0
    elif mode == "angular_only":
        out[:, LINEAR_HEAD_CHANNEL_SLICE] = 0.0
    return out


def load_calibration_residual_features(
    paths: str | Path | Sequence[str | Path],
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """Load one or more calibration-residual feature artifacts.

    The expected JSON payload has a top-level ``feature_names`` list and a
    ``features`` list whose rows contain session identifiers plus ``features``.
    CSV files are also accepted when they include the feature-name columns.
    """
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)
    if not path_list:
        raise ValueError("At least one calibration residual feature path is required.")

    feature_map: Dict[str, np.ndarray] = {}
    feature_names: Optional[List[str]] = None
    sources: List[str] = []

    for raw_path in path_list:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration residual feature artifact not found: {path}")
        sources.append(str(path))
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            names = [c for c in frame.columns if str(c).startswith("feature__")]
            if not names:
                reserved = {
                    "session_key",
                    "participant_id",
                    "session_id",
                    "source_file",
                    "source_file_name",
                    "source_file_stem",
                    "split",
                }
                names = [c for c in frame.columns if c not in reserved]
            rows = frame.to_dict(orient="records")
        else:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            names = [str(v) for v in payload.get("feature_names", [])]
            rows = list(payload.get("features", []))
        if not names:
            raise ValueError(f"Calibration residual feature artifact {path} has no feature names.")
        if feature_names is None:
            feature_names = list(names)
        elif list(names) != feature_names:
            raise ValueError(
                "Calibration residual feature artifacts must use the same feature_names order. "
                f"Expected {feature_names}, got {list(names)} from {path}."
            )
        assert feature_names is not None
        for row in rows:
            if "features" in row:
                vector = np.asarray(row["features"], dtype=np.float32)
            else:
                vector = np.asarray([row.get(name, row.get(f"feature__{name}", 0.0)) for name in feature_names], dtype=np.float32)
            if vector.shape != (len(feature_names),):
                raise ValueError(f"Feature row in {path} has shape {vector.shape}, expected {(len(feature_names),)}.")
            row_keys = [
                row.get("session_key"),
                row.get("source_file"),
                row.get("source_file_name"),
                row.get("source_file_stem"),
                row.get("session_id"),
            ]
            if row.get("participant_id") and row.get("session_id"):
                row_keys.append(f"{row.get('participant_id')}::{row.get('session_id')}")
            for key in row_keys:
                if key is not None and str(key):
                    feature_map[str(key)] = vector
    assert feature_names is not None
    metadata = {"sources": sources, "feature_count": len(feature_names), "session_key_count": len(feature_map)}
    return feature_map, feature_names, metadata


def find_csv_files(data_dir: str | Path) -> List[Path]:
    files = sorted(Path(data_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    return files


def _looks_like_header(values: Sequence[Any]) -> bool:
    tokens = [str(v).strip().lower() for v in values]
    header_hits = 0
    for token in tokens:
        compact = re.sub(r"[^a-z0-9]+", "_", token)
        if any(key in compact for key in ("time", "timestamp", "fms", "acc", "angular", "velocity", "gyro")):
            header_hits += 1
    alpha_long = 0
    for token in tokens:
        if re.search(r"[a-zA-Z]", token) and token not in {"m", "f", "male", "female"}:
            alpha_long += 1
    return header_hits >= 1 or alpha_long >= 2


def read_csv_robust(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a DenseFMS CSV with or without a header row."""
    path = Path(path)
    preview = pd.read_csv(path, header=None, nrows=3)
    has_header = bool(len(preview) and _looks_like_header(preview.iloc[0].tolist()))
    if has_header:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None)
        names = CANONICAL_HEADERLESS_COLUMNS[: df.shape[1]]
        if df.shape[1] > len(names):
            names.extend([f"extra_{idx}" for idx in range(len(names), df.shape[1])])
        df.columns = names
    meta = {
        "source_file": str(path),
        "has_header": has_header,
        "columns": list(df.columns),
        "rows": int(len(df)),
    }
    return df, meta


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _axis_match(normalized: str, axis: str) -> bool:
    parts = [p for p in normalized.split("_") if p]
    return axis in parts or normalized.endswith(axis) or normalized.startswith(axis)


def _score_candidate(name: str, preferred_terms: Sequence[str], axis: Optional[str] = None) -> Tuple[int, int]:
    n = _norm(name)
    score = 0
    for idx, term in enumerate(preferred_terms):
        if term in n:
            score += 20 - idx
    if axis and _axis_match(n, axis):
        score += 30
    if n in {"timestamp", "time", "fms", f"acc_{axis}", f"angular_velocity_{axis}"}:
        score += 50
    return (-score, len(n))


def _choose_candidate(
    columns: Sequence[str],
    predicate,
    label: str,
    preferred_terms: Sequence[str],
    axis: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    candidates = [c for c in columns if predicate(_norm(c))]
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda c: _score_candidate(c, preferred_terms, axis))
    return ranked[0], ranked


def infer_static_columns(columns: Sequence[str]) -> Dict[str, Any]:
    columns = [str(c) for c in columns]
    age_col, age_candidates = _choose_candidate(
        columns,
        lambda n: n in AGE_PATTERNS,
        "age",
        AGE_PATTERNS,
    )
    gender_col, gender_candidates = _choose_candidate(
        columns,
        lambda n: n in GENDER_PATTERNS,
        "gender",
        GENDER_PATTERNS,
    )
    mssq_col, mssq_candidates = _choose_candidate(
        columns,
        lambda n: n in MSSQ_PATTERNS,
        "mssq",
        MSSQ_PATTERNS,
    )
    return {
        "age_column": age_col,
        "gender_column": gender_col,
        "mssq_column": mssq_col,
        "age_candidates": age_candidates,
        "gender_candidates": gender_candidates,
        "mssq_candidates": mssq_candidates,
        "use_static_available": bool(age_col and gender_col),
        "full_static_available": bool(age_col and gender_col and mssq_col),
    }


def normalize_static_features(static_features: Optional[Sequence[str]]) -> List[str]:
    if static_features is None:
        return ["age", "gender"]
    normalized: List[str] = []
    for item in static_features:
        token = str(item).strip().lower()
        if token == "sex":
            token = "gender"
        if token not in SUPPORTED_STATIC_FEATURES:
            raise ValueError(
                f"Unsupported static feature '{item}'. "
                f"Expected a subset of {list(SUPPORTED_STATIC_FEATURES)}."
            )
        if token not in normalized:
            normalized.append(token)
    if not normalized:
        raise ValueError("static_features must contain at least one supported feature when static is enabled.")
    return normalized


def normalize_gender_encoding(gender_encoding: Optional[str] = None) -> str:
    token = str(gender_encoding or "category3").strip().lower().replace("-", "_")
    aliases = {
        "category": "category3",
        "categorical": "category3",
        "categorical3": "category3",
        "category_3": "category3",
        "3d": "category3",
        "three": "category3",
        "onehot3": "category3",
        "one_hot_3": "category3",
        "one_hot_3d": "category3",
        "binary": "binary2",
        "binary_2": "binary2",
        "2d": "binary2",
        "two": "binary2",
        "onehot2": "binary2",
        "one_hot_2": "binary2",
        "one_hot_2d": "binary2",
    }
    token = aliases.get(token, token)
    if token not in SUPPORTED_GENDER_ENCODINGS:
        raise ValueError(f"Unsupported gender_encoding '{gender_encoding}'. Expected one of {list(SUPPORTED_GENDER_ENCODINGS)}.")
    return token


def gender_order_for_encoding(gender_encoding: Optional[str] = None) -> List[str]:
    encoding = normalize_gender_encoding(gender_encoding)
    return list(GENDER_BINARY_ORDER if encoding == "binary2" else GENDER_CATEGORY_ORDER)


def parse_scenario_from_filename(path_or_name: str | Path) -> str:
    """Map DenseFMS filename tokens to a fixed deployment-visible scenario category."""
    stem = Path(path_or_name).stem.lower()
    stem = re.sub(r"^pa\d+_?", "", stem)
    stem = re.sub(r"_\d{1,2}_\d{2}_\d{2}_(am|pm)$", "", stem)
    stem = stem.replace("foward", "forward")
    stem = stem.replace("rof", "reverse_optical_flow")
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if "base" in stem:
        return "base"
    if "white_line" in stem or "whiteline" in stem:
        return "rof_forward_whiteline"
    if "original" in stem:
        return "rof_original"
    if "backward_texture" in stem:
        return "reverse_optical_flow_backward_texture"
    if "forward_texture" in stem:
        return "reverse_optical_flow_forward_texture"
    if "high_density" in stem:
        return "reverse_optical_flow_high_density"
    if "low_density" in stem:
        return "reverse_optical_flow_low_density"
    if "reverse_optical_flow" in stem:
        return "reverse_optical_flow_general"
    return "unknown"


def static_feature_names(static_features: Optional[Sequence[str]], gender_encoding: Optional[str] = None) -> List[str]:
    features = normalize_static_features(static_features)
    names: List[str] = []
    if "age" in features:
        names.append("age_z")
    if "mssq" in features:
        names.append("mssq_z")
    if "gender" in features:
        names.extend([f"gender_{cat}" for cat in gender_order_for_encoding(gender_encoding)])
    if "scenario" in features:
        names.extend([f"scenario_{cat}" for cat in SCENARIO_CATEGORY_ORDER])
    return names


def static_feature_dim(static_features: Optional[Sequence[str]], gender_encoding: Optional[str] = None) -> int:
    return len(static_feature_names(static_features, gender_encoding=gender_encoding))


def infer_column_mapping(columns: Sequence[str]) -> Dict[str, Any]:
    """Infer time, FMS, head-motion, participant, and session columns."""
    columns = [str(c) for c in columns]
    mapping: Dict[str, Any] = {"all_columns": columns, "candidates": {}, "warnings": []}

    fms, fms_candidates = _choose_candidate(
        columns,
        lambda n: "fms" in n,
        "fms",
        ("fms", "score"),
    )
    time_col, time_candidates = _choose_candidate(
        columns,
        lambda n: any(term in n for term in ("timestamp", "time", "elapsed", "second", "sec")),
        "time",
        ("timestamp", "time", "elapsed", "seconds"),
    )
    mapping["fms"] = fms
    mapping["time"] = time_col
    mapping["candidates"]["fms"] = fms_candidates
    mapping["candidates"]["time"] = time_candidates

    head_features: List[str] = []
    missing: List[str] = []
    for prefix, predicate, terms in [
        (
            "acc",
            lambda n: any(term in n for term in ("acceleration", "accelerometer", "accel", "acc")),
            ("acceleration", "accelerometer", "accel", "acc"),
        ),
        (
            "angular_velocity",
            lambda n: (
                any(term in n for term in ("angular", "rotation", "rot", "gyro"))
                and any(term in n for term in ("velocity", "vel", "rate", "gyro"))
            ),
            ("angular_velocity", "angular", "gyro", "rotation", "rot", "velocity", "vel"),
        ),
    ]:
        for axis in ("x", "y", "z"):
            selected, candidates = _choose_candidate(
                columns,
                lambda n, pred=predicate, ax=axis: pred(n) and _axis_match(n, ax),
                f"{prefix}_{axis}",
                terms,
                axis=axis,
            )
            key = f"{prefix}_{axis}"
            mapping["candidates"][key] = candidates
            if selected is None:
                missing.append(key)
            else:
                head_features.append(selected)

    mapping["head_features"] = head_features

    participant, participant_candidates = _choose_candidate(
        columns,
        lambda n: n in PARTICIPANT_PATTERNS or any(n == p or n.endswith(f"_{p}") for p in PARTICIPANT_PATTERNS),
        "participant",
        PARTICIPANT_PATTERNS,
    )
    session, session_candidates = _choose_candidate(
        columns,
        lambda n: n in SESSION_PATTERNS or any(p in n for p in SESSION_PATTERNS),
        "session",
        SESSION_PATTERNS,
    )
    mapping["participant"] = participant
    mapping["session"] = session
    mapping["candidates"]["participant"] = participant_candidates
    mapping["candidates"]["session"] = session_candidates

    static_cols = infer_static_columns(columns)
    age_col = static_cols["age_column"]
    gender_col = static_cols["gender_column"]
    mssq_col = static_cols["mssq_column"]
    mapping["static"] = {
        "age_column": age_col,
        "gender_column": gender_col,
        "mssq_column": mssq_col,
        "source_file_or_location": "session_csv" if age_col or gender_col or mssq_col else None,
        "mssq_source_file_or_location": "session_csv" if mssq_col else None,
        "gender_categories_observed": [],
        "mssq_available_count": 0,
        "mssq_missing_count": 0,
        "static_features_supported": list(SUPPORTED_STATIC_FEATURES),
        "use_static_available": bool(age_col and gender_col),
        "full_static_available": bool(age_col and gender_col and mssq_col),
    }
    mapping["candidates"]["age"] = static_cols["age_candidates"]
    mapping["candidates"]["gender"] = static_cols["gender_candidates"]
    mapping["candidates"]["mssq"] = static_cols["mssq_candidates"]

    if fms is None:
        missing.append("fms")
    if missing:
        raise ValueError(
            "Could not infer required DenseFMS columns. Missing: "
            + ", ".join(missing)
            + f". Available columns: {columns}"
        )
    if len(head_features) != 6:
        raise ValueError(
            f"Expected 6 head-motion features, inferred {len(head_features)} from columns {columns}"
        )
    if time_col is None:
        mapping["warnings"].append("No time column inferred; row index with 0.5s interval will be used.")
    return mapping


def normalize_gender(value: Any, allow_missing: bool = True) -> Optional[str]:
    if pd.isna(value):
        return "unknown" if allow_missing else None
    token = str(value).strip().lower()
    if token in {"m", "male", "man", "boy"}:
        return "male"
    if token in {"f", "female", "woman", "girl"}:
        return "female"
    return "unknown" if allow_missing else None


def _first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def inspect_dataset(data_dir: str | Path, artifacts_dir: str | Path = "artifacts") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    files = find_csv_files(data_dir)
    file_reports: List[Dict[str, Any]] = []
    unique_schemas: Dict[str, int] = {}
    first_columns: Optional[List[str]] = None
    static_source: Optional[str] = None
    static_age_col: Optional[str] = None
    static_gender_col: Optional[str] = None
    static_mssq_col: Optional[str] = None
    static_mssq_source: Optional[str] = None
    mssq_available_count = 0
    mssq_missing_count = 0
    observed_genders: set[str] = set()

    for path in files:
        df, meta = read_csv_robust(path)
        if first_columns is None:
            first_columns = list(df.columns)
        schema_key = "|".join(map(str, df.columns))
        unique_schemas[schema_key] = unique_schemas.get(schema_key, 0) + 1
        missing = df.isna().sum().to_dict()
        file_reports.append(
            {
                **meta,
                "missing_values": {str(k): int(v) for k, v in missing.items()},
            }
        )
        static_mapping = infer_static_columns(df.columns)
        age_col = static_mapping.get("age_column")
        gender_col = static_mapping.get("gender_column")
        mssq_col = static_mapping.get("mssq_column")
        if (age_col or gender_col or mssq_col) and static_source is None:
            static_source = str(path)
            static_age_col = age_col
            static_gender_col = gender_col
            static_mssq_col = mssq_col
        if mssq_col and static_mssq_source is None:
            static_mssq_source = str(path)
        if gender_col and gender_col in df.columns:
            values = df[gender_col].dropna().head(20).tolist()
            for value in values:
                gender = normalize_gender(value, allow_missing=True)
                if gender:
                    observed_genders.add(gender)
        if mssq_col and mssq_col in df.columns:
            mssq_value = _first_non_null(pd.to_numeric(df[mssq_col], errors="coerce"))
            if mssq_value is not None and np.isfinite(mssq_value):
                mssq_available_count += 1
            else:
                mssq_missing_count += 1

    if first_columns is None:
        raise RuntimeError("No CSV files were readable.")
    mapping = infer_column_mapping(first_columns)
    age_final = static_age_col or mapping.get("static", {}).get("age_column")
    gender_final = static_gender_col or mapping.get("static", {}).get("gender_column")
    mssq_final = static_mssq_col or mapping.get("static", {}).get("mssq_column")
    mapping["static"] = {
        "age_column": age_final,
        "gender_column": gender_final,
        "mssq_column": mssq_final,
        "source_file_or_location": static_source or mapping.get("static", {}).get("source_file_or_location"),
        "mssq_source_file_or_location": static_mssq_source or mapping.get("static", {}).get("mssq_source_file_or_location"),
        "gender_categories_observed": sorted(observed_genders),
        "mssq_available_count": int(mssq_available_count),
        "mssq_missing_count": int(mssq_missing_count),
        "static_features_supported": list(SUPPORTED_STATIC_FEATURES),
        "use_static_available": bool(age_final and gender_final),
        "full_static_available": bool(age_final and gender_final and mssq_final and mssq_missing_count == 0),
    }
    report = {
        "data_dir": str(Path(data_dir)),
        "file_count": len(files),
        "total_rows": int(sum(fr["rows"] for fr in file_reports)),
        "unique_schema_count": len(unique_schemas),
        "schema_counts": [{"columns": key.split("|"), "file_count": count} for key, count in unique_schemas.items()],
        "files": file_reports,
        "inferred_mapping": mapping,
    }
    artifacts = Path(artifacts_dir)
    save_json(artifacts / "data_report.json", report)
    save_json(artifacts / "column_mapping.json", mapping)
    return report, mapping


def infer_sampling_interval(sessions: Sequence[DenseFMSSession], default_interval: float = 0.5) -> float:
    diffs: List[float] = []
    for sess in sessions:
        if len(sess.time) > 2:
            d = np.diff(sess.time.astype(np.float64))
            d = d[np.isfinite(d) & (d > 0)]
            if len(d):
                diffs.extend(d.tolist())
    if not diffs:
        return float(default_interval)
    median = float(np.median(diffs))
    # DenseFMS timestamps jitter around 0.5 seconds; preserve the task's 60/20/10
    # step definitions unless a dataset clearly uses a different cadence.
    if abs(median - default_interval) <= 0.05:
        return float(default_interval)
    return median


def parse_participant_from_filename(path: str | Path) -> Optional[str]:
    match = re.search(r"(PA\d+)", Path(path).stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _series_to_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def causal_fill_head_motion(
    head_df: pd.DataFrame,
    neutral_value: float = HEAD_IMPUTATION_NEUTRAL_VALUE,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Fill head-motion NaNs without using later samples to repair earlier rows."""
    numeric = head_df.apply(pd.to_numeric, errors="coerce")
    missing_before = numeric.isna()
    carried = numeric.ffill()
    leading_missing = carried.isna()
    filled_by_ffill = missing_before & ~leading_missing
    filled = carried.fillna(float(neutral_value))
    missing_after = filled.isna()

    report = {
        "strategy": "causal_forward_fill_then_fixed_neutral",
        "neutral_fill_value": float(neutral_value),
        "missing_head_values_before_fill": int(missing_before.to_numpy().sum()),
        "values_filled_by_ffill": int(filled_by_ffill.to_numpy().sum()),
        "leading_values_filled_by_neutral": int(leading_missing.to_numpy().sum()),
        "missing_head_values_after_fill": int(missing_after.to_numpy().sum()),
        "missing_mask_channels_added": 0,
        "per_feature_missing_before": {str(col): int(missing_before[col].sum()) for col in numeric.columns},
        "per_feature_ffill": {str(col): int(filled_by_ffill[col].sum()) for col in numeric.columns},
        "per_feature_neutral": {str(col): int(leading_missing[col].sum()) for col in numeric.columns},
        "per_feature_missing_after": {str(col): int(missing_after[col].sum()) for col in numeric.columns},
    }
    if report["missing_head_values_after_fill"]:
        raise ValueError("Head-motion imputation left missing values after causal fill.")
    return filled.astype(np.float32), missing_before.to_numpy(dtype=np.float32), report


def session_from_csv(
    path: str | Path,
    mapping: Mapping[str, Any],
    default_sampling_interval: float = 0.5,
    max_session_points: Optional[int] = None,
) -> DenseFMSSession:
    """Load and preprocess one CSV as a session without fitting scalers."""
    path = Path(path)
    df, _ = read_csv_robust(path)
    if max_session_points is not None and int(max_session_points) <= 0:
        raise ValueError(f"max_session_points must be positive when provided, got {max_session_points}.")
    missing_cols = [c for c in [mapping.get("fms"), *(mapping.get("head_features") or [])] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing mapped columns {missing_cols}; available columns: {list(df.columns)}")
    time_col = mapping.get("time")
    if time_col and time_col in df.columns:
        time = _series_to_numeric(df, time_col).to_numpy(dtype=np.float64)
        order = np.argsort(np.where(np.isfinite(time), time, np.inf))
        df = df.iloc[order].reset_index(drop=True)
        time = time[order]
    else:
        time = np.arange(len(df), dtype=np.float64) * default_sampling_interval
    original_length = int(len(df))
    if max_session_points is not None and original_length > int(max_session_points):
        keep = int(max_session_points)
        df = df.iloc[:keep].reset_index(drop=True)
        time = time[:keep]

    fms = _series_to_numeric(df, str(mapping["fms"])).to_numpy(dtype=np.float32)
    head_cols = [str(c) for c in mapping["head_features"]]
    head_df = df[head_cols].apply(pd.to_numeric, errors="coerce")
    head_df, head_missing_mask, head_imputation_report = causal_fill_head_motion(head_df)
    if head_df.isna().any().any():
        raise ValueError(f"{path} has head-motion missing values that could not be filled.")
    participant_col = mapping.get("participant")
    if participant_col and participant_col in df.columns and df[participant_col].notna().any():
        participant_id = str(df[participant_col].dropna().iloc[0])
    else:
        participant_id = parse_participant_from_filename(path)
    session_col = mapping.get("session")
    if session_col and session_col in df.columns and df[session_col].notna().any():
        session_id = f"{participant_id or 'unknown'}::{df[session_col].dropna().iloc[0]}::{path.stem}"
    else:
        session_id = path.stem
    static_cfg = mapping.get("static", {})
    age = None
    age_col = static_cfg.get("age_column")
    if age_col and age_col in df.columns:
        age_value = _first_non_null(pd.to_numeric(df[age_col], errors="coerce"))
        age = float(age_value) if age_value is not None and np.isfinite(age_value) else None
    gender = None
    gender_col = static_cfg.get("gender_column")
    if gender_col and gender_col in df.columns:
        gender = normalize_gender(_first_non_null(df[gender_col]), allow_missing=True)
    mssq = None
    mssq_col = static_cfg.get("mssq_column")
    if mssq_col and mssq_col in df.columns:
        mssq_value = _first_non_null(pd.to_numeric(df[mssq_col], errors="coerce"))
        mssq = float(mssq_value) if mssq_value is not None and np.isfinite(mssq_value) else None
    head = head_df.to_numpy(dtype=np.float32)
    return DenseFMSSession(
        head=head,
        fms=fms,
        time=time.astype(np.float32),
        participant_id=participant_id,
        session_id=session_id,
        source_file=str(path),
        head_raw=head.copy(),
        fms_raw=fms.copy(),
        age=age,
        gender=gender,
        mssq=mssq,
        head_missing_mask=head_missing_mask,
        head_imputation_report=head_imputation_report,
        original_length=original_length,
        max_session_points=int(max_session_points) if max_session_points is not None else None,
    )


def load_raw_sessions(
    data_dir: str | Path,
    mapping: Optional[Mapping[str, Any]] = None,
    calibration_seconds: float = 30.0,
    horizon_seconds: float = 5.0,
    default_sampling_interval: float = 0.5,
    max_session_points: Optional[int] = None,
) -> Tuple[List[DenseFMSSession], Dict[str, Any], Dict[str, Any]]:
    files = find_csv_files(data_dir)
    if mapping is None:
        first_df, _ = read_csv_robust(files[0])
        mapping = infer_column_mapping(first_df.columns)
    else:
        mapping = dict(mapping)

    rough_sessions: List[DenseFMSSession] = []
    dropped: List[Dict[str, Any]] = []
    for path in files:
        try:
            rough_sessions.append(
                session_from_csv(
                    path,
                    mapping,
                    default_sampling_interval,
                    max_session_points=max_session_points,
                )
            )
        except ValueError as exc:
            dropped.append({"source_file": str(path), "reason": str(exc)})

    sampling_interval = infer_sampling_interval(rough_sessions, default_sampling_interval)
    calibration_steps = seconds_to_steps(
        calibration_seconds,
        sampling_interval,
        name="calibration_seconds",
        allow_zero=True,
    )
    horizon_steps = seconds_to_steps(horizon_seconds, sampling_interval, name="horizon_seconds")
    min_len = calibration_steps + horizon_steps + 1

    sessions: List[DenseFMSSession] = []
    for sess in rough_sessions:
        if sess.length < min_len:
            dropped.append({"source_file": sess.source_file, "reason": f"too short: {sess.length} < {min_len}"})
            continue
        if np.isnan(sess.fms[:calibration_steps]).any():
            dropped.append({"source_file": sess.source_file, "reason": "missing FMS in calibration period"})
            continue
        sessions.append(sess)

    if not sessions:
        raise RuntimeError(
            "No usable sessions remained after preprocessing. "
            f"Dropped examples: {dropped[:5]}"
        )

    participant_source = "column" if mapping.get("participant") else "filename_regex"
    if not any(sess.participant_id for sess in sessions):
        participant_source = "unavailable"
    info = {
        "sampling_interval": sampling_interval,
        "calibration_steps": calibration_steps,
        "horizon_steps": horizon_steps,
        "dropped_sessions": dropped,
        "participant_source": participant_source,
        "session_count": len(sessions),
        "participant_count": len({s.participant_id for s in sessions if s.participant_id is not None}),
        "max_session_points": int(max_session_points) if max_session_points is not None else None,
        "truncated_session_count": int(
            sum(
                1
                for s in sessions
                if s.original_length is not None and s.original_length > s.length
            )
        ),
        "max_original_session_length": int(max((s.original_length or s.length) for s in sessions)),
        "max_loaded_session_length": int(max(s.length for s in sessions)),
    }
    return sessions, dict(mapping), info


def split_sessions(
    sessions: Sequence[DenseFMSSession],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Dict[str, Any]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    participant_ids = [s.participant_id for s in sessions]
    use_participants = all(pid is not None for pid in participant_ids) and len(set(participant_ids)) >= 3
    group_key = "participant_id" if use_participants else "session_id"
    if not use_participants:
        print("WARNING: participant_id unavailable or too sparse; using session/file-level split.")
    group_to_sessions: Dict[str, List[DenseFMSSession]] = {}
    for sess in sessions:
        group = str(sess.participant_id if use_participants else sess.session_id)
        group_to_sessions.setdefault(group, []).append(sess)

    rng = np.random.default_rng(seed)
    groups = np.array(sorted(group_to_sessions))
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = max(1, n - 2)
        n_val = 1
    train_groups = set(groups[:n_train].tolist())
    val_groups = set(groups[n_train : n_train + n_val].tolist())
    test_groups = set(groups[n_train + n_val :].tolist())
    if not test_groups and n >= 2:
        moved = sorted(train_groups)[-1]
        train_groups.remove(moved)
        test_groups.add(moved)

    split = {"train": [], "val": [], "test": []}
    for group, group_sessions in group_to_sessions.items():
        bucket = "train" if group in train_groups else "val" if group in val_groups else "test"
        split[bucket].extend(group_sessions)

    check_disjoint("train", train_groups, "val", val_groups)
    check_disjoint("train", train_groups, "test", test_groups)
    check_disjoint("val", val_groups, "test", test_groups)
    return {
        "sessions": split,
        "group_key": group_key,
        "groups": {
            "train": sorted(train_groups),
            "val": sorted(val_groups),
            "test": sorted(test_groups),
        },
        "counts": {k: len(v) for k, v in split.items()},
    }


def make_group_kfold_splits(
    sessions: Sequence[DenseFMSSession],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Return optional participant/session GroupKFold split descriptors."""
    from sklearn.model_selection import GroupKFold

    participant_ids = [s.participant_id for s in sessions]
    use_participants = all(pid is not None for pid in participant_ids) and len(set(participant_ids)) >= n_splits
    group_key = "participant_id" if use_participants else "session_id"
    groups = np.asarray([str(s.participant_id if use_participants else s.session_id) for s in sessions])
    if len(set(groups.tolist())) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups for GroupKFold, got {len(set(groups.tolist()))}.")
    indices = np.arange(len(sessions))
    # Shuffle session order deterministically before GroupKFold to avoid filename-order bias.
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    folds = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_order_idx, test_order_idx) in enumerate(splitter.split(order, groups=groups[order])):
        train_idx = order[train_order_idx]
        test_idx = order[test_order_idx]
        train_groups = sorted(set(groups[train_idx].tolist()))
        test_groups = sorted(set(groups[test_idx].tolist()))
        check_disjoint("train", train_groups, "test", test_groups)
        folds.append(
            {
                "fold": fold_idx,
                "group_key": group_key,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_groups": train_groups,
                "test_groups": test_groups,
            }
        )
    return folds


def apply_saved_split(sessions: Sequence[DenseFMSSession], split_info: Mapping[str, Any]) -> Dict[str, List[DenseFMSSession]]:
    group_key = split_info.get("group_key", "participant_id")
    groups = split_info["groups"]
    out = {"train": [], "val": [], "test": []}
    for sess in sessions:
        group = str(sess.participant_id if group_key == "participant_id" else sess.session_id)
        for split_name in out:
            if group in groups.get(split_name, []):
                out[split_name].append(sess)
                break
    return out


def fit_scalers(train_sessions: Sequence[DenseFMSSession], calibration_steps: int, horizon_steps: int) -> Dict[str, Any]:
    heads = np.concatenate([s.head for s in train_sessions], axis=0).astype(np.float64)
    fms = np.concatenate([s.fms for s in train_sessions], axis=0).astype(np.float64)
    head_mean = np.nanmean(heads, axis=0)
    head_std = np.nanstd(heads, axis=0)
    head_std = np.where(head_std < 1e-8, 1.0, head_std)
    observed_f_min = float(np.nanmin(fms))
    observed_f_max = float(np.nanmax(fms))
    # DenseFMS FMS is defined on an absolute 0-20 scale. Use that fixed scale
    # for normalization instead of split-specific observed min/max, so runs are
    # comparable even when a split does not contain the full label range.
    f_min = FMS_SCALE_MIN
    f_max = FMS_SCALE_MAX
    diffs = []
    for sess in train_sessions:
        y = (sess.fms.astype(np.float64) - f_min) / (f_max - f_min)
        for t in range(calibration_steps, max(calibration_steps, sess.length - horizon_steps)):
            if np.isfinite(y[t]) and np.isfinite(y[t + horizon_steps]):
                diffs.append(abs(float(y[t + horizon_steps] - y[t])))
    delta_max = float(np.percentile(diffs, 95)) if diffs else 0.1
    delta_max = max(delta_max, 1e-3)
    return {
        "head": {"mean": head_mean.astype(float).tolist(), "std": head_std.astype(float).tolist()},
        "fms": {
            "min": f_min,
            "max": f_max,
            "scale_source": "fixed_densefms_0_20",
            "observed_train_min": observed_f_min,
            "observed_train_max": observed_f_max,
        },
        "delta_max": delta_max,
    }


def fit_static_scaler(
    train_sessions: Sequence[DenseFMSSession],
    static_features: Optional[Sequence[str]] = None,
    allow_missing_static: bool = False,
    gender_encoding: Optional[str] = None,
) -> Dict[str, Any]:
    features = normalize_static_features(static_features)
    gender_encoding = normalize_gender_encoding(gender_encoding)
    gender_order = gender_order_for_encoding(gender_encoding)
    names = static_feature_names(features, gender_encoding=gender_encoding)
    ages = [float(s.age) for s in train_sessions if s.age is not None and np.isfinite(s.age)]
    if "age" in features and not ages:
        if not allow_missing_static:
            raise ValueError("Static features requested, but no valid age values were found in the training split.")
        ages = [0.0]
    age_mean = float(np.mean(ages)) if ages else 0.0
    age_std = float(np.std(ages)) if ages else 1.0
    if age_std < 1e-8:
        age_std = 1.0
    mssq_values = [float(s.mssq) for s in train_sessions if s.mssq is not None and np.isfinite(s.mssq)]
    if "mssq" in features and not mssq_values:
        if not allow_missing_static:
            raise ValueError("Static features requested, but no valid MSSQ values were found in the training split.")
        mssq_values = [0.0]
    mssq_mean = float(np.mean(mssq_values)) if mssq_values else 0.0
    mssq_std = float(np.std(mssq_values)) if mssq_values else 1.0
    if mssq_std < 1e-8:
        mssq_std = 1.0
    gender_counts = {cat: 0 for cat in GENDER_CATEGORY_ORDER}
    scenario_counts = {cat: 0 for cat in SCENARIO_CATEGORY_ORDER}
    for sess in train_sessions:
        gender = sess.gender if sess.gender in GENDER_CATEGORY_ORDER else "unknown"
        gender_counts[gender] += 1
        scenario = parse_scenario_from_filename(sess.source_file)
        scenario_counts[scenario if scenario in scenario_counts else "unknown"] += 1
    return {
        "static_features": features,
        "static_feature_names": names,
        "static_dim": len(names),
        "age_mean": age_mean,
        "age_std": age_std,
        "mssq_mean": mssq_mean,
        "mssq_std": mssq_std,
        "mssq_train_min": float(np.min(mssq_values)) if mssq_values else float("nan"),
        "mssq_train_max": float(np.max(mssq_values)) if mssq_values else float("nan"),
        "gender_encoding": gender_encoding,
        "gender_category_order": gender_order,
        "gender_categories": gender_order,
        "scenario_category_order": list(SCENARIO_CATEGORY_ORDER),
        "train_scenario_counts": scenario_counts,
        "train_gender_counts": gender_counts,
    }


def static_vector_for_session(
    session: DenseFMSSession,
    static_scaler: Mapping[str, Any],
    static_features: Optional[Sequence[str]] = None,
    allow_missing_static: bool = False,
) -> np.ndarray:
    features = normalize_static_features(static_features or static_scaler.get("static_features", ["age", "gender"]))
    values: List[float] = []
    if "age" in features:
        age = session.age
        if age is None or not np.isfinite(age):
            if not allow_missing_static:
                raise ValueError(f"Missing age for session {session.session_id} ({session.source_file}).")
            age = float(static_scaler["age_mean"])
        values.append((float(age) - float(static_scaler["age_mean"])) / float(static_scaler["age_std"]))
    if "mssq" in features:
        mssq = session.mssq
        if mssq is None or not np.isfinite(mssq):
            if not allow_missing_static:
                raise ValueError(f"Missing MSSQ for session {session.session_id} ({session.source_file}).")
            mssq = float(static_scaler["mssq_mean"])
        values.append((float(mssq) - float(static_scaler["mssq_mean"])) / float(static_scaler["mssq_std"]))
    if "gender" in features:
        gender = session.gender
        if gender not in {"male", "female"}:
            if not allow_missing_static:
                raise ValueError(f"Missing or unknown gender for session {session.session_id} ({session.source_file}).")
            gender = "unknown" if "unknown" in static_scaler.get("gender_category_order", GENDER_CATEGORY_ORDER) else None
        order = list(static_scaler["gender_category_order"])
        values.extend([1.0 if gender == cat else 0.0 for cat in order])
    if "scenario" in features:
        scenario = parse_scenario_from_filename(session.source_file)
        order = list(static_scaler.get("scenario_category_order", SCENARIO_CATEGORY_ORDER))
        if scenario not in order:
            scenario = "unknown"
        values.extend([1.0 if scenario == cat else 0.0 for cat in order])
    return np.asarray(values, dtype=np.float32)


def validate_static_availability(
    sessions: Sequence[DenseFMSSession],
    static_features: Optional[Sequence[str]] = None,
    allow_missing_static: bool = False,
) -> None:
    features = normalize_static_features(static_features)
    missing_age = [s.session_id for s in sessions if "age" in features and (s.age is None or not np.isfinite(s.age))]
    missing_gender = [s.session_id for s in sessions if "gender" in features and s.gender not in {"male", "female"}]
    missing_mssq = [s.session_id for s in sessions if "mssq" in features and (s.mssq is None or not np.isfinite(s.mssq))]
    if not allow_missing_static and (missing_age or missing_gender or missing_mssq):
        details = []
        if missing_age:
            details.append(f"missing age for {len(missing_age)} sessions, examples={missing_age[:5]}")
        if missing_gender:
            details.append(f"missing gender for {len(missing_gender)} sessions, examples={missing_gender[:5]}")
        if missing_mssq:
            details.append(f"missing MSSQ for {len(missing_mssq)} sessions, examples={missing_mssq[:5]}")
        raise ValueError("Static features requested but incomplete: " + "; ".join(details))


def build_static_report(
    splits: Mapping[str, Sequence[DenseFMSSession]],
    static_scaler: Optional[Mapping[str, Any]] = None,
    static_features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    features = normalize_static_features(static_features or (static_scaler or {}).get("static_features", ["age", "gender"]))
    gender_encoding = normalize_gender_encoding((static_scaler or {}).get("gender_encoding", "category3"))
    report: Dict[str, Any] = {
        "static_features": features,
        "gender_encoding": gender_encoding,
        "gender_category_order": gender_order_for_encoding(gender_encoding),
        "splits": {},
    }
    for split_name, sessions in splits.items():
        ages = [float(s.age) for s in sessions if s.age is not None and np.isfinite(s.age)]
        mssqs = [float(s.mssq) for s in sessions if s.mssq is not None and np.isfinite(s.mssq)]
        genders = [s.gender if s.gender in GENDER_CATEGORY_ORDER else "unknown" for s in sessions]
        scenarios = [parse_scenario_from_filename(s.source_file) for s in sessions]
        report["splits"][split_name] = {
            "session_count": len(sessions),
            "age_available": len(ages),
            "age_missing": len(sessions) - len(ages),
            "gender_available": sum(1 for s in sessions if s.gender in GENDER_CATEGORY_ORDER),
            "gender_missing": sum(1 for s in sessions if s.gender not in GENDER_CATEGORY_ORDER),
            "mssq_available": len(mssqs),
            "mssq_missing": len(sessions) - len(mssqs),
            "age_mean_original": float(np.mean(ages)) if ages else float("nan"),
            "age_std_original": float(np.std(ages)) if ages else float("nan"),
            "mssq_mean_original": float(np.mean(mssqs)) if mssqs else float("nan"),
            "mssq_std_original": float(np.std(mssqs)) if mssqs else float("nan"),
            "mssq_min_original": float(np.min(mssqs)) if mssqs else float("nan"),
            "mssq_max_original": float(np.max(mssqs)) if mssqs else float("nan"),
            "gender_counts": {cat: int(sum(g == cat for g in genders)) for cat in GENDER_CATEGORY_ORDER},
            "scenario_counts": {cat: int(sum(s == cat for s in scenarios)) for cat in SCENARIO_CATEGORY_ORDER},
        }
    all_sessions = [s for sessions in splits.values() for s in sessions]
    report["mssq_available_all"] = bool(all_sessions) and all(s.mssq is not None and np.isfinite(s.mssq) for s in all_sessions)
    if static_scaler is not None:
        report["static_scaler"] = dict(static_scaler)
    return report


def transform_sessions(
    sessions: Sequence[DenseFMSSession],
    scalers: Mapping[str, Any],
    use_static: bool = False,
    static_features: Optional[Sequence[str]] = None,
    allow_missing_static: bool = False,
    head_channel_mode: Optional[str] = None,
    calibration_residual_feature_map: Optional[Mapping[str, np.ndarray]] = None,
    calibration_residual_feature_names: Optional[Sequence[str]] = None,
    require_calibration_residual_features: bool = False,
) -> List[DenseFMSSession]:
    mean = np.asarray(scalers["head"]["mean"], dtype=np.float32)
    std = np.asarray(scalers["head"]["std"], dtype=np.float32)
    f_min = float(scalers["fms"]["min"])
    f_max = float(scalers["fms"]["max"])
    denom = max(f_max - f_min, 1e-8)
    head_channel_mode = normalize_head_channel_mode(head_channel_mode)
    if require_calibration_residual_features and calibration_residual_feature_map is None:
        raise ValueError(
            "require_calibration_residual_features=True requires a calibration_residual_feature_map."
        )
    transformed: List[DenseFMSSession] = []
    for sess in sessions:
        head_norm = ((sess.head.astype(np.float32) - mean) / std).astype(np.float32)
        head_norm = apply_head_channel_mode(head_norm, head_channel_mode)
        fms_norm = ((sess.fms.astype(np.float32) - f_min) / denom).astype(np.float32)
        fms_norm = np.clip(fms_norm, 0.0, 1.0)
        static = None
        feature_names = None
        if use_static:
            features = normalize_static_features(static_features or scalers["static"].get("static_features", ["age", "gender"]))
            static = static_vector_for_session(
                sess,
                scalers["static"],
                static_features=features,
                allow_missing_static=allow_missing_static,
            )
            feature_names = static_feature_names(features, gender_encoding=scalers["static"].get("gender_encoding", "category3"))
        residual_features = None
        residual_names = list(calibration_residual_feature_names) if calibration_residual_feature_names is not None else None
        if calibration_residual_feature_map is not None:
            for key in session_identity_keys(sess):
                if key in calibration_residual_feature_map:
                    residual_features = np.asarray(calibration_residual_feature_map[key], dtype=np.float32)
                    break
            if residual_features is None and require_calibration_residual_features:
                raise ValueError(
                    f"Missing calibration residual features for session {sess.session_id} ({sess.source_file})."
                )
            if residual_features is not None and residual_names is not None and residual_features.shape != (len(residual_names),):
                raise ValueError(
                    f"Calibration residual feature dimension mismatch for session {sess.session_id}: "
                    f"got {residual_features.shape}, expected {(len(residual_names),)}."
                )
        transformed.append(
            DenseFMSSession(
                head=head_norm,
                fms=fms_norm,
                time=sess.time.astype(np.float32),
                participant_id=sess.participant_id,
                session_id=sess.session_id,
                source_file=sess.source_file,
                fms_raw=sess.fms_raw if sess.fms_raw is not None else sess.fms.copy(),
                head_raw=sess.head_raw if sess.head_raw is not None else sess.head.copy(),
                age=sess.age,
                gender=sess.gender,
                mssq=sess.mssq,
                static=static,
                static_feature_names=feature_names,
                head_missing_mask=sess.head_missing_mask,
                head_imputation_report=sess.head_imputation_report,
                original_length=sess.original_length,
                max_session_points=sess.max_session_points,
                calibration_residual_features=residual_features,
                calibration_residual_feature_names=residual_names,
            )
        )
    return transformed


class DenseFMSSessionDataset(Dataset):
    def __init__(self, sessions: Sequence[DenseFMSSession]):
        self.sessions = list(sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> DenseFMSSession:
        return self.sessions[idx]


def collate_sessions(batch: Sequence[DenseFMSSession]) -> Dict[str, Any]:
    max_len = max(s.length for s in batch)
    head_dim = batch[0].head.shape[1]
    bsz = len(batch)
    head = torch.zeros((bsz, max_len, head_dim), dtype=torch.float32)
    fms = torch.zeros((bsz, max_len), dtype=torch.float32)
    fms_raw = torch.full((bsz, max_len), float("nan"), dtype=torch.float32)
    time = torch.full((bsz, max_len), float("nan"), dtype=torch.float32)
    lengths = torch.zeros((bsz,), dtype=torch.long)
    include_static = all(sess.static is not None for sess in batch)
    static_dim = int(batch[0].static.shape[0]) if include_static and batch[0].static is not None else 0
    if include_static:
        for sess in batch:
            if sess.static is None or int(sess.static.shape[0]) != static_dim:
                raise ValueError("All static vectors in a batch must have the same dimension.")
    static = torch.zeros((bsz, static_dim), dtype=torch.float32) if include_static else None
    include_residual = all(sess.calibration_residual_features is not None for sess in batch)
    residual_dim = (
        int(batch[0].calibration_residual_features.shape[0])
        if include_residual and batch[0].calibration_residual_features is not None
        else 0
    )
    if include_residual:
        for sess in batch:
            if sess.calibration_residual_features is None or int(sess.calibration_residual_features.shape[0]) != residual_dim:
                raise ValueError("All calibration residual feature vectors in a batch must have the same dimension.")
    residual = torch.zeros((bsz, residual_dim), dtype=torch.float32) if include_residual else None
    residual_mask = torch.ones((bsz, residual_dim), dtype=torch.float32) if include_residual else None
    metadata = []
    for i, sess in enumerate(batch):
        n = sess.length
        head[i, :n] = torch.from_numpy(sess.head.astype(np.float32))
        fms[i, :n] = torch.from_numpy(sess.fms.astype(np.float32))
        raw = sess.fms_raw if sess.fms_raw is not None else sess.fms
        fms_raw[i, :n] = torch.from_numpy(raw.astype(np.float32))
        time[i, :n] = torch.from_numpy(sess.time.astype(np.float32))
        lengths[i] = n
        if include_static and static is not None:
            static[i] = torch.from_numpy(sess.static.astype(np.float32))
        if include_residual and residual is not None:
            assert sess.calibration_residual_features is not None
            residual[i] = torch.from_numpy(sess.calibration_residual_features.astype(np.float32))
        metadata.append(
            {
                "participant_id": sess.participant_id,
                "session_id": sess.session_id,
                "source_file": sess.source_file,
                "age": sess.age,
                "original_age": sess.age,
                "gender": sess.gender,
                "gender_label": sess.gender,
                "mssq": sess.mssq,
                "scenario": parse_scenario_from_filename(sess.source_file),
                "original_mssq": sess.mssq,
                "static_feature_names": sess.static_feature_names,
                "head_imputation_report": sess.head_imputation_report,
                "original_session_length_steps": sess.original_length,
                "max_session_points": sess.max_session_points,
                "calibration_residual_feature_names": sess.calibration_residual_feature_names,
            }
        )
    batch_out = {"head": head, "fms": fms, "fms_raw": fms_raw, "time": time, "lengths": lengths, "metadata": metadata}
    if include_static and static is not None:
        batch_out["static"] = static
    if include_residual and residual is not None and residual_mask is not None:
        batch_out["calibration_residual_features"] = residual
        batch_out["calibration_residual_feature_mask"] = residual_mask
    return batch_out


def valid_prediction_mask(lengths: torch.Tensor, total_len: int, calibration_steps: int, horizon_steps: int) -> torch.Tensor:
    positions = torch.arange(total_len, device=lengths.device).unsqueeze(0)
    return (positions >= calibration_steps) & (positions + horizon_steps < lengths.unsqueeze(1))


def shifted_future_targets(fms: torch.Tensor, horizon_steps: int) -> torch.Tensor:
    target = torch.zeros_like(fms)
    if horizon_steps < fms.shape[1]:
        target[:, :-horizon_steps] = fms[:, horizon_steps:]
    return target


def future_sequence_targets(
    fms: torch.Tensor,
    lengths: torch.Tensor,
    calibration_steps: int,
    horizon_steps: int,
    max_pred_steps: Optional[int] = None,
    prediction_start_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact [B,T_pred] future targets ordered by target time."""
    if lengths.device != fms.device:
        lengths = lengths.to(fms.device)
    start_steps = int(calibration_steps if prediction_start_steps is None else prediction_start_steps)
    max_forecast_t = int(torch.clamp(lengths.max() - horizon_steps, min=start_steps).item())
    pred_steps = max(0, max_forecast_t - start_steps)
    if max_pred_steps is not None:
        pred_steps = min(pred_steps, int(max_pred_steps))
    positions = start_steps + torch.arange(pred_steps, device=fms.device)
    target_positions = positions + horizon_steps
    if pred_steps == 0:
        return fms[:, :0], torch.zeros((fms.shape[0], 0), dtype=torch.bool, device=fms.device)
    safe_idx = target_positions.clamp_max(fms.shape[1] - 1)
    target = fms.index_select(1, safe_idx)
    mask = target_positions.unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask & torch.isfinite(target)
    return target, mask


def future_sequence_times(
    time: torch.Tensor,
    lengths: torch.Tensor,
    calibration_steps: int,
    horizon_steps: int,
    max_pred_steps: Optional[int] = None,
    prediction_start_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact [B,T_pred] target timestamps and their length masks."""
    if lengths.device != time.device:
        lengths = lengths.to(time.device)
    start_steps = int(calibration_steps if prediction_start_steps is None else prediction_start_steps)
    max_forecast_t = int(torch.clamp(lengths.max() - horizon_steps, min=start_steps).item())
    pred_steps = max(0, max_forecast_t - start_steps)
    if max_pred_steps is not None:
        pred_steps = min(pred_steps, int(max_pred_steps))
    positions = start_steps + torch.arange(pred_steps, device=time.device)
    target_positions = positions + horizon_steps
    if pred_steps == 0:
        return time[:, :0], torch.zeros((time.shape[0], 0), dtype=torch.bool, device=time.device)
    safe_idx = target_positions.clamp_max(time.shape[1] - 1)
    target_time = time.index_select(1, safe_idx)
    mask = target_positions.unsqueeze(0) < lengths.unsqueeze(1)
    return target_time, mask


def current_sequence_times(
    time: torch.Tensor,
    lengths: torch.Tensor,
    calibration_steps: int,
    horizon_steps: int,
    max_pred_steps: Optional[int] = None,
    prediction_start_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return compact [B,T_pred] current timestamps for valid prediction positions."""
    if lengths.device != time.device:
        lengths = lengths.to(time.device)
    start_steps = int(calibration_steps if prediction_start_steps is None else prediction_start_steps)
    max_forecast_t = int(torch.clamp(lengths.max() - horizon_steps, min=start_steps).item())
    pred_steps = max(0, max_forecast_t - start_steps)
    if max_pred_steps is not None:
        pred_steps = min(pred_steps, int(max_pred_steps))
    positions = start_steps + torch.arange(pred_steps, device=time.device)
    if pred_steps == 0:
        return time[:, :0], torch.zeros((time.shape[0], 0), dtype=torch.bool, device=time.device)
    safe_idx = positions.clamp_max(time.shape[1] - 1)
    current_time = time.index_select(1, safe_idx)
    mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)) & ((positions + horizon_steps).unsqueeze(0) < lengths.unsqueeze(1))
    return current_time, mask


def run_data_sanity_checks(
    split_info: Mapping[str, Any],
    batch: Mapping[str, Any],
    calibration_steps: int,
    horizon_steps: int,
) -> None:
    """Unit-test-like guards for leakage-sensitive indexing."""
    groups = split_info["groups"]
    check_disjoint("train", groups.get("train", []), "val", groups.get("val", []))
    check_disjoint("train", groups.get("train", []), "test", groups.get("test", []))
    check_disjoint("val", groups.get("val", []), "test", groups.get("test", []))

    head = batch["head"]
    fms = batch["fms"]
    lengths = batch["lengths"]
    assert head.ndim == 3 and fms.ndim == 2, "batch_first tensors expected"
    y_calib = fms[:, :calibration_steps]
    assert y_calib.shape[1] == calibration_steps, "calibration input must use exactly the first C steps"
    mask = valid_prediction_mask(lengths, fms.shape[1], calibration_steps, horizon_steps)
    shifted = shifted_future_targets(fms, horizon_steps)
    valid_idx = mask.nonzero(as_tuple=False)
    if len(valid_idx):
        b, t = valid_idx[0].tolist()
        assert torch.isclose(shifted[b, t], fms[b, t + horizon_steps]), "prediction target is not shifted by K"
    assert not mask[:, :calibration_steps].any(), "loss mask includes calibration steps"
