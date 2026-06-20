"""Inspect DenseFMS CSV files and infer the forecasting column mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.densefms_forecast.data import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DenseFMS dataset CSV files.")
    parser.add_argument("--data_dir", default="./DenseFMS/Dataset")
    parser.add_argument("--artifacts_dir", default="./artifacts")
    args = parser.parse_args()
    report, mapping = inspect_dataset(args.data_dir, args.artifacts_dir)

    print(f"CSV files: {report['file_count']}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Unique schemas: {report['unique_schema_count']}")
    print("Inferred column mapping:")
    print(f"  time: {mapping.get('time')}")
    print(f"  fms: {mapping.get('fms')}")
    print(f"  head_features: {mapping.get('head_features')}")
    print(f"  participant: {mapping.get('participant')}")
    print(f"  session: {mapping.get('session')}")
    print(f"  static: {mapping.get('static')}")
    for key, candidates in mapping.get("candidates", {}).items():
        if len(candidates) > 1:
            print(f"  candidates[{key}]: {candidates} -> {candidates[0]}")
    for schema in report["schema_counts"][:5]:
        print(f"Schema used by {schema['file_count']} files: {schema['columns']}")
    print(f"Saved report to {Path(args.artifacts_dir) / 'data_report.json'}")
    print(f"Saved mapping to {Path(args.artifacts_dir) / 'column_mapping.json'}")


if __name__ == "__main__":
    main()
