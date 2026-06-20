# DenseFMS Forecasting

This folder contains the Python side of the capstone project: DenseFMS data loading, leakage-safe windowing, model definitions, training, evaluation, realtime streaming, and experiment scripts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Place the DenseFMS dataset outside git at:

```text
DenseFMS/Dataset
```

The expected sampling interval is 0.5 seconds. The default public configuration uses the first 420 sampled steps of each session, 90 seconds of calibration, a 30-second recent window, and a 5-second forecast horizon unless a config explicitly changes that.

## Useful Commands

```powershell
python scripts/inspect_densefms.py --data_dir ./DenseFMS/Dataset
python scripts/run_densefms_sanity_tests.py
```

Training outputs are written under `runs/` and are ignored by git.

## Key Files

- `src/densefms_forecast/data.py`: session loading, split handling, leakage-safe windows.
- `src/densefms_forecast/model.py`: forecasting and online-current model definitions.
- `src/densefms_forecast/train.py`: training/evaluation loop and metrics.
- `src/densefms_forecast/realtime.py`: online prefix streaming logic used by the Unity bridge.
- `configs/online_current/`: selected online current-FMS configs.
- `FINAL_MODEL.md`: final warning-policy/checkpoint record.

