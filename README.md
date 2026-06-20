# Capstone VR Cybersickness Prediction

This repository packages a capstone project for real-time cybersickness prediction in VR.
It combines a DenseFMS forecasting pipeline with a Unity RollerCoaster demo that streams head-motion samples to a Python sidecar and displays live FMS/risk feedback.

## Repository Layout

```text
densefms-forecast/   PyTorch DenseFMS training, evaluation, sanity checks, and model configs
unity-demo/          Unity 2021.3 RollerCoaster demo and Python HTTP bridge
docs/assets/         Public screenshots and lightweight visual assets
```

## What Is Included

- DenseFMS sequence dataset/model/training/evaluation code.
- Leakage-aware windowing and sanity-test utilities.
- Selected model/config documentation for online current-FMS and warning-light work.
- Unity RollerCoaster demo project files needed to open the scene.
- Unity-to-Python sidecar bridge at `unity-demo/UnityProject/Tools/unity_realtime_bridge.py`.

## What Is Not Included

- DenseFMS dataset files.
- Training runs, prediction CSVs, generated reports, and checkpoints.
- Unity generated folders such as `Library`, `Temp`, `Obj`, `Logs`, and `UserSettings`.
- Local replay outputs and participant/session artifacts.

Checkpoints are intentionally excluded from git. To run live inference, train or download a compatible checkpoint and place it under:

```text
densefms-forecast/runs/risk_light_state_0521/state_headonly_pos0p5_thr12_seed42/best.pt
```

or set a custom `checkpointPath` in the `DenseFMSRealtimeDemo` Inspector.

## Python Setup

```powershell
cd densefms-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The dataset is expected at:

```text
densefms-forecast/DenseFMS/Dataset
```

The dataset folder is not committed.

## Git LFS

Large Unity textures, audio, and model files are tracked with Git LFS. Before the first push, make sure Git LFS is installed locally:

```powershell
git lfs install
```

## Unity Setup

Open this folder in Unity Hub:

```text
unity-demo/UnityProject
```

Recommended editor:

```text
Unity 2021.3.45f2
```

The RollerCoaster-only notes are in `unity-demo/ROLLER_ONLY_NOTES.md`.

## Demo Flow

1. Start or let Unity start the Python sidecar.
2. Open the RollerCoaster scene.
3. Set participant age, MSSQ, gender, and initial FMS in the Inspector.
4. Start the ride in PC test mode with `Space`, or use Quest Link/OpenXR for headset testing.
5. Unity records head motion every 0.5 seconds and sends samples to the sidecar.
6. After calibration, the sidecar returns online FMS/risk predictions.
7. Unity saves local `samples.csv` and `summary.json` under `Application.persistentDataPath`.

## Notes For Reviewers

This repository is organized for project review and reproducibility of the code path, not for redistributing private datasets or model weights. Reported validation/test metrics in the documentation should be read as experiment records; new model selection must still use validation data only, with test data reserved for final reporting.
