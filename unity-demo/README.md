# Unity RollerCoaster DenseFMS Demo

Open `UnityProject/` with Unity `2021.3.45f2`.

This demo streams RollerCoaster head-motion samples to the Python sidecar in `UnityProject/Tools/unity_realtime_bridge.py`. The sidecar loads the DenseFMS code from the sibling `densefms-forecast/` folder by default.

## Included Project Folders

- `UnityProject/Assets`
- `UnityProject/Packages`
- `UnityProject/ProjectSettings`
- `UnityProject/Tools`

Unity-generated folders such as `Library`, `Temp`, `Obj`, `Logs`, and `UserSettings` are excluded.

## Runtime Requirements

- Unity 2021.3.45f2
- Python environment from `../densefms-forecast/requirements.txt`
- A compatible DenseFMS checkpoint, either in the default ignored path or configured in the `DenseFMSRealtimeDemo` Inspector

Default checkpoint path expected by the sidecar:

```text
../densefms-forecast/runs/risk_light_state_0521/state_headonly_pos0p5_thr12_seed42/best.pt
```

## Notes

`ROLLER_ONLY_NOTES.md` records how the original CET-VR project was reduced and repaired for the RollerCoaster demo. `CET-VR_ORIGINAL_README.md` is kept for upstream context and third-party asset acknowledgements.

