# Roller-only cleanup

This copy was reduced to keep the RollerCoaster scene usable without restoring every external asset from the original CET-VR project.

## Open this project

Open `UnityProject/` with Unity `2021.3.45f2`.

The build scene list now contains only:

- `Assets/Scenes/GameManagerLoader.unity`
- `Assets/Scenes/RollerCoaster.unity`

`GameManagerLoader` loads `RollerCoaster` directly.

## Disabled content

Non-Roller scenes, scripts, and prefabs were moved under:

- `UnityProject/DisabledForRollerOnly/`

Unity does not import or compile files in that folder because it is outside `Assets/`.

## Applied fixes

- Restored PathCreator scripts from the upstream Path-Creator project while keeping this repository's existing `.meta` GUIDs.
- Applied `PathCreator_patch.diff`.
- Added a no-op `DepthOfField` placeholder so old Standard Assets references do not break compile.
- Disabled `GazeContingentDOF` at runtime in the Roller-only build.
- Fixed the RollerCoaster start trigger condition so right trigger and space are gated by `_canStart`.
- Restored a compatible Toon outline shader for the Stand materials that referenced the missing Unity Standard Assets Toon shader GUID.
- Restored `Water4Stereo`, fixed its Unity 2021 compile issues, and changed the unsupported water material to the Built-in Standard shader.
- Added `CET-VR > Roller > Report Error Materials` and `CET-VR > Roller > Repair Error Materials` editor tools.
- Added a local replacement for the missing Standard Assets water prefab GUID used by the Roller scene.
- Repaired 17 null material slots inside `RollerCoaster.unity` with `Assets/Materials/RollerCoaster/RollerFallbackStandard.mat`.
- Imported the old Built-in Pipeline Viking Village package from a local Unity package export.
- Restored the Roller terrain material overrides to `mat_terrain_near_01` and `mat_terrain_far_01`.
- Repaired the three missing Roller terrain tree prototypes with `Assets/Prefabs/Tree.prefab`.

## Validation

Unity batchmode import/compile was run with:

```bash
Unity.exe -batchmode -nographics -quit -projectPath ".\UnityProject" -logFile ".\unity-roller-import.log"
```

It completed successfully with return code `0`.

Material validation was also run with:

```bash
Unity.exe -batchmode -nographics -quit -projectPath ".\UnityProject" -executeMethod RollerMaterialDiagnostics.RepairRollerErrorMaterials -logFile ".\unity-roller-material-repair-verify.log"
```

It completed successfully. The final Roller material repair pass reported `0` remaining Error/unsupported material assets and `0` remaining broken scene material slots.

After importing Viking Village, `Assets/Models/Terrain/terrain_01.fbx`, the terrain textures, and `Assets/Textures/Skies/Daytime/SunsetSkyboxHDR.hdr` are present. The final validation log no longer reports the missing terrain prefab, missing tree prefab, compiler error, shader error, or broken Roller material slot warnings.
