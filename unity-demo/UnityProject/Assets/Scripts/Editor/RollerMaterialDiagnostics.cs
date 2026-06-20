using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class RollerMaterialDiagnostics
{
    private const string RollerScenePath = "Assets/Scenes/RollerCoaster.unity";
    private const string FallbackMaterialPath = "Assets/Materials/RollerCoaster/RollerFallbackStandard.mat";
    private const string TerrainNearMaterialPath = "Assets/Models/Terrain/Materials/mat_terrain_near_01.mat";
    private const string TerrainFarMaterialPath = "Assets/Models/Terrain/Materials/mat_terrain_far_01.mat";
    private const string DefaultTreePrefabPath = "Assets/Prefabs/Tree.prefab";

    private static readonly string[] ExtraMaterialRoots =
    {
        "Assets/Materials/RollerCoaster",
        "Assets/Models/Terrain",
        "Assets/Models/starting_line",
        "Assets/Stand",
        "Assets/Water4Stereo"
    };

    [MenuItem("CET-VR/Roller/Report Error Materials")]
    public static void ReportRollerErrorMaterials()
    {
        var sceneMaterialReport = CollectRollerSceneMaterialReport();
        var materials = CollectRollerMaterials();
        var errorMaterials = FindErrorMaterials(materials);

        Debug.Log($"[RollerMaterialDiagnostics] Checked {sceneMaterialReport.RendererCount} scene renderers and {materials.Count} Roller materials. Null scene material slots: {sceneMaterialReport.NullMaterialSlots}. Error materials: {errorMaterials.Count}");
        foreach (var material in errorMaterials)
        {
            Debug.Log($"[RollerMaterialDiagnostics] Error material: {AssetDatabase.GetAssetPath(material)} shader={GetShaderName(material)}");
        }
    }

    [MenuItem("CET-VR/Roller/Repair Error Materials")]
    public static void RepairRollerErrorMaterials()
    {
        var fallbackShader = Shader.Find("Standard");
        if (fallbackShader == null)
        {
            Debug.LogError("[RollerMaterialDiagnostics] Could not find Built-in Standard shader.");
            return;
        }

        var fallbackMaterial = GetOrCreateFallbackMaterial(fallbackShader);
        var sceneSlotRepairCount = RepairRollerSceneMaterialSlots(fallbackMaterial);
        var terrainRestoreCount = RestoreImportedTerrainMaterialOverrides(fallbackMaterial);
        var treePrototypeRepairCount = RepairMissingTerrainTreePrototypes();
        var materials = CollectRollerMaterials();
        var errorMaterials = FindErrorMaterials(materials);
        foreach (var material in errorMaterials)
        {
            material.shader = fallbackShader;
            material.SetFloat("_Metallic", 0f);
            material.SetFloat("_Glossiness", 0.35f);
            EditorUtility.SetDirty(material);
            Debug.Log($"[RollerMaterialDiagnostics] Repaired material: {AssetDatabase.GetAssetPath(material)}");
        }

        AssetDatabase.SaveAssets();
        Debug.Log($"[RollerMaterialDiagnostics] Repaired {errorMaterials.Count} Roller material assets and {sceneSlotRepairCount} scene material slots with the Built-in Standard shader. Restored {terrainRestoreCount} imported terrain material overrides. Repaired {treePrototypeRepairCount} terrain tree prototypes.");
    }

    private static Material GetOrCreateFallbackMaterial(Shader fallbackShader)
    {
        var fallbackMaterial = AssetDatabase.LoadAssetAtPath<Material>(FallbackMaterialPath);
        if (fallbackMaterial != null)
        {
            return fallbackMaterial;
        }

        fallbackMaterial = new Material(fallbackShader)
        {
            name = "RollerFallbackStandard"
        };

        fallbackMaterial.SetColor("_Color", new Color(0.62f, 0.62f, 0.62f, 1f));
        fallbackMaterial.SetFloat("_Metallic", 0f);
        fallbackMaterial.SetFloat("_Glossiness", 0.3f);
        AssetDatabase.CreateAsset(fallbackMaterial, FallbackMaterialPath);
        AssetDatabase.SaveAssets();
        return fallbackMaterial;
    }

    private static int RepairRollerSceneMaterialSlots(Material fallbackMaterial)
    {
        var scene = OpenRollerScene();
        var repairedSlotCount = 0;

        foreach (var renderer in Object.FindObjectsOfType<Renderer>(true))
        {
            var sharedMaterials = renderer.sharedMaterials;
            var changed = false;

            for (var index = 0; index < sharedMaterials.Length; index++)
            {
                var material = sharedMaterials[index];
                if (material == null || IsErrorMaterial(material))
                {
                    Debug.Log($"[RollerMaterialDiagnostics] Repaired scene renderer material slot: {GetHierarchyPath(renderer.transform)}[{index}] old={GetShaderName(material)}");
                    sharedMaterials[index] = fallbackMaterial;
                    changed = true;
                    repairedSlotCount++;
                }
            }

            if (changed)
            {
                renderer.sharedMaterials = sharedMaterials;
                EditorUtility.SetDirty(renderer);
            }
        }

        foreach (var terrain in Object.FindObjectsOfType<Terrain>(true))
        {
            if (terrain.materialTemplate != null && IsErrorMaterial(terrain.materialTemplate))
            {
                Debug.Log($"[RollerMaterialDiagnostics] Repaired terrain material template: {GetHierarchyPath(terrain.transform)} old={GetShaderName(terrain.materialTemplate)}");
                terrain.materialTemplate = fallbackMaterial;
                EditorUtility.SetDirty(terrain);
                repairedSlotCount++;
            }
        }

        if (repairedSlotCount > 0)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }

        return repairedSlotCount;
    }

    private static int RestoreImportedTerrainMaterialOverrides(Material fallbackMaterial)
    {
        var scene = OpenRollerScene();
        var nearMaterial = AssetDatabase.LoadAssetAtPath<Material>(TerrainNearMaterialPath);
        var farMaterial = AssetDatabase.LoadAssetAtPath<Material>(TerrainFarMaterialPath);
        var restoredCount = 0;

        foreach (var renderer in Object.FindObjectsOfType<Renderer>(true))
        {
            Material expectedMaterial = null;
            if (renderer.name == "terrain_near_01")
            {
                expectedMaterial = nearMaterial;
            }
            else if (renderer.name == "terrain_far_01")
            {
                expectedMaterial = farMaterial;
            }

            if (expectedMaterial == null || !GetHierarchyPath(renderer.transform).Contains("/Terrain/terrain_01/"))
            {
                continue;
            }

            var sharedMaterials = renderer.sharedMaterials;
            var changed = false;
            for (var index = 0; index < sharedMaterials.Length; index++)
            {
                if (sharedMaterials[index] == fallbackMaterial || sharedMaterials[index] == null || IsErrorMaterial(sharedMaterials[index]))
                {
                    sharedMaterials[index] = expectedMaterial;
                    changed = true;
                }
            }

            if (!changed)
            {
                continue;
            }

            renderer.sharedMaterials = sharedMaterials;
            EditorUtility.SetDirty(renderer);
            restoredCount++;
            Debug.Log($"[RollerMaterialDiagnostics] Restored terrain renderer material override: {GetHierarchyPath(renderer.transform)} -> {AssetDatabase.GetAssetPath(expectedMaterial)}");
        }

        if (restoredCount > 0)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }

        return restoredCount;
    }

    private static int RepairMissingTerrainTreePrototypes()
    {
        var scene = OpenRollerScene();
        var defaultTreePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(DefaultTreePrefabPath);
        if (defaultTreePrefab == null)
        {
            Debug.LogWarning($"[RollerMaterialDiagnostics] Default tree prefab is missing: {DefaultTreePrefabPath}");
            return 0;
        }

        var repairedCount = 0;
        foreach (var terrain in Object.FindObjectsOfType<Terrain>(true))
        {
            var terrainData = terrain.terrainData;
            if (terrainData == null)
            {
                continue;
            }

            var treePrototypes = terrainData.treePrototypes;
            var changed = false;
            for (var index = 0; index < treePrototypes.Length; index++)
            {
                if (treePrototypes[index].prefab != null)
                {
                    continue;
                }

                treePrototypes[index].prefab = defaultTreePrefab;
                changed = true;
                repairedCount++;
                Debug.Log($"[RollerMaterialDiagnostics] Repaired terrain tree prototype: {GetHierarchyPath(terrain.transform)}[{index}] -> {DefaultTreePrefabPath}");
            }

            if (!changed)
            {
                continue;
            }

            terrainData.treePrototypes = treePrototypes;
            EditorUtility.SetDirty(terrainData);
            EditorUtility.SetDirty(terrain);
        }

        if (repairedCount > 0)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            AssetDatabase.SaveAssets();
        }

        return repairedCount;
    }

    private static SceneMaterialReport CollectRollerSceneMaterialReport()
    {
        OpenRollerScene();

        var report = new SceneMaterialReport();
        foreach (var renderer in Object.FindObjectsOfType<Renderer>(true))
        {
            report.RendererCount++;
            var sharedMaterials = renderer.sharedMaterials;
            for (var index = 0; index < sharedMaterials.Length; index++)
            {
                var material = sharedMaterials[index];
                if (material == null)
                {
                    report.NullMaterialSlots++;
                    Debug.Log($"[RollerMaterialDiagnostics] Null scene material slot: {GetHierarchyPath(renderer.transform)}[{index}]");
                }
                else if (IsErrorMaterial(material))
                {
                    Debug.Log($"[RollerMaterialDiagnostics] Error scene material slot: {GetHierarchyPath(renderer.transform)}[{index}] material={AssetDatabase.GetAssetPath(material)} shader={GetShaderName(material)}");
                }
            }
        }

        return report;
    }

    private static HashSet<Material> CollectRollerMaterials()
    {
        var materials = new HashSet<Material>();

        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(RollerScenePath) != null)
        {
            OpenRollerScene();
            foreach (var renderer in Object.FindObjectsOfType<Renderer>(true))
            {
                foreach (var material in renderer.sharedMaterials)
                {
                    if (material != null)
                    {
                        materials.Add(material);
                    }
                }
            }
        }

        var guids = AssetDatabase.FindAssets("t:Material", ExtraMaterialRoots);
        foreach (var guid in guids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material != null)
            {
                materials.Add(material);
            }
        }

        return materials;
    }

    private static UnityEngine.SceneManagement.Scene OpenRollerScene()
    {
        return EditorSceneManager.OpenScene(RollerScenePath);
    }

    private static List<Material> FindErrorMaterials(IEnumerable<Material> materials)
    {
        var errorMaterials = new List<Material>();
        foreach (var material in materials)
        {
            if (IsErrorMaterial(material))
            {
                errorMaterials.Add(material);
            }
        }

        errorMaterials.Sort((left, right) => string.CompareOrdinal(AssetDatabase.GetAssetPath(left), AssetDatabase.GetAssetPath(right)));
        return errorMaterials;
    }

    private static bool IsErrorMaterial(Material material)
    {
        if (material == null || material.shader == null)
        {
            return true;
        }

        var shaderName = material.shader.name;
        return shaderName == "Hidden/InternalErrorShader"
            || shaderName.Contains("InternalErrorShader")
            || !material.shader.isSupported;
    }

    private static string GetShaderName(Material material)
    {
        if (material == null)
        {
            return "<null material>";
        }

        return material.shader != null ? material.shader.name : "<null shader>";
    }

    private static string GetHierarchyPath(Transform transform)
    {
        var path = transform.name;
        while (transform.parent != null)
        {
            transform = transform.parent;
            path = transform.name + "/" + path;
        }

        return path;
    }

    private struct SceneMaterialReport
    {
        public int RendererCount;
        public int NullMaterialSlots;
    }
}
