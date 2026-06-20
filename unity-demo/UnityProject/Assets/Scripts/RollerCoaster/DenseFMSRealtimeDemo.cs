using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using XR = UnityEngine.XR;
using XRInputDevice = UnityEngine.XR.InputDevice;

[DefaultExecutionOrder(-50)]
public class DenseFMSRealtimeDemo : MonoBehaviour
{
    enum ParticipantGender
    {
        Male,
        Female
    }

    const float SampleInterval = 0.5f;
    const int ExpectedCalibrationSteps = 240;
    const float FmsKeyStep = 1f;
    const float FmsKeyRepeatDelay = 0.35f;
    const float FmsKeyRepeatInterval = 0.08f;

    static bool sceneHooked;
    static DenseFMSRealtimeDemo activeDemo;

    [Header("Bridge")]
    [SerializeField] string bridgeUrl = "http://127.0.0.1:8765";
    [SerializeField] bool autoStartSidecar = true;
    [SerializeField] string pythonExecutable = "python";
    [SerializeField] string codexRepoPath = "";
    [SerializeField] string checkpointPath = "";
    [SerializeField] string sidecarScriptPath = "";

    [Header("FMS Toast")]
    [SerializeField] string fmsImageResourceFolder = "FMS_Generated_v2_Transparent";
    [Header("Inspector Fallback Inputs")]
    [SerializeField, Min(0.01f)] float participantAge = 20f;
    [SerializeField, Min(0f)] float participantMssq = 0f;
    [SerializeField] ParticipantGender participantGender = ParticipantGender.Male;
    [SerializeField, Range(0f, 20f)] float rawFms = 0f;
    [SerializeField] bool showRuntimeUi = false;
    [SerializeField] bool showLivePredictionMetrics = true;
    [Header("VR UI")]
    [SerializeField] bool useHeadLockedVrCanvas = true;
    [SerializeField, Min(0.25f)] float vrUiDistance = 0.95f;
    [SerializeField, Min(0.0001f)] float vrUiScale = 0.0012f;
    [Header("FMS Runtime Toast")]
    [SerializeField] bool showFmsRuntimeHud = true;
    [SerializeField] Vector3 fmsHudLocalPosition = new Vector3(0f, -0.12f, 0.85f);
    [SerializeField, Min(0.0001f)] float fmsHudScale = 0.0012f;
    [Header("Risk Warning")]
    [SerializeField] bool showRiskWarningIcon = true;
    [SerializeField, Range(0f, 1f)] float riskWarningOnThreshold = 0.8f;
    [SerializeField, Range(0f, 1f)] float riskWarningOffThreshold = 0.6f;
    [Header("Head Motion Sampling")]
    [SerializeField] bool deriveHeadMotionFromSceneTransform = true;
    [SerializeField] bool useXrVelocityFeatureFallback = true;

    RollerCoasterLevelManager manager;
    Canvas canvas;
    RectTransform canvasRect;
    Canvas fmsHudCanvas;
    RectTransform fmsHudCanvasRect;
    Canvas riskWarningCanvas;
    RectTransform riskWarningCanvasRect;
    Camera uiCamera;
    Transform headMotionTransform;
    GameObject preRidePanel;
    GameObject ridePanel;
    GameObject resultPanel;
    Text bridgeStatusText;
    Text rideStatusText;
    Text resultText;
    RawImage chartImage;
    Text fmsValueText;
    Text fmsToastText;
    RawImage fmsToastImage;
    RawImage riskWarningIconImage;
    CanvasGroup fmsToastCanvasGroup;
    Coroutine fmsToastRoutine;
    Texture2D riskWarningIconTexture;

    readonly Queue<DenseFmsSample> pendingSamples = new Queue<DenseFmsSample>();
    readonly List<DenseFmsSample> samples = new List<DenseFmsSample>();
    readonly List<DenseFmsPredictionRow> predictions = new List<DenseFmsPredictionRow>();
    readonly Dictionary<int, Texture2D> fmsToastTextures = new Dictionary<int, Texture2D>();

    bool uiCreated;
    bool bridgeOnline;
    bool bridgeStarting;
    bool rideActive;
    bool sessionStarting;
    bool sessionStarted;
    bool finishing;
    bool resultsSaved;
    bool riskWarningActive;
    int stepIndex;
    float nextSampleAt;
    float nextFmsKeyRepeatAt;
    float currentFmsValue;
    Vector3 previousHeadPosition;
    Quaternion previousHeadRotation;
    Vector3 previousHeadLinearVelocity;
    Vector3 previousXrLinearVelocity;
    float previousHeadSampleTime;
    float previousXrSampleTime;
    bool hasPreviousHeadPose;
    bool hasPreviousHeadLinearVelocity;
    bool hasPreviousXrLinearVelocity;
    bool warnedMissingHeadMotion;
    string sessionId = "";
    string outputDirectory = "";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void RuntimeAttach()
    {
        if (!sceneHooked)
        {
            SceneManager.sceneLoaded += (_, __) => TryAttachToRollerScene();
            sceneHooked = true;
        }
        TryAttachToRollerScene();
    }

    static void TryAttachToRollerScene()
    {
        RollerCoasterLevelManager roller = FindObjectOfType<RollerCoasterLevelManager>();
        if (roller == null)
            return;
        if (roller.GetComponent<DenseFMSRealtimeDemo>() == null)
            roller.gameObject.AddComponent<DenseFMSRealtimeDemo>();
    }

    public static void NotifyRideStarted(RollerCoasterLevelManager roller)
    {
        DenseFMSRealtimeDemo demo = EnsureDemo(roller);
        if (demo != null)
            demo.BeginRide();
    }

    public static void NotifyRideEnding(RollerCoasterLevelManager roller)
    {
        DenseFMSRealtimeDemo demo = EnsureDemo(roller);
        if (demo != null)
            demo.EndRide();
    }

    public static bool CanStartRide(RollerCoasterLevelManager roller)
    {
        DenseFMSRealtimeDemo demo = EnsureDemo(roller);
        return demo == null || demo.ValidatePreRideInputs(true);
    }

    static DenseFMSRealtimeDemo EnsureDemo(RollerCoasterLevelManager roller)
    {
        if (roller == null)
            return null;
        DenseFMSRealtimeDemo demo = roller.GetComponent<DenseFMSRealtimeDemo>();
        if (demo == null)
            demo = roller.gameObject.AddComponent<DenseFMSRealtimeDemo>();
        return demo;
    }

    void Awake()
    {
        manager = GetComponent<RollerCoasterLevelManager>();
        if (manager == null)
            manager = FindObjectOfType<RollerCoasterLevelManager>();
        if (activeDemo != null && activeDemo != this)
        {
            Destroy(this);
            return;
        }
        activeDemo = this;
    }

    void Start()
    {
        CreateFmsHud();
        CreateRiskWarningHud();
        SetRiskWarningActive(false);
        SetFmsValue(CurrentFmsValue());
        ShowCurrentFmsToast();
        if (RuntimeUiEnabled())
            CreateUi();
        StartCoroutine(EnsureBridgeRunning());
    }

    void OnDestroy()
    {
        SaveLocalResultsIfNeeded("destroy");
        if (activeDemo == this)
            activeDemo = null;
    }

    void Update()
    {
        HandleFmsKeyboardInput();
        if (!rideActive)
            return;
        if (Time.time >= nextSampleAt)
        {
            CaptureSample();
            nextSampleAt += SampleInterval;
        }
    }

    void LateUpdate()
    {
        PositionFmsHudCanvas();
        PositionRiskWarningCanvas();
        if (RuntimeUiEnabled())
            PositionVrCanvas();
    }

    void HandleFmsKeyboardInput()
    {
        if (Keyboard.current == null || finishing)
            return;

        bool leftPressed = Keyboard.current[Key.LeftArrow].wasPressedThisFrame;
        bool rightPressed = Keyboard.current[Key.RightArrow].wasPressedThisFrame;
        bool leftHeld = Keyboard.current[Key.LeftArrow].isPressed;
        bool rightHeld = Keyboard.current[Key.RightArrow].isPressed;

        if (leftPressed)
        {
            AdjustFms(-FmsKeyStep);
            nextFmsKeyRepeatAt = Time.unscaledTime + FmsKeyRepeatDelay;
        }
        else if (rightPressed)
        {
            AdjustFms(FmsKeyStep);
            nextFmsKeyRepeatAt = Time.unscaledTime + FmsKeyRepeatDelay;
        }
        else if ((leftHeld || rightHeld) && Time.unscaledTime >= nextFmsKeyRepeatAt)
        {
            AdjustFms(rightHeld ? FmsKeyStep : -FmsKeyStep);
            nextFmsKeyRepeatAt = Time.unscaledTime + FmsKeyRepeatInterval;
        }
    }

    void AdjustFms(float delta)
    {
        if (SetFmsValue(CurrentFmsValue() + delta))
            ShowCurrentFmsToast();
    }

    bool SetFmsValue(float value)
    {
        float clamped = Mathf.Round(Mathf.Clamp(value, 0f, 20f));
        bool changed = !Mathf.Approximately(currentFmsValue, clamped);
        currentFmsValue = clamped;
        rawFms = clamped;
        if (manager != null)
            manager.DenseFmsRawFms = clamped;
        if (fmsValueText != null)
            fmsValueText.text = currentFmsValue.ToString("0", CultureInfo.InvariantCulture);
        return changed;
    }

    void ShowCurrentFmsToast()
    {
        if (showFmsRuntimeHud && (fmsToastText == null || fmsToastCanvasGroup == null))
            CreateFmsHud();
        if (fmsToastText == null || fmsToastCanvasGroup == null)
            return;

        int fmsIndex = Mathf.RoundToInt(currentFmsValue);
        fmsToastText.text = "Current FMS : " + fmsIndex.ToString(CultureInfo.InvariantCulture);
        if (fmsToastImage != null)
        {
            Texture2D texture = LoadFmsToastTexture(fmsIndex);
            fmsToastImage.texture = texture;
            fmsToastImage.enabled = texture != null;
        }
        if (fmsToastRoutine != null)
            StopCoroutine(fmsToastRoutine);
        fmsToastRoutine = StartCoroutine(FadeFmsToast());
    }

    Texture2D LoadFmsToastTexture(int fmsIndex)
    {
        if (fmsToastTextures.TryGetValue(fmsIndex, out Texture2D cached))
            return cached;

        string resourceFolder = string.IsNullOrWhiteSpace(fmsImageResourceFolder) ? "" : fmsImageResourceFolder.Replace("\\", "/").Trim('/');
        string resourcePath = string.IsNullOrEmpty(resourceFolder)
            ? fmsIndex.ToString(CultureInfo.InvariantCulture)
            : resourceFolder + "/" + fmsIndex.ToString(CultureInfo.InvariantCulture);
        Texture2D texture = Resources.Load<Texture2D>(resourcePath);
        if (texture != null)
        {
            texture.wrapMode = TextureWrapMode.Clamp;
            fmsToastTextures[fmsIndex] = texture;
        }
        return texture;
    }

    IEnumerator FadeFmsToast()
    {
        const float holdSeconds = 2.0f;
        const float fadeSeconds = 1.0f;
        fmsToastCanvasGroup.alpha = 1f;
        yield return new WaitForSecondsRealtime(holdSeconds);

        float start = Time.unscaledTime;
        while (Time.unscaledTime - start < fadeSeconds)
        {
            float t = (Time.unscaledTime - start) / fadeSeconds;
            fmsToastCanvasGroup.alpha = 1f - Mathf.Clamp01(t);
            yield return null;
        }
        fmsToastCanvasGroup.alpha = 0f;
        fmsToastRoutine = null;
    }

    void BeginRide()
    {
        if (rideActive || finishing)
            return;
        if (RuntimeUiEnabled())
            CreateUi();
        SetFmsValue(CurrentFmsValue());

        samples.Clear();
        predictions.Clear();
        pendingSamples.Clear();
        stepIndex = 0;
        rideActive = true;
        finishing = false;
        sessionStarted = false;
        sessionStarting = true;
        resultsSaved = false;
        SetRiskWarningActive(false);
        showLivePredictionMetrics = ShowLivePredictionMetricsSetting();
        outputDirectory = "";
        sessionId = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);

        ResetHeadMotionTracking();
        nextSampleAt = Time.time + SampleInterval;

        if (preRidePanel != null)
            preRidePanel.SetActive(false);
        if (ridePanel != null)
            ridePanel.SetActive(true);
        if (resultPanel != null)
            resultPanel.SetActive(false);
        ShowCurrentFmsToast();
        SetRideStatus("Starting DenseFMS session...");
        if (!bridgeOnline && !bridgeStarting)
            StartCoroutine(EnsureBridgeRunning());
        StartCoroutine(StartBridgeSession());
    }

    void EndRide()
    {
        if (finishing)
            return;
        if (RuntimeUiEnabled())
            CreateUi();
        rideActive = false;
        finishing = true;
        SetRiskWarningActive(false);
        StartCoroutine(FinishBridgeSession());
    }

    IEnumerator EnsureBridgeRunning()
    {
        if (bridgeStarting)
            yield break;
        bridgeStarting = true;
        SetBridgeStatus("Checking model bridge...");
        bool healthOk = false;
        yield return GetHealth(ok => healthOk = ok);
        if (!healthOk && autoStartSidecar)
        {
            LaunchSidecar();
            SetBridgeStatus("Starting Python sidecar...");
            float deadline = Time.realtimeSinceStartup + 12f;
            while (!healthOk && Time.realtimeSinceStartup < deadline)
            {
                yield return new WaitForSecondsRealtime(0.75f);
                yield return GetHealth(ok => healthOk = ok);
            }
        }
        bridgeOnline = healthOk;
        SetBridgeStatus(bridgeOnline ? "Model bridge online" : "Model offline; raw data will still be saved");
        bridgeStarting = false;
    }

    IEnumerator StartBridgeSession()
    {
        float waitUntil = Time.realtimeSinceStartup + 15f;
        while (bridgeStarting && Time.realtimeSinceStartup < waitUntil)
            yield return null;
        if (!bridgeOnline && !bridgeStarting)
            yield return EnsureBridgeRunning();
        if (!bridgeOnline)
        {
            sessionStarting = false;
            SetRideStatus("Model offline: collecting raw FMS/tracking only.");
            yield break;
        }
        if (!TryReadStaticInputs(out float age, out float mssq, true))
        {
            sessionStarting = false;
            SetRideStatus("Age and MSSQ are required; model session not started.");
            yield break;
        }

        BridgeStartRequest request = new BridgeStartRequest
        {
            session_id = sessionId,
            age = age,
            mssq = mssq,
            gender = CurrentGender()
        };

        yield return PostJson("/session/start", JsonUtility.ToJson(request), json =>
        {
            BridgeStartResponse response = JsonUtility.FromJson<BridgeStartResponse>(json);
            if (response != null && response.ok)
            {
                sessionStarted = true;
                SetRideStatus("Calibrating 0/" + ExpectedCalibrationSteps + " samples");
                StartCoroutine(PumpPendingSamples());
            }
            else
            {
                bridgeOnline = false;
                SetRideStatus("Model offline: " + ((response != null && response.error != null) ? response.error : "start failed"));
            }
        }, error =>
        {
            bridgeOnline = false;
            SetRideStatus("Model offline: " + error);
        });
        sessionStarting = false;
    }

    IEnumerator PumpPendingSamples()
    {
        while ((rideActive || finishing) && bridgeOnline)
        {
            if (sessionStarted && pendingSamples.Count > 0)
            {
                DenseFmsSample sample = pendingSamples.Peek();
                yield return SendStep(sample);
            }
            else
            {
                yield return null;
            }
        }
    }

    IEnumerator SendStep(DenseFmsSample sample)
    {
        BridgeStepRequest request = new BridgeStepRequest
        {
            step_index = sample.step_index,
            timestamp = sample.timestamp,
            acc_x = sample.acc_x,
            acc_y = sample.acc_y,
            acc_z = sample.acc_z,
            linear_velocity_x = sample.acc_x,
            linear_velocity_y = sample.acc_y,
            linear_velocity_z = sample.acc_z,
            angular_velocity_x = sample.angular_velocity_x,
            angular_velocity_y = sample.angular_velocity_y,
            angular_velocity_z = sample.angular_velocity_z,
            fms_raw = sample.fms_raw
        };
        bool completed = false;
        yield return PostJson("/session/step", JsonUtility.ToJson(request), json =>
        {
            BridgeStepResponse response = JsonUtility.FromJson<BridgeStepResponse>(json);
            if (response != null && response.ok)
            {
                ApplyStepResponse(response);
                if (pendingSamples.Count > 0 && pendingSamples.Peek().step_index == sample.step_index)
                    pendingSamples.Dequeue();
            }
            else
            {
                bridgeOnline = false;
                pendingSamples.Clear();
                SetRiskWarningActive(false);
                SetRideStatus("Model offline: " + ((response != null && response.error != null) ? response.error : "step failed"));
            }
            completed = true;
        }, error =>
        {
            bridgeOnline = false;
            pendingSamples.Clear();
            SetRiskWarningActive(false);
            SetRideStatus("Model offline: " + error);
            completed = true;
        });
        while (!completed)
            yield return null;
    }

    IEnumerator FinishBridgeSession()
    {
        SetRideStatus("Finishing DenseFMS session...");
        if (bridgeOnline && sessionStarted)
        {
            float deadline = Time.realtimeSinceStartup + 10f;
            while (pendingSamples.Count > 0 && Time.realtimeSinceStartup < deadline)
                yield return null;

            yield return PostJson("/session/finish", "{}", json =>
            {
                BridgeFinishResponse response = JsonUtility.FromJson<BridgeFinishResponse>(json);
                if (response != null && response.ok)
                    ApplyFinishResponse(response);
            }, error =>
            {
                SetRideStatus("Finish request failed: " + error);
            });
        }

        DenseFmsSummary summary = BuildSummary();
        SaveResults(summary);
        ShowResults(summary);
        finishing = false;
    }

    void SaveLocalResultsIfNeeded(string reason)
    {
        if (resultsSaved || (samples.Count == 0 && string.IsNullOrEmpty(sessionId)))
            return;

        DenseFmsSummary summary = BuildSummary();
        SaveResults(summary);
        Debug.Log("DenseFMS local results saved on " + reason + ": " + summary.output_directory);
    }

    void CaptureSample()
    {
        TryReadHmdMotion(out Vector3 hmdAcceleration, out Vector3 hmdAngularVelocity);

        DenseFmsSample sample = new DenseFmsSample
        {
            step_index = stepIndex++,
            timestamp = Time.time - (manager != null ? manager.StartTime : 0f),
            fms_raw = CurrentFmsValue(),
            acc_x = hmdAcceleration.x,
            acc_y = hmdAcceleration.y,
            acc_z = hmdAcceleration.z,
            angular_velocity_x = hmdAngularVelocity.x,
            angular_velocity_y = hmdAngularVelocity.y,
            angular_velocity_z = hmdAngularVelocity.z,
            has_prediction = false
        };
        samples.Add(sample);
        if ((bridgeOnline || sessionStarting || sessionStarted) && !finishing)
            pendingSamples.Enqueue(sample);

        if (!bridgeOnline)
            SetRideStatus("Model offline: samples " + samples.Count);
    }

    bool TryReadHmdMotion(out Vector3 acceleration, out Vector3 angularVelocity)
    {
        acceleration = Vector3.zero;
        angularVelocity = Vector3.zero;

        if (deriveHeadMotionFromSceneTransform && TryReadHeadTransformMotion(out acceleration, out angularVelocity))
            return true;

        if (useXrVelocityFeatureFallback && TryReadXrFeatureMotion(out acceleration, out angularVelocity))
            return true;

        if (!warnedMissingHeadMotion)
        {
            Debug.LogWarning("DenseFMS could not read HMD motion; sending zero acceleration/angular velocity until a head transform or XR velocity feature is available.");
            warnedMissingHeadMotion = true;
        }
        return false;
    }

    void ResetHeadMotionTracking()
    {
        hasPreviousHeadPose = false;
        hasPreviousHeadLinearVelocity = false;
        hasPreviousXrLinearVelocity = false;
        previousHeadLinearVelocity = Vector3.zero;
        previousXrLinearVelocity = Vector3.zero;
        Transform head = ResolveHeadMotionTransform();
        if (head == null)
            return;

        previousHeadPosition = head.position;
        previousHeadRotation = head.rotation;
        previousHeadSampleTime = Time.time;
        hasPreviousHeadPose = true;
    }

    bool TryReadHeadTransformMotion(out Vector3 acceleration, out Vector3 angularVelocity)
    {
        acceleration = Vector3.zero;
        angularVelocity = Vector3.zero;

        Transform head = ResolveHeadMotionTransform();
        if (head == null)
            return false;

        float now = Time.time;
        Vector3 position = head.position;
        Quaternion rotation = head.rotation;
        if (!hasPreviousHeadPose)
        {
            previousHeadPosition = position;
            previousHeadRotation = rotation;
            previousHeadSampleTime = now;
            hasPreviousHeadPose = true;
            return false;
        }

        float deltaTime = now - previousHeadSampleTime;
        if (deltaTime <= Mathf.Epsilon)
            return false;

        Vector3 linearVelocity = (position - previousHeadPosition) / deltaTime;
        Vector3 angularVelocityDegrees = CalculateAngularVelocityDegrees(previousHeadRotation, rotation, deltaTime);

        previousHeadPosition = position;
        previousHeadRotation = rotation;
        previousHeadSampleTime = now;

        if (!hasPreviousHeadLinearVelocity)
        {
            previousHeadLinearVelocity = linearVelocity;
            hasPreviousHeadLinearVelocity = true;
            return false;
        }

        acceleration = (linearVelocity - previousHeadLinearVelocity) / deltaTime;
        angularVelocity = angularVelocityDegrees;
        previousHeadLinearVelocity = linearVelocity;
        return true;
    }

    Transform ResolveHeadMotionTransform()
    {
        if (headMotionTransform != null && headMotionTransform.gameObject.activeInHierarchy)
            return headMotionTransform;

        if (manager != null && manager._playerHead != null)
        {
            headMotionTransform = manager._playerHead.transform;
            return headMotionTransform;
        }

        Camera mainCamera = Camera.main;
        if (mainCamera != null && mainCamera.isActiveAndEnabled)
        {
            headMotionTransform = mainCamera.transform;
            return headMotionTransform;
        }

        Camera[] cameras = FindObjectsOfType<Camera>(true);
        foreach (Camera candidate in cameras)
        {
            if (candidate != null && candidate.isActiveAndEnabled && candidate.stereoTargetEye != StereoTargetEyeMask.None)
            {
                headMotionTransform = candidate.transform;
                return headMotionTransform;
            }
        }

        return null;
    }

    static Vector3 CalculateAngularVelocityDegrees(Quaternion previousRotation, Quaternion currentRotation, float deltaTime)
    {
        Quaternion deltaRotation = currentRotation * Quaternion.Inverse(previousRotation);
        deltaRotation.ToAngleAxis(out float angleDegrees, out Vector3 axis);
        if (float.IsNaN(axis.x) || axis.sqrMagnitude < 0.000001f)
            return Vector3.zero;
        if (angleDegrees > 180f)
            angleDegrees -= 360f;
        return axis.normalized * (angleDegrees / deltaTime);
    }

    bool TryReadXrFeatureMotion(out Vector3 acceleration, out Vector3 angularVelocity)
    {
        acceleration = Vector3.zero;
        angularVelocity = Vector3.zero;
        Vector3 linearVelocity;
        Vector3 angularVelocityRaw;
        bool gotLinear;
        bool gotAngular;

        XRInputDevice centerEye = XR.InputDevices.GetDeviceAtXRNode(XR.XRNode.CenterEye);
        if (TryReadMotionFeature(centerEye, XR.CommonUsages.centerEyeVelocity, XR.CommonUsages.centerEyeAngularVelocity, out linearVelocity, out angularVelocityRaw, out gotLinear, out gotAngular))
            return ConvertXrMotion(linearVelocity, angularVelocityRaw, gotLinear, gotAngular, out acceleration, out angularVelocity);

        XRInputDevice head = XR.InputDevices.GetDeviceAtXRNode(XR.XRNode.Head);
        if (TryReadMotionFeature(head, XR.CommonUsages.deviceVelocity, XR.CommonUsages.deviceAngularVelocity, out linearVelocity, out angularVelocityRaw, out gotLinear, out gotAngular))
            return ConvertXrMotion(linearVelocity, angularVelocityRaw, gotLinear, gotAngular, out acceleration, out angularVelocity);

        return false;
    }

    bool ConvertXrMotion(Vector3 linearVelocity, Vector3 angularVelocityRadians, bool gotLinear, bool gotAngular, out Vector3 acceleration, out Vector3 angularVelocity)
    {
        acceleration = Vector3.zero;
        angularVelocity = gotAngular ? angularVelocityRadians * Mathf.Rad2Deg : Vector3.zero;

        if (!gotLinear)
            return gotAngular;

        float now = Time.time;
        if (hasPreviousXrLinearVelocity)
        {
            float deltaTime = now - previousXrSampleTime;
            if (deltaTime > Mathf.Epsilon)
                acceleration = (linearVelocity - previousXrLinearVelocity) / deltaTime;
        }

        previousXrLinearVelocity = linearVelocity;
        previousXrSampleTime = now;
        bool hasAcceleration = hasPreviousXrLinearVelocity;
        hasPreviousXrLinearVelocity = true;
        return hasAcceleration || gotAngular;
    }

    static bool TryReadMotionFeature(
        XRInputDevice device,
        XR.InputFeatureUsage<Vector3> linearVelocityUsage,
        XR.InputFeatureUsage<Vector3> angularVelocityUsage,
        out Vector3 linearVelocity,
        out Vector3 angularVelocity,
        out bool gotLinear,
        out bool gotAngular)
    {
        linearVelocity = Vector3.zero;
        angularVelocity = Vector3.zero;
        gotLinear = false;
        gotAngular = false;

        if (!device.isValid)
            return false;

        gotLinear = device.TryGetFeatureValue(linearVelocityUsage, out linearVelocity);
        gotAngular = device.TryGetFeatureValue(angularVelocityUsage, out angularVelocity);
        if (!gotLinear)
            linearVelocity = Vector3.zero;
        if (!gotAngular)
            angularVelocity = Vector3.zero;
        return gotLinear || gotAngular;
    }

    void ApplyStepResponse(BridgeStepResponse response)
    {
        if (response.has_prediction)
        {
            UpdateRiskWarning(response.p_high_risk_20s_thr12);
            for (int i = samples.Count - 1; i >= 0; i--)
            {
                if (samples[i].step_index == response.step_index)
                {
                    DenseFmsSample sample = samples[i];
                    sample.has_prediction = true;
                    sample.predicted_fms_now = response.predicted_fms_now;
                    sample.fms_absolute_error = response.fms_absolute_error;
                    sample.p_high_risk_20s_thr12 = response.p_high_risk_20s_thr12;
                    samples[i] = sample;
                    break;
                }
            }
            DenseFmsPredictionRow row = new DenseFmsPredictionRow
            {
                step_index = response.step_index,
                timestamp = response.timestamp,
                target_fms_now = response.target_fms_now,
                predicted_fms_now = response.predicted_fms_now,
                fms_absolute_error = response.fms_absolute_error,
                p_high_risk_20s_thr12 = response.p_high_risk_20s_thr12
            };
            predictions.Add(row);
            if (showLivePredictionMetrics)
            {
                SetRideStatus(
                    "FMS " + response.target_fms_now.ToString("0", CultureInfo.InvariantCulture) +
                    " pred " + response.predicted_fms_now.ToString("0.0", CultureInfo.InvariantCulture) +
                    " MAE " + RunningMae().ToString("0.00", CultureInfo.InvariantCulture) +
                    " risk12 " + response.p_high_risk_20s_thr12.ToString("0.00", CultureInfo.InvariantCulture)
                );
            }
            else
            {
                SetRideStatus("Predicting. Samples " + response.sample_count);
            }
        }
        else
        {
            SetRideStatus("Calibrating " + response.sample_count + "/" + ExpectedCalibrationSteps + " samples");
        }
    }

    void ApplyFinishResponse(BridgeFinishResponse response)
    {
        if (response.predictions != null && response.predictions.Length > predictions.Count)
        {
            predictions.Clear();
            predictions.AddRange(response.predictions);
        }
    }

    DenseFmsSummary BuildSummary()
    {
        int predictionCount = 0;
        float absSum = 0f;
        float sqSum = 0f;
        float lastTarget = 0f;
        float lastPred = 0f;
        float lastRisk = 0f;
        foreach (DenseFmsSample sample in samples)
        {
            if (!sample.has_prediction)
                continue;
            predictionCount++;
            absSum += sample.fms_absolute_error;
            sqSum += sample.fms_absolute_error * sample.fms_absolute_error;
            lastTarget = sample.fms_raw;
            lastPred = sample.predicted_fms_now;
            lastRisk = sample.p_high_risk_20s_thr12;
        }
        if (predictionCount == 0)
        {
            foreach (DenseFmsPredictionRow row in predictions)
            {
                predictionCount++;
                absSum += row.fms_absolute_error;
                sqSum += row.fms_absolute_error * row.fms_absolute_error;
                lastTarget = row.target_fms_now;
                lastPred = row.predicted_fms_now;
                lastRisk = row.p_high_risk_20s_thr12;
            }
        }
        return new DenseFmsSummary
        {
            session_id = sessionId,
            created_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
            model_online = bridgeOnline,
            calibration_complete = samples.Count >= ExpectedCalibrationSteps,
            sample_count = samples.Count,
            prediction_count = predictionCount,
            mae = predictionCount > 0 ? absSum / predictionCount : 0f,
            rmse = predictionCount > 0 ? Mathf.Sqrt(sqSum / predictionCount) : 0f,
            last_target_fms = lastTarget,
            last_predicted_fms = lastPred,
            last_p_high_risk_20s_thr12 = lastRisk,
            output_directory = outputDirectory
        };
    }

    float RunningMae()
    {
        float sum = 0f;
        int count = 0;
        foreach (DenseFmsSample sample in samples)
        {
            if (!sample.has_prediction)
                continue;
            sum += sample.fms_absolute_error;
            count++;
        }
        return count > 0 ? sum / count : 0f;
    }

    void SaveResults(DenseFmsSummary summary)
    {
        outputDirectory = Path.Combine(
            Application.persistentDataPath,
            "DenseFMSDemo",
            string.IsNullOrEmpty(sessionId) ? DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) : sessionId
        );
        Directory.CreateDirectory(outputDirectory);
        summary.output_directory = outputDirectory;
        File.WriteAllText(Path.Combine(outputDirectory, "samples.csv"), BuildSamplesCsv(), Encoding.UTF8);
        File.WriteAllText(Path.Combine(outputDirectory, "summary.json"), JsonUtility.ToJson(summary, true), Encoding.UTF8);
        resultsSaved = true;
    }

    string BuildSamplesCsv()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("step_index,timestamp,fms_raw,hmd_acc_x,hmd_acc_y,hmd_acc_z,hmd_angular_velocity_deg_s_x,hmd_angular_velocity_deg_s_y,hmd_angular_velocity_deg_s_z,has_prediction,predicted_fms_now,fms_absolute_error,p_high_risk_20s_thr12");
        foreach (DenseFmsSample s in samples)
        {
            sb.Append(s.step_index).Append(',');
            AppendFloat(sb, s.timestamp).Append(',');
            AppendFloat(sb, s.fms_raw).Append(',');
            AppendFloat(sb, s.acc_x).Append(',');
            AppendFloat(sb, s.acc_y).Append(',');
            AppendFloat(sb, s.acc_z).Append(',');
            AppendFloat(sb, s.angular_velocity_x).Append(',');
            AppendFloat(sb, s.angular_velocity_y).Append(',');
            AppendFloat(sb, s.angular_velocity_z).Append(',');
            sb.Append(s.has_prediction ? "true" : "false").Append(',');
            AppendFloat(sb, s.predicted_fms_now).Append(',');
            AppendFloat(sb, s.fms_absolute_error).Append(',');
            AppendFloat(sb, s.p_high_risk_20s_thr12).AppendLine();
        }
        return sb.ToString();
    }

    static StringBuilder AppendFloat(StringBuilder sb, float value)
    {
        return sb.Append(value.ToString("0.######", CultureInfo.InvariantCulture));
    }

    void ShowResults(DenseFmsSummary summary)
    {
        if (fmsToastRoutine != null)
        {
            StopCoroutine(fmsToastRoutine);
            fmsToastRoutine = null;
        }
        if (fmsToastCanvasGroup != null)
            fmsToastCanvasGroup.alpha = 0f;

        CreateUi(true);
        if (preRidePanel != null)
            preRidePanel.SetActive(false);
        if (ridePanel != null)
            ridePanel.SetActive(false);
        if (resultPanel == null || resultText == null || chartImage == null)
        {
            Debug.Log("DenseFMS demo complete. Results saved to " + summary.output_directory);
            return;
        }

        resultPanel.SetActive(true);
        resultText.text =
            "DenseFMS demo complete\n" +
            "Calibration: " + (summary.calibration_complete ? "complete" : "incomplete") + "\n" +
            "Samples: " + summary.sample_count + "  Predictions: " + summary.prediction_count + "\n" +
            "MAE: " + summary.mae.ToString("0.00", CultureInfo.InvariantCulture) +
            "  RMSE: " + summary.rmse.ToString("0.00", CultureInfo.InvariantCulture) + "\n" +
            "Last FMS: " + summary.last_target_fms.ToString("0", CultureInfo.InvariantCulture) +
            "  Pred: " + summary.last_predicted_fms.ToString("0.0", CultureInfo.InvariantCulture) + "\n" +
            "High risk 20s thr12: " + summary.last_p_high_risk_20s_thr12.ToString("0.00", CultureInfo.InvariantCulture) + "\n" +
            "Saved: " + summary.output_directory;
        chartImage.texture = BuildChartTexture();
    }

    Texture2D BuildChartTexture()
    {
        const int width = 640;
        const int height = 220;
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        Color bg = new Color(0.07f, 0.08f, 0.09f, 1f);
        Color grid = new Color(0.22f, 0.24f, 0.26f, 1f);
        Color actual = new Color(0.2f, 0.8f, 1f, 1f);
        Color predicted = new Color(1f, 0.86f, 0.25f, 1f);
        Color[] pixels = new Color[width * height];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = bg;
        tex.SetPixels(pixels);
        for (int y = 20; y < height; y += 45)
            DrawLine(tex, 0, y, width - 1, y, grid);
        for (int x = 0; x < width; x += 80)
            DrawLine(tex, x, 0, x, height - 1, grid);
        DrawSeries(tex, width, height, false, actual);
        DrawSeries(tex, width, height, true, predicted);
        tex.Apply();
        return tex;
    }

    void DrawSeries(Texture2D tex, int width, int height, bool predicted, Color color)
    {
        if (samples.Count < 2)
            return;
        bool hasPrevious = false;
        int prevX = 0;
        int prevY = 0;
        int maxIndex = Mathf.Max(1, samples.Count - 1);
        for (int i = 0; i < samples.Count; i++)
        {
            DenseFmsSample sample = samples[i];
            if (predicted && !sample.has_prediction)
                continue;
            float value = predicted ? sample.predicted_fms_now : sample.fms_raw;
            int x = Mathf.RoundToInt((float)i / maxIndex * (width - 1));
            int y = Mathf.RoundToInt(Mathf.Clamp01(value / 20f) * (height - 1));
            if (hasPrevious)
                DrawLine(tex, prevX, prevY, x, y, color);
            prevX = x;
            prevY = y;
            hasPrevious = true;
        }
    }

    static void DrawLine(Texture2D tex, int x0, int y0, int x1, int y1, Color color)
    {
        int dx = Mathf.Abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -Mathf.Abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true)
        {
            if (x0 >= 0 && x0 < tex.width && y0 >= 0 && y0 < tex.height)
                tex.SetPixel(x0, y0, color);
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    IEnumerator GetHealth(Action<bool> done)
    {
        using (UnityWebRequest request = UnityWebRequest.Get(bridgeUrl + "/health"))
        {
            request.timeout = 2;
            yield return request.SendWebRequest();
            done(request.result == UnityWebRequest.Result.Success);
        }
    }

    IEnumerator PostJson(string path, string json, Action<string> onSuccess, Action<string> onError)
    {
        byte[] body = Encoding.UTF8.GetBytes(json);
        using (UnityWebRequest request = new UnityWebRequest(bridgeUrl + path, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = 10;
            yield return request.SendWebRequest();
            if (request.result == UnityWebRequest.Result.Success)
                onSuccess(request.downloadHandler.text);
            else
                onError(request.error);
        }
    }

    void LaunchSidecar()
    {
        string script = ResolveSidecarScriptPath();
        if (string.IsNullOrEmpty(script) || !File.Exists(script))
        {
            SetBridgeStatus("Sidecar script not found");
            return;
        }
        try
        {
            System.Diagnostics.ProcessStartInfo info = new System.Diagnostics.ProcessStartInfo
            {
                FileName = pythonExecutable,
                Arguments = BuildSidecarArguments(script),
                WorkingDirectory = Directory.Exists(codexRepoPath) ? codexRepoPath : Path.GetDirectoryName(script),
                UseShellExecute = false,
                CreateNoWindow = true
            };
            System.Diagnostics.Process.Start(info);
        }
        catch (Exception ex)
        {
            SetBridgeStatus("Failed to start sidecar: " + ex.Message);
        }
    }

    string ResolveSidecarScriptPath()
    {
        if (!string.IsNullOrWhiteSpace(sidecarScriptPath))
            return sidecarScriptPath;
        string projectTools = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Tools", "unity_realtime_bridge.py"));
        if (File.Exists(projectTools))
            return projectTools;
        string repoTools = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "Tools", "unity_realtime_bridge.py"));
        return repoTools;
    }

    string BuildSidecarArguments(string script)
    {
        StringBuilder args = new StringBuilder();
        args.Append(Quote(script));
        if (!string.IsNullOrWhiteSpace(codexRepoPath))
            args.Append(" --codex_repo ").Append(Quote(codexRepoPath));
        if (!string.IsNullOrWhiteSpace(checkpointPath))
            args.Append(" --checkpoint ").Append(Quote(checkpointPath));
        args.Append(" --host 127.0.0.1 --port 8765 --device cpu");
        return args.ToString();
    }

    static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    float CurrentFmsValue()
    {
        if (manager != null)
            return Mathf.Round(Mathf.Clamp(manager.DenseFmsRawFms, 0f, 20f));
        return Mathf.Round(Mathf.Clamp(rawFms, 0f, 20f));
    }

    string CurrentGender()
    {
        if (manager != null)
            return manager.DenseFmsParticipantGenderName;
        return participantGender == ParticipantGender.Female ? "female" : "male";
    }

    bool ValidatePreRideInputs(bool showMessage)
    {
        if (RuntimeUiEnabled())
            CreateUi();
        bool valid = TryReadStaticInputs(out _, out _, showMessage);
        if (!valid && showMessage)
        {
            if (preRidePanel != null)
                preRidePanel.SetActive(true);
            if (ridePanel != null)
                ridePanel.SetActive(false);
            if (resultPanel != null)
                resultPanel.SetActive(false);
        }
        return valid;
    }

    bool TryReadStaticInputs(out float age, out float mssq, bool showMessage)
    {
        age = manager != null ? manager.DenseFmsParticipantAge : participantAge;
        mssq = manager != null ? manager.DenseFmsParticipantMssq : participantMssq;
        bool ageOk = IsFinite(age) && age > 0f;
        bool mssqOk = IsFinite(mssq) && mssq >= 0f;
        if (ageOk && mssqOk)
        {
            if (showMessage)
                SetBridgeStatus(bridgeOnline ? "Model bridge online" : "Checking model bridge...");
            return true;
        }

        if (showMessage)
        {
            if (!ageOk && !mssqOk)
                SetBridgeStatus("Enter numeric Age and MSSQ before starting.");
            else if (!ageOk)
                SetBridgeStatus("Enter numeric Age before starting.");
            else
                SetBridgeStatus("Enter numeric MSSQ before starting.");
            Debug.LogWarning("DenseFMS inspector inputs are invalid. Age must be greater than 0 and MSSQ must be 0 or greater.");
        }
        return false;
    }

    bool RuntimeUiEnabled()
    {
        return manager != null ? manager.DenseFmsShowRuntimeUi : showRuntimeUi;
    }

    bool ShowLivePredictionMetricsSetting()
    {
        return manager != null ? manager.DenseFmsShowLivePredictionMetrics : showLivePredictionMetrics;
    }

    static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    void SetBridgeStatus(string message)
    {
        if (bridgeStatusText != null)
            bridgeStatusText.text = message;
    }

    void SetRideStatus(string message)
    {
        if (rideStatusText != null)
            rideStatusText.text = message;
    }

    void UpdateRiskWarning(float riskProbability)
    {
        if (!showRiskWarningIcon || !IsFinite(riskProbability))
            return;

        float onThreshold = Mathf.Clamp01(riskWarningOnThreshold);
        float offThreshold = Mathf.Min(Mathf.Clamp01(riskWarningOffThreshold), onThreshold);
        if (!riskWarningActive && riskProbability >= onThreshold)
            SetRiskWarningActive(true);
        else if (riskWarningActive && riskProbability <= offThreshold)
            SetRiskWarningActive(false);
    }

    void SetRiskWarningActive(bool active)
    {
        riskWarningActive = active;
        if (active && showRiskWarningIcon && riskWarningCanvas == null)
            CreateRiskWarningHud();

        bool visible = showRiskWarningIcon && active;
        if (riskWarningCanvas != null)
        {
            riskWarningCanvas.gameObject.SetActive(visible);
            riskWarningCanvas.enabled = visible;
        }
        if (riskWarningIconImage != null)
            riskWarningIconImage.enabled = visible;
    }

    string InspectorValueSummary()
    {
        float age;
        float mssq;
        TryReadStaticInputs(out age, out mssq, false);
        return "Inspector inputs  Age " + age.ToString("0.##", CultureInfo.InvariantCulture) +
            "  MSSQ " + mssq.ToString("0.##", CultureInfo.InvariantCulture) +
            "  Gender " + CurrentGender();
    }

    void CreateUi(bool force = false)
    {
        if (!force && !RuntimeUiEnabled())
            return;
        if (uiCreated)
            return;
        uiCreated = true;
        Font font = Resources.GetBuiltinResource<Font>("Arial.ttf");

        GameObject canvasObject = new GameObject("DenseFMS Realtime Canvas");
        canvasObject.transform.SetParent(transform, false);
        canvas = canvasObject.AddComponent<Canvas>();
        canvas.renderMode = useHeadLockedVrCanvas ? RenderMode.WorldSpace : RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 2000;
        canvasRect = canvasObject.GetComponent<RectTransform>();
        canvasRect.sizeDelta = new Vector2(1280, 720);
        PositionVrCanvas();
        CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1280, 720);
        canvasObject.AddComponent<GraphicRaycaster>();

        preRidePanel = CreatePanel("DenseFMS PreRide", canvas.transform, new Vector2(12, -12), new Vector2(420, 156), TextAnchor.UpperLeft);
        CreateText("Title", preRidePanel.transform, "DenseFMS realtime demo", font, 18, new Vector2(16, -14), new Vector2(380, 26), TextAnchor.MiddleLeft);
        bridgeStatusText = CreateText("BridgeStatus", preRidePanel.transform, "Checking model bridge...", font, 13, new Vector2(16, -42), new Vector2(388, 22), TextAnchor.MiddleLeft);
        CreateText("InspectorValues", preRidePanel.transform, InspectorValueSummary(), font, 13, new Vector2(16, -72), new Vector2(388, 42), TextAnchor.UpperLeft);
        CreateText("FmsLabel", preRidePanel.transform, "FMS 0-20", font, 13, new Vector2(16, -120), new Vector2(90, 24), TextAnchor.MiddleLeft);
        fmsValueText = CreateValueDisplay("FmsValue", preRidePanel.transform, CurrentFmsValue().ToString("0", CultureInfo.InvariantCulture), font, new Vector2(110, -118), new Vector2(82, 26));

        ridePanel = CreatePanel("DenseFMS Ride", canvas.transform, new Vector2(12, -12), new Vector2(380, 118), TextAnchor.UpperRight);
        ridePanel.GetComponent<RectTransform>().anchorMin = new Vector2(1, 1);
        ridePanel.GetComponent<RectTransform>().anchorMax = new Vector2(1, 1);
        ridePanel.GetComponent<RectTransform>().pivot = new Vector2(1, 1);
        ridePanel.GetComponent<RectTransform>().anchoredPosition = new Vector2(-12, -12);
        CreateText("RideTitle", ridePanel.transform, "DenseFMS live", font, 18, new Vector2(16, -12), new Vector2(340, 28), TextAnchor.MiddleLeft);
        rideStatusText = CreateText("RideStatus", ridePanel.transform, "Waiting for ride start", font, 14, new Vector2(16, -44), new Vector2(348, 54), TextAnchor.UpperLeft);
        ridePanel.SetActive(false);

        resultPanel = CreatePanel("DenseFMS Result", canvas.transform, Vector2.zero, new Vector2(760, 470), TextAnchor.MiddleCenter);
        CreateText("ResultTitle", resultPanel.transform, "DenseFMS results", font, 20, new Vector2(18, -16), new Vector2(700, 30), TextAnchor.MiddleLeft);
        resultText = CreateText("ResultText", resultPanel.transform, "", font, 14, new Vector2(18, -52), new Vector2(700, 132), TextAnchor.UpperLeft);
        GameObject chartObject = CreateUiObject("Chart", resultPanel.transform, new Vector2(18, -198), new Vector2(720, 248));
        chartImage = chartObject.AddComponent<RawImage>();
        chartImage.color = Color.white;
        resultPanel.SetActive(false);

        GameObject toastObject = CreateUiObject("FmsToast", canvas.transform, Vector2.zero, new Vector2(700, 220));
        RectTransform toastRect = toastObject.GetComponent<RectTransform>();
        toastRect.anchorMin = new Vector2(0.5f, 0.5f);
        toastRect.anchorMax = new Vector2(0.5f, 0.5f);
        toastRect.pivot = new Vector2(0.5f, 0.5f);
        fmsToastCanvasGroup = toastObject.AddComponent<CanvasGroup>();
        fmsToastCanvasGroup.alpha = 0f;

        fmsToastText = CreateText("FmsToastText", toastObject.transform, "", font, 42, new Vector2(24, -74), new Vector2(440, 90), TextAnchor.MiddleLeft);
        Shadow toastShadow = fmsToastText.gameObject.AddComponent<Shadow>();
        toastShadow.effectColor = new Color(0f, 0f, 0f, 0.85f);
        toastShadow.effectDistance = new Vector2(2f, -2f);

        GameObject toastImageObject = CreateUiObject("FmsToastImage", toastObject.transform, new Vector2(500, -20), new Vector2(180, 180));
        fmsToastImage = toastImageObject.AddComponent<RawImage>();
        fmsToastImage.color = Color.white;
        fmsToastImage.enabled = false;
    }

    void CreateFmsHud()
    {
        if (!showFmsRuntimeHud || fmsHudCanvas != null)
            return;

        Font font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        GameObject canvasObject = new GameObject("DenseFMS FMS Toast");
        canvasObject.transform.SetParent(transform, false);
        fmsHudCanvas = canvasObject.AddComponent<Canvas>();
        fmsHudCanvas.renderMode = RenderMode.WorldSpace;
        fmsHudCanvas.sortingOrder = 2100;
        fmsHudCanvasRect = canvasObject.GetComponent<RectTransform>();
        fmsHudCanvasRect.sizeDelta = new Vector2(700, 220);

        CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 12f;

        fmsToastCanvasGroup = canvasObject.AddComponent<CanvasGroup>();
        fmsToastCanvasGroup.alpha = 0f;

        GameObject textObject = new GameObject("FmsToastText");
        textObject.transform.SetParent(canvasObject.transform, false);
        RectTransform textRect = textObject.AddComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0f, 1f);
        textRect.anchorMax = new Vector2(0f, 1f);
        textRect.pivot = new Vector2(0f, 1f);
        textRect.anchoredPosition = new Vector2(24, -74);
        textRect.sizeDelta = new Vector2(440, 90);
        fmsToastText = textObject.AddComponent<Text>();
        fmsToastText.font = font;
        fmsToastText.fontSize = 42;
        fmsToastText.color = Color.white;
        fmsToastText.alignment = TextAnchor.MiddleLeft;
        fmsToastText.horizontalOverflow = HorizontalWrapMode.Wrap;
        fmsToastText.verticalOverflow = VerticalWrapMode.Overflow;
        Shadow toastShadow = fmsToastText.gameObject.AddComponent<Shadow>();
        toastShadow.effectColor = new Color(0f, 0f, 0f, 0.85f);
        toastShadow.effectDistance = new Vector2(2f, -2f);

        GameObject imageObject = new GameObject("FmsToastImage");
        imageObject.transform.SetParent(canvasObject.transform, false);
        RectTransform imageRect = imageObject.AddComponent<RectTransform>();
        imageRect.anchorMin = new Vector2(0f, 1f);
        imageRect.anchorMax = new Vector2(0f, 1f);
        imageRect.pivot = new Vector2(0f, 1f);
        imageRect.anchoredPosition = new Vector2(500, -20);
        imageRect.sizeDelta = new Vector2(180, 180);
        fmsToastImage = imageObject.AddComponent<RawImage>();
        fmsToastImage.color = Color.white;
        fmsToastImage.enabled = false;

        PositionFmsHudCanvas();
    }

    void CreateRiskWarningHud()
    {
        if (!showRiskWarningIcon || riskWarningCanvas != null)
            return;

        GameObject canvasObject = new GameObject("DenseFMS Risk Warning Canvas");
        canvasObject.transform.SetParent(transform, false);
        riskWarningCanvas = canvasObject.AddComponent<Canvas>();
        riskWarningCanvas.renderMode = useHeadLockedVrCanvas ? RenderMode.WorldSpace : RenderMode.ScreenSpaceOverlay;
        riskWarningCanvas.sortingOrder = 2200;
        riskWarningCanvasRect = canvasObject.GetComponent<RectTransform>();
        riskWarningCanvasRect.sizeDelta = new Vector2(1280, 720);

        CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1280, 720);
        scaler.dynamicPixelsPerUnit = 12f;

        GameObject iconObject = new GameObject("RiskWarningIcon");
        iconObject.transform.SetParent(canvasObject.transform, false);
        RectTransform iconRect = iconObject.AddComponent<RectTransform>();
        iconRect.anchorMin = new Vector2(0f, 1f);
        iconRect.anchorMax = new Vector2(0f, 1f);
        iconRect.pivot = new Vector2(0f, 1f);
        iconRect.anchoredPosition = new Vector2(28f, -28f);
        iconRect.sizeDelta = new Vector2(96f, 96f);

        riskWarningIconImage = iconObject.AddComponent<RawImage>();
        riskWarningIconImage.texture = BuildRiskWarningIconTexture();
        riskWarningIconImage.color = Color.white;
        riskWarningIconImage.enabled = false;

        Shadow iconShadow = iconObject.AddComponent<Shadow>();
        iconShadow.effectColor = new Color(0f, 0f, 0f, 0.55f);
        iconShadow.effectDistance = new Vector2(2f, -2f);

        PositionRiskWarningCanvas();
        riskWarningCanvas.enabled = false;
        canvasObject.SetActive(false);
    }

    void PositionRiskWarningCanvas()
    {
        if (!showRiskWarningIcon || riskWarningCanvas == null)
            return;

        if (!useHeadLockedVrCanvas)
            return;

        Camera targetCamera = ResolveUiCamera();
        if (targetCamera == null)
            return;

        riskWarningCanvas.worldCamera = targetCamera;
        Transform canvasTransform = riskWarningCanvas.transform;
        if (canvasTransform.parent != targetCamera.transform)
            canvasTransform.SetParent(targetCamera.transform, false);

        canvasTransform.localPosition = new Vector3(0f, 0f, vrUiDistance);
        canvasTransform.localRotation = Quaternion.identity;
        canvasTransform.localScale = Vector3.one * vrUiScale;
        if (riskWarningCanvasRect != null)
            riskWarningCanvasRect.sizeDelta = new Vector2(1280, 720);
    }

    Texture2D BuildRiskWarningIconTexture()
    {
        if (riskWarningIconTexture != null)
            return riskWarningIconTexture;

        const int size = 128;
        Color32 clear = new Color32(255, 255, 255, 0);
        Color32 fill = new Color32(255, 205, 42, 255);
        Color32 border = new Color32(20, 20, 20, 255);
        Color32 mark = new Color32(20, 20, 20, 255);
        Color32[] pixels = new Color32[size * size];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = clear;

        Vector2 a = new Vector2(64f, 116f);
        Vector2 b = new Vector2(12f, 12f);
        Vector2 c = new Vector2(116f, 12f);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                Vector2 p = new Vector2(x + 0.5f, y + 0.5f);
                if (!PointInTriangle(p, a, b, c))
                    continue;

                float edgeDistance = Mathf.Min(
                    DistanceToSegment(p, a, b),
                    Mathf.Min(DistanceToSegment(p, b, c), DistanceToSegment(p, c, a))
                );
                Color32 color = edgeDistance <= 4.0f ? border : fill;
                if ((x >= 59 && x <= 69 && y >= 43 && y <= 84) || Vector2.Distance(p, new Vector2(64f, 29f)) <= 6.0f)
                    color = mark;
                pixels[y * size + x] = color;
            }
        }

        riskWarningIconTexture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        riskWarningIconTexture.SetPixels32(pixels);
        riskWarningIconTexture.wrapMode = TextureWrapMode.Clamp;
        riskWarningIconTexture.filterMode = FilterMode.Bilinear;
        riskWarningIconTexture.Apply(false, true);
        return riskWarningIconTexture;
    }

    static bool PointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
    {
        float d1 = Sign(p, a, b);
        float d2 = Sign(p, b, c);
        float d3 = Sign(p, c, a);
        bool hasNegative = d1 < 0f || d2 < 0f || d3 < 0f;
        bool hasPositive = d1 > 0f || d2 > 0f || d3 > 0f;
        return !(hasNegative && hasPositive);
    }

    static float Sign(Vector2 p1, Vector2 p2, Vector2 p3)
    {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }

    static float DistanceToSegment(Vector2 point, Vector2 a, Vector2 b)
    {
        Vector2 segment = b - a;
        float lengthSquared = segment.sqrMagnitude;
        if (lengthSquared <= Mathf.Epsilon)
            return Vector2.Distance(point, a);

        float t = Mathf.Clamp01(Vector2.Dot(point - a, segment) / lengthSquared);
        return Vector2.Distance(point, a + segment * t);
    }

    void PositionFmsHudCanvas()
    {
        if (!showFmsRuntimeHud || fmsHudCanvas == null)
            return;

        if (!fmsHudCanvas.gameObject.activeSelf)
            fmsHudCanvas.gameObject.SetActive(true);
        fmsHudCanvas.enabled = true;

        Camera targetCamera = ResolveUiCamera();
        if (targetCamera == null)
            return;

        fmsHudCanvas.worldCamera = targetCamera;
        Transform canvasTransform = fmsHudCanvas.transform;
        if (canvasTransform.parent != targetCamera.transform)
            canvasTransform.SetParent(targetCamera.transform, false);

        canvasTransform.localPosition = fmsHudLocalPosition;
        canvasTransform.localRotation = Quaternion.identity;
        canvasTransform.localScale = Vector3.one * fmsHudScale;
        if (fmsHudCanvasRect != null)
            fmsHudCanvasRect.sizeDelta = new Vector2(700, 220);
    }

    void PositionVrCanvas()
    {
        if (!useHeadLockedVrCanvas || canvas == null)
            return;

        Camera targetCamera = ResolveUiCamera();
        if (targetCamera == null)
            return;

        canvas.worldCamera = targetCamera;
        Transform canvasTransform = canvas.transform;
        if (canvasTransform.parent != targetCamera.transform)
            canvasTransform.SetParent(targetCamera.transform, false);

        canvasTransform.localPosition = new Vector3(0f, 0f, vrUiDistance);
        canvasTransform.localRotation = Quaternion.identity;
        canvasTransform.localScale = Vector3.one * vrUiScale;
        if (canvasRect != null)
            canvasRect.sizeDelta = new Vector2(1280, 720);
    }

    Camera ResolveUiCamera()
    {
        if (uiCamera != null && uiCamera.isActiveAndEnabled)
            return uiCamera;

        Camera mainCamera = Camera.main;
        if (mainCamera != null && mainCamera.isActiveAndEnabled)
        {
            uiCamera = mainCamera;
            return uiCamera;
        }

        Camera[] cameras = FindObjectsOfType<Camera>(true);
        foreach (Camera candidate in cameras)
        {
            if (candidate != null && candidate.isActiveAndEnabled && candidate.stereoTargetEye != StereoTargetEyeMask.None)
            {
                uiCamera = candidate;
                return uiCamera;
            }
        }

        foreach (Camera candidate in cameras)
        {
            if (candidate != null && candidate.isActiveAndEnabled)
            {
                uiCamera = candidate;
                return uiCamera;
            }
        }

        return null;
    }

    GameObject CreatePanel(string name, Transform parent, Vector2 anchoredPosition, Vector2 size, TextAnchor anchor)
    {
        GameObject panel = CreateUiObject(name, parent, anchoredPosition, size);
        Image image = panel.AddComponent<Image>();
        image.color = new Color(0.02f, 0.025f, 0.03f, 0.86f);
        RectTransform rect = panel.GetComponent<RectTransform>();
        if (anchor == TextAnchor.MiddleCenter)
        {
            rect.anchorMin = new Vector2(0.5f, 0.5f);
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.pivot = new Vector2(0.5f, 0.5f);
        }
        return panel;
    }

    GameObject CreateUiObject(string name, Transform parent, Vector2 anchoredPosition, Vector2 size)
    {
        GameObject obj = new GameObject(name);
        obj.transform.SetParent(parent, false);
        RectTransform rect = obj.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.pivot = new Vector2(0, 1);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
        return obj;
    }

    Text CreateText(string name, Transform parent, string text, Font font, int size, Vector2 anchoredPosition, Vector2 rectSize, TextAnchor alignment)
    {
        GameObject obj = CreateUiObject(name, parent, anchoredPosition, rectSize);
        Text uiText = obj.AddComponent<Text>();
        uiText.font = font;
        uiText.fontSize = size;
        uiText.color = Color.white;
        uiText.alignment = alignment;
        uiText.horizontalOverflow = HorizontalWrapMode.Wrap;
        uiText.verticalOverflow = VerticalWrapMode.Overflow;
        uiText.text = text;
        return uiText;
    }

    Text CreateValueDisplay(string name, Transform parent, string text, Font font, Vector2 anchoredPosition, Vector2 rectSize)
    {
        GameObject root = CreateUiObject(name, parent, anchoredPosition, rectSize);
        Image image = root.AddComponent<Image>();
        image.color = new Color(1f, 1f, 1f, 0.14f);
        Text valueText = CreateText("Text", root.transform, text, font, 15, new Vector2(0, 0), rectSize, TextAnchor.MiddleCenter);
        valueText.color = Color.white;
        return valueText;
    }

    [Serializable]
    class BridgeStartRequest
    {
        public string session_id;
        public float age;
        public float mssq;
        public string gender;
    }

    [Serializable]
    class BridgeStartResponse
    {
        public bool ok;
        public string status;
        public string error;
    }

    [Serializable]
    class BridgeStepRequest
    {
        public int step_index;
        public float timestamp;
        public float acc_x;
        public float acc_y;
        public float acc_z;
        public float linear_velocity_x;
        public float linear_velocity_y;
        public float linear_velocity_z;
        public float angular_velocity_x;
        public float angular_velocity_y;
        public float angular_velocity_z;
        public float fms_raw;
    }

    [Serializable]
    class BridgeStepResponse
    {
        public bool ok;
        public string status;
        public string error;
        public int sample_count;
        public int remaining_steps;
        public bool calibration_complete;
        public bool has_prediction;
        public int prediction_index;
        public int step_index;
        public float timestamp;
        public float target_fms_now;
        public float predicted_fms_now;
        public float fms_absolute_error;
        public float p_high_risk_20s_thr12;
    }

    [Serializable]
    class BridgeFinishResponse
    {
        public bool ok;
        public string status;
        public string error;
        public int sample_count;
        public int prediction_count;
        public bool calibration_complete;
        public bool has_metrics;
        public float mae;
        public float rmse;
        public float last_target_fms;
        public float last_predicted_fms;
        public float last_p_high_risk_20s_thr12;
        public DenseFmsPredictionRow[] predictions;
    }

    [Serializable]
    struct DenseFmsSample
    {
        public int step_index;
        public float timestamp;
        public float fms_raw;
        public float acc_x;
        public float acc_y;
        public float acc_z;
        public float angular_velocity_x;
        public float angular_velocity_y;
        public float angular_velocity_z;
        public bool has_prediction;
        public float predicted_fms_now;
        public float fms_absolute_error;
        public float p_high_risk_20s_thr12;
    }

    [Serializable]
    public struct DenseFmsPredictionRow
    {
        public int step_index;
        public float timestamp;
        public float target_fms_now;
        public float predicted_fms_now;
        public float fms_absolute_error;
        public float p_high_risk_20s_thr12;
    }

    [Serializable]
    class DenseFmsSummary
    {
        public string session_id;
        public string created_at;
        public bool model_online;
        public bool calibration_complete;
        public int sample_count;
        public int prediction_count;
        public float mae;
        public float rmse;
        public float last_target_fms;
        public float last_predicted_fms;
        public float last_p_high_risk_20s_thr12;
        public string output_directory;
    }
}
