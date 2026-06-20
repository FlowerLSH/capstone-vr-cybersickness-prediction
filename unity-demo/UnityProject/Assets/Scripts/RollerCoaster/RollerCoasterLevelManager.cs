using System.Collections;
using System.Collections.Generic;
using System;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class RollerCoasterLevelManager : LevelManager
{
    public enum DenseFmsParticipantGender
    {
        Male,
        Female
    }

    [SerializeField]
    ButtonSequence sequenceSigns; //sequence
    [SerializeField]
    int _laps = 6; //total number of laps
    [SerializeField]
    bool useDenseFmsDemoRideSettings = true;
    [SerializeField, Min(1)]
    int denseFmsDemoLaps = 2;
    [SerializeField]
    float denseFmsDemoSpeedMultiplier = 1.1428572f; // 4:00 for two laps -> about 3:30
    [SerializeField]
    bool hideLegacySceneUiForDenseFmsDemo = true;
    [SerializeField]
    bool hideLegacyTutorialUiForDenseFmsDemo = true;
    [SerializeField]
    bool hideAllLegacyCanvasesForDenseFmsDemo = true;
    [Header("DenseFMS Inspector Inputs")]
    [SerializeField, Min(0.01f)]
    float denseFmsParticipantAge = 20f;
    [SerializeField, Min(0f)]
    float denseFmsParticipantMssq = 0f;
    [SerializeField]
    DenseFmsParticipantGender denseFmsParticipantGender = DenseFmsParticipantGender.Male;
    [SerializeField, Range(0f, 20f)]
    float denseFmsRawFms = 0f;
    [SerializeField]
    bool denseFmsShowLivePredictionMetrics = true;
    [SerializeField]
    bool denseFmsShowRuntimeUi = false;
    int _currLap = 0; //number of completed laps
    internal bool isLastLap = false;
    [SerializeField]
    PathCreation.Examples.PathFollower pathFollower;
    [SerializeField]
    PathCreation.PathCreator pathCreator; //curve
    [SerializeField]
    AudioSource RC_AudioSource;
    [SerializeField]
    internal Text _time;
    [SerializeField]
    internal Text _lapNum;
    [SerializeField]
    internal Text _totalLaps;
    [SerializeField]
    internal TextMeshPro _startSignText;
    [SerializeField]
    internal Text _speedText;
    [SerializeField]
    internal Text _errorsText;
    [SerializeField]
    internal int _surpassedSigns = 0;
    int _errors = 0;
    [SerializeField]
    private float minLapTime = 60f, maxLapTime = 150f; //minimum and maximum times for a lap (TESTED)
    [SerializeField]
    GameObject tutorialCanvas;
    [SerializeField]
    Text tutorialText;
    [SerializeField]
    AudioClip countSound, goSound;
    [SerializeField]
    AudioSource startSound;
    [SerializeField]
    internal GameObject cartBase;
    [SerializeField]
    internal ButtonSequence signs;
    [SerializeField]
    HideMeshes _hideMeshes;

    internal float DenseFmsParticipantAge
    {
        get { return denseFmsParticipantAge; }
    }

    internal float DenseFmsParticipantMssq
    {
        get { return denseFmsParticipantMssq; }
    }

    internal string DenseFmsParticipantGenderName
    {
        get { return denseFmsParticipantGender == DenseFmsParticipantGender.Female ? "female" : "male"; }
    }

    internal float DenseFmsRawFms
    {
        get { return denseFmsRawFms; }
        set { denseFmsRawFms = Mathf.Clamp(value, 0f, 20f); }
    }

    internal bool DenseFmsShowLivePredictionMetrics
    {
        get { return denseFmsShowLivePredictionMetrics; }
    }

    internal bool DenseFmsShowRuntimeUi
    {
        get { return denseFmsShowRuntimeUi; }
    }

    public float maxSpeed = 30f; //maximum carriage speed
    public float minSpeed = 15f; //minimum carriage speed
    public float speedStep = 2f; //speed increase/decrease step
    private float maxSpeedComplTime, minSpeedComplTime; //completion times at max and min speeds for all laps in total
    bool denseFmsDemoRideSettingsApplied;

    protected override bool SuppressLegacySceneUi
    {
        get { return hideLegacySceneUiForDenseFmsDemo; }
    }

    protected override bool SuppressLegacyTutorialUi
    {
        get { return hideLegacyTutorialUiForDenseFmsDemo; }
    }

    internal override void Awake()
    {
        base.Awake();
        ApplyDenseFmsDemoRideSettings();
        ApplyDemoLegacyUiSuppression();
        if (pathFollower != null)
        {
            pathFollower.Speed = minSpeed;
            pathFollower.enabled = false;
        }

        if (sequenceSigns != null)
        {
            sequenceSigns.OnErrorCommitted.AddListener(ErrorCommitted);
            sequenceSigns.OnSignSurpassed.AddListener(SignSurpassed);
        }
    }

    void ApplyDenseFmsDemoRideSettings()
    {
        if (!useDenseFmsDemoRideSettings || denseFmsDemoRideSettingsApplied)
            return;

        _laps = Mathf.Max(1, denseFmsDemoLaps);
        float multiplier = Mathf.Max(0.01f, denseFmsDemoSpeedMultiplier);
        minSpeed *= multiplier;
        maxSpeed *= multiplier;
        speedStep *= multiplier;
        minLapTime /= multiplier;
        maxLapTime /= multiplier;
        denseFmsDemoRideSettingsApplied = true;
    }

    private void SignSurpassed()
    {
        _surpassedSigns++;
    }

    private void ErrorCommitted()
    {
        _errors++;
        SetUIText(_errorsText, _errors.ToString());
    }

    private void AppendTutorialLapText()
    {
        if (!hideLegacyTutorialUiForDenseFmsDemo && tutorialText != null)
            tutorialText.text += " " + _laps.ToString("0");
    }

    private void SetTutorialCanvasActive(bool active)
    {
        if (hideLegacyTutorialUiForDenseFmsDemo)
            active = false;

        if (tutorialCanvas != null)
            tutorialCanvas.SetActive(active);
    }

    private void SetPressToStartTextActive(bool active)
    {
        if (hideLegacyTutorialUiForDenseFmsDemo)
            active = false;

        if (tutorialCanvas == null)
            return;

        Transform pressToStartText = tutorialCanvas.transform.Find("PressToStartText");
        if (pressToStartText != null)
            pressToStartText.gameObject.SetActive(active);
    }

    private void SetUIText(Text target, string value)
    {
        if (target != null)
            target.text = value;
    }

    private void SetStartSignText(string value, float fontSize = -1f)
    {
        if (_startSignText == null)
            return;

        if (fontSize > 0f)
            _startSignText.fontSize = fontSize;
        _startSignText.text = value;
    }

    private void PlayStartSound(AudioClip clip)
    {
        if (startSound == null || clip == null)
            return;

        startSound.clip = clip;
        startSound.Play();
    }

    private void StopCoasterAudio()
    {
        if (RC_AudioSource == null)
            return;

        RC_AudioSource.pitch = 1f;
        RC_AudioSource.Stop();
    }

    private void PlayCoasterAudio()
    {
        if (RC_AudioSource == null)
            return;

        RC_AudioSource.pitch = 1f;
        RC_AudioSource.Play();
    }

    private void ApplyDemoLegacyUiSuppression()
    {
        ApplyLegacyUiSuppression();
        SetTutorialCanvasActive(false);
        HideNamedLegacyTutorialObjects();
        HideLegacyCanvasesInScene();
    }

    private void HideLegacyCanvasesInScene()
    {
        if (!hideAllLegacyCanvasesForDenseFmsDemo)
            return;

        Canvas[] canvases = FindObjectsOfType<Canvas>(true);
        foreach (Canvas legacyCanvas in canvases)
        {
            if (legacyCanvas == null || IsDenseFmsCanvas(legacyCanvas))
                continue;
            if (legacyCanvas.gameObject.scene != gameObject.scene)
                continue;

            legacyCanvas.enabled = false;
            legacyCanvas.gameObject.SetActive(false);
        }
    }

    private bool IsDenseFmsCanvas(Canvas targetCanvas)
    {
        return targetCanvas != null && targetCanvas.gameObject.name.StartsWith("DenseFMS", StringComparison.Ordinal);
    }

    private void HideNamedLegacyTutorialObjects()
    {
        if (!hideLegacyTutorialUiForDenseFmsDemo || !gameObject.scene.IsValid())
            return;

        GameObject[] roots = gameObject.scene.GetRootGameObjects();
        foreach (GameObject root in roots)
            HideNamedLegacyTutorialObjectsRecursive(root.transform);
    }

    private void HideNamedLegacyTutorialObjectsRecursive(Transform target)
    {
        if (target == null || target.gameObject.name == "DenseFMS Realtime Canvas")
            return;

        if (target.gameObject != gameObject && ContainsLegacyTutorialName(target.gameObject.name))
        {
            target.gameObject.SetActive(false);
            return;
        }

        for (int i = 0; i < target.childCount; i++)
            HideNamedLegacyTutorialObjectsRecursive(target.GetChild(i));
    }

    private bool ContainsLegacyTutorialName(string objectName)
    {
        return objectName.IndexOf("Tutorial", StringComparison.OrdinalIgnoreCase) >= 0
            || objectName.IndexOf("ControllerInputsTutorial", StringComparison.OrdinalIgnoreCase) >= 0
            || objectName.IndexOf("PressToStartText", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    internal override void Start()
    {
        base.Start();
        ApplyDemoLegacyUiSuppression();

        if (pathFollower != null)
        {
            pathFollower.OnLapCompleted.AddListener(LapCompleted);
            pathFollower.updatePosition(0);
        }

        float circuitLen = pathCreator != null && pathCreator.path != null ? pathCreator.path.length : 0f;
        AppendTutorialLapText();
        SetTutorialCanvasActive(true);

        SetPressToStartTextActive(false);
        ApplyDemoLegacyUiSuppression();

        maxSpeedComplTime = minLapTime * _laps;
        minSpeedComplTime = maxLapTime * _laps;

        float length = circuitLen;
        length = circuitLen * _laps;
        Debug.Log("One lap: " + circuitLen + " All laps: " + length);
    }
    internal override void Update()
    {
        base.Update();

        if (!_started && !_ended)
        {
            bool spacePressed = Keyboard.current[Key.Space].wasPressedThisFrame;
            bool debugStartPressed = Keyboard.current[Key.LeftCtrl].isPressed && Keyboard.current[Key.I].wasPressedThisFrame;
            bool triggerPressed = _input != null && (_input.IsLeftTriggerClickedDown || _input.IsRightTriggerClickedDown);

            if (!_canStart && (debugStartPressed || spacePressed))
            {
                if (!DenseFMSRealtimeDemo.CanStartRide(this))
                    return;
                StartCoroutine(StartRide());
                if (spacePressed)
                    _go = true;
            }
            else if (_canStart && (triggerPressed || spacePressed))
            {
                if (DenseFMSRealtimeDemo.CanStartRide(this))
                    _go = true;
            }
        }
        else if (_started && !_ended)
        {

            SetUIText(_time, FloatTimeToString(Time.time - _startTime));
            if (pathFollower != null)
                SetUIText(_speedText, pathFollower.correctSpeed.ToString("0.00"));

        }

    }

    public void LapCompleted(int lap)
    {
        if (_started && !_ended)
        {
            _currLap = lap;
            //if the experience is finished, finish the execution by collecting the results
            if (lap >= _laps)
            {
                StopCoasterAudio();
                if (pathFollower != null)
                {
                    pathFollower.resetPath(true, false);
                    pathFollower.enabled = false;
                }
                if (_hideMeshes != null)
                    _hideMeshes.Show();
                EndGame(5);
            }
            else
            {
                SetUIText(_lapNum, (lap + 1).ToString());
                if (lap == _laps - 1)
                {
                    isLastLap = true; //last lap
                    SetStartSignText("FINISH");
                }
                else if (lap >= 0 && lap < _laps - 1)
                {
                    ShowLapText();
                }

                if (GameManager.Instance != null)
                    GameManager.Instance.ResultToFile(GetResultString(GetResults(Time.time - _startTime), GameManager.Instance._csvSeparator));         
            }
        }
    }
    internal override void Reset()
    {
        if (pathFollower != null)
        {
            pathFollower.Speed = minSpeed;
            pathFollower.resetPath(true);
            pathFollower.enabled = true;
            pathFollower.startTime = Time.time;
            pathFollower.updatePosition(0);
        }
        _currLap = 0;
        SetTutorialCanvasActive(true);
        SetPressToStartTextActive(false);
        SetUIText(_time, "00:00");
        SetUIText(_lapNum, "0");
        SetUIText(_errorsText, "0");
        SetUIText(_speedText, "0");
        _surpassedSigns = 0;
        StopCoasterAudio();
        if (sequenceSigns != null)
            sequenceSigns.Reset();
        SetStartSignText("START", 6f);
        _errors = 0;
        if (_hideMeshes != null)
            _hideMeshes.Show();
        base.Reset();
        ApplyDemoLegacyUiSuppression();

    }
    
    private IEnumerator StartRide()
    {
        //tutorialCanvas.gameObject.SetActive(true);
        SetPressToStartTextActive(true);
        ApplyDemoLegacyUiSuppression();
        _canStart = true;
        yield return new WaitWhile(() => _go == false);
        SetStartSignText("START", 6f);
        StartGame();
    }
    
    public override void StartGame()
    {
        if (!_started)
        {
            base.StartGame();
            DenseFMSRealtimeDemo.NotifyRideStarted(this);
            SetPressToStartTextActive(false);

            SetTutorialCanvasActive(false);
            ApplyDemoLegacyUiSuppression();
            SetUIText(_totalLaps, _laps.ToString());

            if (sequenceSigns != null)
                sequenceSigns.activateFirstSign();
            SetStartSignText("START", 6f);

            PlayCoasterAudio();
            if (pathFollower != null)
            {
                pathFollower.updatePosition(0);
                pathFollower.resetPath(false);
                pathFollower.enabled = true;
                pathFollower.startTime = Time.time;
            }
            if (_hideMeshes != null)
                _hideMeshes.Hide();
            Invoke(nameof(ShowLapText), 1);
        }
    }

    public override void EndGame(float loadAfter = 0)
    {
        DenseFMSRealtimeDemo.NotifyRideEnding(this);
        base.EndGame(Mathf.Max(loadAfter, 60f));
    }

    void ShowLapText()
    {

        SetStartSignText("LAP " + (_currLap+1).ToString());
    }

    //NB: CSV header names must be specified in the levemanager gameobjects
    //from the editor
    internal override List<string> GetResults(float time)
    {
        var res = base.GetResults(time);
        res.Add((_currLap).ToString());
        res.Add(GetAVGSpeed(time).ToString("0.00"));
        return res;
    }

    internal override float GetOperationSpeed(float time)
    {
        if (pathFollower == null)
            return -1f;

        float os = -1f;
        float clampedTime;
       
        float perc = pathFollower.getCircuitLenPercentage();
        maxSpeedComplTime = minLapTime * (_currLap + perc); //laps already completed plus percentage of the last partial lap completed at the stop
        minSpeedComplTime = maxLapTime * (_currLap + perc);
        //time clamped to this interval
        if(time < minSpeedComplTime) 
            time = minSpeedComplTime;
        else if (time > maxSpeedComplTime)
            time = maxSpeedComplTime;
        clampedTime = Mathf.Clamp(time, maxSpeedComplTime, minSpeedComplTime);
        //Debug.Log(maxSpeedComplTime + " " + minSpeedComplTime + " " + _clampedTime);
        //op = 1 if minimum possible time, = 0 if max. time. Otherwise, intermediate values ​​(0.1)
        os = 1f - Mathf.InverseLerp(maxSpeedComplTime, minSpeedComplTime, clampedTime);
        
            //_op = -1;
        return os;
    }
    //average speed maintained by the user
    float GetAVGSpeed(float time)
    {
        if (pathFollower == null || time <= 0f)
            return 0f;

        float dist = pathFollower.getDistanceTravelled();
        return dist / time;
    }

    internal override float GetAccuracy()
    {
        return -1;
    }

    internal override float GetErrors(float time)
    {
        float ep = -1;
        if (_started || _ended)
        {
            if (_surpassedSigns > 0 && sequenceSigns != null)
                ep = (float)sequenceSigns.correctPressures / _surpassedSigns;
            else
                ep = 1;
        }

        return ep;
    }

    public int getTotalLapNumber()
    {
        return _laps;
    }

}
