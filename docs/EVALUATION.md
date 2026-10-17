# Evaluation Protocol

> Phase 3 · branch `polish/phase-3-evaluation`

---

## 1. Definitions

| Term | Definition |
|------|-----------|
| **TP** | Ground-truth event detected: an alert of the same label fires in [start + 0.5 s, end + 1.0 s]. |
| **FN** | Ground-truth event with no matching alert. |
| **FP** | Alert that matches no ground-truth event of the same label. |
| **Detection delay** | First matching alert time − event start time (seconds). |
| **False-alarm rate** | FP alerts per minute of neutral (rest) time. If 0 FP, the 95% upper bound `3 / neutral_minutes` is reported (rule of three). |
| **Frame-level label** | DROWSY / LEFT / RIGHT / DOWN during a cued event; ACTIVE otherwise. Frames within 0.5 s of any ground-truth boundary are excluded. |
| **Neutral time** | Total recording − calibration window (10 s) − sum of event durations. |

Labels: **DROWSY, LEFT, RIGHT, DOWN**. All other states are neutral.

---

## 2. Limitations

- **Simulated cued events.** Subjects consciously close eyes or look away on cue. This is not natural drowsiness. The reported metric is *simulated-event detection*, not real drowsiness detection.
- **Number of subjects.** NOT YET MEASURED sessions recorded; see splits.csv.
- **One camera.** Fixed laptop webcam; no variation in camera angle or distance in Phase 3. Generalization to other cameras is untested.
- **Labels from cue schedule.** Ground truth is the recorder's cue schedule, not a second observer. Human reaction time is absorbed by `margin_s = 0.5 s`.
- **No PERCLOS.** The current detection rule is a 1.5 s EAR hold, not the PERCLOS metric (percentage of eyelid closure over the pupil ≥ 80% for at least 80% of a 1-minute epoch).

---

## 3. Dataset

| Split | Sessions | Notes |
|-------|---------|-------|
| dev | S01_1 | Used for threshold tuning sweeps |
| test | S01_2 | Held-out; reported metrics come from here |

Session IDs are anonymous. See `evaluation/splits.csv`.

---

## 4. Commands

### Record a clip

```powershell
& "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/record_clip.py `
    --subject S01 --session 1 --condition daylight --seed 42
```

### Extract landmarks

```powershell
& "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/extract.py `
    --video data/S01_1_daylight.mp4 `
    --timestamps data/S01_1_daylight_timestamps.csv `
    --out data/S01_1_daylight.npz
```

### Run evaluation

```powershell
& "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/evaluate.py `
    --npz data/S01_1_daylight.npz `
    --labels data/S01_1_daylight_labels.csv
```

### Config overrides (ablations)

```powershell
# Fixed EAR threshold instead of calibrated
& "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/evaluate.py `
    --npz data/S01_1_daylight.npz --labels data/S01_1_daylight_labels.csv `
    --set ear_fixed_threshold=0.25

# EMA disabled
... --set ema_enabled=false

# Head hysteresis disabled
... --set head_hysteresis_enabled=false

# Tighten pitch threshold
... --set head_enter_deg=12.0 head_exit_deg=8.0
```

---

## 5. Results

> NOT YET MEASURED — no recordings have been taken at time of writing.
> Run the commands in §4 to fill this table.

### 5.1 Event-level (test split, default config)

| Label | Recall (n) | 95% CI | Precision | F1 | Mean delay (s) |
|-------|-----------|--------|-----------|-----|----------------|
| DROWSY | NOT YET | — | NOT YET | NOT YET | NOT YET |
| LEFT | NOT YET | — | NOT YET | NOT YET | NOT YET |
| RIGHT | NOT YET | — | NOT YET | NOT YET | NOT YET |
| DOWN | NOT YET | — | NOT YET | NOT YET | NOT YET |

### 5.2 False-alarm rate (test split, default config)

| Metric | Value |
|--------|-------|
| FP / minute neutral | NOT YET MEASURED |
| Neutral time (min) | NOT YET MEASURED |
| 95% upper bound (if 0 FP) | NOT YET MEASURED |

### 5.3 Ablation (dev split — for threshold selection only)

| Ablation | DROWSY recall | FAR (FP/min) |
|----------|--------------|--------------|
| Default config | NOT YET | NOT YET |
| EMA off (`ema_enabled=false`) | NOT YET | NOT YET |
| Debounce off (`stability_duration_s=0`) | NOT YET | NOT YET |
| Hysteresis off (`head_hysteresis_enabled=false`) | NOT YET | NOT YET |
| Fixed EAR 0.25 (`ear_fixed_threshold=0.25`) | NOT YET | NOT YET |

---

## 6. Sweep commands (dev split)

```powershell
# EAR threshold sweep
foreach ($f in 0.70, 0.75, 0.80, 0.85) {
    & "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/evaluate.py `
        --npz data/S01_1_daylight.npz --labels data/S01_1_daylight_labels.csv `
        --set ear_factor=$f
}

# Head pitch threshold sweep
foreach ($d in 10, 12, 15, 18, 20) {
    & "D:\Projects ML\venv_mp\Scripts\python.exe" scripts/evaluate.py `
        --npz data/S01_1_daylight.npz --labels data/S01_1_daylight_labels.csv `
        --set head_enter_deg=$d
}
```

Results are saved to `results/<run_id>.json`. Figures (recall vs FAR) are
written to `results/figures/` by passing `--per-frame` and post-processing the
raw CSV with matplotlib (lazy import, not required by CI).
