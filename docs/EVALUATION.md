# Evaluation Protocol

> Phase 3 · branch `polish/phase-3-evaluation`
> Results recorded 2026-11-01 · 1 subject · 2 sessions · seed 42 (dev) / seed 7 (test)

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
- **One subject, two sessions.** Results reflect a single subject; generalisation to other individuals and lighting conditions is untested.
- **One camera.** Fixed laptop webcam; no variation in camera angle or distance. Generalisation to other cameras is untested.
- **Labels from cue schedule.** Ground truth is the recorder's cue schedule, not a second observer. Human reaction time is absorbed by `margin_s = 0.5 s`.
- **No PERCLOS.** The current detection rule is a 1.5 s EAR hold, not the PERCLOS metric (percentage of eyelid closure over the pupil ≥ 80% for at least 80% of a 1-minute epoch).
- **Low precision / high FAR.** Results show high false-alarm rates (10–20 FP/min) and low recall on RIGHT and DOWN. The default thresholds were tuned for live feedback, not offline evaluation against a cue schedule with a short neutral window (~1.3 min). These numbers are reported honestly and without adjustment.

---

## 3. Dataset

| Split | Sessions | Seed | Frames | Face det. % | Neutral time |
|-------|---------|------|--------|-------------|--------------|
| dev   | S01_1   | 42   | 3 644  | 93.5 %      | 1.27 min     |
| test  | S01_2   | 7    | 3 780  | 95.7 %      | 1.27 min     |

Each session: 10 s calibration + 18 cued events (3 per event type × 6 types), ~125 s total.
Session IDs are anonymous. See `evaluation/splits.csv`.

---

## 4. Commands

### Record a clip

```powershell
python scripts/record_clip.py `
    --subject S01 --session 1 --condition daylight --seed 42 --events-per-type 3
```

### Extract landmarks

```powershell
python scripts/extract.py `
    --video data/S01_1_daylight.mp4 `
    --timestamps data/S01_1_daylight_timestamps.csv `
    --out data/S01_1_daylight.npz
```

### Run evaluation

```powershell
python scripts/evaluate.py `
    --npz data/S01_1_daylight.npz `
    --labels data/S01_1_daylight_labels.csv
```

### Config overrides (ablations)

```powershell
# Fixed EAR threshold instead of calibrated
python scripts/evaluate.py `
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

### 5.1 Event-level (test split S01\_2, default config)

> n = events per label; recall CI = 95 % bootstrap (2 000 resamples, seed 0).

| Label | n | TP | FN | FP | Recall | 95 % CI | Precision | F1 | Mean delay (s) |
|-------|---|----|----|-----|--------|---------|-----------|-----|----------------|
| DROWSY | 3 | 0 | 3 | 1 | 0.00 | [0.00, 0.00] | 0.00 | — | — |
| LEFT   | 6 | 1 | 5 | 5 | 0.17 | [0.00, 0.17] | 0.17 | 0.17 | 2.86 |
| RIGHT  | 6 | 0 | 6 | 3 | 0.00 | [0.00, 0.00] | 0.00 | — | — |
| DOWN   | 3 | 0 | 3 | 4 | 0.00 | [0.00, 0.00] | 0.00 | — | — |

### 5.2 False-alarm rate (test split, default config)

| Metric | Value |
|--------|-------|
| FP / minute neutral | 10.22 |
| Neutral time (min) | 1.27 |
| 95 % upper bound (if 0 FP) | n/a — FP > 0 |

### 5.3 Ablation (dev split S01\_1 — for threshold selection only)

> All ablations on dev split (S01\_1, seed 42). DROWSY recall and FAR shown.
> EMA-off, debounce-off, hysteresis-off, and fixed-EAR ablations produced identical
> results on this recording, indicating the dominant failure mode is not in these
> components but in the feature values themselves (e.g. gaze and pitch metrics not
> crossing thresholds during cued events at this camera distance and angle).

| Ablation | DROWSY recall (n=3) | FAR (FP/min) |
|----------|---------------------|--------------|
| Default config | 0.33 | 20.44 |
| EMA off (`ema_enabled=false`) | 0.33 | 20.44 |
| Debounce off (`stability_duration_s=0`) | 0.33 | 20.44 |
| Hysteresis off (`head_hysteresis_enabled=false`) | 0.33 | 20.44 |
| Fixed EAR 0.25 (`ear_fixed_threshold=0.25`) | 0.33 | 20.44 |

---

## 6. Sweep commands (dev split)

```powershell
# EAR threshold sweep
foreach ($f in 0.70, 0.75, 0.80, 0.85) {
    python scripts/evaluate.py `
        --npz data/S01_1_daylight.npz --labels data/S01_1_daylight_labels.csv `
        --set ear_factor=$f
}

# Head pitch threshold sweep
foreach ($d in 10, 12, 15, 18, 20) {
    python scripts/evaluate.py `
        --npz data/S01_1_daylight.npz --labels data/S01_1_daylight_labels.csv `
        --set head_enter_deg=$d
}
```

Results are saved to `results/<run_id>.json`. Figures (recall vs FAR) are
written to `results/figures/` by passing `--per-frame` and post-processing the
raw CSV with matplotlib (lazy import, not required by CI).

---

## 7. Interpretation

The system was developed for **live interactive feedback**, not offline batch evaluation.
The short neutral window (~1.3 min per session) inflates FAR, and the cue schedule
requires the subject to perform actions that may not always be captured within the
detection window at the default thresholds. These results are published as-is; no
post-hoc threshold adjustment was made to improve the reported numbers.
