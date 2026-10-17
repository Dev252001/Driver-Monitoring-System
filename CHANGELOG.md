# Changelog

All notable behaviour-changing commits are listed here in reverse chronological
order. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — branch `polish/phase-3-evaluation` (Stage 0)

### Fixed — C22: DOWN state unreachable (wrong unit conversion in monitor.py)

**Before (`69c8682` monitor.py lines 216–217):**
```python
enter = CFG.head_enter_delta * (180.0 / 0.08)   # = 0.08 × 2250 = 180 °
exit_ = CFG.head_exit_delta  * (180.0 / 0.08)   # = 0.05 × 2250 = 112.5 °
```
The normalised-coord threshold (0.08) was multiplied by the wrong factor,
producing enter=180° and exit=112.5°.  A 180° pitch is physically impossible,
so DOWN was unreachable since commit `69c8682`.  The Phase 2 summary that
listed C5 as fixed was wrong — the solvePnP pitch values were computed
correctly, but the comparison threshold was broken.

**After:** `Config` gains two degree-valued fields (PROVISIONAL, tuned in
Stage 3 sweep):
- `head_enter_deg = 15.0` (degrees above calibrated neutral → enter DOWN)
- `head_exit_deg  = 10.0` (degrees below enter threshold → leave DOWN)

`monitor.py` now compares `pitch_delta > CFG.head_enter_deg` directly.
No change to the hysteresis comparisons.

**Removed:** `Config.head_enter_delta`, `Config.head_exit_delta` (normalised
units, no longer used); `detection.get_head()` (old normalised-y heuristic,
unused since `69c8682`); its characterization tests in `test_detection.py`.

**Added:** `FrameResult.pitch_delta` field (pitch relative to calibrated
neutral, degrees); shown in main.py overlay as `Pd:±xx.x`.

**Tests:** `TestC22HeadDownReachability`, `TestC22PitchHysteresis`,
`TestC22ConfigSanity` in `tests/test_monitor.py` — 13 new tests.

**grep output for head_enter_delta / head_exit_delta / get_head before deletion:**
```
Drowsiness Detection\config.py:80: head_enter_delta = 0.08
Drowsiness Detection\config.py:85: head_exit_delta  = 0.05
Drowsiness Detection\monitor.py:216: enter = CFG.head_enter_delta * (180.0 / 0.08)
Drowsiness Detection\monitor.py:217: exit_ = CFG.head_exit_delta  * (180.0 / 0.08)
Drowsiness Detection\detection.py:221: def get_head(lm, baseline, prev_down=False):
Drowsiness Detection\tests\test_detection.py:398–477: TestGetHead class
```
All deleted; no production caller remains.

**Other old-unit numbers remaining in production code (reported, not fixed):**
- `config.py:54` `gaze_left_threshold = 0.60` (dimensionless ratio, valid)
- `config.py:60` `gaze_right_threshold = 0.40` (dimensionless ratio, valid)
- `config.py:31` comment: `ln(1-0.4)` in tau derivation (correct comment)

---

### Fixed — C1: EAR pixel-space (aspect-ratio independence)

**Before:** `detection.get_ear(lm, eye)` computed Euclidean distances in
normalised coordinate space.  On a non-square sensor the horizontal and
vertical scales differ: same 58×10 px eye gives EAR ≈ 0.229 at 640×480
(4:3) vs ≈ 0.307 at 1280×720 (16:9) — a 34% difference.

**After:** `get_ear(lm, eye, frame_w=1, frame_h=1)` scales each landmark by
`frame_w` and `frame_h` before computing distances.  The same physical eye at
640×480 and 1280×720 now gives the same EAR (within 1%).  The default
`frame_w=frame_h=1` preserves backward-compat for callers that still pass
normalised coords.

**Threshold change:** None required.  `ear_factor = 0.80` is a multiplier on
the driver's calibrated mean EAR, so the threshold tracks whatever absolute
value `get_ear` produces.  No defaults need to change.

**`monitor.py`:** `get_ear(lm, eye)` → `get_ear(lm, eye, frame_w, frame_h)`;
`frame_w`/`frame_h` are already available from `process_landmarks`.

**Test:** `test_same_physical_eye_same_ear_across_resolutions` in
`TestGetEar` — was `xfail(strict=True)`, now a normal passing test.

---

## [Unreleased] — branch `polish/phase-2-correctness` (Stages C–D)

### Fixed — `ffe8708` `chore(sounds): convert alert sounds mp3->wav (D3)`

**Bug C16 — MP3 not guaranteed cross-platform**

- **Before:** `sounds/*.mp3` loaded by pygame; SDL2's MP3 support is optional
  and silently broken on some Linux builds.
- **After:** `sounds/*.wav` added (44100 Hz, 16-bit, mono); `AudioPlayer`
  tries `.ogg → .wav → .mp3`; the `.wav` files are found first so no
  MP3-deprecation warning fires at runtime.

---

### Fixed — `cb4fe55` `fix(config,detection,monitor): FPS-independent EMA (C7)`

**Bug C7i — fixed EMA alpha is FPS-dependent**

- **Before:** `CFG.ear_ema_alpha = 0.4` applied per-frame regardless of dt,
  so the effective time constant doubled at 15 FPS vs 30 FPS.
- **After:** `alpha = 1 - exp(-dt / tau)` computed per-frame.
  `tau_ear_s = 0.0653 s` reproduces `alpha = 0.400` at 30 FPS.
  `tau_gaze_s = 0.0935 s` reproduces `alpha = 0.300` at 30 FPS.
  Step-response test: values at t = step + tau agree within 2% at 15/30/60 FPS.
- **Removed:** `CFG.ear_ema_alpha`, `CFG.gaze_ema_alpha`.
- **Added:** `CFG.tau_ear_s`, `CFG.tau_gaze_s`.

---

### Fixed — `74462f2` `fix(state): remove transition guard (C7)`

**Bug C7 — DROWSY alert fires only once for sustained state (safety issue)**

- **Before:** transition guard `if stable_state == prev_state: return` blocked
  all repeat alerts — a driver who stayed DROWSY received exactly 1 alert ever.
- **After:** guard removed; cooldown (`cooldown_drowsy_s = 2 s`) rate-limits
  repeats.  A sustained 6 s DROWSY episode now fires 3 alerts.

---

### Fixed — `97d596e` + `3b9bfb1` `fix(detection): gaze corner swap + C1 docs (C2, C1)`

**Bug C2 — gaze corners swapped between eyes**

- **Before:** left iris (474-477) normalised against right eye corners (33, 133);
  right iris (469-472) against left eye corners (362, 263).  On asymmetric eyes
  (right width=0.09, left width=0.11) with centred irises, raw ratio = 0.6919
  instead of 0.500 — enough to trigger a spurious LEFT alert.
- **After:** each iris normalised against its own eye's corners.
  Centred irises give raw = 0.500 for any eye-width asymmetry.

**Finding C1 — EAR aspect-ratio dependence (documented)**

- Added to `get_ear()` docstring: on non-square sensors (640×480 vs 1280×720)
  the same physical eye produces different EAR values.  Mitigation: per-driver
  calibration normalises the threshold to each driver's own distribution.

---

### Fixed — `69c8682` `feat(detection): get_head_pose() via cv2.solvePnP (C3/C4)`

**Finding C5 — head-down normalised y-heuristic replaced by real pose estimation**

- **Before:** `get_head()` compared normalised y-coords — scale-dependent and
  sensitive to face distance.
- **After:** `get_head_pose(lm, frame_w, frame_h)` uses `cv2.solvePnP` with a
  standard 6-point 3D face model; returns `pitch_deg`, `yaw_deg` in degrees.
  `FrameResult` now includes `pitch_deg` and `yaw_deg`.
  Head-down detection uses `pitch_delta > threshold_deg` relative to a neutral
  pitch baseline calibrated during the calibration window.
  cv2 is lazily imported so `detection.py` remains headless-safe.

---

## [Unreleased] — branch `polish/phase-2-correctness` (Stage B)

### Added — `0e67f75` `feat(monitor): implement DriverMonitor pipeline class`

**monitor.py: new `DriverMonitor.process_landmarks(lm, t) → FrameResult`**

- **Before:** all detection + state logic was scattered across 200 lines of
  module-level code in `main.py` with no encapsulation; untestable.
- **After:** `DriverMonitor` owns the full per-frame pipeline; both `main.py`
  and `scripts/evaluate.py` (Phase 3) use the same code path.  `FrameResult`
  dataclass carries all per-frame outputs.
- **Fatigue (C12):** per-second rates (`CFG.fatigue_decay_drowsy_per_s` etc.)
  multiplied by `dt`, so behaviour is identical at 30 FPS and frame-accurate
  at any other rate.  (main.py still had per-frame deltas before this commit.)
- **Injectable backends:** `audio_backend.play()` and `event_logger.log_event()`
  are injected; `NullAudio()` + temp-path `EventLogger` used in all tests.
- **Time injection:** `t` parameter used throughout; no internal `time.time()`
  or `time.monotonic()` calls.

### Fixed — `32188f1` `refactor(main): rewrite to use DriverMonitor.process_landmarks()`

**Bug C14 — no `try/finally` for camera release**

- **Before (`main.py:213`):** `cap.release()` called unconditionally at the
  bottom of the script.  An unhandled exception (e.g. from MediaPipe) would
  skip it, leaving the webcam locked.
- **After:** `cap.release()` and `cv2.destroyAllWindows()` are both in a
  `finally` block, guaranteed to run even on exception.

**Bug C17 — `time.time()` in main.py**

- **Before:** `now = time.time()`, `frame_start = time.time()`.
- **After:** all timestamps in `main.py` use `time.monotonic()`.

**Bug C12 — per-frame fatigue deltas in main.py**

- **Before:** `fatigue_score -= 2` / `−1` / `+0.5` per frame — FPS-dependent.
- **After:** logic moved to `DriverMonitor._update_fatigue()` which multiplies
  per-second rates from CFG by `dt`.  The old per-frame code is removed from
  `main.py`.

Each entry cites the commit hash, the **before** behaviour, the **after**
behaviour, and the reason for the change.

---

## [Unreleased] — branch `polish/phase-2-correctness`

### Fixed — `6d5d3f2` `test(state): rewrite tests using StateManager; add C21 regression`

**Bug C21 — first alert suppressed when clock starts at 0 (video/offline mode)**

- **Before (`ca64640` state.py line 61, initial value):** `last_alerts` was
  initialised to `{"DROWSY": None, …}` but `_cooldown_ok` returned
  `last is None or (now - last) > cooldown_s`. When `now=0.0` and
  `last=None` the first alert correctly fired; however, early implementations
  initialised `last_alerts` to `0` (see original `c643d54` / `30843b7`), which
  suppressed the first alert for `cooldown_s` seconds in offline mode where
  `now` starts at `0.0`.
- **After:** `last_alerts` initialises to `None`; `_cooldown_ok` returns `True`
  when `last is None`, so the first alert always fires regardless of clock
  origin.
- **Reason:** Offline evaluation uses `t = frame_index / fps`, which starts at
  `0.0`. With `last_alerts = 0`, the first drowsy event within the first
  `cooldown_drowsy_s` (2 s) of the video was silently suppressed — a safety
  defect that is only visible in offline mode (live mode uses
  `time.monotonic()`, which starts in the millions of seconds).

---

### Refactored — `ca64640` `refactor(state): extract StateManager class, remove all module-level globals`

**state.py: module-level mutable globals → `StateManager` instance attributes**

- **Before:** `state.py` held `last_alerts`, `prev_state`, `stable_state`,
  `state_count`, `STABILITY_FRAMES` as module-level variables. Tests required
  `importlib.reload(state)` to reset between cases. Different call sites shared
  one global machine.
- **After:** All mutable state lives in `StateManager.__init__`. A backward-
  compatible module-level `handle_state()` shim delegates to a lazily created
  `_manager` singleton so `main.py` still works unchanged.
- **Reason:** Testability (each test creates a fresh `StateManager`) and
  correctness (Phase 3 offline evaluation needs one machine per video stream,
  not a shared global).
- **No behaviour change** in live mode — the singleton shim behaves identically
  to the old globals for a single live session.

---

### Fixed — `ca64640` `refactor(state): extract StateManager class…`

**Bug C8 — frame-count debounce is FPS-dependent (partial fix)**

- **Before (`30843b7` `state.py:16`):** `STABILITY_FRAMES = 5` — an alert fired
  only after 5 consecutive frames with the same non-ACTIVE state. At 30 FPS
  this is ≈ 0.167 s; at 15 FPS it is ≈ 0.333 s; at 60 FPS it is ≈ 0.083 s.
- **After:** The debounce logic compares elapsed wall-time:
  `now - self._stable_state_start < CFG.stability_duration_s` where
  `CFG.stability_duration_s = 0.167 s`. The frame counter
  (`state_count`) is kept as a diagnostic counter only.
- **Reason:** FPS varies with hardware load. A time-based gate (using the
  injected monotonic clock) gives identical behaviour at any frame rate, and is
  directly testable with a synthetic clock.
- **Calibration note:** `0.167 s ≈ 5 / 30 FPS` reproduces the original
  behaviour at 30 FPS. The constant can be tuned per Phase 3 evaluation.

---

### Refactored — `0475513` `refactor(A3): remove import-time side effects; add AudioPlayer, EventLogger classes`

**audio.py: `pygame.mixer.init()` and MP3 loads run at import → lazy init**

- **Before (`c643d54` `audio.py:5`):** `pygame.mixer.init()` ran at import
  time; four `pygame.mixer.Sound(…)` calls also ran at import. Any `import
  audio` failed without a display/audio device and with no sound files present.
  CI, headless evaluation, and unit tests were all broken by this.
- **After:** `PygameAudio._ensure_init()` runs `pygame.mixer.init()` and loads
  sounds on the first call to `play()`. `import audio` never touches audio
  hardware.
- **Reason:** Testability (headless CI) and the architectural requirement that
  no module import has side effects on hardware.

**audio.py: `time.time()` → `time.monotonic()` / injectable clock**

- **Before (`c643d54` `audio.py:32`):** `now = time.time()` — uses wall-clock
  UTC time. A system clock adjustment (NTP sync, DST, sleep/wake) could make
  `now - last_play_time` negative, re-triggering audio immediately.
- **After:** `AudioPlayer` takes an injectable `clock: Callable[[], float]`
  (default `time.monotonic`). Offline tests pass a fake clock.
- **Reason:** Monotonic clocks are guaranteed non-decreasing; they are the
  correct choice for measuring elapsed intervals.

**audio.py: MP3 loading now logs a warning; .ogg/.wav tried first**

- **Before:** `pygame.mixer.Sound("drowsy.mp3")` — MP3 decoding is not
  guaranteed by pygame's SDL2 backend on all platforms.
- **After:** `_load_sound` tries `.ogg`, then `.wav`, then `.mp3`; if MP3 is
  used it emits a `warnings.warn`. Sound files will be converted to `.ogg` in
  Stage D3 (last commit of Phase 2).

**logger.py: CWD-relative path, no header, two columns → absolute path, header, seven columns**

- **Before (`c643d54` `logger.py:5`):** `open("fatigue_log.csv", "a")` — file
  opened relative to the process working directory. If the script was run from
  a different directory, a second log file appeared in that directory. Log rows
  contained only `[datetime, event_name]` — no EAR, no threshold, no FPS, no
  session ID — making the log useless for evaluation.
- **After:** `EventLogger._path` defaults to
  `Path(__file__).parent / "logs" / "fatigue_log.csv"` (project-root-relative,
  independent of CWD). The log directory is created automatically. Header row
  written on first open. Columns: `iso_utc_time, session_id, state, ear,
  ear_threshold, fatigue_score, fps`.
- **Reason:** Evaluation (Phase 3) requires structured, reproducible logs with
  numeric features and ISO-8601 UTC timestamps.

---

### Refactored — `bd85b0e` `refactor(A2): replace flat config constants with frozen Config dataclass`

**config.py: flat module constants → frozen `@dataclass Config` + `CFG` singleton**

- **Before (`c643d54` `config.py`):**
  ```python
  COOLDOWN = 4
  CALIB_FRAMES = 60
  EAR_EMA_ALPHA = 0.4
  EAR_FACTOR = 0.80
  GAZE_HOLD_TIME = 1.4
  ```
  Five constants with no units, no docstrings. The DROWSY cooldown (2 s) and
  `STABILITY_FRAMES` (5) lived in `state.py` itself, not in config.
- **After:** Single frozen `@dataclass Config` with 20+ fields; every field has
  a docstring with units. `CFG = Config()` singleton imported everywhere.
  `cooldown_drowsy_s`, `cooldown_other_s`, `stability_duration_s`,
  `audio_min_gap_s` all centralised.
- **Reason:** Single source of truth for all thresholds; makes threshold
  sensitivity analysis tractable in Phase 3.

---

## [Phase 1] — branch `polish/phase-1-hygiene` (merged to `main`)

Commits `4c4f59d`–`e984003`: hygiene only (`.gitignore`, `.bobignore`,
`requirements.txt`, `LICENSE`, `pyproject.toml`, `agents.md`, `docs/PLAN.md`,
ruff autofixes, untrack `venv/` and `fatigue_log.csv`).

No behaviour change. See individual commit messages for details.

---

## [0.1.0] — initial commits `c643d54`–`a914da4`

Original working implementation. Behaviour is preserved by the Phase 2
refactors above except where explicitly noted.
