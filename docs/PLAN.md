# Project Plan — AI-Based Smart Driver Monitoring System

> Approved 2026-09-01. Execute one phase at a time; each phase is a separate
> branch. Do not start the next phase until the current one is approved.
> Non-negotiable constraints are listed at the bottom of this file.

---

## Pre-work (complete)

Resolve the in-progress interactive rebase, land `.gitignore` on `main`.

---

## Phase 1 — Hygiene (branch: `polish/phase-1-hygiene`)

**Goal:** clean repo state, reproducible install, contributor onboarding.

**Files:** `.gitignore` (extended), `.bobignore`, `requirements.txt` (pinned),
`LICENSE` (MIT 2026 Devashish Pandey), `pyproject.toml` (`[tool.ruff]` +
`[tool.pytest.ini_options]`), `docs/PLAN.md`, `agents.md`.

**Acceptance criteria:**
```
git status              # working tree clean
pip install -r requirements.txt && python -c "import cv2, mediapipe, numpy, pygame; print('OK')"
python -m ruff check .  # zero errors
```

**Effort:** S · **Risks:** None — no behaviour change.

---

## Phase 2 — Correctness + Refactor (branch: `polish/phase-2-correctness`)

**Goal:** fix every CONFIRMED bug; make code testable headless; centralise
all thresholds; introduce the `DriverMonitor` pipeline class.

### Confirmed bugs to fix (test-first: failing test → fix in separate commit)

| ID | Bug | File:line |
|----|-----|-----------|
| C2 | Iris corners swapped: left iris uses right eye corners and vice versa | `detection.py:26-27` |
| C7 | DROWSY alert fires once then silenced indefinitely (safety issue) | `state.py:41-50` |
| C8 | Frame-count debounce is FPS-dependent | `state.py:16` |
| C12 | Fatigue score deltas are per-frame, not per-second | `main.py:152-156` |
| C15 | `audio.py` fails on headless import (side effects at module level) | `audio.py:5-14` |
| C17 | `time.time()` instead of `time.monotonic()` for duration measurement | `audio.py`, `state.py` |
| C18 | Logger: CWD-relative path, no header, no features logged | `logger.py:5` |
| C4 | Ten thresholds scattered outside `config.py` | multiple |
| C1 | EAR on normalised coords: document the calibration mitigation | `detection.py:4-12` |
| C5 | Head-down uses normalised y-delta: document scale dependence | `detection.py:38-53` |
| C13 | Drowsy detection is 1.5 s EAR hold, not PERCLOS: add PERCLOS metric | `main.py:96-102` |
| C14 | No `try/finally` for camera release | `main.py:211-212` |
| C16 | MP3 not guaranteed cross-platform (Phase 2 prep; conversion is last commit of Phase 2) | `audio.py` |

### Architectural decisions (binding for all phases)

#### a. Time injection
Every time-dependent rule (EAR hold duration, debounce, cooldowns, fatigue
score decay, PERCLOS window) receives its current timestamp from the **caller**
as a parameter — never calls `time.time()` or `time.monotonic()` internally.

- Live mode: caller passes `time.monotonic()`.
- Offline evaluation: caller passes `frame_index / fps` (video time).

This makes every algorithm unit-testable with a synthetic clock and ensures
offline evaluation is frame-accurate.

#### b. One pipeline class — `DriverMonitor`
Extract a class `DriverMonitor` (in a new file `monitor.py`) with a single
method:

```python
result: FrameResult = monitor.process(frame: np.ndarray, t: float) -> FrameResult
```

`FrameResult` is a dataclass with fields: `ear`, `ema_ear`, `threshold`,
`gaze`, `head_down`, `pitch_deg`, `yaw_deg`, `state`, `fatigue_score`,
`blink_count`, `alert_fired`.

Both `main.py` (live) and `scripts/evaluate.py` (offline) use
`DriverMonitor.process`. No detection or state logic lives outside this class.

#### c. Head yaw/pitch via `cv2.solvePnP`
`get_head` is replaced by a proper head-pose estimator using `cv2.solvePnP`
with a standard 3-D face model (6-point: nose tip, chin, eye corners, mouth
corners). Returns `pitch_deg` and `yaw_deg`.

The DROWSY/DOWN detection uses pitch; LEFT/RIGHT distraction combines:
- iris gaze ratio (corrected, Phase 2 C2 fix), AND
- yaw magnitude > threshold (both must agree for a sustained LEFT/RIGHT alert).

Document the LEFT/RIGHT convention: after `cv2.flip(frame, 1)`, image-left =
driver-right. Labels refer to driver perspective throughout.

#### d. Test-first workflow
For each confirmed bug:
1. Commit a **failing test** that demonstrates the bug (pytest with synthetic
   landmarks; no cv2/mediapipe/pygame imported at test collection time).
2. Commit the **fix** in a separate commit referencing the test.

Gaze corner-swap verification (run before committing the fix):

```python
# Symmetric frontal face: right eye x=[0.30, 0.39], left eye x=[0.61, 0.70]
# Right iris centre x=0.345, left iris centre x=0.655
# Eye widths: right=0.09, left=0.09
# Old formula (wrong corners): ratio ≈ 0.69  (off-centre even on symmetric face)
# Corrected formula:           ratio ≈ 0.50  (centred, as expected)
# Asymmetric: right width=0.09, left width=0.11 → difference = 0.5*(l-r)*(1/w1-1/w2)
```

Compute and record the actual numbers in the test docstring.

#### e. GitHub Actions CI (last non-audio commit of Phase 2)
File: `.github/workflows/ci.yml`
- Trigger: push and pull_request on `main` and `polish/*` branches.
- Job: `ubuntu-latest`, Python 3.10.
- Steps: `pip install ruff pytest`, `ruff check .`, `pytest tests/`.
- Tests MUST NOT import `cv2`, `mediapipe` or `pygame` at import time.
  Modules that use those libraries must use lazy imports or injectable backends.

#### f. Fatigue score: per-second rates
Convert the per-frame deltas (`-2`, `-1`, `+0.5`) to per-second rates that
reproduce the same behaviour at 30 FPS:

```
FATIGUE_DECAY_DROWSY_PER_S  = -2  * 30 = -60  pts/s
FATIGUE_DECAY_ALERT_PER_S   = -1  * 30 = -30  pts/s
FATIGUE_RECOVER_PER_S       = +0.5 * 30 = +15 pts/s
```

Store rates in `config.py`. `DriverMonitor.process` multiplies by `dt`
(seconds since last frame), so behaviour is identical at 30 FPS and
frame-accurate at any other rate.

#### g. Phase 3 statistics
- Use 95% bootstrap confidence intervals; resample by segment (contiguous
  labelled event) or by subject.
- Tune thresholds on one session/subject set; report on a held-out set.
- Self-recorded clips contain **simulated events** (consciously closing eyes,
  looking away). The reported metric is therefore "simulated eye-closure /
  distraction event detection", not "real drowsiness detection". Label all
  tables accordingly.
- Subject ID column in all logs and labels CSVs. Report per-subject and
  pooled results.

#### h. Citations and AI disclosure
- Do not invent citations. List references as `TODO verify (DOI or URL)` until
  manually confirmed.
- Add a one-line AI-assistance disclosure to the README footer:
  > "Development assisted by IBM Bob (AI coding assistant)."

### Audio format migration (last commit of Phase 2)
Convert `.mp3` → `.ogg` (Vorbis). The `AudioPlayer` loader tries `.ogg`,
then `.wav`, then `.mp3` with a logged warning. This is the **last commit**
of Phase 2.

**Acceptance criteria:**
```
python -m pytest tests/ -v   # all green; no cv2/mediapipe/pygame at import time
python -m ruff check .        # zero errors
python -c "from monitor import DriverMonitor"   # no audio device needed
```

**Effort:** M (~1 day) · **Risks:** Gaze fix changes empirical thresholds;
re-verify live after fix. Document threshold change in commit message.

---

## Phase 3 — Evaluation (branch: `polish/phase-3-evaluation`)

**Goal:** produce real, reproducible numbers and honest error analysis.

**Files:** `main.py` (add `--video`/`--headless` args), `scripts/evaluate.py`,
`scripts/record_clip.py`, updated `logger.py` (structured JSONL),
`docs/EVALUATION.md`, `results/*.json`.

**Dataset notes:** to be supplied by the author in Phase 3. All datasets and
videos of people are never committed to the repository.

**`scripts/evaluate.py` design:**
- Flags: `--video PATH --labels PATH --headless --subject-id ID`
- Runs `DriverMonitor.process(frame, t=frame_index/fps)` for each frame.
- Outputs per-frame CSV: `iso_time, subject_id, frame_id, ear, ema_ear,
  threshold, state, gt_label`.
- Computes: precision, recall, F1 (drowsy vs alert), detection delay,
  false alarms per hour, per-frame latency mean/p95, PERCLOS estimate.
- Ablations: fixed threshold (0.25) vs calibrated; with/without EMA;
  with/without hysteresis; combined yaw+gaze vs gaze-only.
- All metrics report 95% bootstrap CIs.
- Saves `results/session_<date>_<subject_id>.json` with metrics + config +
  git commit hash + library versions.

**Structured log format (JSONL, one object per frame):**
```json
{"iso_time": "2026-09-01T10:00:00.123Z", "session_id": "s01", "subject_id": "dev",
 "frame": 42, "t": 1.4, "ear": 0.31, "ema_ear": 0.29, "threshold": 0.26,
 "gaze": "CENTER", "pitch_deg": -2.1, "yaw_deg": 1.3, "state": "ACTIVE",
 "fatigue": 98.5, "fps": 28.3}
```

**Error analysis conditions** (self-recorded, labelled by author):
glasses, low light, head yaw >30°, partial occlusion, camera angle
(below / above eye level).

**Acceptance criteria:**
```
python scripts/evaluate.py --video clip.mp4 --labels clip_labels.csv --headless
# prints metric table; saves results/session_*.json
```

**Effort:** L (~3 days) · **Risks:** Dataset access time; self-recorded clip
is sufficient for application if scope is honest.

---

## Phase 4 — Documentation (branch: `polish/phase-4-docs`)

**Goal:** README and `docs/EVALUATION.md` suitable for jury inspection.

**README additions:**
1. Problem statement (cite road-safety statistics — TODO verify DOI).
2. Method: EAR formula, cite Soukupová & Čech 2016 (TODO verify DOI);
   PERCLOS definition; note that current implementation is a simplified proxy.
3. Mermaid architecture diagram.
4. Results table from Phase 3 output only; "NOT YET MEASURED" until then.
5. Limitations: normalised-coord EAR; no full PERCLOS; single-subject
   evaluation; gaze ignores yaw (mitigated in Phase 2); MP3 fragility;
   no night vision; not a certified safety system.
6. Ethics: research prototype; not validated for deployment; no face images
   stored; log contains numeric features only; GDPR note.
7. Reproduction steps: exact Python version, install, run commands.
8. Demo GIF instructions.
9. AI-assistance disclosure (see decision h).

**Effort:** M (~1 day after Phase 3) · **Risks:** All numbers must come from
committed `results/*.json` files.

---

## Phase 5 — Learned model (stretch, only after Phases 1–4 complete)

Small classifier (logistic regression or MLP) on a rolling 2 s window of
features: EAR mean/std, blink rate, mouth aspect ratio (MAR), pitch, yaw.
Subject-independent split. Same metrics as Phase 3. Compare rule-based vs
learned in a single table.

**Files:** `scripts/train_model.py`, `scripts/extract_features.py`,
`models/drowsy_clf.pkl`, `tests/test_model.py`.

**Effort:** L (~2 days) · **Prerequisite:** labelled multi-subject data.

---

## Non-negotiable constraints

1. **No fabricated numbers.** Unrun metrics are marked `NOT YET MEASURED`
   with the exact command that produces them.
2. **No history rewriting.** Never `rebase`, `reset --hard`, `push --force`,
   or `stash` on shared history. One branch per phase; small atomic commits
   with Conventional Commit messages (`fix:`, `feat:`, `test:`, `docs:`,
   `chore:`).
3. **Preserve behaviour.** Every behavioural change needs a test or a
   documented measurement. Phase 1 has zero behaviour changes.
4. **Ask before deleting.** Ask before deleting anything other than `venv/`,
   `__pycache__/` and generated logs (untrack with `git rm --cached` only).
   Justify each new dependency.
5. **Offline, laptop webcam.** Must run on a normal laptop, offline.
   Python 3.10.11 (MediaPipe 0.10.9 constraint).
6. **Privacy.** No frames or face images saved by default. Logs contain
   numeric features and events only.
7. **Readable code.** Type hints, short docstrings with formulas and units.
   No clever abstractions.
