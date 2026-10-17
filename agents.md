# agents.md — Contributor Quick-Start

## Project
Real-time driver drowsiness/distraction monitor.
Python 3.10.11 · OpenCV · MediaPipe Face Mesh · NumPy · Pygame.

## Structure
```
main.py          Live webcam loop (delegates all logic to DriverMonitor)
monitor.py       DriverMonitor.process_landmarks(lm, t) → FrameResult pipeline
detection.py     get_ear, gaze_ratio, get_head_pose (solvePnP)
state.py         StateManager alert state machine
audio.py         AudioPlayer, PygameAudio, NullAudio
logger.py        EventLogger CSV logger
config.py        Frozen Config dataclass — all thresholds with units
sounds/          Alert audio files (.ogg after Phase 2 D3)
tests/           Pytest unit tests (headless — no cv2/mediapipe/pygame at import)
scripts/         smoke_check.py; evaluate.py + record_clip.py added in Phase 3
docs/PLAN.md     Full phased plan and architectural decisions
docs/EVALUATION.md  Error analysis (Phase 4)
CHANGELOG.md     Behaviour-changing commits with before/after
results/         Committed summary JSON; results/raw/ is git-ignored
.github/workflows/ci.yml  ruff + pytest on every push/PR
```

## Commands
```powershell
# Run live
.\venv\Scripts\python.exe main.py

# Lint
.\venv\Scripts\python.exe -m ruff check .

# Test
.\venv\Scripts\python.exe -m pytest tests/ -v

# Smoke check (headless import + FaceMesh black-frame test)
.\venv\Scripts\python.exe scripts/smoke_check.py

# Install deps in a fresh venv
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, pygame; print('OK')"
```

## Commit convention (Conventional Commits)
`fix:` bug fix  `feat:` new feature  `test:` tests  `docs:` docs  `chore:` tooling  `ci:` CI
Scope = module name: `fix(state):`, `test(detection):`, `feat(monitor):`.

## Branch convention
One branch per phase: `polish/phase-N-<name>`. Never commit directly to `main`.
Author pushes; never run `git push`, `git rebase`, `git reset --hard`,
`git stash` or any force option.

## Key architectural decisions (see docs/PLAN.md for details)
- **Time injection:** every duration rule takes `t: float` from caller; live
  uses `time.monotonic()`, offline uses `frame_index / fps`.
- **One pipeline:** `DriverMonitor.process_landmarks(lm, t, frame_w, frame_h) → FrameResult`.
- **FPS-independent EMA:** `alpha = 1 - exp(-dt / tau)` — never a fixed alpha.
- **Test-first:** failing test committed → fix in separate commit.
- **No fabricated numbers:** unrun metrics marked `NOT YET MEASURED`.
- **Privacy:** no frames saved; logs contain numeric features only.

## Non-negotiable constraints
See docs/PLAN.md § "Non-negotiable constraints" (7 items).
