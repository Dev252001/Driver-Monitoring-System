# agents.md — Contributor Quick-Start

## Project
Real-time driver drowsiness/distraction monitor.
Python 3.10.11 · OpenCV · MediaPipe Face Mesh · NumPy · Pygame.

## Structure
```
main.py          Live webcam loop
detection.py     EAR, gaze ratio, head pose
state.py         Alert state machine
audio.py         Priority audio playback
logger.py        CSV event logger
config.py        All tuneable parameters
sounds/          Alert audio files (.mp3 → .ogg in Phase 2)
tests/           Pytest unit tests (headless, no cv2/mediapipe/pygame at import)
scripts/         evaluate.py, record_clip.py (added in Phase 3)
docs/PLAN.md     Full phased plan and architectural decisions
results/         Committed summary JSON; results/raw/ is git-ignored
```

## Commands
```powershell
# Run live
.\venv\Scripts\python.exe main.py

# Lint
.\venv\Scripts\python.exe -m ruff check .

# Test
.\venv\Scripts\python.exe -m pytest tests/ -v

# Install deps in a fresh venv
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -c "import cv2, mediapipe, numpy, pygame; print('OK')"
```

## Commit convention (Conventional Commits)
`fix:` bug fix  `feat:` new feature  `test:` tests  `docs:` docs  `chore:` tooling

## Branch convention
One branch per phase: `polish/phase-N-<name>`. Never commit directly to `main`.
Author pushes; never run `git push`, `git rebase`, `git reset --hard`,
`git stash` or any force option.

## Key architectural decisions (see docs/PLAN.md for details)
- **Time injection:** every duration rule takes `t: float` from caller; live
  uses `time.monotonic()`, offline uses `frame_index / fps`.
- **One pipeline:** `DriverMonitor.process(frame, t) -> FrameResult` (Phase 2).
- **Test-first:** failing test → fix in separate commit (Phase 2+).
- **No fabricated numbers:** unrun metrics marked `NOT YET MEASURED`.
- **Privacy:** no frames saved; logs contain numeric features only.

## Non-negotiable constraints
See docs/PLAN.md § "Non-negotiable constraints" (7 items).
