# agents.md — Contributor Quick-Start

## Project
Real-time driver drowsiness/distraction monitor.
Python 3.10.11 · mediapipe==0.10.9 · opencv-contrib-python==4.10.0.84 · numpy==1.26.4 · pygame==2.6.1.
Interpreter: `D:\Projects ML\venv_mp\Scripts\python.exe`

## Structure
```
main.py               Live webcam loop (uses landmark_adapter + DriverMonitor)
monitor.py            DriverMonitor.process_landmarks(lm, t, w, h) → FrameResult
detection.py          get_ear(+pixel-space fix C1), gaze_ratio, get_head_pose
landmark_adapter.py   Shared frame→landmarks adapter (main.py and extract.py)
state.py              StateManager alert state machine
audio.py              AudioPlayer, PygameAudio, NullAudio
logger.py             EventLogger CSV logger
config.py             Frozen Config dataclass — all thresholds with units + ablation flags
evaluation/metrics.py Event-level and frame-level evaluation metrics (pure Python)
evaluation/splits.csv Session split assignments (dev / test; anonymous IDs)
scripts/record_clip.py  Labelled webcam recorder with cue schedule
scripts/extract.py      Video → data/<stem>.npz (landmarks + timestamps)
scripts/evaluate.py     .npz + labels → results/<run_id>.json (metrics + CIs)
tests/                Pytest unit tests (headless — no cv2/mediapipe/pygame at import)
docs/EVALUATION.md    Protocol, definitions, limitations, results table
docs/PLAN.md          Phased plan and architectural decisions
CHANGELOG.md          Behaviour-changing commits with before/after
results/              Committed summary JSON; results/raw/ is git-ignored
data/                 Recordings and extracted .npz — git-ignored
```

## Commands
```powershell
$py = "D:\Projects ML\venv_mp\Scripts\python.exe"
& $py main.py                                                              # live
& $py -m ruff check . ; & $py -m pytest tests/ -v                         # lint+test
& $py scripts/smoke_check.py                                               # smoke
& $py scripts/record_clip.py --subject S01 --session 1 --dry-run          # record
& $py scripts/extract.py --video data/S01_1_day.mp4 --out data/S01_1_day.npz
& $py scripts/evaluate.py --npz data/S01_1_day.npz --labels data/S01_1_day_labels.csv
```

## Commit convention (Conventional Commits)
`fix:` `feat:` `test:` `docs:` `chore:` `ci:`  Scope = module: `fix(monitor):`.

## Key architectural decisions
- **Time injection:** `t: float` from caller; live = `time.monotonic()`, offline = `frame_index/fps`.
- **One pipeline:** `DriverMonitor.process_landmarks(lm, t, frame_w, frame_h) → FrameResult`.
- **Pixel-space EAR:** `get_ear` scales landmarks by `frame_w`/`frame_h` (C1 fix).
- **Degree-based pitch:** `head_enter_deg=15°`, `head_exit_deg=10°` (C22 fix).
- **FPS-independent EMA:** `alpha = 1 - exp(-dt / tau)`.
- **Ablation flags** in Config: `ema_enabled`, `head_hysteresis_enabled`, `use_yaw_for_gaze`, `ear_fixed_threshold`.
- **No fabricated numbers:** unrun metrics marked `NOT YET MEASURED`.
- **Privacy:** no frames saved; logs contain numeric features only.

## Non-negotiable constraints
See docs/PLAN.md § "Non-negotiable constraints" (7 items).
