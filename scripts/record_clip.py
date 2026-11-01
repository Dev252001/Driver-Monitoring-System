"""Record a labelled evaluation clip from the webcam.

Usage: python scripts/record_clip.py --subject S01 --session 1 --condition daylight [--dry-run]
Outputs under data/: <stem>.mp4, <stem>_timestamps.csv, <stem>_labels.csv
Labels CSV: subject_id,session,condition,variant,label,start_s,end_s  (times in seconds)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
NEUTRAL_S: float = 10.0   # calibration window, excluded from metrics
REST_MIN_S: float = 3.0
REST_MAX_S: float = 6.0

# (label, variant, duration_s)
_EVENT_TYPES: list[tuple[str, str, float]] = [
    ("DROWSY", "N/A",  3.0),
    ("LEFT",   "HEAD", 2.0),
    ("LEFT",   "IRIS", 2.0),
    ("RIGHT",  "HEAD", 2.0),
    ("RIGHT",  "IRIS", 2.0),
    ("DOWN",   "N/A",  2.5),
]


@dataclass
class CueEvent:
    """Scheduled cue; times in seconds from end of neutral calibration window."""
    label: str; variant: str; start_s: float; end_s: float  # noqa: E702


def build_schedule(seed: int, events_per_type: int = 8) -> list[CueEvent]:
    """Return a seeded random cue schedule (pure function — no webcam).

    Pool = each event type × events_per_type, shuffled.  Events are separated
    by rest periods drawn uniformly from [REST_MIN_S, REST_MAX_S].
    t=0 is the end of the NEUTRAL_S calibration window.
    """
    rng = random.Random(seed)
    pool = _EVENT_TYPES * events_per_type
    rng.shuffle(pool)
    events: list[CueEvent] = []
    t = 0.0
    for label, variant, dur in pool:
        events.append(CueEvent(label=label, variant=variant, start_s=t, end_s=t + dur))
        t += dur + rng.uniform(REST_MIN_S, REST_MAX_S)
    return events


def write_labels_csv(path: Path, subject_id: str, session: str,
                     condition: str, schedule: list[CueEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject_id", "session", "condition", "variant", "label", "start_s", "end_s"])
        for ev in schedule:
            w.writerow([subject_id, session, condition, ev.variant, ev.label,
                        f"{ev.start_s:.4f}", f"{ev.end_s:.4f}"])


def record(subject: str, session: str, condition: str,
           seed: int, events_per_type: int, dry_run: bool) -> None:
    stem = f"{subject}_{session}_{condition}"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(seed=seed, events_per_type=events_per_type)
    labels_path = _DATA_DIR / f"{stem}_labels.csv"
    write_labels_csv(labels_path, subject, session, condition, schedule)
    total_s = NEUTRAL_S + (schedule[-1].end_s if schedule else 0.0)
    print(f"Subject:{subject} Session:{session} Condition:{condition} Seed:{seed} "
          f"Total:{total_s:.1f}s  Labels->{labels_path}")
    for ev in schedule:
        print(f"  {ev.start_s:6.1f}s  {ev.label:<6} ({ev.variant})")
    if dry_run:
        print("[dry-run] No webcam opened.")
        return

    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Could not open webcam (device 0).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(_DATA_DIR / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ts_path = _DATA_DIR / f"{stem}_timestamps.csv"
    t0: float | None = None; frame_idx = 0  # noqa: E702
    try:
        with ts_path.open("w", newline="") as ts_fh:
            tsw = csv.writer(ts_fh)
            tsw.writerow(["frame", "time_s"])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                now = time.monotonic()
                if t0 is None:
                    t0 = now
                elapsed = now - t0
                if elapsed > total_s + 2.0:
                    break
                tsw.writerow([frame_idx, f"{elapsed:.6f}"])
                vw.write(frame)
                t_rel = elapsed - NEUTRAL_S
                cue_text, color = "REST", (200, 200, 200)
                if elapsed < NEUTRAL_S:
                    cue_text, color = f"CALIBRATE {NEUTRAL_S - elapsed:.1f}s", (0, 255, 255)
                else:
                    for ev in schedule:
                        if ev.start_s <= t_rel <= ev.end_s:
                            cue_text = f"{ev.label}({ev.variant}) {ev.end_s - t_rel:.1f}s"
                            color = (0, 80, 255); break  # noqa: E702
                cv2.putText(frame, cue_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"t={elapsed:.1f}s", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Recording — press q to abort", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_idx += 1
    finally:
        cap.release(); vw.release(); cv2.destroyAllWindows()  # noqa: E702
    print(f"Video->{_DATA_DIR/f'{stem}.mp4'}  Timestamps->{ts_path} ({frame_idx} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description="Record a labelled evaluation clip.")
    p.add_argument("--subject",         default="S01")
    p.add_argument("--session",         default="1")
    p.add_argument("--condition",       default="neutral")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--events-per-type", type=int, default=8, dest="events_per_type")
    p.add_argument("--dry-run",         action="store_true", dest="dry_run")
    a = p.parse_args()
    record(subject=a.subject, session=a.session, condition=a.condition,
           seed=a.seed, events_per_type=a.events_per_type, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
