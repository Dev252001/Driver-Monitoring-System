"""Extract landmarks from a video file (or webcam) to a .npz archive.

Usage::

    python scripts/extract.py --video data/S01_1_daylight.mp4 \\
                               --timestamps data/S01_1_daylight_timestamps.csv \\
                               --out data/S01_1_daylight.npz

Output npz keys:
    landmarks   — float32, shape (N, 478, 3); NaN rows where no face detected
    time_s      — float64, shape (N,); seconds from first frame
    frame_w     — int scalar
    frame_h     — int scalar

Time source:
    If --timestamps CSV is provided: uses per-frame capture times from the CSV.
    Otherwise: frame_index / fps from the video's CAP_PROP_FPS.

Shares landmark extraction logic with main.py via landmark_adapter.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from landmark_adapter import N_LANDMARKS, landmarks_to_array  # noqa: E402


def extract(
    video_path: Path,
    out_path: Path,
    timestamps_path: Path | None = None,
) -> None:
    """Extract per-frame landmarks from a video file.

    Args:
        video_path:      Path to input video (.mp4, .avi, …).
        out_path:        Path to output .npz archive.
        timestamps_path: Optional CSV with columns [frame, time_s].
    """
    import cv2  # noqa: PLC0415

    from landmark_adapter import process_frame  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load timestamps CSV if provided
    ts_map: dict[int, float] | None = None
    if timestamps_path is not None and timestamps_path.exists():
        ts_map = {}
        with timestamps_path.open() as fh:
            for row in csv.DictReader(fh):
                ts_map[int(row["frame"])] = float(row["time_s"])

    # Pre-allocate output arrays (resize if actual frame count differs)
    max_frames = max(n_frames, 1)
    lm_arr = np.full((max_frames, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    time_arr = np.zeros(max_frames, dtype=np.float64)
    frame_w = frame_h = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= max_frames:
            # Resize if video has more frames than CAP_PROP_FRAME_COUNT reported
            extra = np.full((max_frames, N_LANDMARKS, 3), np.nan, dtype=np.float32)
            lm_arr = np.concatenate([lm_arr, extra], axis=0)
            time_arr = np.concatenate([time_arr, np.zeros(max_frames)])
            max_frames *= 2

        t = ts_map[frame_idx] if (ts_map and frame_idx in ts_map) else frame_idx / fps
        time_arr[frame_idx] = t

        lm, fw, fh = process_frame(frame)
        if lm is not None:
            lm_arr[frame_idx] = landmarks_to_array(lm)
        if frame_idx == 0:
            frame_w, frame_h = fw, fh

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx} frames processed…", end="\r")

    cap.release()

    lm_arr = lm_arr[:frame_idx]
    time_arr = time_arr[:frame_idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        landmarks=lm_arr,
        time_s=time_arr,
        frame_w=np.int32(frame_w),
        frame_h=np.int32(frame_h),
    )
    print(f"\nExtracted {frame_idx} frames -> {out_path}")
    n_faces = int(np.sum(~np.isnan(lm_arr[:, 0, 0])))
    print(f"  Face detected in {n_faces}/{frame_idx} frames "
          f"({100*n_faces/max(frame_idx,1):.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract landmarks from video to .npz")
    p.add_argument("--video",      required=True, help="Input video path")
    p.add_argument("--out",        required=True, help="Output .npz path")
    p.add_argument("--timestamps", default=None,  help="Timestamps CSV (optional)")
    a = p.parse_args()
    extract(
        video_path=Path(a.video),
        out_path=Path(a.out),
        timestamps_path=Path(a.timestamps) if a.timestamps else None,
    )


if __name__ == "__main__":
    main()
