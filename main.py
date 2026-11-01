"""Live driver monitoring loop.

Captures frames from the default webcam, decodes face landmarks with
MediaPipe FaceMesh, and delegates all per-frame processing to
``DriverMonitor.process_landmarks(lm, t)``.

Usage::

    python main.py

Press ``q`` to quit.

Notes
-----
- Camera release is guaranteed by a try/finally block (fix C14).
- Timestamps use ``time.monotonic()`` (fix C17).
- All detection and state logic lives in monitor.py; main.py is pure I/O.
- Fatigue deltas are now per-second via CFG (fix C12, in monitor.py).
"""

from __future__ import annotations

import time

import cv2

from audio import AudioPlayer
from landmark_adapter import process_frame
from logger import EventLogger
from monitor import DriverMonitor


def main() -> None:
    """Run the live monitoring loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0)")

    audio = AudioPlayer()
    log = EventLogger()
    monitor = DriverMonitor(audio_backend=audio._backend, event_logger=log)

    try:
        while cap.isOpened():
            frame_start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break

            lm, frame_w, frame_h = process_frame(frame)
            t = time.monotonic()

            if lm is not None:
                result = monitor.process_landmarks(lm, t, frame_w=frame_w, frame_h=frame_h)
                color = (0, 255, 0) if result.state in ("ACTIVE", "CALIBRATING") else (0, 0, 255)
                cv2.putText(frame, f"STATE: {result.state}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"EAR: {result.ema_ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FATIGUE: {int(result.fatigue_score)}%", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"BLINKS: {result.blink_count}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                cv2.putText(frame, f"Pd:{result.pitch_delta:+.1f} Y:{result.yaw_deg:+.1f}",
                            (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
            else:
                cv2.putText(
                    frame, "FACE NOT DETECTED", (150, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                )

            dt = time.monotonic() - frame_start
            fps = int(1 / dt) if dt > 0 else 0
            cv2.putText(
                frame, f"FPS: {fps}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
            )

            cv2.imshow("SMART DRIVER SYSTEM", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
