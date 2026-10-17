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
import mediapipe as mp

from audio import AudioPlayer
from logger import EventLogger
from monitor import DriverMonitor

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh
# refine_landmarks=True is required for iris landmarks 468-477.
# ---------------------------------------------------------------------------
_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


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

            # Pre-process: brighten slightly, mirror so left/right match driver perspective
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _face_mesh.process(rgb)
            t = time.monotonic()

            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    result = monitor.process_landmarks(face.landmark, t)

                    color = (
                        (0, 255, 0)
                        if result.state in ("ACTIVE", "CALIBRATING")
                        else (0, 0, 255)
                    )
                    cv2.putText(
                        frame, f"STATE: {result.state}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    )
                    cv2.putText(
                        frame, f"EAR: {result.ema_ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )
                    cv2.putText(
                        frame, f"FATIGUE: {int(result.fatigue_score)}%", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                    )
                    cv2.putText(
                        frame, f"BLINKS: {result.blink_count}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2,
                    )

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
