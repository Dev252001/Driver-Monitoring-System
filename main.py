import cv2
import time
import mediapipe as mp
import numpy as np

from detection import get_ear, get_gaze, get_head
from state import handle_state
from config import *

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

ema_ear = 0.25
ema_ratio = 0.5

ear_samples = []
head_samples = []

HEAD_BASELINE = None
gaze_start_time = None
drowsy_start_time = None

last_gaze = "CENTER"
gaze_switch_count = 0
last_switch_time = 0

turn_mode = False
turn_start_time = 0

head_down = False

# -------- NEW --------
fatigue_score = 100
last_active_time = time.time()

blink_count = 0
prev_eye_closed = False
last_blink_reset = time.time()

while cap.isOpened():
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    state = "CALIBRATING"   # default — overwritten below once face is found

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    now = time.time()

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            lm = face.landmark

            # -------- EAR --------
            ear = (get_ear(lm,[33,160,158,133,153,144]) +
                   get_ear(lm,[362,385,387,263,373,380])) / 2

            ema_ear = EAR_EMA_ALPHA * ear + (1 - EAR_EMA_ALPHA) * ema_ear

            # -------- GAZE --------
            gaze, ema_ratio = get_gaze(lm, ema_ratio)

            # -------- HEAD --------
            head_down, diff = get_head(lm, HEAD_BASELINE, head_down)

            # DEBUG (optional)
            # print("HEAD_DIFF:", diff)

            # -------- CALIBRATION --------
            if len(ear_samples) < CALIB_FRAMES:
                ear_samples.append(ear)
                head_samples.append(diff)
                state = "CALIBRATING"

                if len(ear_samples) == CALIB_FRAMES:
                    print("Calibration Complete")

            else:
                EAR_THRESHOLD = np.mean(ear_samples) * EAR_FACTOR
                HEAD_BASELINE = np.mean(head_samples)

                # -------- DROWSY --------
                if ema_ear < EAR_THRESHOLD * 1.1:

                    if not prev_eye_closed:
                        blink_count += 1
                        prev_eye_closed = True

                    if drowsy_start_time is None:
                        drowsy_start_time = now

                    if now - drowsy_start_time > 1.5:
                        state = "DROWSY"
                    else:
                        state = "ACTIVE"

                else:
                    drowsy_start_time = None
                    prev_eye_closed = False

                    # -------- HEAD --------
                    if head_down:
                        state = "DOWN"

                    # -------- GAZE --------
                    elif gaze in ["LEFT", "RIGHT"]:

                        if gaze_start_time is None:
                            gaze_start_time = now
                            last_gaze = gaze
                            gaze_switch_count = 0

                        if gaze != last_gaze:
                            gaze_switch_count += 1
                            last_switch_time = now
                            gaze_start_time = now

                        if gaze_switch_count >= 2 and (now - last_switch_time < 1.5):
                            turn_mode = True
                            turn_start_time = now

                        if turn_mode:
                            state = "ACTIVE"
                        else:
                            if now - gaze_start_time > GAZE_HOLD_TIME:
                                state = gaze
                            else:
                                state = "ACTIVE"

                        last_gaze = gaze

                    else:
                        gaze_start_time = None
                        last_gaze = "CENTER"
                        gaze_switch_count = 0
                        turn_mode = False
                        state = "ACTIVE"

            # -------- TURN EXIT --------
            if turn_mode and (now - turn_start_time > 2):
                turn_mode = False

            # -------- FATIGUE --------
            if state == "DROWSY":
                fatigue_score -= 2
            elif state in ["LEFT", "RIGHT", "DOWN"]:
                fatigue_score -= 1
            else:
                fatigue_score += 0.5

            # blink reset every 30 sec
            if now - last_blink_reset > 30:
                blink_count = 0
                last_blink_reset = now

            if blink_count > 20:
                fatigue_score -= 0.2

            fatigue_score = max(0, min(100, fatigue_score))

            # -------- ATTENTION --------
            if state == "ACTIVE":
                last_active_time = now

            attention_gap = now - last_active_time

            # -------- ALERT --------
            handle_state(state, now, len(ear_samples) >= CALIB_FRAMES)

            # -------- UI --------
            color = (0,255,0) if state in ["ACTIVE","CALIBRATING"] else (0,0,255)

            cv2.putText(frame, f"STATE: {state}", (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f"EAR: {ema_ear:.2f}", (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(frame, f"FATIGUE: {int(fatigue_score)}%", (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.putText(frame, f"INATTENTIVE: {int(attention_gap)}s", (30,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.putText(frame, f"BLINKS: {blink_count}", (30,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)

    else:
        cv2.putText(frame, "FACE NOT DETECTED", (150,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # -------- FPS --------
    dt = time.time() - frame_start
    fps = int(1/dt) if dt > 0 else 0

    cv2.putText(frame, f"FPS: {fps}", (500,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("SMART DRIVER SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()