from audio import speak
from config import COOLDOWN
from logger import log_event

last_alerts = {
    "DROWSY": 0,
    "LEFT": 0,
    "RIGHT": 0,
    "DOWN": 0
}

prev_state = "ACTIVE"
stable_state = "ACTIVE"
state_count = 0

STABILITY_FRAMES = 5

def handle_state(state, now, calibrated):
    global prev_state, stable_state, state_count

    if not calibrated or state in ["ACTIVE", "CALIBRATING"]:
        # reset so the next alert state is never silently skipped
        if state == "ACTIVE":
            prev_state = "ACTIVE"
        stable_state = state
        state_count = 0
        return

    # stability check
    if state == stable_state:
        state_count += 1
    else:
        stable_state = state
        state_count = 1

    # wait for stable frames
    if state_count < STABILITY_FRAMES:
        return

    # same state → ignore
    if stable_state == prev_state:
        return

    # DROWSY priority + cooldown
    if stable_state == "DROWSY":
        if now - last_alerts["DROWSY"] > 2:   # shorter cooldown
            speak("DROWSY")
            log_event("DROWSY")
            last_alerts["DROWSY"] = now
        prev_state = "DROWSY"
        return

    # normal cooldown
    if now - last_alerts[stable_state] > COOLDOWN:
        speak(stable_state)
        log_event(stable_state)
        last_alerts[stable_state] = now

    prev_state = stable_state