from audio import speak
from config import CFG
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
_stable_state_start: float = 0.0  # monotonic time when current stable_state began


def handle_state(state: str, now: float, calibrated: bool) -> None:
    global prev_state, stable_state, state_count, _stable_state_start

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
        _stable_state_start = now

    # wait for stability_duration_s (replaces old STABILITY_FRAMES=5)
    if now - _stable_state_start < CFG.stability_duration_s:
        return

    # same state → ignore
    if stable_state == prev_state:
        return

    # DROWSY priority + cooldown
    if stable_state == "DROWSY":
        if now - last_alerts["DROWSY"] > CFG.cooldown_drowsy_s:
            speak("DROWSY")
            log_event("DROWSY")
            last_alerts["DROWSY"] = now
        prev_state = "DROWSY"
        return

    # normal cooldown
    if now - last_alerts[stable_state] > CFG.cooldown_other_s:
        speak(stable_state)
        log_event(stable_state)
        last_alerts[stable_state] = now

    prev_state = stable_state
