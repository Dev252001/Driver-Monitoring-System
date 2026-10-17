import os
import time

import pygame

pygame.mixer.init()

BASE = os.path.dirname(os.path.abspath(__file__))

sounds = {
    "DROWSY": pygame.mixer.Sound(os.path.join(BASE, "sounds", "drowsy.mp3")),
    "LEFT": pygame.mixer.Sound(os.path.join(BASE, "sounds", "left.mp3")),
    "RIGHT": pygame.mixer.Sound(os.path.join(BASE, "sounds", "right.mp3")),
    "DOWN": pygame.mixer.Sound(os.path.join(BASE, "sounds", "down.mp3")),
}

priority = {
    "DROWSY": 3,
    "DOWN": 2,
    "LEFT": 1,
    "RIGHT": 1
}

channel = None
current_state = None
last_play_time = 0

MIN_AUDIO_GAP = 1.5

def speak(state):
    global channel, current_state, last_play_time

    if state not in sounds:
        return

    now = time.time()

    # reset if finished
    if channel is not None and not channel.get_busy():
        current_state = None

    # same state → ignore
    if state == current_state:
        return

    # priority check
    if current_state is not None:
        if priority[state] < priority[current_state]:
            return

    # gap control (except DROWSY)
    if state != "DROWSY" and (now - last_play_time < MIN_AUDIO_GAP):
        return

    # interrupt only if higher priority
    if channel is not None and channel.get_busy():
        if priority[state] > priority.get(current_state, 0):
            channel.stop()
        else:
            return

    # play
    channel = sounds[state].play()

    current_state = state
    last_play_time = now
