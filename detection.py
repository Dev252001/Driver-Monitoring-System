import numpy as np


def get_ear(lm, eye):
    p2 = np.array([lm[eye[1]].x, lm[eye[1]].y])
    p6 = np.array([lm[eye[5]].x, lm[eye[5]].y])
    p3 = np.array([lm[eye[2]].x, lm[eye[2]].y])
    p5 = np.array([lm[eye[4]].x, lm[eye[4]].y])
    p1 = np.array([lm[eye[0]].x, lm[eye[0]].y])
    p4 = np.array([lm[eye[3]].x, lm[eye[3]].y])

    den = 2 * np.linalg.norm(p1-p4) + 1e-6
    return (np.linalg.norm(p2-p6)+np.linalg.norm(p3-p5)) / den


def iris_center(lm, pts):
    return np.array([
        np.mean([lm[i].x for i in pts]),
        np.mean([lm[i].y for i in pts])
    ])


def get_gaze(lm, ema_ratio):
    left_iris = iris_center(lm, [474, 475, 476, 477])
    right_iris = iris_center(lm, [469, 470, 471, 472])

    ratio = ((left_iris[0] - lm[33].x) / (lm[133].x - lm[33].x + 1e-6) +
             (right_iris[0] - lm[362].x) / (lm[263].x - lm[362].x + 1e-6)) / 2

    ema_ratio = 0.3 * ratio + 0.7 * ema_ratio

    if ema_ratio > 0.60:
        return "LEFT", ema_ratio
    elif ema_ratio < 0.40:
        return "RIGHT", ema_ratio
    else:
        return "CENTER", ema_ratio

def get_head(lm, baseline, prev_down=False):
    nose = lm[1].y
    eye = (lm[33].y + lm[263].y) / 2
    diff = nose - eye

    if baseline is None:
        return False, diff

    # 🔥 hysteresis (stable detection)
    enter = baseline + 0.08
    exit_  = baseline + 0.05

    if prev_down:
        return (diff > exit_), diff
    else:
        return (diff > enter), diff
