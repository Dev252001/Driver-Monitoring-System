"""Evaluation metrics for the driver monitoring system.

All functions are pure Python (no video, no mediapipe).  They operate on
lists of ground-truth events and monitor alert records.

Definitions
-----------
Labels:  DROWSY | LEFT | RIGHT | DOWN.  Anything else is neutral.
event:   ground-truth interval (label, start_s, end_s) from the labels CSV.
alert:   (label, time_s) fired by the monitor.

Event-level (per label):
  TP  — a ground-truth event is *detected* if an alert of the same label fires
        in [event.start_s + margin_s,  event.end_s + tol_s].
        Only the FIRST matching alert is consumed (one TP per event).
  FN  — a ground-truth event with no matching alert.
  FP  — an alert that matches no ground-truth event of the same label.
  delay — first_match_time - event.start_s  (defined only for TPs).

False-alarm rate:
  FP per minute of neutral (rest) time.
  If FP == 0, the 95% upper bound  3 / neutral_minutes  is reported
  (one-sided rule of three).

Frame-level:
  Confusion matrix between ground-truth label (or ACTIVE for neutral frames)
  and the monitor's stable state.  Frames within boundary_s of any ground-truth
  boundary are excluded to avoid transition artefacts.

Uncertainty:
  Seeded 95% bootstrap CIs (n_boot=2000) that resample *events* (or segments
  for frame-level metrics), never individual frames.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ── Constants ────────────────────────────────────────────────────────────────
ALERT_LABELS: frozenset[str] = frozenset({"DROWSY", "LEFT", "RIGHT", "DOWN"})


@dataclass
class GroundTruthEvent:
    """One labelled ground-truth interval."""

    label: str
    start_s: float
    end_s: float


@dataclass
class Alert:
    """One alert fired by the monitor."""

    label: str
    time_s: float


@dataclass
class EventMetrics:
    """Event-level detection metrics for one label (or pooled)."""

    label: str
    tp: int
    fn: int
    fp: int
    delays: list[float]           # delay in seconds for each TP
    recall: float                 # tp / (tp + fn);  NaN if tp+fn==0
    precision: float              # tp / (tp + fp);  NaN if tp+fp==0
    f1: float                     # harmonic mean of recall and precision
    mean_delay_s: float | None    # mean(delays) or None if no TPs
    recall_ci: tuple[float, float] | None = field(default=None)
    precision_ci: tuple[float, float] | None = field(default=None)


@dataclass
class FalseAlarmRate:
    """FP per minute of neutral time."""

    fp_count: int
    neutral_minutes: float
    rate: float                          # fp / neutral_minutes
    upper_bound_95: float | None         # 3 / neutral_minutes if fp == 0, else None
    ci: tuple[float, float] | None = field(default=None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float:
    """Return num/den, or float('nan') if den == 0."""
    return num / den if den != 0 else float("nan")


def _f1(recall: float, precision: float) -> float:
    s = recall + precision
    if s == 0 or s != s:   # NaN guard
        return float("nan")
    return 2 * recall * precision / s


# ── Event-level metrics ───────────────────────────────────────────────────────

def event_metrics(
    events: list[GroundTruthEvent],
    alerts: list[Alert],
    label: str,
    margin_s: float = 0.5,
    tol_s: float = 1.0,
) -> EventMetrics:
    """Compute TP/FN/FP/delay for one label.

    Matching rule:
      An alert matches an event if alert.time_s ∈ [event.start_s + margin_s,
      event.end_s + tol_s].  Each alert and each event is consumed at most once
      (greedy earliest-alert-first matching within each event).

    Args:
        events:   All ground-truth events (filtered to `label` internally).
        alerts:   All monitor alerts (filtered to `label` internally).
        label:    The label to evaluate.
        margin_s: Earliest alert time relative to event start (default 0.5 s).
        tol_s:    Latest alert time relative to event end (default 1.0 s).
    """
    ev_label = [e for e in events if e.label == label]
    al_label = sorted([a for a in alerts if a.label == label], key=lambda a: a.time_s)

    matched_alerts: set[int] = set()
    tp, fn = 0, 0
    delays: list[float] = []

    for ev in ev_label:
        lo = ev.start_s + margin_s
        hi = ev.end_s + tol_s
        match_idx: int | None = None
        for i, al in enumerate(al_label):
            if i in matched_alerts:
                continue
            if lo <= al.time_s <= hi:
                match_idx = i
                break
        if match_idx is not None:
            tp += 1
            delays.append(al_label[match_idx].time_s - ev.start_s)
            matched_alerts.add(match_idx)
        else:
            fn += 1

    fp = len(al_label) - len(matched_alerts)

    recall = _safe_div(tp, tp + fn)
    precision = _safe_div(tp, tp + fp)
    return EventMetrics(
        label=label,
        tp=tp, fn=fn, fp=fp,
        delays=delays,
        recall=recall,
        precision=precision,
        f1=_f1(recall, precision),
        mean_delay_s=sum(delays) / len(delays) if delays else None,
    )


def false_alarm_rate(
    fp_count: int,
    neutral_seconds: float,
) -> FalseAlarmRate:
    """Compute FP per minute of neutral (rest) time.

    If fp_count == 0, rate is 0 and upper_bound_95 = 3 / neutral_minutes
    (one-sided 95% rule of three).

    Args:
        fp_count:        Number of false-positive alerts.
        neutral_seconds: Total neutral (rest) time in seconds.
    """
    minutes = neutral_seconds / 60.0
    rate = _safe_div(fp_count, minutes)
    ub = (3.0 / minutes) if fp_count == 0 and minutes > 0 else None
    return FalseAlarmRate(fp_count=fp_count, neutral_minutes=minutes,
                          rate=rate, upper_bound_95=ub)


# ── Frame-level metrics ────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    """One frame's ground-truth and monitor state."""

    time_s: float
    gt_label: str    # DROWSY | LEFT | RIGHT | DOWN | ACTIVE (neutral)
    pred_label: str  # monitor's stable state for this frame


def frame_confusion_matrix(
    frames: list[FrameRecord],
    events: list[GroundTruthEvent],
    boundary_s: float = 0.5,
) -> dict[tuple[str, str], int]:
    """Build a confusion matrix {(gt, pred): count} excluding boundary frames.

    Frames within boundary_s of any ground-truth event boundary are excluded
    to avoid classification artefacts near transitions.

    Args:
        frames:     Per-frame (time_s, gt_label, pred_label) records.
        events:     Ground-truth events (used to locate boundaries).
        boundary_s: Exclusion radius around each boundary (default 0.5 s).
    """
    # Collect all boundary times
    boundaries: list[float] = []
    for ev in events:
        boundaries.append(ev.start_s)
        boundaries.append(ev.end_s)

    matrix: dict[tuple[str, str], int] = {}
    for fr in frames:
        # Skip frames near a boundary
        near_boundary = any(abs(fr.time_s - b) < boundary_s for b in boundaries)
        if near_boundary:
            continue
        key = (fr.gt_label, fr.pred_label)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


# ── Bootstrap CIs ─────────────────────────────────────────────────────────────

def _bootstrap_recall_precision(
    events: list[GroundTruthEvent],
    alerts: list[Alert],
    label: str,
    margin_s: float,
    tol_s: float,
    n_boot: int,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return (recall_ci, precision_ci) via event-resampling bootstrap."""
    ev_label = [e for e in events if e.label == label]
    rng = random.Random(seed)

    recalls, precisions = [], []
    for _ in range(n_boot):
        sample = [ev_label[rng.randrange(len(ev_label))] for _ in ev_label]
        m = event_metrics(sample, alerts, label, margin_s, tol_s)
        if m.recall == m.recall:   # not NaN
            recalls.append(m.recall)
        if m.precision == m.precision:
            precisions.append(m.precision)

    def _ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        vals.sort()
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        return lo, hi

    return _ci(recalls), _ci(precisions)


def add_bootstrap_cis(
    metrics: EventMetrics,
    events: list[GroundTruthEvent],
    alerts: list[Alert],
    margin_s: float = 0.5,
    tol_s: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> EventMetrics:
    """Return a copy of metrics with recall_ci and precision_ci populated."""
    ev_label = [e for e in events if e.label == metrics.label]
    if not ev_label:
        return metrics
    recall_ci, precision_ci = _bootstrap_recall_precision(
        events, alerts, metrics.label, margin_s, tol_s, n_boot, seed
    )
    import dataclasses  # noqa: PLC0415
    return dataclasses.replace(metrics, recall_ci=recall_ci, precision_ci=precision_ci)


# ── Neutral-time helper ────────────────────────────────────────────────────────

def neutral_seconds(
    total_duration_s: float,
    events: list[GroundTruthEvent],
    calibration_s: float = 0.0,
) -> float:
    """Return seconds of neutral (non-event, non-calibration) time.

    Args:
        total_duration_s: Total recording duration in seconds.
        events:           Ground-truth events (any label).
        calibration_s:    Calibration window at the start to exclude.
    """
    covered = calibration_s
    for ev in events:
        covered += ev.end_s - ev.start_s
    return max(0.0, total_duration_s - covered)
