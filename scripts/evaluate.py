"""Evaluate a pre-extracted .npz against ground-truth labels.

Usage::

    python scripts/evaluate.py \\
        --npz data/S01_1_daylight.npz \\
        --labels data/S01_1_daylight_labels.csv \\
        --set ear_factor=0.75 head_enter_deg=12.0

Options:
    --npz          Path to .npz from extract.py
    --labels       Path to labels CSV (subject_id,session,condition,variant,label,start_s,end_s)
    --set          key=value config overrides (dataclasses.replace); repeatable
    --per-frame    Also write per-frame CSV to results/raw/<run_id>_frames.csv
    --margin-s     Alert window margin before event start (default 0.5)
    --tol-s        Alert window tolerance after event end (default 1.0)
    --out-dir      Results directory (default results/)
    --seed         Bootstrap seed (default 0)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from config import CFG, Config  # noqa: E402
from landmark_adapter import array_to_landmarks  # noqa: E402
from metrics import (  # noqa: E402
    Alert,
    FrameRecord,
    GroundTruthEvent,
    add_bootstrap_cis,
    event_metrics,
    false_alarm_rate,
    frame_confusion_matrix,
    neutral_seconds,
)


# ── Labels from cue schedule ─────────────────────────────────────────────────
def _load_labels(path: Path) -> list[GroundTruthEvent]:
    events = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["label"] in {"DROWSY", "LEFT", "RIGHT", "DOWN"}:
                events.append(GroundTruthEvent(
                    label=row["label"],
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                ))
    return events


def _gt_label_at(t: float, events: list[GroundTruthEvent]) -> str:
    """Return the ground-truth label at time t (seconds from first frame)."""
    for ev in events:
        if ev.start_s <= t <= ev.end_s:
            return ev.label
    return "ACTIVE"


# ── Config override parsing ───────────────────────────────────────────────────
def _apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply key=value overrides to a Config via dataclasses.replace."""
    kw = {}
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    for item in overrides:
        if "=" not in item:
            sys.exit(f"ERROR: --set value must be key=value, got: {item!r}")
        key, val = item.split("=", 1)
        if key not in fields:
            sys.exit(f"ERROR: unknown config field: {key!r}")
        ftype = fields[key].type
        # Simple type coercion
        try:
            if ftype in ("bool", bool):
                kw[key] = val.lower() in ("1", "true", "yes")
            elif "float | None" in str(ftype) or ftype is type(None):
                kw[key] = None if val.lower() == "none" else float(val)
            elif "float" in str(ftype):
                kw[key] = float(val)
            elif "int" in str(ftype):
                kw[key] = int(val)
            else:
                kw[key] = val
        except ValueError:
            sys.exit(f"ERROR: cannot parse {val!r} for field {key!r} (type {ftype})")
    return dataclasses.replace(cfg, **kw)


# ── Run evaluation ────────────────────────────────────────────────────────────
def run(
    npz_path: Path,
    labels_path: Path,
    cfg: Config,
    margin_s: float,
    tol_s: float,
    per_frame: bool,
    out_dir: Path,
    seed: int,
) -> dict:
    """Load npz + labels, replay DriverMonitor, compute metrics.

    Returns a dict suitable for JSON serialisation.
    """
    from audio import NullAudio  # noqa: PLC0415
    from monitor import DriverMonitor  # noqa: PLC0415

    data = np.load(str(npz_path), allow_pickle=False)
    lm_arr: np.ndarray = data["landmarks"]       # (N, 478, 3) float32
    time_s: np.ndarray = data["time_s"]          # (N,) float64
    frame_w = int(data["frame_w"])
    frame_h = int(data["frame_h"])
    n_frames = len(time_s)

    events = _load_labels(labels_path)

    # Inject modified CFG singleton for this run
    import config as _config_mod  # noqa: PLC0415
    orig_cfg = _config_mod.CFG
    _config_mod.CFG = cfg

    alerts: list[Alert] = []
    frame_records: list[FrameRecord] = []

    Path(out_dir) / "_tmp_eval_log.csv"
    audio = NullAudio()

    class _AlertCollector:
        def log_event(self, state, **_):
            pass   # alerts collected via alert_fired flag below

    mon = DriverMonitor(audio_backend=audio, event_logger=_AlertCollector())

    try:
        for i in range(n_frames):
            t = float(time_s[i])
            row = lm_arr[i]
            if np.any(np.isnan(row)):
                lm = None
            else:
                lm = array_to_landmarks(row)

            if lm is None:
                continue

            result = mon.process_landmarks(lm, t, frame_w=frame_w, frame_h=frame_h)

            if result.alert_fired and result.state in ("DROWSY", "LEFT", "RIGHT", "DOWN"):
                alerts.append(Alert(label=result.state, time_s=t))

            gt = _gt_label_at(t, events)
            frame_records.append(FrameRecord(time_s=t, gt_label=gt, pred_label=result.state))
    finally:
        _config_mod.CFG = orig_cfg

    # ── Metrics ───────────────────────────────────────────────────────────────
    total_s = float(time_s[-1]) if len(time_s) > 0 else 0.0
    ns = neutral_seconds(total_s, events, calibration_s=10.0)
    all_fp = sum(
        event_metrics(events, alerts, lbl).fp
        for lbl in ("DROWSY", "LEFT", "RIGHT", "DOWN")
    )
    far = false_alarm_rate(all_fp, ns)
    cm = frame_confusion_matrix(frame_records, events)

    per_label = {}
    for lbl in ("DROWSY", "LEFT", "RIGHT", "DOWN"):
        m = event_metrics(events, alerts, lbl, margin_s, tol_s)
        m = add_bootstrap_cis(m, events, alerts, margin_s, tol_s, n_boot=2000, seed=seed)
        per_label[lbl] = {
            "tp": m.tp, "fn": m.fn, "fp": m.fp,
            "recall": m.recall, "recall_ci": list(m.recall_ci) if m.recall_ci else None,
            "precision": m.precision,
            "precision_ci": list(m.precision_ci) if m.precision_ci else None,
            "f1": m.f1,
            "mean_delay_s": m.mean_delay_s,
            "n_events": m.tp + m.fn,
        }

    result_dict = {
        "npz": str(npz_path),
        "labels": str(labels_path),
        "config": dataclasses.asdict(cfg),
        "margin_s": margin_s,
        "tol_s": tol_s,
        "n_frames": n_frames,
        "total_s": total_s,
        "neutral_minutes": far.neutral_minutes,
        "false_alarm_rate": far.rate,
        "false_alarm_upper_95": far.upper_bound_95,
        "per_label": per_label,
        "frame_confusion_matrix": {f"{k[0]}->{k[1]}": v for k, v in cm.items()},
        "seed": seed,
    }

    # ── Write errors CSV ──────────────────────────────────────────────────────
    run_id = hashlib.md5(f"{npz_path}{labels_path}{_time.time()}".encode()).hexdigest()[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / f"{run_id}_errors.csv"
    with errors_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "label", "time_s", "event_start_s", "event_end_s"])
        for lbl in ("DROWSY", "LEFT", "RIGHT", "DOWN"):
            event_metrics(events, alerts, lbl, margin_s, tol_s)
            # FNs: events without a matching alert
            ev_label = [e for e in events if e.label == lbl]
            matched = set()
            al_label = sorted([a for a in alerts if a.label == lbl], key=lambda a: a.time_s)
            for ev in ev_label:
                lo, hi = ev.start_s + margin_s, ev.end_s + tol_s
                for j, a in enumerate(al_label):
                    if j not in matched and lo <= a.time_s <= hi:
                        matched.add(j)
                        break
                else:
                    w.writerow(["FN", lbl, "", ev.start_s, ev.end_s])
            for j, a in enumerate(al_label):
                if j not in matched:
                    w.writerow(["FP", lbl, a.time_s, "", ""])

    # ── Write per-frame CSV (optional) ────────────────────────────────────────
    if per_frame:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        frames_path = raw_dir / f"{run_id}_frames.csv"
        with frames_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "gt_label", "pred_label"])
            for fr in frame_records:
                w.writerow([f"{fr.time_s:.4f}", fr.gt_label, fr.pred_label])

    result_json = out_dir / f"{run_id}.json"
    with result_json.open("w") as fh:
        json.dump(result_dict, fh, indent=2, allow_nan=True)

    print(f"Run {run_id}")
    print(f"  Results → {result_json}")
    print(f"  Errors  → {errors_path}")
    for lbl, m in per_label.items():
        r_lo = f"{m['recall_ci'][0]:.2f}" if m['recall_ci'] else "—"
        r_hi = f"{m['recall_ci'][1]:.2f}" if m['recall_ci'] else "—"
        print(f"  {lbl:<8} recall={m['recall']:.2f} [{r_lo},{r_hi}]  "
              f"precision={m['precision']:.2f}  f1={m['f1']:.2f}  "
              f"tp={m['tp']} fn={m['fn']} fp={m['fp']}  "
              f"delay={m['mean_delay_s']:.2f}s" if m['mean_delay_s'] else
              f"  {lbl:<8} recall={m['recall']:.2f}  precision={m['precision']:.2f}  "
              f"tp={m['tp']} fn={m['fn']} fp={m['fp']}")
    print(f"  FAR={far.rate:.3f} FP/min  neutral={far.neutral_minutes:.1f} min")
    if far.upper_bound_95:
        print(f"  (0 FP; 95% upper bound = {far.upper_bound_95:.3f} FP/min)")
    return result_dict


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate .npz landmarks against labels.")
    p.add_argument("--npz",       required=True)
    p.add_argument("--labels",    required=True)
    p.add_argument("--set",       action="append", default=[], metavar="key=value",
                   help="Config overrides: e.g. --set ear_factor=0.75")
    p.add_argument("--per-frame", action="store_true", dest="per_frame")
    p.add_argument("--margin-s",  type=float, default=0.5, dest="margin_s")
    p.add_argument("--tol-s",     type=float, default=1.0, dest="tol_s")
    p.add_argument("--out-dir",   default="results", dest="out_dir")
    p.add_argument("--seed",      type=int, default=0)
    a = p.parse_args()
    cfg = _apply_overrides(CFG, a.set)
    run(
        npz_path=Path(a.npz),
        labels_path=Path(a.labels),
        cfg=cfg,
        margin_s=a.margin_s,
        tol_s=a.tol_s,
        per_frame=a.per_frame,
        out_dir=Path(a.out_dir),
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
