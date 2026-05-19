"""
IPT validation metrics: frame-level and event-level micro / macro F1.

Event-level evaluation follows eval_table1_guzheng.py:
shared onset + per-class IPT frames, mir_eval note matching with class encoded as pitch.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz

from config import FEATURE_RATE, NUM_LABELS

from lib import extract_notes

# Guzheng Tech99 IPT index order (matches load_data technique ids)
IPT_NAMES = [
    "vibrato",    # chanyin 0
    "plucks",     # boxian 1
    "UP",         # shanghua 2
    "DP",         # xiahua 3
    "glissando",  # huazhi 4
    "tremolo",    # yaozhi 5
    "PN",         # dianyin 6
]


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def accumulate_frame_counts(pred_bin: np.ndarray, tar_bin: np.ndarray) -> Tuple[int, int, int]:
    """pred_bin, tar_bin: (C, T) with values in {0, 1}."""
    tp = int(np.logical_and(pred_bin == 1, tar_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, tar_bin == 0).sum())
    fn = int(np.logical_and(pred_bin == 0, tar_bin == 1).sum())
    return tp, fp, fn


def events_from_shared_onset(
    onset_prob: np.ndarray,
    frame_prob: np.ndarray,
    onset_th: float = 0.5,
    frame_th: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    onset_prob: (1, T) or (T,) probabilities
    frame_prob: (C, T) probabilities
    Returns (intervals_seconds, pitches_hz) for mir_eval.
    """
    if onset_prob.ndim == 1:
        onset_prob = onset_prob.reshape(1, -1)
    pitches, intervals = extract_notes(
        torch.from_numpy(onset_prob.astype(np.float32)).transpose(-1, -2),
        torch.from_numpy(frame_prob.astype(np.float32)).transpose(-1, -2),
        onset_threshold=onset_th,
        frame_threshold=frame_th,
    )
    if intervals.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    intervals = (intervals / FEATURE_RATE).reshape(-1, 2)
    pitches_hz = np.array([midi_to_hz(21 + int(p)) for p in pitches], dtype=np.float64)
    return intervals.astype(np.float64), pitches_hz


def concat_event_lists_with_offsets(
    per_chunk_events: Sequence[Tuple[np.ndarray, np.ndarray, float]],
    gap_seconds: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate (intervals, pitches, duration_sec) lists with time offsets."""
    all_intervals: List[np.ndarray] = []
    all_pitches: List[np.ndarray] = []
    offset = 0.0
    for intervals, pitches, dur in per_chunk_events:
        if intervals.shape[0] > 0:
            shifted = intervals.copy()
            shifted[:, 0] += offset
            shifted[:, 1] += offset
            all_intervals.append(shifted)
            all_pitches.append(pitches)
        offset += float(dur) + float(gap_seconds)
    if not all_intervals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(all_intervals, axis=0), np.concatenate(all_pitches, axis=0)


def evaluate_ipt_chunks(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    onset_th: float = 0.5,
    frame_th: float = 0.5,
    onset_tolerance: float = 0.05,
    event_gap_seconds: float = 1.0,
) -> Dict:
    """
    Compute frame / event MI-F1 and MA-F1 on validation chunks.

    Each chunk: pred_ipt (C, T), tar_ipt (C, T), onset (1, T).
    """
    assert len(pred_ipt_chunks) == len(tar_ipt_chunks) == len(pred_onset_chunks) == len(tar_onset_chunks)
    n_chunks = len(pred_ipt_chunks)

    frame_tp = frame_fp = frame_fn = 0
    frame_tp_c = np.zeros((NUM_LABELS,), dtype=np.int64)
    frame_fp_c = np.zeros((NUM_LABELS,), dtype=np.int64)
    frame_fn_c = np.zeros((NUM_LABELS,), dtype=np.int64)

    ref_events_all: List[Tuple[np.ndarray, np.ndarray, float]] = []
    est_events_all: List[Tuple[np.ndarray, np.ndarray, float]] = []
    ref_events_c: List[List[Tuple[np.ndarray, np.ndarray, float]]] = [[] for _ in range(NUM_LABELS)]
    est_events_c: List[List[Tuple[np.ndarray, np.ndarray, float]]] = [[] for _ in range(NUM_LABELS)]

    for i in range(n_chunks):
        pred_IPT = np.asarray(pred_ipt_chunks[i], dtype=np.float32)
        tar_IPT = np.asarray(tar_ipt_chunks[i], dtype=np.float32)
        pred_onset = np.asarray(pred_onset_chunks[i], dtype=np.float32)
        tar_onset = np.asarray(tar_onset_chunks[i], dtype=np.float32)

        if pred_IPT.ndim == 1:
            pred_IPT = pred_IPT.reshape(1, -1)
        if tar_IPT.ndim == 1:
            tar_IPT = tar_IPT.reshape(1, -1)
        if pred_onset.ndim == 1:
            pred_onset = pred_onset.reshape(1, -1)
        if tar_onset.ndim == 1:
            tar_onset = tar_onset.reshape(1, -1)

        t_len = min(pred_IPT.shape[-1], tar_IPT.shape[-1])
        pred_IPT = pred_IPT[:, :t_len]
        tar_IPT = tar_IPT[:, :t_len]
        pred_onset = pred_onset[:, :t_len]
        tar_onset = tar_onset[:, :t_len]

        pred_bin = (pred_IPT > frame_th).astype(np.uint8)
        tar_bin = (tar_IPT > 0.5).astype(np.uint8)

        tp, fp, fn = accumulate_frame_counts(pred_bin, tar_bin)
        frame_tp += tp
        frame_fp += fp
        frame_fn += fn
        for c in range(NUM_LABELS):
            tpc, fpc, fnc = accumulate_frame_counts(pred_bin[c : c + 1], tar_bin[c : c + 1])
            frame_tp_c[c] += tpc
            frame_fp_c[c] += fpc
            frame_fn_c[c] += fnc

        dur_sec = t_len / float(FEATURE_RATE)
        ref_int, ref_pitch = events_from_shared_onset(
            tar_onset, tar_IPT, onset_th=onset_th, frame_th=frame_th
        )
        est_int, est_pitch = events_from_shared_onset(
            pred_onset, pred_IPT, onset_th=onset_th, frame_th=frame_th
        )
        ref_events_all.append((ref_int, ref_pitch, dur_sec))
        est_events_all.append((est_int, est_pitch, dur_sec))

        for c in range(NUM_LABELS):
            ref_i_c, ref_p_c = events_from_shared_onset(
                tar_onset, tar_IPT[c : c + 1], onset_th=onset_th, frame_th=frame_th
            )
            est_i_c, est_p_c = events_from_shared_onset(
                pred_onset, pred_IPT[c : c + 1], onset_th=onset_th, frame_th=frame_th
            )
            ref_events_c[c].append((ref_i_c, ref_p_c, dur_sec))
            est_events_c[c].append((est_i_c, est_p_c, dur_sec))

    frame_mi_f1 = f1_from_counts(frame_tp, frame_fp, frame_fn)
    frame_class_f1 = [
        f1_from_counts(int(frame_tp_c[c]), int(frame_fp_c[c]), int(frame_fn_c[c]))
        for c in range(NUM_LABELS)
    ]
    frame_ma_f1 = float(np.mean(frame_class_f1)) if frame_class_f1 else 0.0

    ref_int_all, ref_pitch_all = concat_event_lists_with_offsets(
        ref_events_all, gap_seconds=event_gap_seconds
    )
    est_int_all, est_pitch_all = concat_event_lists_with_offsets(
        est_events_all, gap_seconds=event_gap_seconds
    )
    if ref_int_all.shape[0] == 0 and est_int_all.shape[0] == 0:
        event_mi_f1 = 0.0
    else:
        _, _, event_mi_f1, _ = evaluate_notes(
            ref_int_all,
            ref_pitch_all,
            est_int_all,
            est_pitch_all,
            offset_ratio=None,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=0,
        )

    event_class_f1: List[float] = []
    for c in range(NUM_LABELS):
        ref_i, ref_p = concat_event_lists_with_offsets(
            ref_events_c[c], gap_seconds=event_gap_seconds
        )
        est_i, est_p = concat_event_lists_with_offsets(
            est_events_c[c], gap_seconds=event_gap_seconds
        )
        if ref_i.shape[0] == 0 and est_i.shape[0] == 0:
            event_class_f1.append(0.0)
            continue
        _, _, f_c, _ = evaluate_notes(
            ref_i,
            ref_p,
            est_i,
            est_p,
            offset_ratio=None,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=0,
        )
        event_class_f1.append(float(f_c))
    event_ma_f1 = float(np.mean(event_class_f1)) if event_class_f1 else 0.0

    return {
        "frame": {
            "micro_f1": frame_mi_f1,
            "macro_f1": frame_ma_f1,
            "per_class_f1": frame_class_f1,
        },
        "event": {
            "micro_f1": event_mi_f1,
            "macro_f1": event_ma_f1,
            "per_class_f1": event_class_f1,
        },
        "thresholds": {
            "onset": onset_th,
            "frame": frame_th,
            "onset_tolerance": onset_tolerance,
        },
        "n_chunks": n_chunks,
    }


def format_ipt_metrics_report(
    metrics: Dict,
    epoch: Optional[int] = None,
    title: Optional[str] = None,
) -> str:
    """Human-readable table for logging."""
    lines: List[str] = []
    if title is None:
        title = "IPT validation metrics"
        if epoch is not None:
            title += f" (epoch {epoch})"
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)

    fr = metrics["frame"]
    ev = metrics["event"]
    lines.append(
        f"FRAME  [overall]  MI-F1: {fr['micro_f1'] * 100:5.1f}%   MA-F1: {fr['macro_f1'] * 100:5.1f}%"
    )
    lines.append(
        f"EVENT  [overall]  MI-F1: {ev['micro_f1'] * 100:5.1f}%   MA-F1: {ev['macro_f1'] * 100:5.1f}%"
    )
    lines.append("-" * 72)
    lines.append(f"{'class':<12} {'frame F1':>10} {'event F1':>10}")
    for name, f_fr, f_ev in zip(IPT_NAMES, fr["per_class_f1"], ev["per_class_f1"]):
        lines.append(f"{name:<12} {f_fr * 100:9.1f}% {f_ev * 100:9.1f}%")
    lines.append("=" * 72)
    return "\n".join(lines)


def metrics_to_json_serializable(metrics: Dict) -> Dict:
    """Convert metrics dict for json.dump."""
    out = {
        "frame": {
            "micro_f1": float(metrics["frame"]["micro_f1"]),
            "macro_f1": float(metrics["frame"]["macro_f1"]),
            "per_class_f1": {
                IPT_NAMES[i]: float(metrics["frame"]["per_class_f1"][i])
                for i in range(NUM_LABELS)
            },
        },
        "event": {
            "micro_f1": float(metrics["event"]["micro_f1"]),
            "macro_f1": float(metrics["event"]["macro_f1"]),
            "per_class_f1": {
                IPT_NAMES[i]: float(metrics["event"]["per_class_f1"][i])
                for i in range(NUM_LABELS)
            },
        },
        "thresholds": metrics.get("thresholds", {}),
        "n_chunks": int(metrics.get("n_chunks", 0)),
    }
    return out


def format_ipt_test_report(
    metrics: Dict,
    *,
    model_name: str,
    checkpoint_path: str,
    test_time: str,
    dataset: str,
    test_group: str,
) -> str:
    """Full text report for offline test scripts (metadata + metrics table)."""
    header = [
        f"model_name      : {model_name}",
        f"checkpoint      : {checkpoint_path}",
        f"test_time       : {test_time}",
        f"dataset         : {dataset}",
        f"test_split      : {test_group}",
        f"n_test_chunks   : {metrics.get('n_chunks', 0)}",
        f"onset_threshold : {metrics['thresholds'].get('onset', 0.5)}",
        f"frame_threshold : {metrics['thresholds'].get('frame', 0.5)}",
        f"onset_tolerance : {metrics['thresholds'].get('onset_tolerance', 0.05)}",
        "",
    ]
    body = format_ipt_metrics_report(metrics, title="IPT test metrics")
    return "\n".join(header) + body + "\n"


def append_metrics_jsonl(path: str, epoch: int, metrics: Dict, extra: Optional[Dict] = None) -> None:
    rec = {"epoch": epoch, **metrics_to_json_serializable(metrics)}
    if extra:
        rec.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
