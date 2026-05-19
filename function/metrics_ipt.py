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

from config import (
    FEATURE_RATE,
    NUM_LABELS,
    THRESHOLD_SWEEP_VALUES,
)

from lib import (
    compute_metrics_with_note,
    compute_metrics_with_note_no_infer,
    extract_notes,
)

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


def _frame_th_for_class(
    c: int,
    frame_th: float,
    frame_th_per_class: Optional[Sequence[float]],
) -> float:
    if frame_th_per_class is not None:
        return float(frame_th_per_class[c])
    return float(frame_th)


def _onset_th_for_class(
    c: int,
    onset_th: float,
    onset_th_per_class: Optional[Sequence[float]],
) -> float:
    if onset_th_per_class is not None:
        return float(onset_th_per_class[c])
    return float(onset_th)


def evaluate_ipt_chunks(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    onset_th: float = 0.5,
    frame_th: float = 0.5,
    onset_tolerance: float = 0.05,
    event_gap_seconds: float = 1.0,
    frame_th_per_class: Optional[Sequence[float]] = None,
    onset_th_per_class: Optional[Sequence[float]] = None,
    return_chunk_event_f1: bool = False,
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
    chunk_event_f1: List[List[float]] = (
        [[0.0] * NUM_LABELS for _ in range(n_chunks)] if return_chunk_event_f1 else []
    )

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

        pred_bin = np.zeros_like(pred_IPT, dtype=np.uint8)
        tar_bin = (tar_IPT > 0.5).astype(np.uint8)
        for c in range(NUM_LABELS):
            th_c = _frame_th_for_class(c, frame_th, frame_th_per_class)
            pred_bin[c] = (pred_IPT[c] > th_c).astype(np.uint8)

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
            oth = _onset_th_for_class(c, onset_th, onset_th_per_class)
            fth = _frame_th_for_class(c, frame_th, frame_th_per_class)
            ref_i_c, ref_p_c = events_from_shared_onset(
                tar_onset, tar_IPT[c : c + 1], onset_th=oth, frame_th=fth
            )
            est_i_c, est_p_c = events_from_shared_onset(
                pred_onset, pred_IPT[c : c + 1], onset_th=oth, frame_th=fth
            )
            ref_events_c[c].append((ref_i_c, ref_p_c, dur_sec))
            est_events_c[c].append((est_i_c, est_p_c, dur_sec))
            if return_chunk_event_f1:
                if ref_i_c.shape[0] == 0 and est_i_c.shape[0] == 0:
                    chunk_event_f1[i][c] = 0.0
                else:
                    _, _, f_ch, _ = evaluate_notes(
                        ref_i_c,
                        ref_p_c,
                        est_i_c,
                        est_p_c,
                        offset_ratio=None,
                        onset_tolerance=onset_tolerance,
                        pitch_tolerance=0,
                    )
                    chunk_event_f1[i][c] = float(f_ch)

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
            "onset_per_class": list(onset_th_per_class) if onset_th_per_class else None,
            "frame_per_class": list(frame_th_per_class) if frame_th_per_class else None,
            "onset_tolerance": onset_tolerance,
        },
        "n_chunks": n_chunks,
        "chunk_event_f1": chunk_event_f1,
    }


def _event_f1_for_class_concat(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    class_idx: int,
    onset_th: float,
    frame_th: float,
    onset_tolerance: float = 0.05,
    event_gap_seconds: float = 1.0,
) -> float:
    ref_events_c: List[Tuple[np.ndarray, np.ndarray, float]] = []
    est_events_c: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(len(pred_ipt_chunks)):
        pred_IPT = np.asarray(pred_ipt_chunks[i], dtype=np.float32)
        tar_IPT = np.asarray(tar_ipt_chunks[i], dtype=np.float32)
        pred_onset = np.asarray(pred_onset_chunks[i], dtype=np.float32)
        tar_onset = np.asarray(tar_onset_chunks[i], dtype=np.float32)
        if pred_onset.ndim == 1:
            pred_onset = pred_onset.reshape(1, -1)
        if tar_onset.ndim == 1:
            tar_onset = tar_onset.reshape(1, -1)
        t_len = min(pred_IPT.shape[-1], tar_IPT.shape[-1])
        dur_sec = t_len / float(FEATURE_RATE)
        ref_i, ref_p = events_from_shared_onset(
            tar_onset[:, :t_len],
            tar_IPT[:, :t_len][class_idx : class_idx + 1],
            onset_th=onset_th,
            frame_th=frame_th,
        )
        est_i, est_p = events_from_shared_onset(
            pred_onset[:, :t_len],
            pred_IPT[:, :t_len][class_idx : class_idx + 1],
            onset_th=onset_th,
            frame_th=frame_th,
        )
        ref_events_c.append((ref_i, ref_p, dur_sec))
        est_events_c.append((est_i, est_p, dur_sec))
    ref_i, ref_p = concat_event_lists_with_offsets(ref_events_c, gap_seconds=event_gap_seconds)
    est_i, est_p = concat_event_lists_with_offsets(est_events_c, gap_seconds=event_gap_seconds)
    if ref_i.shape[0] == 0 and est_i.shape[0] == 0:
        return 0.0
    _, _, f1, _ = evaluate_notes(
        ref_i,
        ref_p,
        est_i,
        est_p,
        offset_ratio=None,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=0,
    )
    return float(f1)


def sweep_thresholds_per_class(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    grid: Optional[Sequence[float]] = None,
    onset_tolerance: float = 0.05,
    event_gap_seconds: float = 1.0,
    focus_classes: Optional[Sequence[int]] = None,
) -> Dict:
    """Grid search onset_th x frame_th per class; maximize class event F1."""
    if grid is None:
        grid = list(THRESHOLD_SWEEP_VALUES)
    classes = list(range(NUM_LABELS)) if focus_classes is None else list(focus_classes)

    best_onset = [0.5] * NUM_LABELS
    best_frame = [0.5] * NUM_LABELS
    best_f1 = [0.0] * NUM_LABELS
    sweep_log: Dict[str, Dict] = {}

    for c in classes:
        best_f1_c = -1.0
        best_ot, best_ft = 0.5, 0.5
        for onset_th in grid:
            for frame_th in grid:
                f1 = _event_f1_for_class_concat(
                    pred_ipt_chunks,
                    tar_ipt_chunks,
                    pred_onset_chunks,
                    tar_onset_chunks,
                    c,
                    float(onset_th),
                    float(frame_th),
                    onset_tolerance=onset_tolerance,
                    event_gap_seconds=event_gap_seconds,
                )
                if f1 > best_f1_c:
                    best_f1_c, best_ot, best_ft = f1, float(onset_th), float(frame_th)
        best_onset[c] = best_ot
        best_frame[c] = best_ft
        best_f1[c] = best_f1_c
        sweep_log[IPT_NAMES[c]] = {
            "best_onset_th": best_ot,
            "best_frame_th": best_ft,
            "best_event_f1": best_f1_c,
        }

    return {
        "onset_th_per_class": best_onset,
        "frame_th_per_class": best_frame,
        "event_f1_per_class": best_f1,
        "grid": list(grid),
        "per_class": sweep_log,
    }


def _concat_chunks(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((NUM_LABELS, 0), dtype=np.float32)
    return np.concatenate([np.asarray(c) for c in chunks], axis=-1)


def evaluate_note_prepost(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    onset_th: float = 0.5,
    frame_th: float = 0.5,
) -> Dict:
    """
    Before post-processing: no onset inference (IPT-only events).
    After post-processing: shared onset + IPT frames (paper pipeline).
    """
    pred_IPT = _concat_chunks(pred_ipt_chunks)
    tar_IPT = _concat_chunks(tar_ipt_chunks)
    pred_onset = _concat_chunks(pred_onset_chunks)
    tar_onset = _concat_chunks(tar_onset_chunks)
    if pred_onset.shape[0] > 1:
        pred_onset = pred_onset[:1]
    if tar_onset.shape[0] > 1:
        tar_onset = tar_onset[:1]

    pred_bin = pred_IPT.copy()
    pred_bin[pred_bin > frame_th] = 1
    pred_bin[pred_bin <= frame_th] = 0
    pred_onset_bin = pred_onset.copy()
    pred_onset_bin[pred_onset_bin > onset_th] = 1
    pred_onset_bin[pred_onset_bin <= onset_th] = 0

    m_before, _ = compute_metrics_with_note_no_infer(
        pred_bin, tar_IPT, pred_onset_bin, tar_onset
    )
    m_after, _ = compute_metrics_with_note(
        pred_bin, tar_IPT, pred_onset_bin, tar_onset
    )

    onset_row = tar_onset[0:1, :]
    per_class_note_after = []
    per_class_frame_after = []
    for c in range(NUM_LABELS):
        mc, _ = compute_metrics_with_note(
            pred_bin[c : c + 1],
            tar_IPT[c : c + 1],
            pred_onset_bin,
            onset_row,
        )
        per_class_note_after.append(float(mc["metric/note/f1"][0]))
        per_class_frame_after.append(float(mc["metric/IPT_frame/f1"][0]))

    macro_note_f1 = float(np.mean(per_class_note_after))

    def _pack(m):
        return {
            "note_f1": float(m["metric/note/f1"][0]),
            "note_precision": float(m["metric/note/precision"][0]),
            "note_recall": float(m["metric/note/recall"][0]),
            "frame_f1": float(m["metric/IPT_frame/f1"][0]),
            "frame_precision": float(m["metric/IPT_frame/precision"][0]),
            "frame_recall": float(m["metric/IPT_frame/recall"][0]),
        }

    return {
        "before": _pack(m_before),
        "after": _pack(m_after),
        "macro_note_f1": macro_note_f1,
        "per_class_note_f1_after": per_class_note_after,
        "per_class_frame_f1_after": per_class_frame_after,
    }


def run_full_validation_metrics(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    *,
    do_threshold_sweep: bool = True,
    onset_tolerance: float = 0.05,
    event_gap_seconds: float = 1.0,
    default_onset_th: float = 0.5,
    default_frame_th: float = 0.5,
    sweep_focus_classes: Optional[Sequence[int]] = None,
) -> Dict:
    """IPT table metrics + threshold sweep + before/after note metrics."""
    sweep = None
    onset_th_pc = None
    frame_th_pc = None
    if do_threshold_sweep:
        sweep = sweep_thresholds_per_class(
            pred_ipt_chunks,
            tar_ipt_chunks,
            pred_onset_chunks,
            tar_onset_chunks,
            onset_tolerance=onset_tolerance,
            event_gap_seconds=event_gap_seconds,
            focus_classes=sweep_focus_classes,
        )
        onset_th_pc = sweep["onset_th_per_class"]
        frame_th_pc = sweep["frame_th_per_class"]

    ipt_default = evaluate_ipt_chunks(
        pred_ipt_chunks,
        tar_ipt_chunks,
        pred_onset_chunks,
        tar_onset_chunks,
        onset_th=default_onset_th,
        frame_th=default_frame_th,
        onset_tolerance=onset_tolerance,
        event_gap_seconds=event_gap_seconds,
        return_chunk_event_f1=True,
    )
    ipt_tuned = evaluate_ipt_chunks(
        pred_ipt_chunks,
        tar_ipt_chunks,
        pred_onset_chunks,
        tar_onset_chunks,
        onset_th=default_onset_th,
        frame_th=default_frame_th,
        onset_th_per_class=onset_th_pc,
        frame_th_per_class=frame_th_pc,
        onset_tolerance=onset_tolerance,
        event_gap_seconds=event_gap_seconds,
    ) if sweep is not None else ipt_default

    prepost = evaluate_note_prepost(
        pred_ipt_chunks,
        tar_ipt_chunks,
        pred_onset_chunks,
        tar_onset_chunks,
        onset_th=default_onset_th,
        frame_th=default_frame_th,
    )

    return {
        "ipt_default_thresholds": ipt_default,
        "ipt_swept_thresholds": ipt_tuned,
        "threshold_sweep": sweep,
        "prepost": prepost,
    }


def format_prepost_report(prepost: Dict) -> str:
    """Before / after onset post-processing (mir_eval note + frame)."""
    lines = [
        "-" * 72,
        "NOTE METRICS — before post-processing (no onset inference)",
        "-" * 72,
    ]
    b = prepost["before"]
    lines.append(
        f"  note F1: {b['note_f1'] * 100:5.1f}%   frame F1: {b['frame_f1'] * 100:5.1f}%"
    )
    lines.append(
        f"  note P/R: {b['note_precision'] * 100:.1f}% / {b['note_recall'] * 100:.1f}%"
    )
    lines.extend(
        [
            "-" * 72,
            "NOTE METRICS — after post-processing (shared onset + IPT)",
            "-" * 72,
        ]
    )
    a = prepost["after"]
    lines.append(
        f"  note F1: {a['note_f1'] * 100:5.1f}%   frame F1: {a['frame_f1'] * 100:5.1f}%"
    )
    lines.append(
        f"  note P/R: {a['note_precision'] * 100:.1f}% / {a['note_recall'] * 100:.1f}%"
    )
    lines.append(f"  macro_note_f1 (mean per-class note F1): {prepost['macro_note_f1'] * 100:.1f}%")
    lines.append(f"{'class':<12} {'note F1':>10} {'frame F1':>10}")
    for name, nf, ff in zip(
        IPT_NAMES,
        prepost["per_class_note_f1_after"],
        prepost["per_class_frame_f1_after"],
    ):
        lines.append(f"{name:<12} {nf * 100:9.1f}% {ff * 100:9.1f}%")
    return "\n".join(lines)


def format_threshold_sweep_report(sweep: Optional[Dict]) -> str:
    if sweep is None:
        return ""
    lines = ["-" * 72, "THRESHOLD SWEEP (per-class best event F1)", "-" * 72]
    lines.append(f"grid: {sweep.get('grid')}")
    for c, name in enumerate(IPT_NAMES):
        lines.append(
            f"  {name:<12} onset_th={sweep['onset_th_per_class'][c]:.1f}  "
            f"frame_th={sweep['frame_th_per_class'][c]:.1f}  "
            f"event_f1={sweep['event_f1_per_class'][c] * 100:.1f}%"
        )
    return "\n".join(lines)


def format_full_evaluation_report(
    bundle: Dict,
    epoch: Optional[int] = None,
) -> str:
    """Table-1 IPT metrics (default + swept) + pre/post note block."""
    parts = []
    title_suffix = f" (epoch {epoch})" if epoch is not None else ""
    parts.append(
        format_ipt_metrics_report(
            bundle["ipt_default_thresholds"],
            title=f"IPT metrics @ default thresholds{title_suffix}",
        )
    )
    if bundle.get("ipt_swept_thresholds") is not bundle.get("ipt_default_thresholds"):
        parts.append(
            format_ipt_metrics_report(
                bundle["ipt_swept_thresholds"],
                title=f"IPT metrics @ swept per-class thresholds{title_suffix}",
            )
        )
    parts.append(format_threshold_sweep_report(bundle.get("threshold_sweep")))
    parts.append(format_prepost_report(bundle["prepost"]))
    return "\n\n".join(p for p in parts if p)


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
    """Convert single IPT metrics dict for json.dump."""
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


def bundle_to_json_serializable(bundle: Dict) -> Dict:
    """Full validation / test bundle for json logging."""
    out = {
        "ipt_default": metrics_to_json_serializable(bundle["ipt_default_thresholds"]),
        "prepost": bundle["prepost"],
    }
    if bundle.get("ipt_swept_thresholds") is not None:
        out["ipt_swept"] = metrics_to_json_serializable(bundle["ipt_swept_thresholds"])
    if bundle.get("threshold_sweep") is not None:
        out["threshold_sweep"] = bundle["threshold_sweep"]
    return out


def format_ipt_test_report(
    bundle: Dict,
    *,
    model_name: str,
    checkpoint_path: str,
    test_time: str,
    dataset: str,
    test_group: str,
) -> str:
    """Full text report for offline test scripts (metadata + all metric blocks)."""
    n_chunks = bundle.get("ipt_default_thresholds", {}).get("n_chunks", 0)
    header = [
        f"model_name      : {model_name}",
        f"checkpoint      : {checkpoint_path}",
        f"test_time       : {test_time}",
        f"dataset         : {dataset}",
        f"test_split      : {test_group}",
        f"n_chunks        : {n_chunks}",
        "",
    ]
    body = format_full_evaluation_report(bundle)
    return "\n".join(header) + body + "\n"


def append_metrics_jsonl(path: str, epoch: int, bundle: Dict, extra: Optional[Dict] = None) -> None:
    rec = {"epoch": epoch, **bundle_to_json_serializable(bundle)}
    if extra:
        rec.update(extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
