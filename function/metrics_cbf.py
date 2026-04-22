import numpy as np
from scipy.stats import hmean
from collections import defaultdict

from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes

from config_cbf import FEATURE_RATE


def _f1_from_pr(p: float, r: float, eps: float = 1e-12) -> float:
    return float(hmean([p + eps, r + eps]) - eps)


def frame_prf_per_class(pred_bin: np.ndarray, tar_bin: np.ndarray):
    """
    pred_bin, tar_bin: [C, T] binary {0,1}
    Returns dict with per-class precision/recall/f1 arrays (len C)
    """
    assert pred_bin.shape == tar_bin.shape and pred_bin.ndim == 2
    C, T = pred_bin.shape
    eps = 1e-12
    prec = np.zeros(C, dtype=np.float64)
    rec = np.zeros(C, dtype=np.float64)
    f1 = np.zeros(C, dtype=np.float64)
    for c in range(C):
        p = pred_bin[c]
        t = tar_bin[c]
        tp = float(np.sum((p == 1) & (t == 1)))
        fp = float(np.sum((p == 1) & (t == 0)))
        fn = float(np.sum((p == 0) & (t == 1)))
        pr = tp / (tp + fp + eps)
        rr = tp / (tp + fn + eps)
        prec[c] = pr
        rec[c] = rr
        f1[c] = _f1_from_pr(pr, rr, eps=eps)
    return {"precision": prec, "recall": rec, "f1": f1}


def frame_micro_macro_f1(pred_bin: np.ndarray, tar_bin: np.ndarray):
    """
    pred_bin, tar_bin: [C, T] binary
    Returns: (micro_f1, macro_f1)
    """
    assert pred_bin.shape == tar_bin.shape and pred_bin.ndim == 2
    eps = 1e-12
    tp = float(np.sum((pred_bin == 1) & (tar_bin == 1)))
    fp = float(np.sum((pred_bin == 1) & (tar_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (tar_bin == 1)))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    micro = _f1_from_pr(p, r, eps=eps)
    per = frame_prf_per_class(pred_bin, tar_bin)
    macro = float(np.mean(per["f1"]))
    return micro, macro


def _binary_to_intervals(x_1d: np.ndarray):
    """
    x_1d: [T] binary
    Returns intervals as [N,2] frame indices [on, off) with off exclusive.
    """
    assert x_1d.ndim == 1
    x = x_1d.astype(np.int8)
    T = x.shape[0]
    intervals = []
    in_seg = False
    start = 0
    for i in range(T):
        if not in_seg and x[i] == 1:
            in_seg = True
            start = i
        elif in_seg and x[i] == 0:
            intervals.append([start, i])
            in_seg = False
    if in_seg:
        intervals.append([start, T])
    return np.asarray(intervals, dtype=np.int64)


def event_prf_per_class(pred_bin: np.ndarray, tar_bin: np.ndarray, onset_tolerance_s: float = 0.05):
    """
    Event-level metrics per class using mir_eval's note matching on intervals only.
    pred_bin, tar_bin: [C, T] binary
    Returns dict with per-class precision/recall/f1 arrays (len C)
    """
    assert pred_bin.shape == tar_bin.shape and pred_bin.ndim == 2
    C, T = pred_bin.shape
    prec = np.zeros(C, dtype=np.float64)
    rec = np.zeros(C, dtype=np.float64)
    f1 = np.zeros(C, dtype=np.float64)
    for c in range(C):
        ref_int = _binary_to_intervals(tar_bin[c])
        est_int = _binary_to_intervals(pred_bin[c])

        # to seconds
        ref = (ref_int / FEATURE_RATE).astype(np.float64)
        est = (est_int / FEATURE_RATE).astype(np.float64)

        # dummy pitches (single class)
        p_ref = np.zeros((len(ref),), dtype=np.float64)
        p_est = np.zeros((len(est),), dtype=np.float64)

        if len(ref) == 0 and len(est) == 0:
            prec[c] = 0.0
            rec[c] = 0.0
            f1[c] = 0.0
            continue

        p, r, ff, _ = evaluate_notes(
            ref,
            p_ref,
            est,
            p_est,
            offset_ratio=None,
            onset_tolerance=onset_tolerance_s,
            pitch_tolerance=0,
        )
        prec[c] = p
        rec[c] = r
        f1[c] = ff
    return {"precision": prec, "recall": rec, "f1": f1}


def event_micro_macro_f1(pred_bin: np.ndarray, tar_bin: np.ndarray, onset_tolerance_s: float = 0.05):
    """
    Micro-F1 at event level: sum TP/FP/FN across classes using interval matching.
    Macro-F1: mean of per-class event F1.
    """
    assert pred_bin.shape == tar_bin.shape and pred_bin.ndim == 2
    per = event_prf_per_class(pred_bin, tar_bin, onset_tolerance_s=onset_tolerance_s)
    macro = float(np.mean(per["f1"]))

    # micro: accumulate counts by turning pr/rec back to counts via matching with mir_eval:
    # we approximate by summing TP/FP/FN inferred from P/R and n_ref/n_est per class.
    # This avoids re-implementing matching logic.
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for c in range(pred_bin.shape[0]):
        ref_n = len(_binary_to_intervals(tar_bin[c]))
        est_n = len(_binary_to_intervals(pred_bin[c]))
        p = float(per["precision"][c])
        r = float(per["recall"][c])
        # Handle empty cases
        if ref_n == 0 and est_n == 0:
            continue
        # TP from recall
        tp_c = r * ref_n
        fn_c = ref_n - tp_c
        # FP from precision (if est_n>0)
        if est_n > 0:
            tp_from_p = p * est_n
            # choose consistent TP estimate
            tp_c = min(tp_c, tp_from_p)
            fp_c = est_n - tp_c
        else:
            fp_c = 0.0
        tp += tp_c
        fp += fp_c
        fn += fn_c

    eps = 1e-12
    p_micro = tp / (tp + fp + eps)
    r_micro = tp / (tp + fn + eps)
    micro = _f1_from_pr(p_micro, r_micro, eps=eps)
    return micro, macro

