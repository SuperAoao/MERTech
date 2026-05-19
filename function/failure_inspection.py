"""Plot validation chunks where frame F1 is high but event (note) F1 is low."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import FEATURE_RATE, NUM_LABELS
from metrics_ipt import IPT_NAMES


def _frame_f1_chunk(pred_c: np.ndarray, tar_c: np.ndarray, frame_th: float) -> float:
    p = (pred_c > frame_th).astype(np.uint8)
    t = (tar_c > 0.5).astype(np.uint8)
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def record_failure_modes(
    pred_ipt_chunks: Sequence[np.ndarray],
    tar_ipt_chunks: Sequence[np.ndarray],
    pred_onset_chunks: Sequence[np.ndarray],
    tar_onset_chunks: Sequence[np.ndarray],
    event_f1_per_chunk_class: Sequence[Sequence[float]],
    *,
    save_dir: str,
    epoch: int,
    frame_th_per_class: Optional[Sequence[float]] = None,
    default_frame_th: float = 0.5,
    frame_f1_min: float = 0.8,
    event_f1_max: float = 0.4,
    max_plots_per_class: int = 3,
    focus_classes: Optional[Sequence[int]] = None,
) -> List[str]:
    """
    Save PNGs for chunks with high frame F1 but low event F1 for a given class.
    Returns list of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    if focus_classes is None:
        focus_classes = list(range(NUM_LABELS))

    saved: List[str] = []
    for c in focus_classes:
        count = 0
        frame_th = (
            float(frame_th_per_class[c])
            if frame_th_per_class is not None
            else default_frame_th
        )
        for i in range(len(pred_ipt_chunks)):
            if count >= max_plots_per_class:
                break
            ff1 = _frame_f1_chunk(
                pred_ipt_chunks[i][c], tar_ipt_chunks[i][c], frame_th
            )
            ef1 = float(event_f1_per_chunk_class[i][c])
            if ff1 < frame_f1_min or ef1 > event_f1_max:
                continue

            pred_o = np.asarray(pred_onset_chunks[i]).reshape(-1)
            tar_o = np.asarray(tar_onset_chunks[i]).reshape(-1)
            pred_i = np.asarray(pred_ipt_chunks[i][c]).reshape(-1)
            tar_i = np.asarray(tar_ipt_chunks[i][c]).reshape(-1)
            t_sec = np.arange(len(pred_o)) / float(FEATURE_RATE)

            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            fig.suptitle(
                f"epoch {epoch} chunk {i} class {c} ({IPT_NAMES[c]}) "
                f"frame_f1={ff1:.2f} event_f1={ef1:.2f}"
            )
            axes[0].plot(t_sec, tar_o, label="GT onset", color="C0", alpha=0.8)
            axes[0].plot(t_sec, pred_o, label="Pred onset", color="C1", alpha=0.8)
            axes[0].set_ylabel("onset prob")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)

            axes[1].fill_between(t_sec, 0, tar_i, alpha=0.25, color="C0", label="GT IPT")
            axes[1].plot(t_sec, pred_i, label="Pred IPT", color="C1", linewidth=1.0)
            axes[1].axhline(frame_th, color="gray", linestyle="--", linewidth=0.8, label="frame_th")
            axes[1].set_xlabel("time (s)")
            axes[1].set_ylabel("IPT prob")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)

            out_path = os.path.join(
                save_dir,
                f"epoch{epoch:04d}_chunk{i:04d}_{IPT_NAMES[c]}_ff1{ff1:.2f}_ef1{ef1:.2f}.png",
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            saved.append(out_path)
            count += 1
    return saved
