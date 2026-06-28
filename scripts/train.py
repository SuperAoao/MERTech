#!/usr/bin/env python3
"""Train Guzheng IPT model from a YAML experiment config."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "function"))

from experiment_config import (  # noqa: E402
    REPO_ROOT,
    apply_experiment_config,
    build_run_dir,
    finalize_run_meta,
    find_best_checkpoint,
    load_experiment_config,
    load_sslnet_state_dict,
    snapshot_run,
)
from fit import Trainer  # noqa: E402
from lib import Data2Torch2  # noqa: E402
from load_data import load  # noqa: E402
from model import SSLNet  # noqa: E402
import config as config_mod  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_weight(Ytr: np.ndarray) -> torch.Tensor:
    mp = Ytr[:].sum(0).sum(0)
    mmp = mp.astype(np.float32) / mp.sum()
    cc = ((mmp.mean() / mmp) * ((1 - mmp) / (1 - mmp.mean()))) ** 0.3
    return torch.from_numpy(cc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MERTech Guzheng IPT model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "fpt_combined_pn.yaml"),
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Override run output directory (default: auto under runs/guzheng/)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(REPO_ROOT / "runs" / "guzheng"),
        help="Root directory for auto-generated run folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_experiment_config(config_path)
    training = cfg["training"]

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = build_run_dir(cfg, runs_root=args.runs_root)

    apply_experiment_config(cfg, run_dir=str(run_dir))
    snapshot_run(cfg, run_dir, config_path, argv=sys.argv)
    set_seed(config_mod.SEED)

    out_model_fn = str(run_dir)
    if not out_model_fn.endswith(os.sep):
        out_model_fn += os.sep

    wav_dir = config_mod.DATASET + "/data"
    csv_dir = config_mod.DATASET + "/labels"
    Xtr, Ytr, Ytr_p, Ytr_o, avg, std = load(wav_dir, csv_dir, ["train"])
    Xva, Yva, Yva_p, Yva_o, _, _ = load(wav_dir, csv_dir, ["validation"], avg, std)
    print("finishing data loading...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config_mod.URL, trust_remote_code=True
    )
    Xtrs = processor(Xtr, sampling_rate=config_mod.MERT_SAMPLE_RATE, return_tensors="pt")
    Xvas = processor(Xva, sampling_rate=config_mod.MERT_SAMPLE_RATE, return_tensors="pt")

    t_kwargs = {
        "batch_size": config_mod.BATCH_SIZE,
        "num_workers": 2,
        "pin_memory": True,
        "drop_last": True,
    }
    v_kwargs = {"batch_size": 1, "num_workers": 2, "pin_memory": True}
    tr_loader = torch.utils.data.DataLoader(
        Data2Torch2([Xtrs["input_values"], Ytr, Ytr_p, Ytr_o]),
        shuffle=True,
        **t_kwargs,
    )
    va_loader = torch.utils.data.DataLoader(
        Data2Torch2([Xvas["input_values"], Yva, Yva_p, Yva_o]),
        **v_kwargs,
    )
    print("finishing data building...")

    print(
        "[experiment] id=%s run_dir=%s"
        % (config_mod.EXPERIMENT_ID, run_dir)
    )
    fpt_extra = ""
    if config_mod.USE_FPT:
        fpt_extra = (
            " | levels=%d layers=%d heads=%d dropout=%s"
            % (
                config_mod.FPT_LEVELS,
                config_mod.FPT_NUM_LAYERS,
                config_mod.FPT_NUM_HEADS,
                config_mod.FPT_DROPOUT,
            )
        )
    print(
        "[FPT] enabled=%s%s | onset_bypass=%s | early_stop=%s best_metric=%s"
        % (
            config_mod.USE_FPT,
            fpt_extra,
            config_mod.USE_FPT_ONSET_BYPASS and config_mod.USE_FPT,
            config_mod.ENABLE_EARLY_STOPPING,
            config_mod.BEST_CHECKPOINT_METRIC,
        )
    )
    if config_mod.USE_PN_HEAD:
        print(
            "[PN head] context=%d hidden=%d fusion_alpha=%.2f "
            "loss_weight=%.2f pos_weight=%.1f pluck_gate=%s"
            % (
                config_mod.PN_HEAD_CONTEXT,
                config_mod.PN_HEAD_HIDDEN,
                config_mod.PN_FUSION_ALPHA,
                config_mod.PN_HEAD_LOSS_WEIGHT,
                config_mod.PN_HEAD_POS_WEIGHT,
                config_mod.PN_HEAD_USE_PLUCK_GATE,
            )
        )

    model = SSLNet(
        url=config_mod.URL,
        class_num=config_mod.NUM_LABELS * (config_mod.MAX_MIDI - config_mod.MIN_MIDI + 1),
        weight_sum=1,
        freeze_all=config_mod.FREEZE_ALL,
    ).to(config_mod.device)
    print(model)

    inverse_feq = get_weight(Ytr.transpose(0, 2, 1))
    trainer = Trainer(
        model,
        float(training["lr"]),
        int(training["max_epochs"]),
        out_model_fn,
        validation_interval=int(training["validation_interval"]),
        save_interval=int(training["save_interval"]),
    )
    trainer.fit(tr_loader, va_loader, inverse_feq)

    best_ckpt = find_best_checkpoint(run_dir)
    finalize_run_meta(
        run_dir,
        status="finished",
        best_checkpoint=str(best_ckpt) if best_ckpt else None,
        save_name=config_mod.saveName,
    )
    print("Run finished. run_dir=%s" % run_dir)
    if best_ckpt:
        print("Best checkpoint: %s" % best_ckpt)
    print("Test with: python scripts/test.py --run %s" % run_dir)


if __name__ == "__main__":
    main()
