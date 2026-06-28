#!/usr/bin/env python3
"""Evaluate a trained Guzheng IPT checkpoint on the test split."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "function"))

from experiment_config import (  # noqa: E402
    REPO_ROOT,
    apply_experiment_config,
    find_best_checkpoint,
    load_experiment_config,
    load_sslnet_state_dict,
    resolve_config_for_run,
)
from load_data import load  # noqa: E402
from metrics_ipt import (  # noqa: E402
    bundle_to_json_serializable,
    format_ipt_test_report,
    run_full_validation_metrics,
)
from model import SSLNet  # noqa: E402
import config as config_mod  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)


def model_name_from_run(run_dir: Path, checkpoint_path: Path) -> str:
    ckpt_base = checkpoint_path.name
    return f"{run_dir.name}_{ckpt_base}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MERTech Guzheng IPT model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run",
        type=str,
        help="Run directory (loads config.yaml snapshot from this run)",
    )
    group.add_argument(
        "--config",
        type=str,
        help="Experiment YAML only (no run snapshot; for legacy checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: newest best_e_* under --run)",
    )
    parser.add_argument(
        "--test-group",
        type=str,
        default="test",
        help="Comma-separated label splits to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write reports (default: <run>/test or output/test_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (default: from config)",
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace, run_dir: Path | None) -> Path:
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt
    if run_dir is None:
        raise ValueError("--checkpoint is required when not using --run")
    ckpt = find_best_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No best_e_* checkpoint found under: {run_dir}")
    return ckpt.resolve()


def main() -> None:
    args = parse_args()
    run_dir: Path | None = None

    if args.run:
        run_dir = Path(args.run).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        cfg = resolve_config_for_run(run_dir)
        apply_experiment_config(cfg, run_dir=str(run_dir))
    else:
        from experiment_config import load_experiment_config

        config_path = Path(args.config).resolve()
        cfg = load_experiment_config(config_path)
        apply_experiment_config(cfg, run_dir=None)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    checkpoint_path = resolve_checkpoint(args, run_dir)
    test_groups = [g.strip() for g in args.test_group.split(",") if g.strip()]

    if args.output_dir:
        results_dir = Path(args.output_dir).resolve()
    elif run_dir is not None:
        results_dir = run_dir / "test"
    else:
        results_dir = REPO_ROOT / "output" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_time = datetime.now()
    test_time_str = test_time.strftime("%Y%m%d_%H%M%S")
    model_name = (
        model_name_from_run(run_dir, checkpoint_path)
        if run_dir is not None
        else f"{config_mod.EXPERIMENT_ID}_{checkpoint_path.name}"
    )

    model = SSLNet(
        url=config_mod.URL,
        class_num=config_mod.NUM_LABELS * (config_mod.MAX_MIDI - config_mod.MIN_MIDI + 1),
        weight_sum=1,
        freeze_all=config_mod.FREEZE_ALL,
    ).to(config_mod.device)
    load_sslnet_state_dict(model, checkpoint_path)
    model.eval()
    print("finishing loading model")

    wav_dir = config_mod.DATASET + "/data"
    csv_dir = config_mod.DATASET + "/labels"
    Xte, Yte, _, Yte_o, _, _ = load(wav_dir, csv_dir, test_groups, None, None)
    print("finishing loading dataset")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config_mod.URL, trust_remote_code=True
    )

    pred_ipt_chunks = []
    tar_ipt_chunks = []
    pred_onset_chunks = []
    tar_onset_chunks = []

    print("start predicting...")
    with torch.no_grad():
        for i, x in enumerate(Xte):
            data = (
                processor(x, sampling_rate=config_mod.MERT_SAMPLE_RATE, return_tensors="pt")[
                    "input_values"
                ]
                .float()
                .to(config_mod.device)
            )
            target = Yte[i]
            target_o = Yte_o[i]
            IPT_pred, _, onset_pred = model(data)
            f_pred = torch.sigmoid(IPT_pred.squeeze(0)).detach().cpu().numpy()
            o_pred = torch.sigmoid(onset_pred.squeeze(0)).detach().cpu().numpy()

            t_len = target.shape[-1]
            pred_ipt_chunks.append(f_pred[:, :t_len].copy())
            tar_ipt_chunks.append(np.asarray(target[:, :t_len], dtype=np.float32))
            pred_onset_chunks.append(o_pred[:, :t_len].copy())
            tar_onset_chunks.append(np.asarray(target_o[:, :t_len], dtype=np.float32))

    val_bundle = run_full_validation_metrics(
        pred_ipt_chunks,
        tar_ipt_chunks,
        pred_onset_chunks,
        tar_onset_chunks,
        do_threshold_sweep=True,
        onset_tolerance=config_mod.EVAL_ONSET_TOLERANCE,
        event_gap_seconds=config_mod.EVAL_EVENT_GAP_SECONDS,
        default_onset_th=config_mod.EVAL_ONSET_THRESHOLD,
        default_frame_th=config_mod.EVAL_FRAME_THRESHOLD,
        sweep_focus_classes=config_mod.THRESHOLD_SWEEP_FOCUS_CLASSES,
    )

    report = format_ipt_test_report(
        val_bundle,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        test_time=test_time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset=config_mod.DATASET,
        test_group=",".join(test_groups),
    )

    out_txt = results_dir / safe_filename(f"{model_name}_{test_time_str}.txt")
    out_json = results_dir / safe_filename(f"{model_name}_{test_time_str}.json")

    result_payload = {
        "experiment_id": config_mod.EXPERIMENT_ID,
        "model_name": model_name,
        "save_name": config_mod.saveName,
        "run_dir": str(run_dir) if run_dir else None,
        "checkpoint": str(checkpoint_path),
        "test_time": test_time.isoformat(),
        "dataset": config_mod.DATASET,
        "test_group": test_groups,
        "metrics": bundle_to_json_serializable(val_bundle),
    }

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(report)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    if run_dir is not None:
        latest_path = run_dir / "test" / "latest.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    print(report)
    print("Saved report: %s" % out_txt)
    print("Saved JSON  : %s" % out_json)


if __name__ == "__main__":
    main()
