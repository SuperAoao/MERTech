"""
Experiment configuration: load YAML, apply to function.config, manage run directories.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import torch

import config as config_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "guzheng"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def default_experiment_dict() -> Dict[str, Any]:
    """Build a nested experiment dict from current function.config defaults."""
    return {
        "experiment_id": "default",
        "seed": 42,
        "dataset": config_mod.DATASET,
        "model": {
            "url": config_mod.URL,
            "freeze_all": config_mod.FREEZE_ALL,
            "use_fpt": config_mod.USE_FPT,
            "fpt_levels": config_mod.FPT_LEVELS,
            "fpt_num_layers": config_mod.FPT_NUM_LAYERS,
            "fpt_num_heads": config_mod.FPT_NUM_HEADS,
            "fpt_dropout": config_mod.FPT_DROPOUT,
            "use_pn_head": config_mod.USE_PN_HEAD,
            "pn_head_context": config_mod.PN_HEAD_CONTEXT,
            "pn_head_hidden": config_mod.PN_HEAD_HIDDEN,
            "pn_head_dropout": config_mod.PN_HEAD_DROPOUT,
            "pn_head_use_pluck_gate": config_mod.PN_HEAD_USE_PLUCK_GATE,
            "pn_head_use_onset_gate": config_mod.PN_HEAD_USE_ONSET_GATE,
            "pn_fusion_alpha": config_mod.PN_FUSION_ALPHA,
        },
        "training": {
            "lr": 1e-3,
            "max_epochs": 10000,
            "batch_size": config_mod.BATCH_SIZE,
            "validation_interval": 5,
            "save_interval": 100,
            "early_stopping": config_mod.EARLY_STOPPING,
            "enable_early_stopping": config_mod.ENABLE_EARLY_STOPPING,
            "best_checkpoint_metric": config_mod.BEST_CHECKPOINT_METRIC,
        },
        "loss": {
            "pitch_weight": config_mod.PITCH_LOSS_WEIGHT,
            "onset_weight": config_mod.ONSET_LOSS_WEIGHT,
            "pn_head_weight": config_mod.PN_HEAD_LOSS_WEIGHT,
            "pn_head_pos_weight": config_mod.PN_HEAD_POS_WEIGHT,
        },
        "eval": {
            "onset_threshold": config_mod.EVAL_ONSET_THRESHOLD,
            "frame_threshold": config_mod.EVAL_FRAME_THRESHOLD,
            "onset_tolerance": config_mod.EVAL_ONSET_TOLERANCE,
            "event_gap_seconds": config_mod.EVAL_EVENT_GAP_SECONDS,
            "threshold_sweep_values": list(config_mod.THRESHOLD_SWEEP_VALUES),
            "threshold_sweep_every_epoch": config_mod.THRESHOLD_SWEEP_EVERY_EPOCH,
            "threshold_sweep_focus_classes": config_mod.THRESHOLD_SWEEP_FOCUS_CLASSES,
            "failure_inspection": config_mod.FAILURE_INSPECTION,
            "failure_frame_f1_min": config_mod.FAILURE_FRAME_F1_MIN,
            "failure_event_f1_max": config_mod.FAILURE_EVENT_F1_MAX,
            "failure_max_plots_per_class": config_mod.FAILURE_MAX_PLOTS_PER_CLASS,
            "failure_focus_classes": list(config_mod.FAILURE_FOCUS_CLASSES),
        },
        "runtime": {
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "use_visdom": config_mod.USE_VISDOM,
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(default_experiment_dict(), user_cfg)


def apply_experiment_config(cfg: Dict[str, Any], *, run_dir: Optional[str] = None) -> None:
    """Patch function.config module globals from a nested experiment dict."""
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    loss = cfg.get("loss", {})
    eval_cfg = cfg.get("eval", {})
    runtime = cfg.get("runtime", {})

    url = model.get("url", config_mod.URL)
    mert_tail = url.split("/")[-1]
    experiment_id = cfg.get("experiment_id", "experiment")

    config_mod.EXPERIMENT_ID = experiment_id
    config_mod.RUN_DIR = run_dir
    config_mod.SEED = int(cfg.get("seed", 42))
    config_mod.DATASET = cfg.get("dataset", config_mod.DATASET)

    config_mod.URL = url
    config_mod.MERT_SAMPLE_RATE = 24000 if "MERT" in url else 16000
    config_mod.FREEZE_ALL = bool(model.get("freeze_all", config_mod.FREEZE_ALL))
    config_mod.USE_FPT = bool(model.get("use_fpt", config_mod.USE_FPT))
    config_mod.FPT_LEVELS = int(model.get("fpt_levels", config_mod.FPT_LEVELS))
    config_mod.FPT_NUM_LAYERS = int(model.get("fpt_num_layers", config_mod.FPT_NUM_LAYERS))
    config_mod.FPT_NUM_HEADS = int(model.get("fpt_num_heads", config_mod.FPT_NUM_HEADS))
    config_mod.FPT_DROPOUT = float(model.get("fpt_dropout", config_mod.FPT_DROPOUT))

    config_mod.USE_PN_HEAD = bool(model.get("use_pn_head", config_mod.USE_PN_HEAD))
    config_mod.PN_HEAD_CONTEXT = int(model.get("pn_head_context", config_mod.PN_HEAD_CONTEXT))
    config_mod.PN_HEAD_HIDDEN = int(model.get("pn_head_hidden", config_mod.PN_HEAD_HIDDEN))
    config_mod.PN_HEAD_DROPOUT = float(model.get("pn_head_dropout", config_mod.PN_HEAD_DROPOUT))
    config_mod.PN_HEAD_USE_PLUCK_GATE = bool(
        model.get("pn_head_use_pluck_gate", config_mod.PN_HEAD_USE_PLUCK_GATE)
    )
    config_mod.PN_HEAD_USE_ONSET_GATE = bool(
        model.get("pn_head_use_onset_gate", config_mod.PN_HEAD_USE_ONSET_GATE)
    )
    config_mod.PN_FUSION_ALPHA = float(model.get("pn_fusion_alpha", config_mod.PN_FUSION_ALPHA))

    config_mod.BATCH_SIZE = int(training.get("batch_size", config_mod.BATCH_SIZE))
    config_mod.EARLY_STOPPING = int(training.get("early_stopping", config_mod.EARLY_STOPPING))
    config_mod.ENABLE_EARLY_STOPPING = bool(
        training.get("enable_early_stopping", config_mod.ENABLE_EARLY_STOPPING)
    )
    config_mod.BEST_CHECKPOINT_METRIC = str(
        training.get("best_checkpoint_metric", config_mod.BEST_CHECKPOINT_METRIC)
    )

    config_mod.PITCH_LOSS_WEIGHT = float(loss.get("pitch_weight", config_mod.PITCH_LOSS_WEIGHT))
    config_mod.ONSET_LOSS_WEIGHT = float(loss.get("onset_weight", config_mod.ONSET_LOSS_WEIGHT))
    config_mod.PN_HEAD_LOSS_WEIGHT = float(
        loss.get("pn_head_weight", config_mod.PN_HEAD_LOSS_WEIGHT)
    )
    config_mod.PN_HEAD_POS_WEIGHT = float(
        loss.get("pn_head_pos_weight", config_mod.PN_HEAD_POS_WEIGHT)
    )

    config_mod.EVAL_ONSET_THRESHOLD = float(
        eval_cfg.get("onset_threshold", config_mod.EVAL_ONSET_THRESHOLD)
    )
    config_mod.EVAL_FRAME_THRESHOLD = float(
        eval_cfg.get("frame_threshold", config_mod.EVAL_FRAME_THRESHOLD)
    )
    config_mod.EVAL_ONSET_TOLERANCE = float(
        eval_cfg.get("onset_tolerance", config_mod.EVAL_ONSET_TOLERANCE)
    )
    config_mod.EVAL_EVENT_GAP_SECONDS = float(
        eval_cfg.get("event_gap_seconds", config_mod.EVAL_EVENT_GAP_SECONDS)
    )
    config_mod.THRESHOLD_SWEEP_VALUES = list(
        eval_cfg.get("threshold_sweep_values", config_mod.THRESHOLD_SWEEP_VALUES)
    )
    config_mod.THRESHOLD_SWEEP_EVERY_EPOCH = bool(
        eval_cfg.get("threshold_sweep_every_epoch", config_mod.THRESHOLD_SWEEP_EVERY_EPOCH)
    )
    focus = eval_cfg.get("threshold_sweep_focus_classes", config_mod.THRESHOLD_SWEEP_FOCUS_CLASSES)
    config_mod.THRESHOLD_SWEEP_FOCUS_CLASSES = None if focus is None else list(focus)

    config_mod.FAILURE_INSPECTION = bool(
        eval_cfg.get("failure_inspection", config_mod.FAILURE_INSPECTION)
    )
    config_mod.FAILURE_FRAME_F1_MIN = float(
        eval_cfg.get("failure_frame_f1_min", config_mod.FAILURE_FRAME_F1_MIN)
    )
    config_mod.FAILURE_EVENT_F1_MAX = float(
        eval_cfg.get("failure_event_f1_max", config_mod.FAILURE_EVENT_F1_MAX)
    )
    config_mod.FAILURE_MAX_PLOTS_PER_CLASS = int(
        eval_cfg.get("failure_max_plots_per_class", config_mod.FAILURE_MAX_PLOTS_PER_CLASS)
    )
    config_mod.FAILURE_FOCUS_CLASSES = list(
        eval_cfg.get("failure_focus_classes", config_mod.FAILURE_FOCUS_CLASSES)
    )

    config_mod.USE_VISDOM = bool(runtime.get("use_visdom", config_mod.USE_VISDOM))
    config_mod.saveName = f"{experiment_id}_{mert_tail}"

    if runtime.get("cuda_device") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_device"])


def build_run_dir(cfg: Dict[str, Any], runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    """Create a unique run directory under runs/guzheng/{date}_{experiment_id}."""
    runs_root = Path(runs_root)
    experiment_id = cfg.get("experiment_id", "experiment")
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    base = runs_root / f"{date_prefix}_{experiment_id}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs_root / f"{date_prefix}_{experiment_id}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def snapshot_run(
    cfg: Dict[str, Any],
    run_dir: str | Path,
    source_config_path: str | Path,
    argv: Optional[List[str]] = None,
) -> Path:
    """Write config snapshot and run_meta.json into the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "config.yaml"
    shutil.copy2(source_config_path, snapshot_path)

    meta = {
        "experiment_id": cfg.get("experiment_id"),
        "started_at": datetime.now().isoformat(),
        "source_config": str(Path(source_config_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "command": " ".join(argv) if argv else None,
        "git_commit": _git_commit(),
        "status": "running",
    }
    meta_path = run_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def finalize_run_meta(run_dir: str | Path, **extra: Any) -> None:
    meta_path = Path(run_dir) / "run_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    meta.update(extra)
    meta["finished_at"] = datetime.now().isoformat()
    meta["status"] = extra.get("status", "finished")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def find_best_checkpoint(run_dir: str | Path) -> Optional[Path]:
    """Return the newest best_e_* checkpoint under a run directory."""
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.glob("best_e_*"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    legacy = sorted(run_dir.glob("checkpoints/best_e_*"), key=lambda p: p.stat().st_mtime)
    return legacy[-1] if legacy else None


def resolve_config_for_run(run_dir: str | Path) -> Dict[str, Any]:
    """Load experiment config from a run directory snapshot."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml in run directory: {run_dir}")
    return load_experiment_config(config_path)


def load_sslnet_state_dict(model, checkpoint_path: str | Path) -> None:
    """
    Load weights into SSLNet. Allows partial load when checkpoint lacks pn_head.*
    (e.g. finetune from FPT-only run with use_pn_head=true in config).
    """
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    missing = [k for k in model_state if k not in state]
    unexpected = [k for k in state if k not in model_state]
    if missing and all(k.startswith("pn_head.") for k in missing):
        model.load_state_dict(state, strict=False)
        print(
            "Loaded checkpoint with randomly initialized PN head "
            "(%d missing keys: %s...)"
            % (len(missing), ", ".join(missing[:3]))
        )
        if unexpected:
            print("Unexpected keys ignored: %s" % unexpected[:5])
        return
    model.load_state_dict(state, strict=True)
