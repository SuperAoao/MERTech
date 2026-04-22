import os
import sys
import datetime
import random
import json

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

sys.path.append("./function")

from function.config_cbf import URL, device, BATCH_SIZE, DATASET_ROOT
from function.model_cbf import SSLNetCBF_IPT
from function.load_data_cbf import list_players, load_cbf_players
from function.lib import Data2Torch2
from function.fit_cbf import TrainerCBF
from function.splits_cbf import load_player_folds


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_player_folds(players: list[str]):
    """
    5-fold CV with player-based 8:2 split (10 players -> 2 test players per fold).
    Default fold pairing: (1,2), (3,4), (5,6), (7,8), (9,10).
    """
    assert len(players) == 10, f"expected 10 players, got {len(players)}"
    folds = []
    for i in range(0, 10, 2):
        test_players = [players[i], players[i + 1]]
        train_players = [p for p in players if p not in test_players]
        folds.append((train_players, test_players))
    assert len(folds) == 5
    return folds


def compute_class_weights(Ytr: np.ndarray) -> torch.Tensor:
    mp = Ytr.sum(axis=(0, 2)).astype(np.float32)
    mp = np.maximum(mp, 1.0)
    freq = mp / mp.sum()
    cc = (freq.mean() / freq) ** 0.3
    return torch.from_numpy(cc).float().to(device)


def main():
    set_seed(42)
    if not os.path.isdir(DATASET_ROOT):
        raise RuntimeError(f"missing dataset dir: {DATASET_ROOT}")

    split_path = os.path.join("data", "splits", "cbf_player_folds_8_2_5fold.json")
    if not os.path.isfile(split_path):
        raise RuntimeError(f"missing split file: {split_path}")
    players, folds = load_player_folds(split_path)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "model", "cbf", f"mertech_cbf_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("split_file:", split_path)
    print("players:", players)
    print("saving to:", out_dir)

    results = {
        "split_file": split_path,
        "dataset_root": DATASET_ROOT,
        "timestamp": ts,
        "folds": [],
    }

    for fd in folds:
        fold_idx = fd.fold
        train_players = fd.train_players
        test_players = fd.test_players
        print(f"\n=== Fold {fold_idx}/5 ===")
        print("train_players:", train_players)
        print("test_players :", test_players)

        Xtr, Ytr, Ytr_p, Ytr_o = load_cbf_players(train_players, root=DATASET_ROOT)
        Xte, Yte, Yte_p, Yte_o = load_cbf_players(test_players, root=DATASET_ROOT)

        Xtrs = processor(list(Xtr), sampling_rate=24000, return_tensors="pt")["input_values"]
        Xtes = processor(list(Xte), sampling_rate=24000, return_tensors="pt")["input_values"]

        t_kwargs = {"batch_size": BATCH_SIZE, "num_workers": 2, "pin_memory": True, "drop_last": True}
        v_kwargs = {"batch_size": 1, "num_workers": 2, "pin_memory": True}
        tr_loader = torch.utils.data.DataLoader(
            Data2Torch2([Xtrs, Ytr, Ytr_p, Ytr_o]), shuffle=True, **t_kwargs
        )
        te_loader = torch.utils.data.DataLoader(
            Data2Torch2([Xtes, Yte, Yte_p, Yte_o]), shuffle=False, **v_kwargs
        )

        class_weights = compute_class_weights(Ytr)

        fold_rec = {
            "fold": fold_idx,
            "train_players": train_players,
            "test_players": test_players,
            "models": {},
        }

        for mode in ["finetune", "probing"]:
            freeze_all = mode == "probing"
            model = SSLNetCBF_IPT(url=URL, weight_sum=True, freeze_all=freeze_all).to(device)
            trainer = TrainerCBF(model, epochs=30, lr=1e-3, use_onset_loss=False)
            trainer.fit(tr_loader, te_loader, class_weights)

            metrics = trainer.evaluate(te_loader)

            save_path = os.path.join(out_dir, f"{mode}_fold{fold_idx}.pt")
            torch.save(model.state_dict(), save_path)
            print("saved:", save_path)
            print(
                f"fold={fold_idx} model={mode} "
                f"FRAME(mi/ma)=({metrics['frame_micro_f1']:.4f},{metrics['frame_macro_f1']:.4f}) "
                f"EVENT(mi/ma)=({metrics['event_micro_f1']:.4f},{metrics['event_macro_f1']:.4f})"
            )

            fold_rec["models"][mode] = {
                "checkpoint": save_path,
                **{k: float(v) for k, v in metrics.items()},
            }

        results["folds"].append(fold_rec)

    metrics_path = os.path.join(out_dir, "metrics_train.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved metrics:", metrics_path)


if __name__ == "__main__":
    main()

