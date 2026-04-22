import os
import sys
import datetime
import random

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

sys.path.append("./function")

from function.config_cbf import (
    URL,
    device,
    BATCH_SIZE,
    DATASET_ROOT,
    NUM_LABELS,
)
from function.model_cbf import SSLNetCBF_IPT
from function.load_data_cbf import list_players, load_cbf_players
from function.lib import Data2Torch2
from function.fit_cbf import TrainerCBF


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
    """
    Ytr: [N, C, T] 0/1
    Return a weight per class similar to original code's idea.
    """
    # sum over N and T -> [C]
    mp = Ytr.sum(axis=(0, 2)).astype(np.float32)
    mp = np.maximum(mp, 1.0)  # avoid div-by-zero
    freq = mp / mp.sum()
    cc = (freq.mean() / freq) ** 0.3
    return torch.from_numpy(cc).float().to(device)


def main():
    set_seed(42)
    if not os.path.isdir(DATASET_ROOT):
        raise RuntimeError(f"missing dataset dir: {DATASET_ROOT}")

    players = list_players(DATASET_ROOT)
    folds = build_player_folds(players)

    print("players:", players)
    print("num_folds:", len(folds))

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "model", "cbf", f"mertech_cbf_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {"finetune": [], "probing": []}
    for fold_idx, (train_players, test_players) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_idx}/5 ===")
        print("train_players:", train_players)
        print("test_players :", test_players)

        # load & chunk
        Xtr, Ytr, Ytr_p, Ytr_o = load_cbf_players(train_players, root=DATASET_ROOT)
        Xte, Yte, Yte_p, Yte_o = load_cbf_players(test_players, root=DATASET_ROOT)

        # feature extractor -> torch tensors
        Xtrs = processor(list(Xtr), sampling_rate=24000, return_tensors="pt")["input_values"]
        Xtes = processor(list(Xte), sampling_rate=24000, return_tensors="pt")["input_values"]

        # dataloaders
        t_kwargs = {"batch_size": BATCH_SIZE, "num_workers": 2, "pin_memory": True, "drop_last": True}
        v_kwargs = {"batch_size": 1, "num_workers": 2, "pin_memory": True}
        tr_loader = torch.utils.data.DataLoader(Data2Torch2([Xtrs, Ytr, Ytr_p, Ytr_o]), shuffle=True, **t_kwargs)
        te_loader = torch.utils.data.DataLoader(Data2Torch2([Xtes, Yte, Yte_p, Yte_o]), shuffle=False, **v_kwargs)

        class_weights = compute_class_weights(Ytr)

        for mode in ["finetune", "probing"]:
            print(f"\n--- model={mode} ---")
            freeze_all = (mode == "probing")
            model = SSLNetCBF_IPT(url=URL, weight_sum=True, freeze_all=freeze_all).to(device)

            # train; use test set as validation to report fold score (player-based CV)
            trainer = TrainerCBF(model, epochs=30, lr=1e-3, use_onset_loss=False)
            result = trainer.fit(tr_loader, te_loader, class_weights)

            metrics = trainer.evaluate(te_loader)
            print(
                f"fold={fold_idx} model={mode} "
                f"best_epoch={result.best_epoch} best_val_frame_mi_f1={result.best_f1:.4f} "
                f"FRAME(mi/ma)=({metrics['frame_micro_f1']:.4f},{metrics['frame_macro_f1']:.4f}) "
                f"EVENT(mi/ma)=({metrics['event_micro_f1']:.4f},{metrics['event_macro_f1']:.4f})"
            )

            # save model
            save_path = os.path.join(out_dir, f"{mode}_fold{fold_idx}.pt")
            torch.save(model.state_dict(), save_path)

            all_results[mode].append(metrics)

    print("\n=== 5-fold summary ===")
    for mode in ["finetune", "probing"]:
        fr_mi = [m["frame_micro_f1"] for m in all_results[mode]]
        fr_ma = [m["frame_macro_f1"] for m in all_results[mode]]
        ev_mi = [m["event_micro_f1"] for m in all_results[mode]]
        ev_ma = [m["event_macro_f1"] for m in all_results[mode]]
        print(f"\nmodel={mode}")
        print("FRAME micro-F1 mean/std:", float(np.mean(fr_mi)), float(np.std(fr_mi)))
        print("FRAME macro-F1 mean/std:", float(np.mean(fr_ma)), float(np.std(fr_ma)))
        print("EVENT micro-F1 mean/std:", float(np.mean(ev_mi)), float(np.std(ev_mi)))
        print("EVENT macro-F1 mean/std:", float(np.mean(ev_ma)), float(np.std(ev_ma)))


if __name__ == "__main__":
    main()

