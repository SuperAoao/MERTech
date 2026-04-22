import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config_cbf import device, INITIAL_LR, MOMENTUM, WEIGHT_DECAY, GRAD_CLIP_NORM
from lib import sp_loss, onset_loss
from metrics_cbf import frame_micro_macro_f1, event_micro_macro_f1


@dataclass
class TrainResult:
    best_f1: float
    best_epoch: int


class TrainerCBF:
    """
    Minimal trainer for CBF:
    - SGD(momentum=0.9), lr=1e-3
    - gradient clipping (L2 norm)
    - cosine LR schedule
    - optimize technique BCE + (optional) onset BCE
    """

    def __init__(self, model, epochs: int, lr: float = INITIAL_LR, use_onset_loss: bool = True):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.use_onset_loss = use_onset_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        preds = []
        tars = []

        for x, y, _, y_o in loader:
            x = x.to(device)
            y = y.to(device)
            y_o = y_o.to(device)
            tech_logits, _ = self.model(x)  # [B, C, T]
            preds.append(torch.sigmoid(tech_logits).detach().cpu().numpy())
            tars.append(y.detach().cpu().numpy())

        pred = np.concatenate(preds, axis=0)  # [N, C, T]
        tar = np.concatenate(tars, axis=0)  # [N, C, T]

        # concatenate along time like prior scripts: [C, N*T]
        pred_ct = np.transpose(pred, (1, 0, 2)).reshape((pred.shape[1], -1))
        tar_ct = np.transpose(tar, (1, 0, 2)).reshape((tar.shape[1], -1))

        threshold = 0.5
        pred_bin = (pred_ct > threshold).astype(np.int8)
        tar_bin = (tar_ct > 0.5).astype(np.int8)

        fr_mi, fr_ma = frame_micro_macro_f1(pred_bin, tar_bin)
        ev_mi, ev_ma = event_micro_macro_f1(pred_bin, tar_bin, onset_tolerance_s=0.05)

        return {
            "frame_micro_f1": fr_mi,
            "frame_macro_f1": fr_ma,
            "event_micro_f1": ev_mi,
            "event_macro_f1": ev_ma,
        }

    def fit(self, tr_loader: DataLoader, va_loader: DataLoader, class_weights: torch.Tensor):
        st = time.time()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.SGD(model_parameters, lr=self.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        # Cosine scheduler (epoch-based)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_f1 = -1.0
        best_epoch = 0

        for e in range(1, self.epochs + 1):
            self.model.train()
            loss_total = 0.0

            for batch_idx, (x, y, _, y_o) in enumerate(tr_loader):
                x = x.to(device)
                y = y.to(device)
                y_o = y_o.to(device)

                tech_logits, onset_logits = self.model(x)

                loss_tech = sp_loss(tech_logits, y, class_weights)
                if self.use_onset_loss:
                    loss_on = onset_loss(onset_logits, y_o)
                    loss = loss_tech + 0.5 * loss_on
                else:
                    loss = loss_tech

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                loss_total += float(loss.detach().cpu())

            scheduler.step()

            # validation each epoch
            metrics = self.evaluate(va_loader)
            f1 = float(metrics["frame_micro_f1"])
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = e

            print(
                f"epoch={e}/{self.epochs} "
                f"loss={loss_total/len(tr_loader):.4f} "
                f"val_frame_mi_f1={metrics['frame_micro_f1']:.4f} "
                f"val_frame_ma_f1={metrics['frame_macro_f1']:.4f} "
                f"val_event_mi_f1={metrics['event_micro_f1']:.4f} "
                f"val_event_ma_f1={metrics['event_macro_f1']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"elapsed_s={int(time.time()-st)}"
            )

        return TrainResult(best_f1=best_f1, best_epoch=best_epoch)

