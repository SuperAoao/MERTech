import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from lib import *
from config import *
import config as config_runtime
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import math
from glob import glob
import os
from metrics_ipt import (
    IPT_NAMES,
    run_full_validation_metrics,
    format_full_evaluation_report,
    append_metrics_jsonl,
)
from failure_inspection import record_failure_modes


class _NoOpVisdom:
    """Skip plotting when Visdom server is off or USE_VISDOM=False."""

    def line(self, *args, **kwargs):
        pass


def _make_visdom():
    if not USE_VISDOM:
        return _NoOpVisdom()
    try:
        from visdom import Visdom

        viz = Visdom()
        if getattr(viz, "check_connection", lambda: True)():
            return viz
        print("Visdom: no server at localhost:8097 — training without live plots.")
    except Exception as exc:
        print("Visdom disabled (%s) — training without live plots." % exc)
    return _NoOpVisdom()


def _ipt_threshold_metrics(val_bundle):
    """Per-class IPT metrics with swept thresholds when available."""
    if val_bundle.get("threshold_sweep"):
        return val_bundle["ipt_swept_thresholds"]
    return val_bundle["ipt_default_thresholds"]


def _pn_frame_f1(val_bundle) -> float:
    """PN frame F1 from default thresholds only (aligned with default test report)."""
    ipt = val_bundle["ipt_default_thresholds"]
    pn_idx = IPT_NAMES.index("PN")
    return float(ipt["frame"]["per_class_f1"][pn_idx])


def _pn_event_f1(val_bundle) -> float:
    """PN event F1 from default thresholds only (aligned with default test report)."""
    ipt = val_bundle["ipt_default_thresholds"]
    pn_idx = IPT_NAMES.index("PN")
    return float(ipt["event"]["per_class_f1"][pn_idx])


def _onset_peak_f1(val_bundle) -> float:
    """Onset rising-edge peak F1 (default onset_th, EVAL_ONSET_TOLERANCE matching)."""
    onset = val_bundle.get("onset")
    if onset is None:
        return 0.0
    return float(onset["peak"]["f1"])


def _checkpoint_score_breakdown(eva_result, val_bundle):
    """
    Return (score, details) for checkpoint selection and jsonl logging.

    Supported BEST_CHECKPOINT_METRIC values:
      ipt            — legacy IPT micro frame F1
      pitch          — pitch frame F1
      combined       — (IPT + pitch + PN frame + PN event) / 4
                       where PN metrics use default thresholds, not swept thresholds
      combined_onset — (IPT + pitch + PN frame + PN event + onset peak) / 5
      pn_frame       — (IPT + pitch + 2 * PN frame) / 4  (PN-weighted composite)
                       where PN frame uses default thresholds, not swept thresholds
    """
    m = BEST_CHECKPOINT_METRIC.lower()
    ipt_frame = float(eva_result[3])
    pitch_frame = float(eva_result[7])
    pn_frame = _pn_frame_f1(val_bundle)
    onset_peak = _onset_peak_f1(val_bundle)

    details = {
        "checkpoint_metric": m,
        "ipt_frame_f1": ipt_frame,
        "pitch_frame_f1": pitch_frame,
        "pn_frame_f1": pn_frame,
        "onset_peak_f1": onset_peak,
    }

    if m == "pitch":
        score = pitch_frame
    elif m == "combined":
        pn_event = _pn_event_f1(val_bundle)
        details["pn_event_f1"] = pn_event
        score = (ipt_frame + pitch_frame + pn_frame + pn_event) / 4.0
    elif m == "combined_onset":
        pn_event = _pn_event_f1(val_bundle)
        details["pn_event_f1"] = pn_event
        if val_bundle.get("onset") is None:
            print(
                "Warning: combined_onset metric requires onset metrics in val_bundle; "
                "onset_peak_f1 treated as 0.0"
            )
        score = (
            ipt_frame + pitch_frame + pn_frame + pn_event + onset_peak
        ) / 5.0
    elif m == "pn_frame":
        details["pn_frame_weight"] = 2
        score = (ipt_frame + pitch_frame + 2.0 * pn_frame) / 4.0
    elif m == "ipt":
        score = ipt_frame
    else:
        print(
            "Warning: unknown BEST_CHECKPOINT_METRIC=%r; using ipt"
            % (BEST_CHECKPOINT_METRIC,)
        )
        details["checkpoint_metric"] = "ipt"
        score = ipt_frame

    details["checkpoint_score"] = float(score)
    return float(score), details


def _validation_score_for_checkpoint(eva_result, val_bundle):
    score, _ = _checkpoint_score_breakdown(eva_result, val_bundle)
    return score


def _format_checkpoint_log_line(details: dict) -> str:
    metric = details.get("checkpoint_metric", "?")
    score = details.get("checkpoint_score", 0.0)
    if metric == "pn_frame":
        return (
            "best_ckpt_metric (pn_frame): %.4f  "
            "[IPT frame=%.4f  pitch=%.4f  PN frame=%.4f  (PN weight=%d)]"
            % (
                score,
                details["ipt_frame_f1"],
                details["pitch_frame_f1"],
                details["pn_frame_f1"],
                details.get("pn_frame_weight", 2),
            )
        )
    if metric == "combined":
        return (
            "best_ckpt_metric (combined): %.4f  "
            "[IPT frame=%.4f  pitch=%.4f  PN frame=%.4f  PN event=%.4f]"
            % (
                score,
                details["ipt_frame_f1"],
                details["pitch_frame_f1"],
                details["pn_frame_f1"],
                details.get("pn_event_f1", 0.0),
            )
        )
    if metric == "combined_onset":
        return (
            "best_ckpt_metric (combined_onset): %.4f  "
            "[IPT frame=%.4f  pitch=%.4f  PN frame=%.4f  PN event=%.4f  "
            "onset peak=%.4f]"
            % (
                score,
                details["ipt_frame_f1"],
                details["pitch_frame_f1"],
                details["pn_frame_f1"],
                details.get("pn_event_f1", 0.0),
                details.get("onset_peak_f1", 0.0),
            )
        )
    return "best_ckpt_metric (%s): %.4f" % (metric, score)


class Trainer:
    def __init__(self, model, lr, epoch, save_fn, validation_interval, save_interval):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        self.metrics_log_path = os.path.join(save_fn, "ipt_metrics.jsonl")
        self.failure_dir = os.path.join(save_fn, "failure_inspection")
        self.best_frame_th_per_class = [EVAL_FRAME_THRESHOLD] * NUM_LABELS
        self.best_onset_th_per_class = [EVAL_ONSET_THRESHOLD] * NUM_LABELS

    def evaluate_epoch(self, loader, b_size, we, epoch: int = 0):
        """
        Validation: losses, legacy frame F1, full IPT bundle (sweep + pre/post note).
        """
        all_pred = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        all_tar = np.zeros((b_size, NUM_LABELS, int(LENGTH)))
        p_pred = np.zeros((b_size, MAX_MIDI - MIN_MIDI + 1, int(LENGTH)))
        pitch_tar = np.zeros((b_size, MAX_MIDI - MIN_MIDI + 1, int(LENGTH)))

        pred_ipt_chunks = []
        tar_ipt_chunks = []
        pred_onset_chunks = []
        tar_onset_chunks = []

        loss_IPT = 0.0
        loss_pitch = 0.0
        loss_onset = 0.0

        self.model.eval()
        ds = 0
        with torch.no_grad():
            for idx, _input in enumerate(loader):
                data = Variable(_input[0].to(device))
                target = Variable(_input[1].to(device))
                target_p = Variable(_input[2].to(device))
                target_o = Variable(_input[3].to(device))
                IPT_pred, pitch_pred, onset_pred = self.model(data)

                loss = sp_loss(IPT_pred, target, we)
                loss_p = pitch_loss(pitch_pred, target_p)
                loss_o = onset_loss(onset_pred, target_o)
                loss_IPT += loss.data
                loss_pitch += loss_p.data
                loss_onset += loss_o.data

                batch_ipt_pred = F.sigmoid(IPT_pred).data.cpu().numpy()
                batch_onset_pred = F.sigmoid(onset_pred).data.cpu().numpy()

                all_tar[ds : ds + len(target)] = target.data.cpu().numpy()
                all_pred[ds : ds + len(target)] = batch_ipt_pred
                pitch_tar[ds : ds + len(target)] = target_p.data.cpu().numpy()
                p_pred[ds : ds + len(target)] = F.sigmoid(pitch_pred).data.cpu().numpy()

                for b in range(len(target)):
                    t_len = int(target.shape[-1])
                    pred_ipt_chunks.append(batch_ipt_pred[b, :, :t_len].copy())
                    tar_ipt_chunks.append(target[b].data.cpu().numpy()[:, :t_len].copy())
                    pred_onset_chunks.append(batch_onset_pred[b, :, :t_len].copy())
                    tar_onset_chunks.append(target_o[b].data.cpu().numpy()[:, :t_len].copy())

                ds += len(target)

        threshold = EVAL_FRAME_THRESHOLD
        pred_inst = np.transpose(all_pred, (1, 0, 2)).reshape((NUM_LABELS, -1))
        tar_inst = np.transpose(all_tar, (1, 0, 2)).reshape((NUM_LABELS, -1))
        pred_pitch = np.transpose(p_pred, (1, 0, 2)).reshape((MAX_MIDI - MIN_MIDI + 1, -1))
        tar_pitch = np.transpose(pitch_tar, (1, 0, 2)).reshape((MAX_MIDI - MIN_MIDI + 1, -1))
        tar_inst_bin = (tar_inst > 0.5).astype(np.float64)
        pred_inst[pred_inst > threshold] = 1
        pred_inst[pred_inst <= threshold] = 0
        pred_pitch[pred_pitch > threshold] = 1
        pred_pitch[pred_pitch <= threshold] = 0

        metrics = compute_metrics(pred_inst, tar_inst_bin)
        metrics_pitch = compute_pitch_metrics(pred_pitch, tar_pitch)

        do_sweep = THRESHOLD_SWEEP_EVERY_EPOCH
        val_bundle = run_full_validation_metrics(
            pred_ipt_chunks,
            tar_ipt_chunks,
            pred_onset_chunks,
            tar_onset_chunks,
            do_threshold_sweep=do_sweep,
            onset_tolerance=EVAL_ONSET_TOLERANCE,
            event_gap_seconds=EVAL_EVENT_GAP_SECONDS,
            default_onset_th=EVAL_ONSET_THRESHOLD,
            default_frame_th=EVAL_FRAME_THRESHOLD,
            sweep_focus_classes=THRESHOLD_SWEEP_FOCUS_CLASSES,
        )

        if val_bundle.get("threshold_sweep") is not None:
            self.best_frame_th_per_class = list(
                val_bundle["threshold_sweep"]["frame_th_per_class"]
            )
            self.best_onset_th_per_class = list(
                val_bundle["threshold_sweep"]["onset_th_per_class"]
            )

        if FAILURE_INSPECTION:
            chunk_ef1 = val_bundle["ipt_default_thresholds"].get("chunk_event_f1", [])
            if chunk_ef1:
                saved = record_failure_modes(
                    pred_ipt_chunks,
                    tar_ipt_chunks,
                    pred_onset_chunks,
                    tar_onset_chunks,
                    chunk_ef1,
                    save_dir=self.failure_dir,
                    epoch=epoch,
                    frame_th_per_class=self.best_frame_th_per_class,
                    default_frame_th=EVAL_FRAME_THRESHOLD,
                    frame_f1_min=FAILURE_FRAME_F1_MIN,
                    event_f1_max=FAILURE_EVENT_F1_MAX,
                    max_plots_per_class=FAILURE_MAX_PLOTS_PER_CLASS,
                    focus_classes=FAILURE_FOCUS_CLASSES,
                )
                if saved:
                    print(f"Failure inspection: saved {len(saved)} plot(s) under {self.failure_dir}")

        eva_result = (
            loss_IPT / b_size,
            metrics["metric/IPT_frame/precision"][0],
            metrics["metric/IPT_frame/recall"][0],
            metrics["metric/IPT_frame/f1"][0],
            loss_pitch / b_size,
            metrics_pitch["metric/pitch_frame/precision"][0],
            metrics_pitch["metric/pitch_frame/recall"][0],
            metrics_pitch["metric/pitch_frame/f1"][0],
            loss_onset / b_size,
        )
        return eva_result, val_bundle

    def Tester(self, loader, b_size, we):
        eva_result, _ = self.evaluate_epoch(loader, b_size, we, epoch=0)
        return eva_result

    def _log_ipt_metrics(self, epoch: int, val_bundle: dict, eva_result=None):
        print(format_full_evaluation_report(val_bundle, epoch=epoch))
        extra = None
        if eva_result is not None:
            _, ckpt_details = _checkpoint_score_breakdown(eva_result, val_bundle)
            extra = {
                "legacy_ipt_frame_f1": float(eva_result[3]),
                "legacy_pitch_frame_f1": float(eva_result[7]),
                "valid_onset_loss": float(eva_result[8]),
                **ckpt_details,
            }
            if val_bundle.get("onset") is not None:
                extra["onset_frame_f1"] = float(val_bundle["onset"]["frame"]["f1"])
                extra["onset_peak_f1"] = float(val_bundle["onset"]["peak"]["f1"])
        append_metrics_jsonl(self.metrics_log_path, epoch, val_bundle, extra=extra)

    def fit(self, tr_loader, va_loader, we):
        st = time.time()
        lr = self.lr

        viz = _make_visdom()
        viz.line([[0.0, 0.0]], [0], win="IPT_loss_" + saveName, opts=dict(title="IPT_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        viz.line([[0.0]], [0], win="IPT_precision_" + saveName, opts=dict(title="IPT_precision_" + saveName, legend=['valid_IPT_precision']))
        viz.line([[0.0]], [0], win="IPT_recall_" + saveName, opts=dict(title="IPT_recall_" + saveName, legend=['valid_IPT_recall']))
        viz.line([[0.0]], [0], win="IPT_F1_" + saveName, opts=dict(title="IPT_F1_" + saveName, legend=['valid_IPT_F1']))
        viz.line([[0.0, 0.0]], [0], win="pitch_loss_" + saveName, opts=dict(title="pitch_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        viz.line([[0.0]], [0], win="pitch_precision_" + saveName, opts=dict(title="pitch_precision_" + saveName, legend=['valid_pitch_precision']))
        viz.line([[0.0]], [0], win="pitch_recall_" + saveName, opts=dict(title="pitch_recall_" + saveName, legend=['valid_pitch_recall']))
        viz.line([[0.0]], [0], win="pitch_F1_" + saveName, opts=dict(title="pitch_F1_" + saveName, legend=['valid_pitch_F1']))
        viz.line([[0.0, 0.0]], [0], win="onset_loss_" + saveName, opts=dict(title="onset_loss_" + saveName, legend=['train_loss', 'valid_loss']))
        viz.line([[0.0]], [0], win="onset_frame_F1_" + saveName, opts=dict(title="onset_frame_F1_" + saveName, legend=['onset_frame_F1']))
        viz.line([[0.0]], [0], win="onset_peak_F1_" + saveName, opts=dict(title="onset_peak_F1_" + saveName, legend=['onset_peak_F1']))
        viz.line([[0.0]], [0], win="IPT_event_MA_" + saveName, opts=dict(title="IPT_event_MA_" + saveName, legend=['event_MA_F1']))
        viz.line([[0.0]], [0], win="IPT_frame_MA_" + saveName, opts=dict(title="IPT_frame_MA_" + saveName, legend=['frame_MA_F1']))
        viz.line([[0.0]], [0], win="macro_note_f1_" + saveName, opts=dict(title="macro_note_f1_" + saveName, legend=['macro_note_f1']))
        best_acc = 0.0
        last_best_epoch = 1

        for e in range(1, self.epoch + 1):
            if TWO_STEP and (e > LIN_EPOCH) and FREEZE_ALL:
                for p in self.model.frontend.parameters():
                    p.requires_grad = True
                    self.model.frontend.model.feature_extractor._freeze_parameters()

            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
            lrf = 0.01
            epochs = 100
            lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

            loss_total_p = 0
            loss_total_i = 0
            loss_total_o = 0
            loss_total_pn = 0
            print('\n==> Training Epoch #%d lr=%4f' % (e, lr))

            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                data = Variable(_input[0].to(device))
                target = Variable(_input[1].to(device))
                target_p = Variable(_input[2].to(device))
                target_o = Variable(_input[3].to(device))

                IPT_pred, pitch_pred, onset_pred = self.model(data)

                loss = sp_loss(IPT_pred, target, we)
                loss_p = pitch_loss(pitch_pred, target_p)
                loss_o = onset_loss(onset_pred, target_o)
                loss_all = (
                    loss
                    + PITCH_LOSS_WEIGHT * loss_p
                    + ONSET_LOSS_WEIGHT * loss_o
                )
                if config_runtime.USE_PN_HEAD and self.model._pn_aux_logit is not None:
                    loss_pn = pn_head_loss(self.model._pn_aux_logit, target)
                    loss_all = loss_all + config_runtime.PN_HEAD_LOSS_WEIGHT * loss_pn
                    loss_total_pn += loss_pn.data
                loss_total_i += loss.data
                loss_total_p += loss_p.data
                loss_total_o += loss_o.data

                optimizer.zero_grad()
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), 3)
                optimizer.step()
                scheduler.step()

                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                    % (e, self.epoch, batch_idx + 1, len(tr_loader), loss.data, time.time() - st)
                )
                sys.stdout.flush()

            print('\n')
            print(loss_total_i / len(tr_loader))
            print(loss_total_p / len(tr_loader))
            print(loss_total_o / len(tr_loader))
            if config_runtime.USE_PN_HEAD:
                print(loss_total_pn / len(tr_loader))

            # Full validation (metrics, sweep, failure plots) every validation_interval epochs
            if e % self.validation_interval == 1:
                print(self.save_fn)
                print(
                    "Running validation (interval=%d, epoch %d)..."
                    % (self.validation_interval, e)
                )
                eva_result, val_bundle = self.evaluate_epoch(
                    va_loader, len(va_loader.dataset), we, epoch=e
                )
                self._log_ipt_metrics(e, val_bundle, eva_result=eva_result)
                self.model.train()

                ipt = (
                    val_bundle["ipt_swept_thresholds"]
                    if val_bundle.get("threshold_sweep")
                    else val_bundle["ipt_default_thresholds"]
                )
                macro_note = val_bundle["prepost"]["macro_note_f1"]

                viz.line(
                    [[float(loss_total_i / len(tr_loader.dataset)), float(eva_result[0])]],
                    [e - 1],
                    win="IPT_loss_" + saveName,
                    update='append',
                )
                viz.line([[float(eva_result[1])]], [e - 1], win="IPT_precision_" + saveName, update='append')
                viz.line([[float(eva_result[2])]], [e - 1], win="IPT_recall_" + saveName, update='append')
                viz.line([[float(eva_result[3])]], [e - 1], win="IPT_F1_" + saveName, update='append')
                viz.line(
                    [[float(loss_total_p / len(tr_loader.dataset)), float(eva_result[4])]],
                    [e - 1],
                    win="pitch_loss_" + saveName,
                    update='append',
                )
                viz.line([[float(eva_result[5])]], [e - 1], win="pitch_precision_" + saveName, update='append')
                viz.line([[float(eva_result[6])]], [e - 1], win="pitch_recall_" + saveName, update='append')
                viz.line([[float(eva_result[7])]], [e - 1], win="pitch_F1_" + saveName, update='append')
                viz.line(
                    [[float(loss_total_o / len(tr_loader.dataset)), float(eva_result[8])]],
                    [e - 1],
                    win="onset_loss_" + saveName,
                    update='append',
                )
                if val_bundle.get("onset") is not None:
                    viz.line(
                        [[float(val_bundle["onset"]["frame"]["f1"])]],
                        [e - 1],
                        win="onset_frame_F1_" + saveName,
                        update='append',
                    )
                    viz.line(
                        [[float(val_bundle["onset"]["peak"]["f1"])]],
                        [e - 1],
                        win="onset_peak_F1_" + saveName,
                        update='append',
                    )
                viz.line(
                    [[float(ipt["frame"]["macro_f1"])]],
                    [e - 1],
                    win="IPT_frame_MA_" + saveName,
                    update='append',
                )
                viz.line(
                    [[float(ipt["event"]["macro_f1"])]],
                    [e - 1],
                    win="IPT_event_MA_" + saveName,
                    update='append',
                )
                viz.line([[float(macro_note)]], [e - 1], win="macro_note_f1_" + saveName, update='append')

                print("IPT_F1 (legacy multipitch):", eva_result[3])
                print("pitch_F1:", eva_result[7])
                if val_bundle.get("onset") is not None:
                    onset_m = val_bundle["onset"]
                    print(
                        "onset frame F1: %.4f  onset peak F1: %.4f"
                        % (onset_m["frame"]["f1"], onset_m["peak"]["f1"])
                    )
                print(
                    "note F1 (after post-proc): %.4f  macro_note_f1: %.4f"
                    % (
                        val_bundle["prepost"]["after"]["note_f1"],
                        macro_note,
                    )
                )
                score, ckpt_details = _checkpoint_score_breakdown(eva_result, val_bundle)
                print(_format_checkpoint_log_line(ckpt_details))

                if score > best_acc:
                    best_acc = score
                    last_best_epoch = e
                    rm_lst = glob(self.save_fn + 'best_*')
                    for p in rm_lst:
                        os.remove(p)
                    torch.save(self.model.state_dict(), self.save_fn + 'best' + '_e_%d' % (e - 1))
                elif ENABLE_EARLY_STOPPING and (e - last_best_epoch >= EARLY_STOPPING):
                    print('Early stopping at epoch {}...'.format(e))
                    break
