import os
import csv
from glob import glob
from dataclasses import dataclass

import numpy as np
import librosa
import torch
import torchaudio.transforms as T

from config_cbf import (
    DATASET_ROOT,
    CBF_TECH_TO_IDX,
    FEATURE_RATE,
    TIME_LENGTH,
    SAMPLE_RATE,
    MERT_SAMPLE_RATE,
    HOPS_IN_ONSET,
    NUM_LABELS,
)


@dataclass(frozen=True)
class CBFTrack:
    player: str
    audio_path: str
    csv_paths: list[str]


def list_players(root=DATASET_ROOT):
    players = []
    for name in os.listdir(root):
        if name.lower().startswith("player"):
            players.append(name)
    return sorted(players, key=lambda s: int(s.replace("Player", "")))


def _resample_if_needed(y_np: np.ndarray, orig_sr: int, target_sr: int):
    if orig_sr == target_sr:
        return y_np
    resampler = T.Resample(orig_sr, target_sr)
    y = torch.from_numpy(y_np)
    y_rs = resampler(y)
    return y_rs.numpy()


def load_wav_mert_sr(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        # librosa will usually respect the sr argument, but keep this safe.
        pass
    y = _resample_if_needed(y, SAMPLE_RATE, MERT_SAMPLE_RATE)
    return y.astype(np.float32)


def _parse_on_off_csv(csv_path: str):
    """
    CSV format examples (2 columns, no header):
      0.827210884,on_Vibrato
      2.766077098,off_Vibrato
    """
    events = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            t = float(row[0])
            tag = row[1].strip()
            events.append((t, tag))
    events.sort(key=lambda x: x[0])
    return events


def _tech_from_tag(tag: str) -> str:
    # tag looks like "on_Vibrato" / "off_FT"
    if "_" not in tag:
        return tag
    return tag.split("_", 1)[1]


def build_labels_for_track(audio_samples: np.ndarray, csv_paths: list[str]):
    """
    Returns:
      tech_label: (NUM_LABELS, n_steps) 0/1
      onset_label: (1, n_steps) 0/1  (onset of ANY technique segment)
    """
    # n_steps: map samples -> seconds -> frames
    n_steps = int(FEATURE_RATE * len(audio_samples) / MERT_SAMPLE_RATE)
    tech_label = np.zeros((NUM_LABELS, n_steps), dtype=np.int8)
    onset_label = np.zeros((1, n_steps), dtype=np.int8)

    for csv_path in csv_paths:
        events = _parse_on_off_csv(csv_path)
        # pair on/off in order; if missing off, close at end
        on_time = None
        tech_name = None
        for t, tag in events:
            is_on = tag.lower().startswith("on_")
            is_off = tag.lower().startswith("off_")
            if is_on:
                on_time = t
                tech_name = _tech_from_tag(tag)
            elif is_off and on_time is not None:
                off_time = t
                name = _tech_from_tag(tag) or tech_name
                if name in CBF_TECH_TO_IDX:
                    k = CBF_TECH_TO_IDX[name]
                    left = int(round(on_time * FEATURE_RATE))
                    right = int(round(off_time * FEATURE_RATE))
                    left = max(0, min(n_steps, left))
                    right = max(0, min(n_steps, right))
                    if right > left:
                        tech_label[k, left:right] = 1
                        onset_right = min(n_steps, left + HOPS_IN_ONSET)
                        onset_label[0, left:onset_right] = 1
                on_time = None
                tech_name = None

        # if last "on" has no off
        if on_time is not None and tech_name in CBF_TECH_TO_IDX:
            k = CBF_TECH_TO_IDX[tech_name]
            left = int(round(on_time * FEATURE_RATE))
            left = max(0, min(n_steps, left))
            tech_label[k, left:n_steps] = 1
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            onset_label[0, left:onset_right] = 1

    return tech_label, onset_label


def _chunk_wav(y: np.ndarray):
    s = int(MERT_SAMPLE_RATE * TIME_LENGTH)
    length = int(np.ceil(len(y) / s) * s)
    if length > len(y):
        y = np.concatenate([y, np.zeros((length - len(y)), dtype=y.dtype)], axis=0)
    chunks = []
    for i in range(int(length / s)):
        chunks.append(y[i * s : (i + 1) * s])
    return np.stack(chunks, axis=0)  # [N, s]


def _chunk_label(mat: np.ndarray):
    # mat: [C, n_steps]
    s = int(FEATURE_RATE * TIME_LENGTH)
    n_steps = mat.shape[1]
    length = int(np.ceil(n_steps / s) * s)
    if length > n_steps:
        pad = np.zeros((mat.shape[0], length - n_steps), dtype=mat.dtype)
        mat = np.concatenate([mat, pad], axis=1)
    chunks = []
    for i in range(int(length / s)):
        chunks.append(mat[:, i * s : (i + 1) * s])
    return np.stack(chunks, axis=0)  # [N, C, s]


def _collect_track_csvs(audio_path: str) -> list[str]:
    """
    For a wav, collect:
    - same-stem csv (Iso)
    - stem + '_tech_*.csv' (Piece)
    """
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    exact = os.path.join(folder, f"{stem}.csv")
    tech_glob = os.path.join(folder, f"{stem}_tech_*.csv")
    csvs = []
    if os.path.isfile(exact):
        csvs.append(exact)
    csvs.extend(sorted(glob(tech_glob)))
    return csvs


def list_tracks_for_players(players: list[str], root=DATASET_ROOT) -> list[CBFTrack]:
    tracks: list[CBFTrack] = []
    for player in players:
        for sub in ["Iso", "Piece"]:
            wavs = sorted(glob(os.path.join(root, player, sub, "*.wav")))
            for wav in wavs:
                csvs = _collect_track_csvs(wav)
                tracks.append(CBFTrack(player=player, audio_path=wav, csv_paths=csvs))
    return tracks


def load_cbf_players(players: list[str], root=DATASET_ROOT):
    """
    Load CBF tracks for a set of players, returning chunked arrays:
      X: float32 [N, s_samples]
      Y: int8   [N, 7, s_frames]
      Y_o: int8 [N, 1, s_frames]
      Y_p: dummy zeros [N, 1, s_frames] (kept for compatibility with existing Dataset wrappers)
    """
    tracks = list_tracks_for_players(players, root=root)
    X_list = []
    Y_list = []
    Yo_list = []

    for tr in tracks:
        y = load_wav_mert_sr(tr.audio_path)  # [samples] @ MERT_SAMPLE_RATE
        tech_label, onset_label = build_labels_for_track(y, tr.csv_paths)

        x_chunks = _chunk_wav(y)  # [n, samples]
        y_chunks = _chunk_label(tech_label)  # [n, 7, frames]
        o_chunks = _chunk_label(onset_label)  # [n, 1, frames]

        assert x_chunks.shape[0] == y_chunks.shape[0] == o_chunks.shape[0]
        X_list.append(x_chunks)
        Y_list.append(y_chunks)
        Yo_list.append(o_chunks)

    if len(X_list) == 0:
        raise RuntimeError("No tracks found for players: " + ",".join(players))

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    Y_o = np.concatenate(Yo_list, axis=0)
    Y_p = np.zeros((Y.shape[0], 1, Y.shape[-1]), dtype=np.int8)

    return X, Y, Y_p, Y_o

