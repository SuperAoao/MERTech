#config
import torch

URL = "m-a-p/MERT-v1-95M" #model URL
TIME_LENGTH = 5 #5 seconds
LENGTH = 375 #number of frame in 3 seconds,225
NUM_LABELS = 7 #number of IPTs
BATCH_SIZE = 10
SAMPLE_RATE = 44100 #Raw audio sampling rate
MERT_SAMPLE_RATE = 24000 if "MERT" in URL else 16000 #input audio sampling rate of MERT
# ao: 75 frames per second
FEATURE_RATE = 75 # FEATURE_RATE = 1000//ZHEN_LENGTH，Sampling rate of feature extracted from MERT
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TWO_STEP = False # Whether two-step finetuning
LIN_EPOCH = 5 #If fine-tuning is done in two steps, which epochs should we start fine-tuning the pre-trained model
FREEZE_ALL = False # Whether to freeze all parameters of the self-supervised pre-training model
EARLY_STOPPING = 1000  # patience in epochs (only used when ENABLE_EARLY_STOPPING is True)

# --- Optional Feature Pyramid Transformer (Option A) ---
# Inserts a multi-scale temporal context module between MERT frontend and task heads.
USE_FPT = True
# number of pyramid levels (including the finest/original resolution)
FPT_LEVELS = 3
# transformer encoder layers per level
FPT_NUM_LAYERS = 1
# attention heads (must divide MERT hidden dim: 768/1024)
FPT_NUM_HEADS = 8
FPT_DROPOUT = 0.1

# Baseline (USE_FPT=False): early stopping on IPT frame F1.
# FPT (USE_FPT=True): early stopping on combined score (IPT + pitch + PN frame/event F1).
if USE_FPT:
    ENABLE_EARLY_STOPPING = True
    BEST_CHECKPOINT_METRIC = "combined"
else:
    ENABLE_EARLY_STOPPING = True
    BEST_CHECKPOINT_METRIC = "ipt"

# Training loss: loss + PITCH_LOSS_WEIGHT * loss_p + ONSET_LOSS_WEIGHT * loss_o
PITCH_LOSS_WEIGHT = 0.5
ONSET_LOSS_WEIGHT = 1.0

saveName = "mul_onset7_pitch_IPT_share_weight_weighted_loss-" + URL.split("/")[-1] #name of the model to save and load
DATASET = "Guzheng_Tech99"

MIN_MIDI = 36 #音域内最低音的midi值 C2 36
MAX_MIDI = 87 #音域内最高音的midi值 Eb6 87
HOPS_IN_ONSET = 1 #onset跨越几帧

# Per-epoch IPT validation callback (frame / event MI-F1 & MA-F1)
EVAL_ONSET_THRESHOLD = 0.5
EVAL_FRAME_THRESHOLD = 0.5
EVAL_ONSET_TOLERANCE = 0.05
EVAL_EVENT_GAP_SECONDS = 1.0

# Per-class threshold sweep on validation (onset_th x frame_th grid)
THRESHOLD_SWEEP_VALUES = [0.3, 0.4, 0.5, 0.6]
THRESHOLD_SWEEP_EVERY_EPOCH = True
# None = sweep all 7 IPT classes; or e.g. [5, 6] for scarce classes only
THRESHOLD_SWEEP_FOCUS_CLASSES = None

# Failure-mode plots: high frame F1 but low event F1
FAILURE_INSPECTION = True
FAILURE_FRAME_F1_MIN = 0.8
FAILURE_EVENT_F1_MAX = 0.4
FAILURE_MAX_PLOTS_PER_CLASS = 3
FAILURE_FOCUS_CLASSES = [5, 6]

# Live training curves via Visdom (requires: python -m visdom.server on port 8097)
USE_VISDOM = False
