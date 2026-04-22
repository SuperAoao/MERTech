# config for CBFdataset experiments (player-based CV)
import torch

# Dataset description
DATASET_ROOT = "CBFdataset"
CBF_NUM_PLAYERS = 10
CBF_TOTAL_WAVS = 80  # monophonic CBF performances
CBF_TOTAL_HOURS = 2.6

# Techniques (7 classes)
# Paper naming: vibrato, tremolo, trill, flutter-tongue, acciaccatura, portamento, glissando
CBF_TECHNIQUES = [
    "Vibrato",
    "Tremolo",
    "Trill",
    "FT",  # flutter-tongue
    "Acciacatura",
    "Portamento",
    "Glissando",
]
NUM_LABELS = 7
CBF_TECH_TO_IDX = {name: i for i, name in enumerate(CBF_TECHNIQUES)}

# Audio / framing
URL = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 44100
MERT_SAMPLE_RATE = 24000 if "MERT" in URL else 16000
FEATURE_RATE = 75  # frames per second for labels
TIME_LENGTH = 5  # seconds per chunk
LENGTH = FEATURE_RATE * TIME_LENGTH  # 375 frames per chunk
HOPS_IN_ONSET = 1  # onset spans how many frames

# Model output factorization (pitch bins not used for CBF; keep 1 bin to satisfy shapes)
MIN_MIDI = 0
MAX_MIDI = 0

# Training hyperparameters (requested)
BATCH_SIZE = 10
INITIAL_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 3.0

# Scheduler (cosine)
USE_COSINE_SCHEDULER = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

