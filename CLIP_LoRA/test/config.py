import os
import torch
from pathlib import Path

# --- System & Hardware ---
CWD = Path(os.getcwd())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# --- Paths ---
DATA_ROOT = Path("cache")
RESULTS_FOLDER = Path("results")
ZERO_SHOT_RESULTS_FILE = RESULTS_FOLDER / "lora_zero_shot_results.json"
LINEAR_PROBE_RESULTS_FILE = RESULTS_FOLDER / "lora_linear_probe_results.json"
FEW_SHOT_RESULTS_FILE = RESULTS_FOLDER / "lora_few_shot_results.json"

# Ensure directories exist
DATA_ROOT.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# --- Model Hyperparameters ---
MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "openai"
BATCH_SIZE = 128

# --- Analysis Hyperparameters ---
LOGISTIC_REGRESSION_C = 0.316
K_SHOTS = [1, 2, 4, 8, 16]