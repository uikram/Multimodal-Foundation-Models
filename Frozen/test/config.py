import os
import torch
from pathlib import Path
from dataclasses import dataclass

# --- General System Paths (Relative to test/ directory) ---
# Assuming 'cache' and 'results' are in the project root (one level up)
DATA_ROOT = Path("../cache") 
RESULTS_FOLDER = Path("../results")

# Ensure directories exist
DATA_ROOT.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

# --- Analysis Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 128 # For evaluation
LOGISTIC_REGRESSION_C = 0.316
K_SHOTS = [1, 2, 4, 8, 16]

# --- Frozen Model Configuration ---
@dataclass
class FrozenConfig:
    """Configuration for Frozen model."""
    vision_encoder_name: str = "resnet50"
    language_model_name: str = "gpt2-large"
    
    visual_prefix_length: int = 2
    vision_hidden_dim: int = 2048
    lm_hidden_dim: int = 1280
    
    # Training params (kept for compatibility, though not used in eval)
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    num_epochs: int = 3
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 16 
    
    seed: int = 42
    image_size: int = 224
    max_caption_length: int = 128
    data_root: Path = Path("../../conceptual_captions_data")
    
    device: str = DEVICE
    num_workers: int = 4
    fp16: bool = True