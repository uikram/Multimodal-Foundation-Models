import os
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    """Central configuration for CLIP LoRA Fine-tuning."""
    
    # --- Paths ---
    # Defines where all external files (datasets, models) live relative to this script
    base_dir: Path = Path(__file__).parent.resolve()
    data_root: Path = base_dir / "../../conceptual_captions_data"  # Adjust as needed
    output_dir: Path = base_dir / "clip_lora_checkpoints"
    cache_dir: Path = base_dir / "cache"
    
    # Dataset specific paths
    image_dir: Path = data_root 
    annotation_file: Path = data_root / "train.jsonl"

    # --- Model & Training ---
    model_id: str = "openai/clip-vit-base-patch32"
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 77
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # --- LoRA Hyperparameters ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = ("q_proj", "v_proj")

    def __post_init__(self):
        """Create directories and set environment variables for caching."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Force libraries to use local cache
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TORCH_HOME'] = str(self.cache_dir)
        os.environ['XDG_CACHE_HOME'] = str(self.cache_dir)