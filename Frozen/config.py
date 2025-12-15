from dataclasses import dataclass
import torch
from pathlib import Path

@dataclass
class FrozenConfig:
    """Configuration for Frozen model training experiment."""
    
    # --- Model Architecture ---
    vision_encoder_name: str = "resnet50"
    # Options: "gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/gpt-neo-2.7B"
    language_model_name: str = "gpt2-large"
    
    visual_prefix_length: int = 2        # Number of visual tokens to prepend
    vision_hidden_dim: int = 2048        # ResNet50 output dim
    lm_hidden_dim: int = 1280            # GPT2-large=1280, GPT2=768
    
    # --- Training Hyperparameters ---
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    num_epochs: int = 3
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 16 
    
    # --- Data Processing ---
    seed: int = 42
    image_size: int = 224
    max_caption_length: int = 128
    data_root: Path = Path("../../conceptual_captions_data")
    
    # --- System & Logging ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    fp16: bool = True               # Mixed precision training
    output_dir: Path = Path("./frozen_outputs")
    checkpoint_dir: Path = Path("./frozen_checkpoints")

    def __post_init__(self):
        # Ensure paths are Path objects and exist
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)