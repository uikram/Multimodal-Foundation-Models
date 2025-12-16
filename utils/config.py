"""
Unified configuration system for all models.
"""

import os
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class BaseConfig:
    """Base configuration with common parameters."""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())
    data_root: Path = field(default_factory=lambda: Path("cache"))
    results_dir: Path = field(default_factory=lambda: Path("results_attained"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Training - MATCH OLD CODE DEFAULTS
    batch_size: int = 8  # ← FIXED: was 128, now 8 (from old FrozenConfig)
    num_epochs: int = 3
    learning_rate: float = 3e-4  # ← FIXED: was 5e-5, now 3e-4 (from old code)
    
    def __post_init__(self):
        """Ensure all directories exist."""
        self.data_root = Path(self.data_root)
        self.results_dir = Path(self.results_dir)
        self.plots_dir = Path(self.plots_dir)
        self.cache_dir = Path(self.cache_dir)
        
        for dir_path in [self.results_dir, self.plots_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TORCH_HOME'] = str(self.cache_dir)
        os.environ['XDG_CACHE_HOME'] = str(self.cache_dir)

@dataclass
class CLIPConfig(BaseConfig):
    """Configuration for CLIP Baseline."""
    
    model_name: str = "ViT-B-32"
    pretrained_tag: str = "openai"
    batch_size: int = 128  # CLIP uses larger batch
    learning_rate: float = 5e-5
    logistic_regression_c: float = 0.316
    k_shots: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

@dataclass
class CLIPLoRAConfig(BaseConfig):
    """Configuration for CLIP + LoRA."""
    
    model_id: str = "openai/clip-vit-base-patch32"
    batch_size: int = 64  # ← FIXED: LoRA uses 64 (from old code)
    learning_rate: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    max_length: int = 77
    output_dir: Path = field(default_factory=lambda: Path("clip_lora_checkpoints"))
    
    # Conceptual Captions paths - will be overridden by YAML
    image_dir: Path = field(default_factory=lambda: Path("../conceptual_captions_data"))
    annotation_file: Path = field(default_factory=lambda: Path("../conceptual_captions_data/train.jsonl"))
    
    def __post_init__(self):
        super().__post_init__()
        self.output_dir = Path(self.output_dir)
        self.image_dir = Path(self.image_dir)
        self.annotation_file = Path(self.annotation_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class FrozenConfig(BaseConfig):
    """Configuration for Frozen CLIP."""
    
    vision_encoder_name: str = "resnet50"
    language_model_name: str = "gpt2-large"
    visual_prefix_length: int = 2
    vision_hidden_dim: int = 2048
    lm_hidden_dim: int = 1280
    
    # Training params - from old code
    batch_size: int = 8  # ← Explicitly set for Frozen
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 16
    
    # Data
    image_size: int = 224
    max_caption_length: int = 128
    fp16: bool = True
    
    output_dir: Path = field(default_factory=lambda: Path("frozen_outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("frozen_checkpoints"))
    
    # Conceptual Captions paths - will be overridden by YAML
    train_image_dir: Path = field(default_factory=lambda: Path("../conceptual_captions_data"))
    train_file: Path = field(default_factory=lambda: Path("../conceptual_captions_data/train.jsonl"))
    val_image_dir: Path = field(default_factory=lambda: Path("../conceptual_captions_data/validation"))
    val_file: Path = field(default_factory=lambda: Path("../conceptual_captions_data/validation.jsonl"))
    
    def __post_init__(self):
        super().__post_init__()
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.train_image_dir = Path(self.train_image_dir)
        self.train_file = Path(self.train_file)
        self.val_image_dir = Path(self.val_image_dir)
        self.val_file = Path(self.val_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

def load_config_from_yaml(yaml_path: str, model_type: str):
    """
    Load configuration from YAML file.
    YAML values take PRIORITY and override defaults!
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested YAML structure (training, model, data sections)
    flat_dict = {}
    for section in ['model', 'system', 'data', 'training', 'dataset', 'optimizer', 'evaluation']:
        if section in config_dict:
            flat_dict.update(config_dict[section])
    
    # Remove None values to let defaults take effect
    flat_dict = {k: v for k, v in flat_dict.items() if v is not None}
    
    # Create config with YAML overrides
    if model_type == 'clip':
        config = CLIPConfig(**flat_dict)
    elif model_type == 'clip_lora':
        config = CLIPLoRAConfig(**flat_dict)
    elif model_type == 'frozen':
        config = FrozenConfig(**flat_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"✓ Loaded config from {yaml_path}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    
    return config
