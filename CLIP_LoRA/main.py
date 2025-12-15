import os
import torch
import warnings
from torch.utils.data import DataLoader

# Local imports
from config import Config
from utils import seed_everything
from dataset import ConceptualCaptionsDataset
from model import get_model_and_processor
from trainer import Trainer

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The channel dimension is ambiguous")

def main():
    # 1. Initialize Config
    config = Config()
    
    # 2. Set Random Seed (Critical for reproducibility)
    seed_everything(config.seed)
    
    # 3. Load Model & Processor
    # Environment variables in Config ensure this goes to ./cache
    model, processor = get_model_and_processor(config)
    
    # 4. Prepare Dataset
    if not config.annotation_file.exists():
        print(f"CRITICAL ERROR: File not found {config.annotation_file}")
        return

    dataset = ConceptualCaptionsDataset(config, processor)
    
    # 5. Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True 
    )
    
    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 7. Start Training
    trainer = Trainer(model, train_loader, optimizer, config)
    trainer.run()

if __name__ == "__main__":
    main()