from config import FrozenConfig
from trainer import FrozenTrainer
from utils import seed_everything

def main():
    # 1. Setup Configuration
    config = FrozenConfig(
        seed=42, 
        batch_size=8,
        gradient_accumulation_steps=16,
        num_epochs=3
    )
    
    # 2. Set Random Seeds 
    seed_everything(config.seed)
    
    # 3. Initialize Trainer
    trainer = FrozenTrainer(config)
    
    # 4. Start Training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")

if __name__ == "__main__":
    main()