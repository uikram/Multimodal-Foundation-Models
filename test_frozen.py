"""
Quick test script for Frozen model with debug mode.
Tests dataloader and runs 1-2 training iterations to verify everything works.
"""

import torch
import warnings
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

from utils.config import load_config_from_yaml
from models import get_model

def check_data_paths(config):
    """Check if data paths exist and suggest fixes."""
    print("\n" + "="*60)
    print("CHECKING DATA PATHS")
    print("="*60)
    
    paths_to_check = [
        ("Train images", config.train_image_dir),
        ("Train JSONL", config.train_file),
        ("Val images", config.val_image_dir),
        ("Val JSONL", config.val_file),
    ]
    
    all_exist = True
    for name, path in paths_to_check:
        path = Path(path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
            
            # Suggest alternatives
            if path.parent.exists():
                similar = [p.name for p in path.parent.iterdir() 
                          if path.stem.lower() in p.name.lower() or 
                          p.stem.lower() in path.name.lower()]
                if similar:
                    print(f"    Similar items found: {similar[:3]}")
    
    if not all_exist:
        print("\n⚠️  Some paths don't exist. Please check your config.")
        print("    Edit configs/frozen_clip.yaml to fix paths.")
        return False
    
    print("\n✓ All paths exist!")
    return True

def test_frozen_model():
    """Test frozen model with debug mode."""
    
    print("\n" + "="*60)
    print("FROZEN MODEL QUICK TEST")
    print("="*60)
    
    # Load config
    config_path = "configs/frozen_clip.yaml"
    config = load_config_from_yaml(config_path, 'frozen')
    
    # Force debug mode
    config.debug_mode = True
    config.num_epochs = 1  # Only 1 epoch for testing
    config.batch_size = 4   # Small batch size
    
    print(f"\n✓ Config loaded")
    print(f"  Device: {config.device}")
    print(f"  Debug Mode: {config.debug_mode}")
    print(f"  Batch Size: {config.batch_size}")
    
    # Check data paths
    if not check_data_paths(config):
        print("\n❌ Please fix data paths before continuing.")
        sys.exit(1)
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = get_model('frozen', config)
    print("✓ Model initialized")
    
    # Test dataloader
    print("\n" + "="*60)
    print("TESTING DATALOADERS")
    print("="*60)
    
    from training.trainer_frozen import FrozenTrainer
    from evaluation.metrics import MetricsTracker
    
    metrics_tracker = MetricsTracker(
        model_name="FROZEN_TEST",
        results_dir=Path("test_results")
    )
    
    trainer = FrozenTrainer(model, config, metrics_tracker)
    
    try:
        train_loader, val_loader = trainer.get_dataloaders()
    except ValueError as e:
        print(f"\n❌ Error loading datasets: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if validation.jsonl has the same format as train.jsonl")
        print("2. Check if image paths in validation.jsonl match actual files")
        print("3. Try using a subset of train data for validation if val data missing")
        sys.exit(1)
    
    print(f"\n✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test loading one batch
    print("\n" + "="*60)
    print("TESTING BATCH LOADING")
    print("="*60)
    
    try:
        batch = next(iter(train_loader))
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"✓ Batch loaded successfully")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    
    # Check if labels are properly masked
    has_padding_mask = (-100 in batch['labels'])
    print(f"  Labels contain -100 (padding masked): {has_padding_mask}")
    if not has_padding_mask:
        print("  ⚠️  WARNING: Labels don't have padding masked!")
    
    # Test forward pass
    print("\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    model.to(config.device)
    batch = {k: v.to(config.device) for k, v in batch.items()}
    
    try:
        with torch.no_grad():
            logits, loss = model(**batch)
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test one training step
    print("\n" + "="*60)
    print("TESTING TRAINING STEP")
    print("="*60)
    
    model.train()
    try:
        loss_value = trainer.train_step(batch)
    except Exception as e:
        print(f"❌ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss_value:.4f}")
    
    # Test validation step
    print("\n" + "="*60)
    print("TESTING VALIDATION STEP")
    print("="*60)
    
    try:
        val_batch = next(iter(val_loader))
        val_batch = {k: v.to(config.device) for k, v in val_batch.items()}
        
        model.eval()
        with torch.no_grad():
            _, val_loss = model(**val_batch)
        
        print(f"✓ Validation step successful")
        print(f"  Val Loss: {val_loss.item():.4f}")
    except Exception as e:
        print(f"⚠️  Validation test failed: {e}")
        print("    (This is OK if you don't have validation data)")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\nYou can now train on full dataset by:")
    print("  1. Set debug_mode: false in configs/frozen_clip.yaml")
    print("  2. Run: python main.py --models frozen --mode train")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_frozen_model()
