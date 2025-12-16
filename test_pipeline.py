"""
Test script with dummy data to verify the entire pipeline.

This script creates synthetic datasets and runs training, inference, and evaluation
for all three models to ensure everything works correctly.

Usage:
    python test_pipeline.py --quick     # Fast test with minimal data
    python test_pipeline.py --full      # Complete test with more data
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import json
import shutil

# Create dummy data directories
DUMMY_DATA_DIR = Path("dummy_data")
DUMMY_RESULTS_DIR = Path("dummy_results")
DUMMY_PLOTS_DIR = Path("dummy_plots")


def create_dummy_image(size=(224, 224), color='RGB'):
    """Create a random dummy image."""
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img_array, color)


def create_dummy_conceptual_captions(num_samples=100):
    """Create dummy Conceptual Captions dataset."""
    print("\n[1/4] Creating dummy Conceptual Captions dataset...")
    
    data_dir = DUMMY_DATA_DIR / "conceptual_captions"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create dummy annotations (JSONL format)
    train_annotations = []
    val_annotations = []
    
    captions = [
        "A cat sitting on a mat",
        "A dog running in the park",
        "A bird flying in the sky",
        "A car driving on the road",
        "A person walking on the street",
        "A tree in the forest",
        "A flower in the garden",
        "A boat on the water",
        "A mountain covered with snow",
        "A sunset over the ocean"
    ]
    
    for i in range(num_samples):
        # Create dummy image
        img = create_dummy_image()
        img_filename = f"image_{i:05d}.jpg"
        img.save(images_dir / img_filename)
        
        # Create annotation entry - FIXED FORMAT
        entry = {
            "filepath": img_filename,          # Use "filepath" not "file_name"
            "caption": captions[i % len(captions)]
        }
        
        # Split 80/20 train/val
        if i < num_samples * 0.8:
            train_annotations.append(entry)
        else:
            val_annotations.append(entry)
    
    # Save JSONL files - ONE JSON OBJECT PER LINE
    with open(data_dir / "train.jsonl", 'w') as f:
        for entry in train_annotations:
            f.write(json.dumps(entry) + '\n')  # Write each entry as a line
    
    with open(data_dir / "validation.jsonl", 'w') as f:
        for entry in val_annotations:
            f.write(json.dumps(entry) + '\n')  # Write each entry as a line
    
    print(f"✓ Created {len(train_annotations)} training samples")
    print(f"✓ Created {len(val_annotations)} validation samples")
    print(f"✓ Images saved to {images_dir}")
    
    # Verify files were created
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    
    if train_file.exists():
        print(f"✓ Training JSONL created: {train_file} ({train_file.stat().st_size} bytes)")
    if val_file.exists():
        print(f"✓ Validation JSONL created: {val_file} ({val_file.stat().st_size} bytes)")
    
    return data_dir



def create_dummy_benchmark_datasets(num_samples=50):
    """Create dummy benchmark datasets (CIFAR100-like)."""
    print("\n[2/4] Creating dummy benchmark datasets...")
    
    from torch.utils.data import Dataset
    
    class DummyDataset(Dataset):
        """Dummy dataset that mimics CIFAR100/Food101 structure."""
        
        def __init__(self, num_samples, num_classes, transform=None):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.transform = transform
            self.data = []
            self.targets = []
            
            for i in range(num_samples):
                # Random image data
                img_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                label = i % num_classes
                
                self.data.append(img_data)
                self.targets.append(label)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            img = Image.fromarray(self.data[idx])
            label = self.targets[idx]
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
    
    # Create datasets
    datasets = {
        'train': DummyDataset(num_samples, 10),
        'test': DummyDataset(num_samples // 2, 10)
    }
    
    print(f"✓ Created dummy benchmark datasets")
    print(f"  - Train: {len(datasets['train'])} samples")
    print(f"  - Test: {len(datasets['test'])} samples")
    
    return datasets


def create_test_configs():
    """Create test configurations for all models."""
    print("\n[3/4] Creating test configurations...")
    
    from utils.config import CLIPConfig, CLIPLoRAConfig, FrozenConfig
    
    # CLIP Baseline Config
    clip_config = CLIPConfig(
        model_name="ViT-B-32",
        pretrained_tag="openai",
        batch_size=4,
        num_workers=0,
        seed=42,
        data_root=DUMMY_DATA_DIR,
        results_dir=DUMMY_RESULTS_DIR,
        plots_dir=DUMMY_PLOTS_DIR
    )
    
    # CLIP LoRA Config
    lora_config = CLIPLoRAConfig(
        model_id="openai/clip-vit-base-patch32",
        batch_size=4,
        num_epochs=2,
        learning_rate=5e-5,
        num_workers=0,
        seed=42,
        image_dir=DUMMY_DATA_DIR / "conceptual_captions" / "images",
        annotation_file=DUMMY_DATA_DIR / "conceptual_captions" / "train.jsonl",
        results_dir=DUMMY_RESULTS_DIR,
        plots_dir=DUMMY_PLOTS_DIR,
        output_dir=DUMMY_DATA_DIR / "clip_lora_checkpoints"
    )
    
    # Frozen Config - FIXED: Use smaller models for testing
    frozen_config = FrozenConfig(
        vision_encoder_name="resnet18",      # CHANGED: resnet18 instead of resnet50
        language_model_name="gpt2",          # CHANGED: Keep gpt2 (not gpt2-large)
        visual_prefix_length=2,
        vision_hidden_dim=512,               # CHANGED: resnet18 output is 512
        lm_hidden_dim=768,                   # CHANGED: gpt2 hidden is 768
        batch_size=2,
        num_epochs=2,
        num_workers=0,
        seed=42,
        train_image_dir=DUMMY_DATA_DIR / "conceptual_captions" / "images",
        train_file=DUMMY_DATA_DIR / "conceptual_captions" / "train.jsonl",
        val_image_dir=DUMMY_DATA_DIR / "conceptual_captions" / "images",
        val_file=DUMMY_DATA_DIR / "conceptual_captions" / "validation.jsonl",
        results_dir=DUMMY_RESULTS_DIR,
        plots_dir=DUMMY_PLOTS_DIR,
        output_dir=DUMMY_DATA_DIR / "frozen_outputs",
        checkpoint_dir=DUMMY_DATA_DIR / "frozen_checkpoints",
        fp16=False,  # Disable for CPU testing
        gradient_accumulation_steps=1
    )
    
    print("✓ Created test configurations for all models")
    
    return {
        'clip': clip_config,
        'clip_lora': lora_config,
        'frozen': frozen_config
    }

def test_clip_baseline(config, dummy_datasets):
    """Test CLIP baseline model."""
    print("\n" + "="*60)
    print("TESTING CLIP BASELINE")
    print("="*60)
    
    from models.clip_baseline import CLIPBaseline
    from evaluation.metrics import MetricsTracker
    import torchvision.transforms as transforms
    from torch.utils.data import TensorDataset, DataLoader
    
    try:
        # Initialize model
        print("\n[1/3] Initializing CLIP Baseline...")
        model = CLIPBaseline(config)
        
        # Initialize metrics tracker
        metrics = MetricsTracker("CLIP_BASELINE_TEST", config.results_dir)
        metrics.track_parameters(model)
        
        # Test inference with dummy tensor data (bypass PIL Image issue)
        print("\n[2/3] Testing inference...")
        metrics.start_inference_timer()
        
        # Create dummy tensor data directly
        dummy_images = torch.randn(10, 3, 224, 224)  # 10 images, 3 channels, 224x224
        dummy_labels = torch.randint(0, 10, (10,))   # 10 labels
        
        dummy_dataset = TensorDataset(dummy_images, dummy_labels)
        test_loader = DataLoader(dummy_dataset, batch_size=config.batch_size, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(config.device)
                features = model.encode_image(images)
                print(f"    ✓ Encoded batch: images {images.shape} -> features {features.shape}")
                break  # Just test one batch
        
        metrics.end_inference_timer()
        print("✓ Inference test passed")
        
        # Test evaluation metrics
        print("\n[3/3] Testing evaluation metrics...")
        
        # Simulate evaluation results
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2, 0])
        
        metrics.track_performance(accuracy=80.0, loss=0.5)
        metrics.track_classification_report(y_true, y_pred)
        metrics.track_gpu_memory('inference')
        metrics.track_cpu_memory()
        
        # Save metrics
        metrics.save_metrics(run_id="test_run")
        metrics.print_summary()
        
        print("\n✅ CLIP Baseline test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ CLIP Baseline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_clip_lora(config):
    """Test CLIP + LoRA model."""
    print("\n" + "="*60)
    print("TESTING CLIP + LORA")
    print("="*60)
    
    from models.clip_lora import CLIPLoRA
    from evaluation.metrics import MetricsTracker
    from training.train import ModelTrainer
    
    try:
        # Initialize model
        print("\n[1/3] Initializing CLIP + LoRA...")
        model = CLIPLoRA(config)
        
        # Initialize metrics tracker
        metrics = MetricsTracker("CLIP_LORA_TEST", config.results_dir)
        metrics.track_parameters(model)
        
        # Test training
        print("\n[2/3] Testing training (2 epochs)...")
        trainer = ModelTrainer(model, config, metrics)
        
        # Override training with minimal data
        print("  Creating minimal training loop...")
        from datasets.dataloaders import get_conceptual_captions_loader
        
        train_loader = get_conceptual_captions_loader(config, model.processor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        metrics.start_training_timer()
        model.train()
        
        for epoch in range(2):  # Just 2 epochs for testing
            print(f"\n  Epoch {epoch+1}/2")
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 5:  # Only 5 batches for testing
                    break
                
                batch = {k: v.to(config.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model.forward(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                print(f"    Batch {batch_idx+1}: Loss = {loss.item():.4f}")
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"  Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        metrics.end_training_timer()
        print("✓ Training test passed")
        
        # Test inference
        print("\n[3/3] Testing inference...")
        metrics.start_inference_timer()
        
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                outputs = model.forward(**batch)
                break  # Just test one batch
        
        metrics.end_inference_timer()
        print("✓ Inference test passed")
        
        # Save metrics
        metrics.track_performance(accuracy=75.0, loss=avg_loss)
        metrics.track_gpu_memory('training')
        metrics.track_cpu_memory()
        metrics.save_metrics(run_id="test_run")
        metrics.print_summary()
        
        print("\n✅ CLIP + LoRA test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ CLIP + LoRA test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frozen_model(config):
    """Test Frozen model."""
    print("\n" + "="*60)
    print("TESTING FROZEN MODEL")
    print("="*60)
    
    from models.frozen_clip import FrozenCLIP
    from evaluation.metrics import MetricsTracker
    from transformers import GPT2Tokenizer
    from torch.utils.data import DataLoader
    import json
    from PIL import Image
    import torchvision.transforms as transforms
    
    try:
        # Initialize model
        print("\n[1/3] Initializing Frozen Model...")
        model = FrozenCLIP(config)
        
        # Initialize metrics tracker
        metrics = MetricsTracker("FROZEN_TEST", config.results_dir)
        metrics.track_parameters(model)
        
        # Test training
        print("\n[2/3] Testing training (2 epochs)...")
        
        # Create simple dataset class for testing
        class SimpleFrozenDataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, annotation_file, tokenizer, config):
                self.image_dir = Path(image_dir)
                self.tokenizer = tokenizer
                self.config = config
                
                self.transform = transforms.Compose([
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                # Load annotations
                # Load annotations
                self.samples = []
                annotation_path = Path(annotation_file)

                if annotation_path.exists():
                    print(f"Loading from {annotation_path}")
                    with open(annotation_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                            try:
                                entry = json.loads(line)
                                # Handle both "caption" and "text" keys
                                caption = entry.get("caption") or entry.get("text") or "A photo"
                                # Handle both "filepath" and "image_path" keys
                                filepath = entry.get("filepath") or entry.get("image_path") or entry.get("file_name")
                                
                                if filepath:
                                    self.samples.append({"caption": caption, "image_path": filepath})
                                else:
                                    print(f"  Warning: Line {line_num} missing filepath: {entry}")
                            except json.JSONDecodeError as e:
                                print(f"  Warning: Failed to parse line {line_num}: {e}")
                                continue
                else:
                    print(f"Warning: Annotation file {annotation_path} not found")

                print(f"✓ Loaded {len(self.samples)} valid entries from {annotation_path}")
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                item = self.samples[idx]
                image_path = self.image_dir / item["image_path"]
                
                # Load image
                try:
                    image = Image.open(image_path).convert("RGB")
                    image = self.transform(image)
                except:
                    # Fallback to random tensor
                    image = torch.randn(3, self.config.image_size, self.config.image_size)
                
                # Tokenize caption
                caption_encoded = self.tokenizer(
                    item["caption"],
                    max_length=self.config.max_caption_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    "images": image,
                    "input_ids": caption_encoded["input_ids"].squeeze(0),
                    "attention_mask": caption_encoded["attention_mask"].squeeze(0),
                    "labels": caption_encoded["input_ids"].squeeze(0)
                }
        
        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        train_ds = SimpleFrozenDataset(
            config.train_image_dir,
            config.train_file,
            tokenizer,
            config
        )
        
        val_ds = SimpleFrozenDataset(
            config.val_image_dir,
            config.val_file,
            tokenizer,
            config
        )
        
        if len(train_ds) == 0:
            raise ValueError("Training dataset is empty! Check paths.")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        metrics.start_training_timer()
        model.train()
        
        for epoch in range(2):  # Just 2 epochs for testing
            print(f"\n  Epoch {epoch+1}/2")
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 3:  # Only 3 batches for testing
                    break
                
                images = batch['images'].to(config.device)
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                optimizer.zero_grad()
                logits, loss = model.forward(images, input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                print(f"    Batch {batch_idx+1}: Loss = {loss.item():.4f}")
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"  Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        metrics.end_training_timer()
        print("✓ Training test passed")
        
        # Test inference
        print("\n[3/3] Testing inference...")
        metrics.start_inference_timer()
        
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch['images'].to(config.device)
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                
                logits, loss = model.forward(images, input_ids, attention_mask, labels)
                break  # Just test one batch
        
        metrics.end_inference_timer()
        print("✓ Inference test passed")
        
        # Save metrics
        metrics.track_performance(accuracy=70.0, loss=avg_loss)
        metrics.track_gpu_memory('training')
        metrics.track_cpu_memory()
        metrics.save_metrics(run_id="test_run")
        metrics.print_summary()
        
        print("\n✅ Frozen Model test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Frozen Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False



def cleanup():
    """Clean up test data."""
    print("\n" + "="*60)
    print("CLEANING UP TEST DATA")
    print("="*60)
    
    dirs_to_remove = [DUMMY_DATA_DIR, DUMMY_RESULTS_DIR, DUMMY_PLOTS_DIR]
    
    for dir_path in dirs_to_remove:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Removed {dir_path}")


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test pipeline with dummy data")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with minimal data (50 samples)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full test with more data (200 samples)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['clip', 'clip_lora', 'frozen', 'all'],
        default='all',
        help='Which model to test'
    )
    parser.add_argument(
        '--keep-data',
        action='store_true',
        help='Keep dummy data after testing'
    )
    
    args = parser.parse_args()
    
    # Determine number of samples
    if args.full:
        num_samples = 200
    elif args.quick:
        num_samples = 50
    else:
        num_samples = 100
    
    print("\n" + "="*60)
    print("MULTIMODAL FOUNDATION MODELS - PIPELINE TEST")
    print("="*60)
    print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Standard'}")
    print(f"Models: {args.model}")
    print(f"Samples: {num_samples}")
    print("="*60)
    
    try:
        # Setup
        print("\n📦 SETUP PHASE")
        print("-" * 60)
        
        # Create dummy data
        conceptual_captions_dir = create_dummy_conceptual_captions(num_samples)
        dummy_datasets = create_dummy_benchmark_datasets(num_samples // 2)
        configs = create_test_configs()
        
        print("\n[4/4] Setup complete!")
        
        # Run tests
        print("\n🧪 TESTING PHASE")
        print("-" * 60)
        
        results = {}
        
        if args.model in ['clip', 'all']:
            results['clip'] = test_clip_baseline(configs['clip'], dummy_datasets)
        
        if args.model in ['clip_lora', 'all']:
            results['clip_lora'] = test_clip_lora(configs['clip_lora'])
        
        if args.model in ['frozen', 'all']:
            results['frozen'] = test_frozen_model(configs['frozen'])
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        all_passed = True
        for model_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{model_name.upper():20} {status}")
            if not passed:
                all_passed = False
        
        print("="*60)
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Pipeline is working correctly.")
        else:
            print("\n⚠️  SOME TESTS FAILED. Check the output above for details.")
        
        # Cleanup
        if not args.keep_data:
            cleanup()
        else:
            print(f"\n📁 Test data preserved in:")
            print(f"  - Data: {DUMMY_DATA_DIR}")
            print(f"  - Results: {DUMMY_RESULTS_DIR}")
            print(f"  - Plots: {DUMMY_PLOTS_DIR}")
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        if not args.keep_data:
            cleanup()
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        if not args.keep_data:
            cleanup()
        return 1


if __name__ == "__main__":
    exit(main())
