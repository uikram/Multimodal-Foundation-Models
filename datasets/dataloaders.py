"""
Unified dataset loading for all models.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ===== Conceptual Captions Dataset for CLIP =====

class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions dataset for CLIP training.
    Auto-fixes corrupted JSONL files on-the-fly.
    """
    
    def __init__(self, config, processor):
        self.image_dir = Path(config.image_dir)
        self.processor = processor
        self.max_length = config.max_length
        self.samples = []
        
        annotation_file = Path(config.annotation_file)
        
        print(f"Loading annotations from: {annotation_file}")
        print(f"Image directory: {self.image_dir}")
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Parse JSONL with robust error handling and line cleaning
        with open(annotation_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(tqdm(f, desc="Loading samples"), 1):
                # Clean the line
                line = raw_line.strip()
                if not line:
                    continue
                
                try:
                    # Try to parse JSON
                    entry = json.loads(line)
                    
                    # Get caption and filepath with flexible key names
                    caption = (
                        entry.get("caption") or 
                        entry.get("text") or 
                        entry.get("description")
                    )
                    filepath = (
                        entry.get("filepath") or 
                        entry.get("image_path") or 
                        entry.get("file_name") or 
                        entry.get("image")
                    )
                    
                    if caption and filepath:
                        self.samples.append({
                            "caption": caption,
                            "image_path": filepath
                        })
                
                except json.JSONDecodeError as e:
                    if line_num <= 10:
                        print(f"  ⚠️  Skipped line {line_num}: {str(e)[:50]}")
                    continue
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ No valid samples found in {annotation_file}. Check JSON format!")
        
        print(f"✓ Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with retry logic."""
        attempts = 0
        max_attempts = len(self.samples)
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            
            # Full image path
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    attempts += 1
                    continue
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process with CLIP processor
                inputs = self.processor(
                    text=[item["caption"]],
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                
                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0)
                }
            
            except Exception as e:
                attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts")


# ===== Conceptual Captions Dataset for Frozen Model =====

class FrozenConceptualCaptionsDataset(torch.utils.data.Dataset):
    """
    Dataset for Frozen model training (ResNet50 + GPT-2).
    Auto-fixes corrupted JSONL files on-the-fly.
    """
    
    def __init__(self, image_dir, annotation_file, tokenizer, config):
        from torchvision import transforms
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.config = config
        
        # Image transforms (ResNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations with robust error handling
        self.samples = []
        annotation_path = Path(annotation_file)
        
        print(f"[FrozenDataset] Loading annotations from: {annotation_path}")
        print(f"[FrozenDataset] Image directory: {self.image_dir}")
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Parse JSONL with on-the-fly corruption fixes
        error_count = 0
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(f, 1):
                # Clean the line
                line = raw_line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    # Handle flexible key names
                    caption = (
                        entry.get("caption") or 
                        entry.get("text") or 
                        entry.get("description")
                    )
                    
                    rel_path = (
                        entry.get("filepath") or 
                        entry.get("image_path") or 
                        entry.get("file_name") or
                        entry.get("image")
                    )
                    
                    if caption and rel_path:
                        self.samples.append({
                            "caption": caption,
                            "image_path": rel_path
                        })
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"[FrozenDataset] ⚠️  Skipped line {line_num}: {str(e)[:50]}")
                    continue
        
        if len(self.samples) == 0:
            raise ValueError(
                f"❌ No valid samples found in {annotation_path}.\n"
                f"   Total lines with errors: {error_count}\n"
                f"   Check JSON format and image paths!"
            )
        
        print(f"[FrozenDataset] ✓ Loaded {len(self.samples):,} samples ({error_count} errors/skipped)")
        
        # Verify first image exists
        first_path = self.image_dir / self.samples[0]["image_path"]
        if not first_path.exists():
            print(f"[FrozenDataset] ⚠️  First image not found: {first_path}")
            print(f"[FrozenDataset]    Using fallback retry logic during training...")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with intelligent retry logic."""
        attempts = 0
        max_attempts = len(self.samples)
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    attempts += 1
                    continue
                    
                # Load and transform image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                    
                image = self.transform(image)
                
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
            except Exception as e:
                attempts += 1
                continue
        
        # Raise error if all attempts fail
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts starting from index {idx}")


# ===== Dataset Factory for Benchmark Datasets =====

class DatasetFactory:
    """Factory for loading benchmark datasets."""
    
    @staticmethod
    def get_zeroshot_config(transform, data_root):
        """Get zero-shot evaluation datasets."""
        from utils.templates import (
            CIFAR100_CLASS_NAMES, FOOD101_CLASS_NAMES, FLOWERS102_CLASS_NAMES,
            DESCRIBEABLETEXTURES_CLASS_NAMES, EUROSAT_CLASS_NAMES,
            CIFAR100_TEMPLATES, FOOD101_TEMPLATES, FLOWERS102_TEMPLATES,
            DESCRIBEABLETEXTURES_TEMPLATES, EUROSAT_TEMPLATES
        )
        
        return {
            "CIFAR100": {
                "dataset": CIFAR100(root=data_root, download=True, train=False, transform=transform),
                "class_names": CIFAR100_CLASS_NAMES,
                "templates": CIFAR100_TEMPLATES
            },
            "Food101": {
                "dataset": Food101(root=data_root, download=True, split='test', transform=transform),
                "class_names": FOOD101_CLASS_NAMES,
                "templates": FOOD101_TEMPLATES
            },
            "Flowers102": {
                "dataset": Flowers102(root=data_root, download=True, split='test', transform=transform),
                "class_names": FLOWERS102_CLASS_NAMES,
                "templates": FLOWERS102_TEMPLATES
            },
            "DTD": {
                "dataset": DTD(root=data_root, download=True, split='test', transform=transform),
                "class_names": DESCRIBEABLETEXTURES_CLASS_NAMES,
                "templates": DESCRIBEABLETEXTURES_TEMPLATES
            },
            "EuroSAT": {
                "dataset": EuroSAT(root=data_root, download=True, transform=transform),
                "class_names": EUROSAT_CLASS_NAMES,
                "templates": EUROSAT_TEMPLATES
            }
        }
    
    @staticmethod
    def get_linear_probe_datasets(transform, data_root):
        """Get linear probe datasets."""
        return {
            "CIFAR100": {
                "train": CIFAR100(root=data_root, download=True, train=True, transform=transform),
                "test": CIFAR100(root=data_root, download=True, train=False, transform=transform)
            },
            "Food101": {
                "train": Food101(root=data_root, download=True, split='train', transform=transform),
                "test": Food101(root=data_root, download=True, split='test', transform=transform)
            },
            "Flowers102": {
                "train": Flowers102(root=data_root, download=True, split='train', transform=transform),
                "test": Flowers102(root=data_root, download=True, split='test', transform=transform)
            },
            "DTD": {
                "train": DTD(root=data_root, download=True, split='train', transform=transform),
                "test": DTD(root=data_root, download=True, split='test', transform=transform)
            },
            "EuroSAT": {
                "train": EuroSAT(root=data_root, download=True, transform=transform),
                "test": EuroSAT(root=data_root, download=True, transform=transform)
            }
        }


# ===== Dataloader Functions =====

def get_conceptual_captions_loader(config, processor):
    """Get Conceptual Captions dataloader for CLIP+LoRA."""
    dataset = ConceptualCaptionsDataset(config, processor)
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return loader


def get_frozen_dataset_loader(config):
    """Get Conceptual Captions dataloaders for Frozen model."""
    from transformers import GPT2Tokenizer
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("[FrozenDataLoader] Initializing training dataset...")
    # Create train dataset
    train_dataset = FrozenConceptualCaptionsDataset(
        config.train_image_dir,
        config.train_file,
        tokenizer,
        config
    )
    
    print("[FrozenDataLoader] Initializing validation dataset...")
    # Create val dataset
    val_dataset = FrozenConceptualCaptionsDataset(
        config.val_image_dir,
        config.val_file,
        tokenizer,
        config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"[FrozenDataLoader] ✓ Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    return train_loader, val_loader
