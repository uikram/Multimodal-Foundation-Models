"""
Unified dataset loading for all models.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
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
    
    def __init__(self, config, processor, debug_mode=False, max_samples=None):
        self.image_dir = Path(config.image_dir)
        self.processor = processor
        self.max_length = config.max_length
        self.samples = []
        self.debug_mode = debug_mode
        
        annotation_file = Path(config.annotation_file)
        
        print(f"Loading annotations from: {annotation_file}")
        print(f"Image directory: {self.image_dir}")
        
        if debug_mode:
            print("🔧 DEBUG MODE: Loading limited samples for testing")
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Parse JSONL with robust error handling and line cleaning
        with open(annotation_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(tqdm(f, desc="Loading samples", disable=debug_mode), 1):
                # In debug mode, limit samples
                if debug_mode and max_samples and len(self.samples) >= max_samples:
                    break
                    
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
                        print(f"Skipped line {line_num}: {str(e)[:50]}")
                    continue
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {annotation_file}. Check JSON format!")
        
        print(f"✓ Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with retry logic."""
        attempts = 0
        max_attempts = min(100, len(self.samples))  # Limit retry attempts
        
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
                if self.debug_mode and attempts < 5:
                    print(f"Error loading {image_path}: {e}")
                attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts starting from idx {idx}")


# ===== Conceptual Captions Dataset for Frozen Model =====

class FrozenConceptualCaptionsDataset(torch.utils.data.Dataset):
    """
    Dataset for Frozen model training (ResNet50 + GPT-2).
    Auto-fixes corrupted JSONL files on-the-fly.
    """
    
    def __init__(self, image_dir, annotation_file, tokenizer, config, debug_mode=False, max_samples=None):
        from torchvision import transforms
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.debug_mode = debug_mode

        # [PAPER FIX] Pad to square logic (Section 4 of paper)
        # Prevents "squashing" objects or cropping out important details.
        def pad_to_square(img):
            w, h = img.size
            if w == h: return img
            max_size = max(w, h)
            # Create black background (0,0,0) as padding
            new_img = Image.new('RGB', (max_size, max_size), (0, 0, 0)) 
            # Paste image in the center
            new_img.paste(img, ((max_size - w) // 2, (max_size - h) // 2))
            return new_img

        self.transform = transforms.Compose([
            transforms.Lambda(pad_to_square), # <--- THE FIX
            transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load annotations with robust error handling
        self.samples = []
        annotation_path = Path(annotation_file)
        
        print(f"[FrozenDataset] Loading annotations from: {annotation_path}")
        print(f"[FrozenDataset] Image directory: {self.image_dir}")
        
        if debug_mode:
            print("🔧 [FrozenDataset] DEBUG MODE: Loading limited samples for testing")
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        if not self.image_dir.exists():
            print(f"[FrozenDataset] Image directory not found: {self.image_dir}")
            # Try to create it or suggest alternatives
            parent_dir = self.image_dir.parent
            if parent_dir.exists():
                available_dirs = [d.name for d in parent_dir.iterdir() if d.is_dir()]
                print(f"    Available directories in {parent_dir}:")
                for d in available_dirs[:10]:
                    print(f"      - {d}")
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # 🔍 Inspect JSONL in debug mode
        if debug_mode:
            inspect_jsonl_file(annotation_path, max_lines=3)
        
        # Parse JSONL with on-the-fly corruption fixes
        error_count = 0
        valid_image_count = 0
        missing_image_count = 0
        
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(f, 1):
                # In debug mode, limit samples
                if debug_mode and max_samples and len(self.samples) >= max_samples:
                    print(f"[FrozenDataset] Reached max_samples limit: {max_samples}")
                    break
                
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
                        # In debug mode, verify image exists before adding
                        if debug_mode:
                            test_path = self.image_dir / rel_path
                            if test_path.exists():
                                self.samples.append({
                                    "caption": caption,
                                    "image_path": rel_path
                                })
                                valid_image_count += 1
                            else:
                                missing_image_count += 1
                                if missing_image_count <= 3:
                                    print(f"  ⚠️  Missing: {test_path}")
                        else:
                            self.samples.append({
                                "caption": caption,
                                "image_path": rel_path
                            })
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"[FrozenDataset] ⚠️  Skipped line {line_num}: {str(e)[:50]}")
                    continue
        
        # More informative error message
        if len(self.samples) == 0:
            error_msg = f"No valid samples found in {annotation_path}.\n"
            error_msg += f"   Total lines with JSON errors: {error_count}\n"
            if debug_mode:
                error_msg += f"   Images missing from disk: {missing_image_count}\n"
            error_msg += f"   Image directory checked: {self.image_dir}\n"
            error_msg += f"\n   Possible issues:\n"
            error_msg += f"   1. JSONL file may be empty or corrupted\n"
            error_msg += f"   2. Image paths in JSONL don't match actual file locations\n"
            error_msg += f"   3. Wrong image_dir path in config\n"
            error_msg += f"\n   Run with debug_mode=True to see detailed inspection."
            raise ValueError(error_msg)
        
        print(f"[FrozenDataset] ✓ Loaded {len(self.samples):,} samples ({error_count} JSON errors)")
        if debug_mode:
            print(f"[FrozenDataset] ✓ Verified {valid_image_count} images exist on disk")
            if missing_image_count > 0:
                print(f"[FrozenDataset] {missing_image_count} images missing from disk")
        
        # Verify first few images exist
        if len(self.samples) > 0:
            print(f"[FrozenDataset] Verifying first 3 image paths...")
            for i in range(min(3, len(self.samples))):
                test_path = self.image_dir / self.samples[i]["image_path"]
                status = "✓" if test_path.exists() else "✗"
                print(f"  {status} {test_path}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with intelligent retry logic."""
        attempts = 0
        max_attempts = min(100, len(self.samples))  # Limit retry attempts
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    if self.debug_mode and attempts < 5:
                        print(f"⚠️  Image not found: {image_path}")
                    attempts += 1
                    continue
                    
                # Load and transform image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                    
                image = self.transform(image)
                
                # ✅ FIX: Tokenize caption with proper label masking
                caption_encoded = self.tokenizer(
                    item["caption"],
                    max_length=self.config.max_caption_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = caption_encoded["input_ids"].squeeze(0)
                attention_mask = caption_encoded["attention_mask"].squeeze(0)
                
                # ✅ CRITICAL FIX: Mask padding tokens in labels
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Ignore padding in loss
                
                return {
                    "images": image,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels  # ← Now properly masked!
                }
                
            except Exception as e:
                if self.debug_mode and attempts < 5:
                    print(f"⚠️  Error loading {image_path}: {e}")
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

def get_conceptual_captions_loader(config, processor, debug_mode=False):
    """Get Conceptual Captions dataloader for CLIP+LoRA."""
    max_samples = 500 if debug_mode else None
    dataset = ConceptualCaptionsDataset(config, processor, debug_mode=debug_mode, max_samples=max_samples)
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if not debug_mode else 0,  # Single worker in debug
        pin_memory=True
    )
    
    return loader


def get_frozen_dataset_loader(config, debug_mode=False):
    """Get Conceptual Captions dataloaders for Frozen model."""
    from transformers import GPT2Tokenizer
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Limit samples in debug mode
    max_samples = 500 if debug_mode else None
    
    print("[FrozenDataLoader] Initializing training dataset...")
    # Create train dataset
    train_dataset = FrozenConceptualCaptionsDataset(
        config.train_image_dir,
        config.train_file,
        tokenizer,
        config,
        debug_mode=debug_mode,
        max_samples=max_samples
    )
    
    print("[FrozenDataLoader] Initializing validation dataset...")
    # Create val dataset (smaller in debug mode)
    val_max_samples = 100 if debug_mode else None
    val_dataset = FrozenConceptualCaptionsDataset(
        config.val_image_dir,
        config.val_file,
        tokenizer,
        config,
        debug_mode=debug_mode,
        max_samples=val_max_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if not debug_mode else 0,  # Single worker in debug
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if not debug_mode else 0,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"[FrozenDataLoader] ✓ Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    return train_loader, val_loader

def inspect_jsonl_file(filepath, max_lines=5):
    """Inspect JSONL file and print sample entries for debugging."""
    print(f"\n🔍 Inspecting: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
            print(f"  Total lines: {len(lines)}")
            
            if len(lines) == 0:
                print("File is empty!")
                return
            
            print(f"  First {min(max_lines, len(lines))} entries:")
            for i, line in enumerate(lines[:max_lines]):
                try:
                    entry = json.loads(line)
                    print(f"    [{i+1}] Keys: {list(entry.keys())}")
                    # Show first entry in detail
                    if i == 0:
                        for key, val in entry.items():
                            val_str = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
                            print(f"        {key}: {val_str}")
                except json.JSONDecodeError as e:
                    print(f"    [{i+1}] ✗ JSON Error: {str(e)[:50]}")
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
