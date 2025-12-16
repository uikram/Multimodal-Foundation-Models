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

# ===== Conceptual Captions Dataset =====

Image.MAX_IMAGE_PIXELS = None


class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions dataset.
    
    Expected structure:
        conceptual_captions_data/
        ├── train/              <- images folder
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── train.jsonl         <- annotations
        └── validation.jsonl
    
    JSONL format: {"filepath": "image1.jpg", "caption": "A cat"}
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
        
        # Parse JSONL
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading samples"), 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Get caption and filepath
                    caption = entry.get("caption") or entry.get("text")
                    filepath = entry.get("filepath") or entry.get("image_path") or entry.get("file_name")
                    
                    if caption and filepath:
                        self.samples.append({
                            "caption": caption,
                            "image_path": filepath
                        })
                
                except json.JSONDecodeError:
                    if line_num <= 10:
                        print(f"Warning: Failed to parse line {line_num}")
                    continue
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found!")
        
        print(f"✓ Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            
            # Full image path
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Process with CLIP
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
                # if attempts < 3:
                    # print(f"Warning: Failed to load {image_path}: {e}")
                # attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load sample after {attempts} attempts")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        attempts = 0
        max_attempts = len(self.samples)
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"File not found: {image_path}")
                
                image = Image.open(image_path).convert("RGB")
                
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
            
            except (OSError, Exception):
                attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load any images starting from index {idx}")

    # Dataset class
    class FrozenConceptualCaptionsDataset(torch.utils.data.Dataset):
        """Dataset for Frozen model training."""
        
        def __init__(self, image_dir, annotation_file, tokenizer, config):
            self.image_dir = Path(image_dir)
            self.tokenizer = tokenizer
            self.config = config
            
            # Image transforms
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Load annotations
            self.samples = []
            annotation_path = Path(annotation_file)
            
            print(f"Loading annotations from: {annotation_path}")
            print(f"Image directory: {self.image_dir}")
            
            if not annotation_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
            with open(annotation_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
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
                    except json.JSONDecodeError:
                        continue
            
            if len(self.samples) == 0:
                raise ValueError(f"No valid samples found in {annotation_path}")
            
            print(f"✓ Loaded {len(self.samples):,} samples")
            
            # Verify first image
            first_path = self.image_dir / self.samples[0]["image_path"]
            if not first_path.exists():
                print(f"⚠️  Warning: First image not found: {first_path}")
                print(f"   Check if image_dir ({self.image_dir}) is correct")

        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            # Infinite retry logic
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
                        
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert("RGB")
                        
                    image = self.transform(image)
                    
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
                except Exception:
                    attempts += 1
                    continue
            
            # Dummy fallback
            return {
                "images": torch.zeros(3, self.config.image_size, self.config.image_size),
                "input_ids": torch.zeros(self.config.max_caption_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.config.max_caption_length, dtype=torch.long),
                "labels": torch.zeros(self.config.max_caption_length, dtype=torch.long)
            }


# ===== Dataset Factory =====

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


def get_conceptual_captions_loader(config, processor):
    """Get Conceptual Captions dataloader for CLIP+LoRA."""
    from datasets.conceptual_captions import ConceptualCaptionsDataset
    from torch.utils.data import DataLoader
    
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
    from torch.utils.data import DataLoader
    import torch
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    class FrozenConceptualCaptionsDataset(torch.utils.data.Dataset):
        """Dataset for Frozen model training."""
        
        def __init__(self, image_dir, annotation_file, tokenizer, config):
            import json
            from pathlib import Path
            from PIL import Image
            import torchvision.transforms as transforms
            
            self.image_dir = Path(image_dir)
            self.tokenizer = tokenizer
            self.config = config
            
            # Image transforms
            self.transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Load annotations
            self.samples = []
            with open(annotation_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    caption = entry.get("caption") or entry.get("text")
                    filepath = entry.get("filepath") or entry.get("image_path")
                    if caption and filepath:
                        self.samples.append({"caption": caption, "image_path": filepath})
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            from PIL import Image
            
            item = self.samples[idx]
            image_path = self.image_dir / item["image_path"]
            
            # Load image
            image = Image.open(image_path).convert("RGB")
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
    
    # Create train and val datasets
    train_dataset = FrozenConceptualCaptionsDataset(
        config.train_image_dir,
        config.train_file,
        tokenizer,
        config
    )
    
    val_dataset = FrozenConceptualCaptionsDataset(
        config.val_image_dir,
        config.val_file,
        tokenizer,
        config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader