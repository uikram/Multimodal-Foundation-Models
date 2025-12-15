import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict

from config import FrozenConfig

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, annotations_file: Path, image_dir: Path, tokenizer, config: FrozenConfig):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.config = config
        
        self.entries = []
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                for line in f:
                    self.entries.append(json.loads(line))
            print(f"Dataset loaded from {annotations_file}. Total entries: {len(self.entries)}")
        else:
            print(f"Warning: Annotation file {annotations_file} not found.")

        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Robust loop to handle missing or corrupted images
        while True:
            entry = self.entries[idx]
            img_path = self.image_dir / entry['file_name']

            try:
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing: {img_path}")

                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                
                encoding = self.tokenizer(
                    entry['text'],
                    max_length=self.config.max_caption_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100 # Ignore padding in loss
                
                return {
                    'images': image,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

            except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
                # Move to next index if current is invalid
                idx = (idx + 1) % len(self.entries)