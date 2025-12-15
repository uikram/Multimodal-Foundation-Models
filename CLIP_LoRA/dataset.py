import os
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPProcessor

# Prevent DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None 

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, config, processor: CLIPProcessor):
        self.image_dir = config.image_dir
        self.processor = processor
        self.max_length = config.max_length
        self.samples = []
        
        print(f"Loading annotations from: {config.annotation_file}")
        
        if not config.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {config.annotation_file}")

        # Robust JSONL parsing
        with open(config.annotation_file, 'r') as f:
            for line in tqdm(f, desc="Parsing JSONL"):
                try:
                    entry = json.loads(line.strip())
                    # Handle flexible key names
                    caption = entry.get("caption") or entry.get("text")
                    rel_path = entry.get("filepath") or entry.get("image_path") or entry.get("file_name")
                    
                    if caption and rel_path:
                        self.samples.append({
                            "caption": caption,
                            "image_path": rel_path
                        })
                except json.JSONDecodeError:
                    continue 
                    
        print(f"Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We use a counter to ensure we don't loop forever if the whole dataset is broken
        attempts = 0
        max_attempts = len(self.samples) 
        
        while attempts < max_attempts:
            # If the current idx fails, try the next one
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            image_path = self.image_dir / item["image_path"]
            
            try:
                # 1. Check if file exists
                if not image_path.exists():
                    # Force an error so we jump to the except block
                    raise FileNotFoundError(f"File not found: {image_path}")

                # 2. Try to open image
                image = Image.open(image_path).convert("RGB")
                
                # 3. Process image and text
                inputs = self.processor(
                    text=[item["caption"]],
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                
                # 4. SUCCESS: Return the dictionary
                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0)
                }
                
            except (OSError, Exception):
                # 5. FAILURE: Increment attempt counter and loop again
                attempts += 1
                continue
        
        # If we exit the while loop, it means we tried EVERY image and failed.
        raise RuntimeError(f"Failed to load ANY images. Checked {attempts} files starting from index {idx}. Check your data path: {self.image_dir}")