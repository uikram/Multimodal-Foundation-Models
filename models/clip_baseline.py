"""
CLIP Baseline Model - Hugging Face implementation for fair comparison.
"""

import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPBaseline:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print(f"Loading CLIP Baseline (Hugging Face): {config.model_name}")
        
        # Load model and processor from Hugging Face
        self.model = CLIPModel.from_pretrained(config.model_name)
        self.processor = CLIPProcessor.from_pretrained(config.model_name)
        
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, texts):
        """
        Tokenize texts using the model's processor.
        Returns a dictionary containing input_ids and attention_mask.
        """
        inputs = self.processor(
            text=texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def encode_image(self, images):
        """Encode images to feature vectors."""
        # Assuming images are already tensors (pixel_values) from the dataloader
        return self.model.get_image_features(pixel_values=images)
    
    def encode_text(self, tokenized_inputs):
        """Encode text tokens to feature vectors."""
        # Unpack the dictionary (input_ids, attention_mask)
        return self.model.get_text_features(**tokenized_inputs)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        
    def train(self):
        """Set model to training mode."""
        self.model.train()
        
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self