"""
CLIP + LoRA Model - CLIP with Low-Rank Adaptation for efficient fine-tuning.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

class CLIPLoRA:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print(f"Loading CLIP + LoRA: {config.model_id}")
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained(
            config.model_id,
            cache_dir=config.cache_dir
        )
        
        # Load base model
        base_model = CLIPModel.from_pretrained(
            config.model_id,
            cache_dir=config.cache_dir
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, peft_config)
        self.model.to(self.device)
        
        print("LoRA Configuration:")
        self.model.print_trainable_parameters()
        
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs, return_loss=True)
    
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
    
    def save_pretrained(self, path):
        """Save LoRA adapter weights."""
        self.model.save_pretrained(path)

    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def encode_image(self, images):
        """Encode images to feature vectors."""
        vision_outputs = self.model.get_image_features(pixel_values=images)
        return vision_outputs
    
    def encode_text(self, text_tokens):
        """Encode text tokens to feature vectors."""
        text_outputs = self.model.get_text_features(input_ids=text_tokens)
        return text_outputs
