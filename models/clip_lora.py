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
        
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs, return_loss=True)
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def parameters(self):
        return self.model.parameters()
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def encode_image(self, images):
        vision_outputs = self.model.get_image_features(pixel_values=images)
        return vision_outputs
    
    def encode_text(self, tokenized_inputs):
        """Encode text using input_ids and attention_mask."""
        # Support both dictionary input and raw tensors for backward compatibility if needed
        if isinstance(tokenized_inputs, dict):
            return self.model.get_text_features(**tokenized_inputs)
        else:
            # Fallback if just input_ids passed
            return self.model.get_text_features(input_ids=tokenized_inputs)