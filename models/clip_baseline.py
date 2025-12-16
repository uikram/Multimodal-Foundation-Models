"""
CLIP Baseline Model - Pretrained OpenAI CLIP for zero-shot and linear probe evaluation.
"""

import torch
import open_clip

class CLIPBaseline:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print(f"Loading CLIP Baseline: {config.model_name}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained_tag,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(config.model_name)
        
    def encode_image(self, images):
        """Encode images to feature vectors."""
        return self.model.encode_image(images)
    
    def encode_text(self, text_tokens):
        """Encode text tokens to feature vectors."""
        return self.model.encode_text(text_tokens)
    
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
