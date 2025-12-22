"""
Frozen CLIP Model - Frozen language model with trainable vision encoder.
"""

import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms

class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50 with projection."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        self.backbone = timm.create_model(
            config.vision_encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer
        self.projection = nn.Linear(
            config.vision_hidden_dim,
            config.visual_prefix_length * config.lm_hidden_dim
        )
    
    def forward(self, images):
        features = self.backbone(images)  # [B, 2048, 7, 7]
        features = self.pool(features).flatten(1)  # [B, 2048]
        prefix = self.projection(features)  # [B, prefix_len * lm_dim]
        return prefix.view(-1, self.config.visual_prefix_length, self.config.lm_hidden_dim)


class FrozenCLIP(nn.Module):
    """Frozen architecture with trainable vision encoder and frozen LLM."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Vision encoder
        print(f"Loading vision encoder: {config.vision_encoder_name}...")
        self.vision_encoder = VisionEncoder(config).to(self.device)
        # FIX: Match old evaluation transform (Bicubic Interpolation)
        from torchvision.transforms import InterpolationMode
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Language model
        print(f"Loading LLM: {config.language_model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            config.language_model_name,
            cache_dir=config.cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model = GPT2LMHeadModel.from_pretrained(
            config.language_model_name,
            cache_dir=config.cache_dir
        )
        self.language_model.to(self.device)
        
        # Freeze language model
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Print parameter counts
        frozen_params = sum(p.numel() for p in self.language_model.parameters())
        trainable_params = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        print(f"Frozen Params: {frozen_params:,} | Trainable Params: {trainable_params:,}")
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        batch_size = images.size(0)
        
        # Encode images
        visual_prefix = self.vision_encoder(images)  # [B, prefix_len, hidden_dim]
        
        # Get text embeddings
        text_embeds = self.language_model.transformer.wte(input_ids)
        
        # Concatenate [visual_prefix, text]
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        visual_attention = torch.ones(
            batch_size, self.config.visual_prefix_length,
            dtype=torch.long,
            device=self.device
        )
        combined_mask = torch.cat([visual_attention, attention_mask], dim=1)
        
        # Forward through LM (DON'T pass labels here!)
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Compute loss manually if labels provided
        loss = None
        if labels is not None:
            # Shift logits: skip visual prefix tokens and align for next-token prediction
            shift_logits = logits[:, self.config.visual_prefix_length:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    # def encode_image(self, images):
    #     """Encode images to feature vectors for evaluation."""
    #     self.vision_encoder.eval()
    #     with torch.no_grad():
    #         # Get features from backbone
    #         features = self.vision_encoder.backbone(images)
    #         features = self.vision_encoder.pool(features).flatten(1)
    #         features = features / features.norm(dim=-1, keepdim=True)
    #     return features

    def encode_image(self, images):
        """Encode images to feature vectors for evaluation."""
        self.vision_encoder.eval()
        with torch.no_grad():
            # CORRECTION: Call the module directly to include Projection Layer
            features = self.vision_encoder(images) 
            
            # CORRECTION: Flatten [Batch, Prefix_Len, Dim] -> [Batch, Features]
            if features.dim() == 3:
                features = features.flatten(start_dim=1)
                
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def encode_text(self, text_tokens):
        """Encode text to feature vectors for evaluation."""
        self.language_model.eval()
        with torch.no_grad():
            # Get GPT-2 outputs with hidden states
            outputs = self.language_model(
                input_ids=text_tokens,
                attention_mask=(text_tokens != self.tokenizer.pad_token_id),
                output_hidden_states=True  # Request hidden states!
            )
            
            # Use last hidden state from hidden_states tuple
            hidden_states = outputs.hidden_states[-1]
            
            # Average pooling over sequence length
            attention_mask = (text_tokens != self.tokenizer.pad_token_id).unsqueeze(-1)
            text_embeds = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            
            # Normalize for similarity comparisons
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds
    
    def eval(self):
        """Set model to evaluation mode."""
        self.vision_encoder.eval()
        self.language_model.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode."""
        self.vision_encoder.train(mode)
        self.language_model.eval()  # Keep LM frozen
        return self
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        self.vision_encoder = self.vision_encoder.to(device)
        self.language_model = self.language_model.to(device)
        return self
    
    def parameters(self):
        """Return trainable parameters only."""
        return self.vision_encoder.parameters()
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.language_model.parameters())
        return {
            'total': trainable + frozen,
            'trainable': trainable,
            'frozen': frozen
        }