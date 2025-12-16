"""
Frozen CLIP Model - Frozen LLM with trainable vision encoder.
"""

import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM

class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = timm.create_model(
            config.vision_encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(
            config.vision_hidden_dim,
            config.visual_prefix_length * config.lm_hidden_dim
        )
    
    def forward(self, images):
        features = self.backbone(images)
        features = self.pool(features).flatten(1)
        prefix = self.projection(features)
        return prefix.view(
            -1,
            self.config.visual_prefix_length,
            self.config.lm_hidden_dim
        )

class FrozenCLIP(nn.Module):  # FIXED: Inherit from nn.Module
    """Frozen architecture with trainable vision encoder and frozen LLM."""
    
    def __init__(self, config):
        super().__init__()  # FIXED: Call super().__init__()
        self.config = config
        self.device = config.device
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config).to(self.device)
        
        # Language model
        print(f"Loading LLM: {config.language_model_name}...")
        if "gpt-neo" in config.language_model_name.lower():
            self.language_model = GPTNeoForCausalLM.from_pretrained(
                config.language_model_name,
                cache_dir=config.cache_dir
            )
        else:
            self.language_model = GPT2LMHeadModel.from_pretrained(
                config.language_model_name,
                cache_dir=config.cache_dir
            )
        self.language_model.to(self.device)
        torch.cuda.empty_cache()
        print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # Freeze language model
        for param in self.language_model.parameters():
            param.requires_grad = False

        self._log_params()
    
    def _log_params(self):
        total_frozen = sum(p.numel() for p in self.language_model.parameters())
        total_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        print(f"Frozen Params: {total_frozen:,} | Trainable Params: {total_trainable:,}")
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        batch_size = images.size(0)
        
        # Encode images and text
        visual_prefix = self.vision_encoder(images)
        text_embeds = self.language_model.transformer.wte(input_ids)
        
        # Concatenate
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        prefix_mask = torch.ones(
            (batch_size, self.config.visual_prefix_length),
            dtype=torch.long,
            device=images.device
        )
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward pass
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, self.config.visual_prefix_length:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total_frozen = sum(p.numel() for p in self.language_model.parameters())
        total_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        return {
            'total': total_frozen + total_trainable,
            'trainable': total_trainable
        }
