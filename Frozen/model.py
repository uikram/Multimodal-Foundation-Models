import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import GPT2LMHeadModel, GPTNeoForCausalLM
from typing import Optional, Tuple, List

from config import FrozenConfig

class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50 (proxy for NF-ResNet-50)."""
    
    def __init__(self, config: FrozenConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50; remove classification head
        self.backbone = timm.create_model(
            config.vision_encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool='' 
        )
        
        # Global average pooling and Projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(
            config.vision_hidden_dim,
            config.visual_prefix_length * config.lm_hidden_dim
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)            # [B, 2048, 7, 7]
        features = self.pool(features).flatten(1)   # [B, 2048]
        prefix = self.projection(features)          # [B, prefix_len * lm_dim]
        
        return prefix.view(
            -1, 
            self.config.visual_prefix_length, 
            self.config.lm_hidden_dim
        )

class FrozenModel(nn.Module):
    """
    Frozen Architecture: Trainable Vision Encoder + Frozen LLM.
    """
    
    def __init__(self, config: FrozenConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config)
        
        print(f"Loading LLM: {config.language_model_name}...")
        if "gpt-neo" in config.language_model_name.lower():
            self.language_model = GPTNeoForCausalLM.from_pretrained(config.language_model_name)
        else:
            self.language_model = GPT2LMHeadModel.from_pretrained(config.language_model_name)
        
        # Freeze the Language Model
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        self._log_params()
        
    def _log_params(self):
        total_frozen = sum(p.numel() for p in self.language_model.parameters())
        total_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        print(f"Model Initialized. Frozen Params: {total_frozen:,} | Trainable Params: {total_trainable:,}")

    def forward(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size = images.size(0)
        
        # 1. Embed Images and Text
        visual_prefix = self.vision_encoder(images)
        text_embeds = self.language_model.transformer.wte(input_ids)
        
        # 2. Concatenate: [Visual Prefix, Text]
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        
        # 3. Handle Attention Masks
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        prefix_mask = torch.ones(
            (batch_size, self.config.visual_prefix_length),
            dtype=torch.long,
            device=images.device
        )
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 4. Forward Pass (LLM)
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # 5. Compute Loss (if labels provided)
        loss = None
        if labels is not None:
            # Shift logits so token t predicts t+1
            shift_logits = logits[:, self.config.visual_prefix_length:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss

    def generate(self, images: torch.Tensor, tokenizer, max_length: int = 50, temperature: float = 1.0) -> List[str]:
        """Simple greedy/sampling generation for inference."""
        self.eval()
        with torch.no_grad():
            visual_prefix = self.vision_encoder(images)
            input_ids = torch.full(
                (images.size(0), 1),
                tokenizer.bos_token_id or tokenizer.eos_token_id,
                dtype=torch.long, 
                device=images.device
            )
            
            for _ in range(max_length):
                text_embeds = self.language_model.transformer.wte(input_ids)
                combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
                
                outputs = self.language_model(inputs_embeds=combined_embeds)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if (next_token == tokenizer.eos_token_id).all():
                    break
                    
            return tokenizer.batch_decode(input_ids, skip_special_tokens=True)