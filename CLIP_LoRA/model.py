import torch
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

def get_model_and_processor(config):
    """
    Loads CLIP model and applies LoRA.
    """
    print(f"Loading {config.model_id}...")
    
    # 1. Load Processor
    processor = CLIPProcessor.from_pretrained(
        config.model_id, 
        cache_dir=config.cache_dir
    )
    
    # 2. Load Base Model
    base_model = CLIPModel.from_pretrained(
        config.model_id, 
        cache_dir=config.cache_dir
    )
    
    # 3. Configure LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none"
    )
    
    # 4. Inject LoRA adapters
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(config.device)
    
    return model, processor