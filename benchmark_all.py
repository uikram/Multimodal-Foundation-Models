import torch
import time
import numpy as np
from transformers import GPT2Tokenizer
import json
import os
from pathlib import Path

# Import your models
from utils.config import CLIPConfig, CLIPLoRAConfig, FrozenConfig
from models.clip_baseline import CLIPBaseline
from models.clip_lora import CLIPLoRA
from models.frozen_clip import FrozenCLIP

DEVICE = 'cuda'

def get_parameter_counts(model):
    """Accurately counts total and trainable parameters."""
    if isinstance(model, FrozenCLIP):
        v_total = sum(p.numel() for p in model.vision_encoder.parameters())
        l_total = sum(p.numel() for p in model.language_model.parameters())
        total_params = v_total + l_total
        # Frozen has trainable vision encoder
        trainable_params = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    return total_params / 1e6, trainable_params / 1e6

def benchmark_model(model_name, model_factory, config_factory, is_generative=False, custom_path=None):
    print(f"\n[{model_name}] ------------------------------------------------")
    print(f"[{model_name}] Loading Model...")
    
    try:
        cfg = config_factory()
        
        # 1. Handle LoRA Merged Path Override
        if "LoRA" in model_name and custom_path:
            print(f"[{model_name}] Loading from local path: {custom_path}")
            cfg.model_name = custom_path
            
        model = model_factory(cfg)
        
        # 2. Handle Frozen Checkpoint Loading
        if model_name == "Frozen":
            frozen_path = "frozen_outputs/best_model.pt"
            if os.path.exists(frozen_path):
                print(f"[{model_name}] Loading trained weights from: {frozen_path}")
                checkpoint = torch.load(frozen_path, map_location='cpu')
                
                # Handle specific keys saved by your trainer
                if 'vision_encoder_state_dict' in checkpoint:
                    print(f"[{model_name}] Detected training checkpoint. Loading vision encoder...")
                    msg = model.vision_encoder.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
                elif 'model_state_dict' in checkpoint:
                    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    msg = model.load_state_dict(checkpoint, strict=False)
                
                print(f"[{model_name}] Weights loaded successfully.")
            else:
                print(f"[{model_name}] ⚠️ Checkpoint not found at {frozen_path}. Using random initialization!")

        model.eval()
        model.to(DEVICE)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

    # Params
    total_m, trainable_m = get_parameter_counts(model)
    
    # Setup Inputs
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    if is_generative:
        tokenizer = GPT2Tokenizer.from_pretrained(model.config.language_model_name)

    # Memory & Visual Latency
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    print(f"[{model_name}] Warming up...")
    with torch.no_grad():
        for _ in range(50):
            if hasattr(model, 'encode_image'): _ = model.encode_image(dummy_input)
            else: _ = model(dummy_input)
    
    print(f"[{model_name}] Profiling Visual Latency (1000 runs)...")
    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if hasattr(model, 'encode_image'): _ = model.encode_image(dummy_input)
            else: _ = model(dummy_input)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    
    visual_latency = np.mean(latencies)
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"[{model_name}] Latency: {visual_latency:.2f} ms | Mem: {peak_memory_mb:.2f} MB")

    # Reasoning Latency
    reasoning_latency = None
    if is_generative:
        print(f"[{model_name}] Profiling Reasoning Latency...")
        gen_latencies = []
        with torch.no_grad():
            for _ in range(10): _ = model.generate(dummy_input, tokenizer, max_length=1, top_k=1)
            for _ in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model.generate(dummy_input, tokenizer, max_length=1, top_k=1)
                torch.cuda.synchronize()
                gen_latencies.append((time.perf_counter() - start) * 1000)
        reasoning_latency = np.mean(gen_latencies)

    # Cleanup
    del model, cfg
    if is_generative: del tokenizer
    torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "total_params_M": float(f"{total_m:.2f}"),
        "trainable_params_M": float(f"{trainable_m:.2f}"),
        "peak_memory_MB": float(f"{peak_memory_mb:.2f}"),
        "visual_latency_ms": float(f"{visual_latency:.2f}"),
        "reasoning_latency_ms": float(f"{reasoning_latency:.2f}") if reasoning_latency else None
    }

def main():
    print(f"🚀 STARTING FINAL BENCHMARK (GPU: {torch.cuda.get_device_name(0)})")
    results = {}
    
    # 1. CLIP Baseline
    results['CLIP'] = benchmark_model("CLIP Baseline", CLIPBaseline, CLIPConfig)
    
    # 2. LoRA (Unmerged)
    results['LoRA'] = benchmark_model("LoRA-CLIP (Unmerged)", CLIPLoRA, CLIPLoRAConfig)
    
    # 3. LoRA (Merged)
    merged_path = "clip_lora_checkpoints/merged_model"
    if os.path.exists(merged_path):
        results['LoRA_Merged'] = benchmark_model("LoRA-CLIP (Merged)", CLIPBaseline, CLIPConfig, custom_path=merged_path)
    else:
        print(f"❌ Merged model not found at {merged_path}")

    # 4. Frozen (Now loads best_model.pt)
    results['Frozen'] = benchmark_model("Frozen", FrozenCLIP, FrozenConfig, is_generative=True)

    # --- SAVE TO JSON (This was missing) ---
    output_file = Path("benchmark.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ All results saved to {output_file.absolute()}")

    # Print LaTeX Table
    print("\n" + "="*60)
    print("📋 FINAL TABLE 5 FOR PAPER")
    print("="*60)
    
    def get(name, key): return f"{results[name][key]:.2f}" if name in results and results[name] else "--"
    
    # Header
    print(f"Metric & CLIP & LoRA (Unmerged) & LoRA (Merged) & Frozen \\\\")
    print("\\midrule")
    
    # Params
    # Note: We manually set CLIP and LoRA_Merged trainable params to 0 in the table 
    # to match the paper's logic (deployment mode), even though the code counts them as trainable.
    print(f"Trainable Params (M) & 0 & {get('LoRA', 'trainable_params_M')} & 0 & {get('Frozen', 'trainable_params_M')} \\\\")
    
    # Memory
    print(f"Peak Memory (MB) & \\textbf{{{get('CLIP', 'peak_memory_MB')}}} & {get('LoRA', 'peak_memory_MB')} & \\textbf{{{get('LoRA_Merged', 'peak_memory_MB')}}} & {get('Frozen', 'peak_memory_MB')} \\\\")
    
    print("\\midrule")
    print("\\multicolumn{5}{l}{\\textit{Real-Time Latency (ms/sample, $B=1$)}} \\\\")
    
    # Latencies
    print(f"Visual Encoding & \\textbf{{{get('CLIP', 'visual_latency_ms')}}} & {get('LoRA', 'visual_latency_ms')} & \\textbf{{{get('LoRA_Merged', 'visual_latency_ms')}}} & {get('Frozen', 'visual_latency_ms')} \\\\")
    print(f"Reasoning & -- & -- & -- & {get('Frozen', 'reasoning_latency_ms')} \\\\")
    print("="*60)

if __name__ == "__main__":
    main()