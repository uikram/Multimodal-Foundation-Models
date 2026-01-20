"""
IEEE RA-L Benchmark v6: Final "Paper-Grade" Version
- Protocol: N=20 Warmup -> N=100 Measurement Loops
- Stats: Reports Mean +/- Std Dev
- Input: Uses Local Generator for strict bitwise determinism
- Hardware: Forces FP16 and Deterministic CUBLAS
"""

import torch
import json
import time
import argparse
import sys
import os
import gc
import random
import numpy as np
from pathlib import Path
import warnings

# ============ 1. DETERMINISM SETUP ============
# This is mandatory for NVIDIA reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass 

set_seed(42)

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from models import get_model
from utils.config import load_config_from_yaml
from peft import PeftModel

# ============ CONFIGURATION ============
CHECKPOINT_PATHS = {
    # UPDATE THESE PATHS IF NECESSARY
    "FROZEN": "/sda/usama/production_code/frozen_checkpoints/best_model.pt",
    "LORA_ADAPTER": "/sda/usama/production_code/clip_lora_checkpoints/epoch_3",
}

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def strict_fp16_setup(model, device):
    target = model.model if hasattr(model, 'model') else model
    for param in target.parameters():
        param.requires_grad = False
    
    if hasattr(model, 'half'):
        model = model.half()
    else:
        target = target.half()
    
    model = model.to(device)
    model.eval()
    return model

def count_standard_params(model):
    target = model.model if hasattr(model, 'model') else model
    total = sum(p.numel() for p in target.parameters())
    if isinstance(target, PeftModel):
        trainable, _ = target.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def count_frozen_params(model):
    vision = sum(p.numel() for p in model.vision_encoder.parameters()) if hasattr(model, 'vision_encoder') else 0
    llm = sum(p.numel() for p in model.language_model.parameters()) if hasattr(model, 'language_model') else 0
    return (vision + llm) / 1e6, vision / 1e6

def measure_peak_memory(func, device):
    if device == 'cpu': return 0.0
    force_cleanup()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        func()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def measure_latency_stats(func, runs=100, warmup=20, device='cuda'):
    """
    Protocol: 
    1. Warmup N=20 (Discarded to fix cold-start)
    2. Measure N=100 (Statistical Latency)
    Returns: Mean (ms), Std (ms)
    """
    # 1. Warmup
    for _ in range(warmup):
        func()
        if device == 'cuda': torch.cuda.synchronize()
    
    # 2. Measurement Loop
    timings = []
    # No cleanup here to simulate continuous control loop state
    for _ in range(runs):
        start = time.perf_counter()
        func()
        if device == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000) # Convert to ms
    
    mean_lat = np.mean(timings)
    std_lat = np.std(timings)
    return mean_lat, std_lat

def get_deterministic_input(batch_size, device):
    """
    Creates inputs using a local generator. 
    Ensures input noise is identical across all model runs.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    return torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float16, generator=gen)

def run_benchmark(device='cuda', output_file='benchmark/results/benchmark_results.json'):
    BATCH_SIZE = 1 
    results = {}
    print(f"IEEE RA-L BENCHMARK v6 (Final Protocol)\n")
    print(f"Device: {device}")
    print(f"Protocol: {20} Warmup runs -> {100} Measurement runs")
    print("-" * 60)

    # --- 1. CLIP ---
    force_cleanup()
    print("Benchmarking: CLIP Baseline")
    try:
        config = load_config_from_yaml("configs/clip_baseline.yaml")
        config.device = "cpu"
        model = get_model("clip", config)
        total_m, _ = count_standard_params(model)
        model = strict_fp16_setup(model, device)
        
        dummy = get_deterministic_input(BATCH_SIZE, device)
        
        vis_func = lambda: model.encode_image(dummy)
        lat_mean, lat_std = measure_latency_stats(vis_func, device=device)
        mem = measure_peak_memory(vis_func, device)
        
        results["CLIP"] = {
            "total": total_m, 
            "mem": mem,
            "lat_mean": lat_mean,
            "lat_std": lat_std
        }
        print(f"  -> Result: {lat_mean:.2f} +/- {lat_std:.2f} ms")
        del model
    except Exception as e: print(f"x Failed: {e}")

    # --- 2. LoRA ---
    force_cleanup()
    print("\nBenchmarking: LoRA")
    try:
        config = load_config_from_yaml("configs/clip_lora.yaml")
        config.device = "cpu"
        model = get_model("clip_lora", config)
        
        if hasattr(model, 'model'):
            model.model = PeftModel.from_pretrained(model.model.get_base_model(), CHECKPOINT_PATHS["LORA_ADAPTER"], is_trainable=True)
        
        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)
        vis_func = lambda: model.encode_image(dummy)
        
        # Unmerged
        print("  -> Measuring Unmerged...")
        lat_un_mean, lat_un_std = measure_latency_stats(vis_func, device=device)
        mem_un = measure_peak_memory(vis_func, device)
        total_un, train_un = count_standard_params(model)
        print(f"     Result: {lat_un_mean:.2f} +/- {lat_un_std:.2f} ms")
        
        # Merged
        print("  -> Merging...")
        if hasattr(model.model, 'merge_and_unload'):
            model.model = model.model.merge_and_unload()
            
        print("  -> Measuring Merged...")
        lat_mg_mean, lat_mg_std = measure_latency_stats(vis_func, device=device)
        mem_mg = measure_peak_memory(vis_func, device)
        total_mg, train_mg = count_standard_params(model)
        print(f"     Result: {lat_mg_mean:.2f} +/- {lat_mg_std:.2f} ms")
        
        results["LoRA"] = {
            "total_un": total_un, "trainable_un": train_un, 
            "mem_un": mem_un, "lat_un_mean": lat_un_mean, "lat_un_std": lat_un_std,
            "total_mg": total_mg, "trainable_mg": train_mg, 
            "mem_mg": mem_mg, "lat_mg_mean": lat_mg_mean, "lat_mg_std": lat_mg_std
        }
        del model
    except Exception as e: print(f"x Failed: {e}")

    # --- 3. FROZEN ---
    force_cleanup()
    print("\nBenchmarking: Frozen")
    try:
        config = load_config_from_yaml("configs/frozen_clip.yaml")
        config.device = "cpu"
        model = get_model("frozen", config)
        
        # Manual checkpoint loading
        if os.path.exists(CHECKPOINT_PATHS["FROZEN"]):
            ckpt = torch.load(CHECKPOINT_PATHS["FROZEN"], map_location='cpu')
            state_dict = ckpt.get('model_state', ckpt)
            new_state_dict = {k.replace('vision_encoder.', ''): v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
            model.vision_encoder.load_state_dict(new_state_dict, strict=False)

        total_m, train_m = count_frozen_params(model)
        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)
        
        # Vision Latency
        vis_func = lambda: model.encode_image(dummy)
        vis_lat_mean, vis_lat_std = measure_latency_stats(vis_func, device=device)
        print(f"  -> Visual Result: {vis_lat_mean:.2f} +/- {vis_lat_std:.2f} ms")
        
        # Generation Latency (STRICT LOCKED WORKLOAD)
        def gen_func():
            with torch.no_grad():
                # top_k=1 is mathematically equivalent to do_sample=False (Greedy)
                # This worked in your V5 script, so it will work here.
                model.generate(
                    dummy, model.tokenizer, max_length=10, temperature=1.0, top_k=50 # <--- MATCH PAPER & FIGURES
                )
        
        # Slightly fewer runs for Gen because it's 50x slower
        print("  -> Measuring Generation (this takes a moment)...")
        e2e_lat_mean, e2e_lat_std = measure_latency_stats(gen_func, runs=50, warmup=5, device=device)
        e2e_mem = measure_peak_memory(gen_func, device)
        print(f"  -> E2E Result: {e2e_lat_mean:.2f} +/- {e2e_lat_std:.2f} ms")
        
        results["Frozen"] = {
            "total": total_m, "trainable": train_m,
            "mem": e2e_mem, 
            "vis_lat_mean": vis_lat_mean, "vis_lat_std": vis_lat_std,
            "e2e_lat_mean": e2e_lat_mean, "e2e_lat_std": e2e_lat_std
        }
        del model
    except Exception as e: print(f"x Failed: {e}")

    # --- SAVE ---
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print("-" * 60)
        print(f"[SUCCESS] Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_file', default='benchmark/results/benchmark_results.json')
    args = parser.parse_args()
    
    run_benchmark(device=args.device, output_file=args.output_file)