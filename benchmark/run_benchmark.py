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
    Returns: Dictionary with Mean, Std, p50, p95, p99 (ms)
    """
    # 1. Warmup
    for _ in range(warmup):
        func()
        if device == 'cuda': torch.cuda.synchronize()
    
    # 2. Measurement Loop
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        if device == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000) # Convert to ms
    
    # === MODIFICATION START: Calculate Percentiles ===
    stats = {
        "mean": np.mean(timings),
        "std": np.std(timings),
        "p50": np.percentile(timings, 50),
        "p95": np.percentile(timings, 95),
        "p99": np.percentile(timings, 99)
    }
    # === MODIFICATION END ===
    return stats

def get_deterministic_input(batch_size, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    return torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float16, generator=gen)

def run_benchmark(device='cuda', output_file='benchmark/results/benchmark_results.json'):
    BATCH_SIZE = 1 
    results = {}
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
        model = strict_fp16_setup(model, device)
        
        dummy = get_deterministic_input(BATCH_SIZE, device)
        
        vis_func = lambda: model.encode_image(dummy)
        
        # === MODIFICATION: Unpack dictionary ===
        stats = measure_latency_stats(vis_func, device=device)
        mem = measure_peak_memory(vis_func, device)
        
        results["CLIP"] = {
            "mem": mem,
            **stats # Unpacks mean, std, p50, p95, p99
        }
        print(f"  -> Result: {stats['mean']:.2f} +/- {stats['std']:.2f} ms (p99: {stats['p99']:.2f})")
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
        stats_un = measure_latency_stats(vis_func, device=device)
        mem_un = measure_peak_memory(vis_func, device)
        print(f"     Result: {stats_un['mean']:.2f} ms (p99: {stats_un['p99']:.2f})")
        
        # Merged
        print("  -> Merging...")
        if hasattr(model.model, 'merge_and_unload'):
            model.model = model.model.merge_and_unload()
            
        print("  -> Measuring Merged...")
        stats_mg = measure_latency_stats(vis_func, device=device)
        mem_mg = measure_peak_memory(vis_func, device)
        print(f"     Result: {stats_mg['mean']:.2f} ms (p99: {stats_mg['p99']:.2f})")
        
        results["LoRA"] = {
            "unmerged": {"mem": mem_un, **stats_un},
            "merged": {"mem": mem_mg, **stats_mg}
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
        
        if os.path.exists(CHECKPOINT_PATHS["FROZEN"]):
            ckpt = torch.load(CHECKPOINT_PATHS["FROZEN"], map_location='cpu')
            state_dict = ckpt.get('model_state', ckpt)
            new_state_dict = {k.replace('vision_encoder.', ''): v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
            model.vision_encoder.load_state_dict(new_state_dict, strict=False)

        model = strict_fp16_setup(model, device)
        dummy = get_deterministic_input(BATCH_SIZE, device)
        
        print("  -> Offloading LLM to CPU for strict vision memory measurement...")
        model.language_model.to("cpu") 
        force_cleanup()
        
        # Vision Latency
        vis_func = lambda: model.encode_image(dummy)
        vis_mem = measure_peak_memory(vis_func, device) 
        stats_vis = measure_latency_stats(vis_func, device=device)
        print(f"  -> Visual Result: {stats_vis['mean']:.2f} ms (p99: {stats_vis['p99']:.2f}) | Mem: {vis_mem:.2f} MB")
        
        print("  -> Reloading LLM to GPU for E2E generation...")
        model.language_model.to(device)
        force_cleanup()

        # E2E Latency
        def gen_func():
            with torch.no_grad():
                model.generate(
                    dummy, model.tokenizer, max_length=10, temperature=1.0, top_k=50
                )
        
        print("  -> Measuring Generation...")
        stats_e2e = measure_latency_stats(gen_func, runs=50, warmup=5, device=device)
        e2e_mem = measure_peak_memory(gen_func, device)
        print(f"  -> E2E Result: {stats_e2e['mean']:.2f} ms (p99: {stats_e2e['p99']:.2f})")
        
        results["Frozen"] = {
            "vision": {"mem": vis_mem, **stats_vis},
            "e2e": {"mem": e2e_mem, **stats_e2e}
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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_file', default='benchmark/results/benchmark_results_0_3.json')
    args = parser.parse_args()
    
    run_benchmark(device=args.device, output_file=args.output_file)