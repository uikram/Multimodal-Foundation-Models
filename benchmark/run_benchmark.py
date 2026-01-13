"""
IEEE RA-L Benchmark v5: Final Verified
- Explicitly sums Frozen submodules (Vision + LLM)
- Strict timing (allocations outside loop)
- Validated LoRA Merge/Unmerge comparison
- SAVES RESULTS TO JSON (Fixed)
"""

import torch
import json
import time
import argparse
import sys
import os
import gc 
from pathlib import Path
import warnings

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from models import get_model
from utils.config import load_config_from_yaml
from peft import PeftModel

# ============ CRITICAL SETUP ============
torch.set_float32_matmul_precision('highest') 
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CHECKPOINT_PATHS = {
    "FROZEN": "/sda/usama/production_code/frozen_checkpoints/best_model.pt",
    "LORA_ADAPTER": "/sda/usama/production_code/clip_lora_checkpoints/epoch_3",
}

def force_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

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

def count_frozen_params(model):
    # 1. Vision Encoder
    vision = 0
    if hasattr(model, 'vision_encoder'):
        vision = sum(p.numel() for p in model.vision_encoder.parameters())
    # 2. Language Model
    llm = 0
    if hasattr(model, 'language_model'):
        llm = sum(p.numel() for p in model.language_model.parameters())
    total = vision + llm
    trainable = vision 
    return total / 1e6, trainable / 1e6

def count_standard_params(model):
    target = model.model if hasattr(model, 'model') else model
    total = sum(p.numel() for p in target.parameters())
    if isinstance(target, PeftModel):
        trainable, _ = target.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def measure_peak_memory(func, device):
    if device == 'cpu': return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        func()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def measure_latency(func, runs=50, warmup=10, device='cuda'):
    for _ in range(warmup):
        func()
        if device == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        func()
        if device == 'cuda': torch.cuda.synchronize()
    end = time.perf_counter()
    return ((end - start) / runs) * 1000

# ADDED output_file ARGUMENT HERE
def run_benchmark(device='cuda', output_file='benchmark/results/benchmark_results.json'):
    BATCH_SIZE = 1 
    results = {}
    print(f"IEEE RA-L BENCHMARK v5 (Final Verified)\n")

    # --- 1. CLIP ---
    force_cleanup()
    print("Benchmarking: CLIP Baseline")
    try:
        config = load_config_from_yaml("configs/clip_baseline.yaml")
        config.device = "cpu"
        model = get_model("clip", config)
        total_m, _ = count_standard_params(model)
        model = strict_fp16_setup(model, device)
        dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device, dtype=torch.float16)
        vis_func = lambda: model.encode_image(dummy)
        results["CLIP"] = {
            "total": total_m, "trainable": 0.0,
            "mem": measure_peak_memory(vis_func, device),
            "lat": measure_latency(vis_func, device=device)
        }
        del model
    except Exception as e: print(f"x Failed: {e}")

    # --- 2. LoRA ---
    force_cleanup()
    print("\nBenchmarking: LoRA")
    try:
        config = load_config_from_yaml("configs/clip_lora.yaml")
        config.device = "cpu"
        model = get_model("clip_lora", config)
        adapter_path = CHECKPOINT_PATHS["LORA_ADAPTER"]
        if hasattr(model, 'model'):
            model.model = PeftModel.from_pretrained(model.model.get_base_model(), adapter_path, is_trainable=True)
        
        total_un, train_un = count_standard_params(model)
        model = strict_fp16_setup(model, device)
        dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device, dtype=torch.float16)
        vis_func = lambda: model.encode_image(dummy)
        
        print("  -> Measuring Unmerged...")
        lat_un = measure_latency(vis_func, device=device)
        mem_un = measure_peak_memory(vis_func, device)
        
        print("  -> Merging...")
        if hasattr(model.model, 'merge_and_unload'):
            model.model = model.model.merge_and_unload()
        
        total_mg, train_mg = count_standard_params(model)
        print("  -> Measuring Merged...")
        lat_mg = measure_latency(vis_func, device=device)
        mem_mg = measure_peak_memory(vis_func, device)
        
        results["LoRA"] = {
            "total_un": total_un, "trainable_un": train_un, "mem_un": mem_un, "lat_un": lat_un,
            "total_mg": total_mg, "trainable_mg": train_mg, "mem_mg": mem_mg, "lat_mg": lat_mg
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
        ckpt_path = CHECKPOINT_PATHS["FROZEN"]
        if os.path.exists(ckpt_path):
             checkpoint = torch.load(ckpt_path, map_location='cpu')
             state_dict = checkpoint.get('model_state', checkpoint)
             new_state_dict = {k.replace('vision_encoder.', ''): v for k, v in state_dict.items() if k.startswith('vision_encoder.')}
             new_state_dict.update({k: v for k, v in state_dict.items() if not k.startswith('language_model.') and not k.startswith('vision_encoder.')})
             model.vision_encoder.load_state_dict(new_state_dict, strict=False)

        total_m, train_m = count_frozen_params(model)
        print(f"  -> Params Computed: Total={total_m:.2f}M | Trainable={train_m:.2f}M")
        
        model = strict_fp16_setup(model, device)
        dummy = torch.randn(BATCH_SIZE, 3, 224, 224, device=device, dtype=torch.float16)
        vis_func = lambda: model.encode_image(dummy)
        vis_lat = measure_latency(vis_func, device=device)
        
        def gen_func():
            with torch.no_grad():
                model.generate(dummy, model.tokenizer, max_length=10, top_k=1)
        e2e_lat = measure_latency(gen_func, runs=10, device=device)
        e2e_mem = measure_peak_memory(gen_func, device) 
        
        results["Frozen"] = {
            "total": total_m, "trainable": train_m,
            "mem": e2e_mem, "vis_lat": vis_lat, "e2e_lat": e2e_lat
        }
        del model
    except Exception as e: print(f"x Failed: {e}")

    # ==========================================
    # SAVE JSON RESULTS (THIS IS THE FIX)
    # ==========================================
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Saved results to {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving JSON: {e}")

    # ==========================================
    # LATEX OUTPUT
    # ==========================================
    def get_r(model, key, default="--"):
        try: return f"{results[model][key]:.2f}"
        except: return default

#     print("\n" + "="*60)
#     print(r"""
# \begin{table}[t]
# \centering
# \caption{Efficiency comparison on NVIDIA RTX A5000 (batch size $=1$).}
# \label{tab:efficiency}
# \begin{threeparttable}
# \setlength{\tabcolsep}{5pt}
# \begin{tabular}{lccc}
# \toprule
# \textbf{Metric} & \textbf{CLIP (Baseline)} & \textbf{Frozen (ResNet+LLM)} & \textbf{LoRA-CLIP (Merged $|$ Unmerged)} \\
# \midrule
# """ + f"""Total params (M) & {get_r('CLIP', 'total')} & {get_r('Frozen', 'total')}\\tnote{{a}} & {get_r('LoRA', 'total_mg')} $|$ {get_r('LoRA', 'total_un')} \\\\
# Trainable params (M) & {get_r('CLIP', 'trainable')} & {get_r('Frozen', 'trainable')}\\tnote{{b}} & {get_r('LoRA', 'trainable_mg')} $|$ {get_r('LoRA', 'trainable_un')} \\\\
# Peak allocated GPU memory (MB) & {get_r('CLIP', 'mem')} & {get_r('Frozen', 'mem')}\\tnote{{c}} & {get_r('LoRA', 'mem_mg')} $|$ {get_r('LoRA', 'mem_un')} \\\\
# \\addlinespace
# Vision latency (ms) & {get_r('CLIP', 'lat')} & {get_r('Frozen', 'vis_lat')} & {get_r('LoRA', 'lat_mg')} $|$ {get_r('LoRA', 'lat_un')} \\\\
# End-to-end generation latency (ms) & -- & {get_r('Frozen', 'e2e_lat')}\\tnote{{d}} & -- \\\\
# \\bottomrule
# \\end{{tabular}}
# \\begin{{tablenotes}}[flushleft]
# \\item[a] Total parameters include ResNet-50 vision encoder and GPT2-Large language model.
# \\item[b] Trainable parameters correspond to the vision-side component; the language model remains frozen.
# \\item[c] Peak allocated GPU tensor memory measured via \\texttt{{torch.cuda.max\_memory\_allocated()}}.
# \\item[d] End-to-end generation latency measured using \\texttt{{generate()}} (10 tokens) and includes the full multimodal pipeline.
# \\end{{tablenotes}}
# \\end{{threeparttable}}
# \\end{{table}}
# """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_file', default='benchmark/results/benchmark_results.json')
    args = parser.parse_args()
    
    run_benchmark(device=args.device, output_file=args.output_file)