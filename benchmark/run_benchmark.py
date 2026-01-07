"""
Comprehensive Model Benchmarking Script.
Measures Parameters, Memory, Latency (Visual/Text/Gen), and Throughput on Dummy Data.
"""

import torch
import json
import time
import argparse
import sys
import os
from pathlib import Path
import warnings
# Add parent directory to path to import 'models' and 'utils'
sys.path.append(str(Path(__file__).parent.parent))
# Suppress specific PEFT warning when reloading adapters on an initialized model
warnings.filterwarnings("ignore", message=".*Already found a `peft_config` attribute.*")
warnings.filterwarnings("ignore", message=".*Already found a peft_config attribute.*")
from models import get_model
from utils.config import load_config_from_yaml


def count_parameters(model):
    """Count total and trainable parameters (Millions)."""
    # 1. Unwrap wrapper classes
    if hasattr(model, 'model') and hasattr(model.model, 'named_parameters'):
        target_model = model.model
    else:
        target_model = model

    # 2. Robust count
    try:
        total = sum(p.numel() for name, p in target_model.named_parameters())
        trainable = sum(p.numel() for name, p in target_model.named_parameters() if p.requires_grad)
    except AttributeError:
        print(f"Warning: Could not find named_parameters on {type(target_model)}")
        total = 0
        trainable = 0
    
    return total / 1e6, trainable / 1e6


def measure_peak_memory(model, func, device):
    """Measure peak GPU memory usage (MB)."""
    if device == 'cpu': return 0.0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        with torch.no_grad():
            func()
    except Exception as e:
        print(f"Memory measurement warning: {e}")
        
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    torch.cuda.empty_cache()
    return peak


def measure_latency_and_throughput(func, batch_size, device='cuda', warmup=10, runs=50):
    """Measure average latency (ms) and throughput (imgs/sec)."""
    # Warmup
    for _ in range(warmup):
        func()
        if device == 'cuda': torch.cuda.synchronize()
        
    # Measure
    start = time.perf_counter()
    for _ in range(runs):
        func()
        if device == 'cuda': torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    avg_latency_ms = (total_time / runs) * 1000
    throughput = (batch_size * runs) / total_time
    
    return avg_latency_ms, throughput


def find_latest_lora_checkpoint(base_dir):
    """Finds the latest epoch folder in the checkpoint directory."""
    path = Path(base_dir)
    if not path.exists(): return None
    
    # Look for folders like 'epoch_1', 'epoch_2'
    epochs = []
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith('epoch_'):
            try:
                num = int(d.name.split('_')[1])
                epochs.append((num, d))
            except: pass
            
    if not epochs: return None
    
    # Sort by epoch number and return the highest
    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0][1]


# --- MAIN BENCHMARK LOGIC ---

def run_benchmark(device='cuda', output_file='benchmark/results/benchmark_results.json'):
    results = {}
    print(f"\n{'='*60}")
    print(f"BENCHMARKING EFFICIENCY METRICS")
    print(f"Device: {device.upper()}")
    print(f"{'='*60}\n")

    # Define scenarios: (key, config_path, pretty_name)
    scenarios = [
        ("clip", "configs/clip_baseline.yaml", "CLIP"),
        ("clip_lora", "configs/clip_lora.yaml", "LoRA"),
        ("frozen", "configs/frozen_clip.yaml", "Frozen"),
    ]

    for model_key, config_path, pretty_name in scenarios:
        print(f"Benchmarking: {pretty_name}...")
        
        try:
            # 1. Load Config
            config = load_config_from_yaml(config_path)
            config.device = device
            
            # --- MODEL LOADING LOGIC ---
            if model_key == "clip_lora":
                # LoRA Loading Logic
                model = get_model(model_key, config)
                ckpt_dir = "clip_lora_checkpoints"
                latest_ckpt = find_latest_lora_checkpoint(ckpt_dir)
                
                if latest_ckpt:
                    print(f"   Found trained checkpoint: {latest_ckpt}")
                    from peft import PeftModel
                    if hasattr(model, 'model') and isinstance(model.model, PeftModel):
                        try:
                            # FIX: Use from_pretrained for robustness
                            model.model = PeftModel.from_pretrained(
                                model.model.get_base_model(),
                                str(latest_ckpt),
                                is_trainable=False
                            )
                            print(f"Successfully loaded adapters from {latest_ckpt}")
                        except Exception as e:
                            print(f"Could not load adapter: {e}")
                else:
                    print(f"No trained checkpoints found in {ckpt_dir}")

            elif model_key == "frozen":
                # Frozen Loading Logic
                model = get_model(model_key, config)
                
                possible_paths = [
                    Path("frozen_checkpoints/best_model.pt"),
                    Path("checkpoints/frozen/best_model.pt"),
                    Path("results_attained/FROZEN/best_model.pt"),
                ]
                
                loaded = False
                for ckpt_path in possible_paths:
                    if ckpt_path.exists():
                        print(f"   Found trained checkpoint: {ckpt_path}")
                        try:
                            # FIX: weights_only=False to allow loading dictionary structure
                            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                            state_dict = checkpoint.get('model_state', checkpoint)
                            
                            # FIX: Handle key prefixes (remove 'vision_encoder.' if loading into submodule)
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith('vision_encoder.'):
                                    new_state_dict[k.replace('vision_encoder.', '')] = v
                                elif not k.startswith('language_model.'):
                                    # Include keys that don't belong to LLM
                                    new_state_dict[k] = v
                            
                            model.vision_encoder.load_state_dict(new_state_dict, strict=False)
                                
                            print(f"Successfully loaded Frozen weights from {ckpt_path}")
                            loaded = True
                            break
                        except Exception as e:
                            print(f"Found checkpoint but failed to load: {e}")
                
                if not loaded:
                    print("WARNING: No 'best_model.pt' found. Benchmarking UNTRAINED Frozen model.")

            else:
                # CLIP Baseline
                model = get_model(model_key, config)

            model = model.to(device)
            model.eval()
            
            # 3. Prepare Dummy Data
            batch_size = 1
            dummy_image = torch.randn(1, 3, 224, 224).to(device)
            
            # 4. Count Parameters
            total_m, train_m = count_parameters(model)
            
            # 5. Define Test Function
            if hasattr(model, 'encode_image'):
                vis_func = lambda: model.encode_image(dummy_image)
            else:
                vis_func = lambda: model(dummy_image)

            # 6. Metrics
            peak_mem = measure_peak_memory(model, vis_func, device)
            vis_lat, vis_through = measure_latency_and_throughput(vis_func, 1, device)
            
            # 7. Reasoning (Frozen Only)
            reasoning_lat = None
            if model_key == "frozen":
                try:
                    def gen_func():
                        with torch.no_grad():
                             if hasattr(model, 'generate'):
                                 _ = model.generate(dummy_image, model.tokenizer, max_length=5)
                    reasoning_lat, _ = measure_latency_and_throughput(gen_func, 1, device, warmup=2, runs=10)
                except Exception as e:
                    print(f"    ⚠️ Reasoning latency skipped: {e}")

            # Store Result
            results[pretty_name] = {
                "model": pretty_name, 
                "total_params_M": round(total_m, 2),
                "trainable_params_M": round(train_m, 2),
                "peak_memory_MB": round(peak_mem, 2),
                "visual_latency_ms": round(vis_lat, 2),
                "reasoning_latency_ms": round(reasoning_lat, 2) if reasoning_lat else None,
                "throughput_imgs_sec": round(vis_through, 2) 
            }
            
            print(f"Done (Vis Latency: {vis_lat:.2f}ms)")
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    # --- SPECIAL: LoRA MERGED ---
    print(f"Benchmarking: LoRA_Merged...")
    try:
        merged_model_path = "clip_lora_checkpoints/merged_model"
        
        if os.path.exists(merged_model_path):
            print(f"   Loading Merged Model from: {merged_model_path}")
            config = load_config_from_yaml("configs/clip_baseline.yaml")
            config.device = device
            
            # FIX: Set BOTH model_name and model_id to ensure CLIPBaseline picks it up
            config.model_name = merged_model_path 
            config.model_id = merged_model_path
            
            model = get_model("clip", config).to(device)
            model.eval()
            
            dummy_image = torch.randn(1, 3, 224, 224).to(device)
            vis_func = lambda: model.encode_image(dummy_image)
            
            total_m, train_m = count_parameters(model)
            peak_mem = measure_peak_memory(model, vis_func, device)
            vis_lat, vis_through = measure_latency_and_throughput(vis_func, 1, device)
            
            results["LoRA_Merged"] = {
                "model": "LoRA-CLIP (Merged)",
                "total_params_M": round(total_m, 2),
                "trainable_params_M": round(total_m, 2),
                "peak_memory_MB": round(peak_mem, 2),
                "visual_latency_ms": round(vis_lat, 2),
                "reasoning_latency_ms": None,
                "throughput_imgs_sec": round(vis_through, 2)
            }
            print(f"Done")
            del model
            torch.cuda.empty_cache()
        else:
            print(f"Failed: Merged model folder not found at {merged_model_path}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # --- SAVE ---
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nBenchmarking Complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    run_benchmark(device=args.device)