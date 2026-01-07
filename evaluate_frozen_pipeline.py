"""
Complete Evaluation & Visualization Pipeline for Frozen Models.
1. Evaluates all checkpoints in a folder (Linear Probe + Few-Shot).
2. Saves individual JSON results.
3. Generates comparison plots across all datasets.
"""

import torch
import json
import argparse
import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from collections import defaultdict

# Ensure project root is in path
sys.path.append(os.getcwd())

# Import project modules
try:
    from utils.config import load_config_from_yaml
    from models import get_model
    from datasets.dataloaders import DatasetFactory
    from evaluation.evaluate import ModelEvaluator
except ImportError as e:
    print(f"❌ Project modules not found: {e}")
    print("Make sure you are running this script from the project root directory.")
    sys.exit(1)

# Mock metrics tracker for offline eval
class OfflineMetrics:
    def start_evaluation_timer(self): pass
    def end_evaluation_timer(self): pass
    def start_inference_timer(self): pass
    def end_inference_timer(self): pass
    def track_gpu_memory(self, *args): pass

def get_eval_transforms(image_size=224):
    """Standard transforms for Frozen/CLIP models."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_checkpoint_robust(model, ckpt_path, device):
    """
    Robustly loads weights from various checkpoint formats.
    """
    try:
        print(f"Loading checkpoint: {ckpt_path}")
        
        # FIX 1: Set weights_only=True to fix the FutureWarning
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(ckpt_path, map_location=device)
            
        state_dict = None
        epoch = 0
        step = 0

        # Extract metadata if available
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('global_step', 0) or checkpoint.get('step', 0)

        # --- Detect Checkpoint Structure ---
        
        # Case A: Nested inside 'vision_encoder_state_dict'
        if isinstance(checkpoint, dict) and 'vision_encoder_state_dict' in checkpoint:
            print("  -> Detected 'vision_encoder_state_dict' wrapper.")
            state_dict = checkpoint['vision_encoder_state_dict']

        # Case B: Standard 'model_state_dict'
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("  -> Detected 'model_state_dict' key.")
            state_dict = checkpoint['model_state_dict']

        # Case C: 'model_state' (FIX FOR YOUR ERROR)
        elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            print("  -> Detected 'model_state' key.")
            state_dict = checkpoint['model_state']

        # Case D: Flat dictionary (just weights)
        else:
            print("  -> Detected flat weight dictionary.")
            state_dict = checkpoint

        # --- Load the Weights with Smart Prefix Handling ---
        if state_dict:
            # Check the first key to see if it has the "vision_encoder." prefix
            first_key = next(iter(state_dict.keys()))
            
            if first_key.startswith("vision_encoder."):
                print("  -> Keys are prefixed. Loading into Full Model.")
                msg = model.load_state_dict(state_dict, strict=False)
            else:
                print("  -> Keys are NOT prefixed. Loading into model.vision_encoder...")
                if hasattr(model, 'vision_encoder'):
                    msg = model.vision_encoder.load_state_dict(state_dict, strict=False)
                else:
                    print("  ❌ Error: Keys lack prefix but model has no 'vision_encoder' attribute.")
                    return None, None
            
            print(f"  -> Load Status: {msg}")
        else:
            print("  ❌ Error: Could not extract state_dict.")
            return None, None

        return epoch, step

    except Exception as e:
        print(f"❌ Error loading {ckpt_path}: {e}")
        return None, None

def visualize_results(output_dir):
    """
    Parses all metrics_*.json files in output_dir and generates comparison plots.
    """
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    json_files = sorted(list(Path(output_dir).glob("metrics_*.json")))
    if not json_files:
        print("No results found to visualize.")
        return

    # Data structure: data[dataset][metric] = [(step, value), ...]
    plot_data = defaultdict(lambda: defaultdict(list))
    
    print(f"Found {len(json_files)} result files. Parsing...")
    
    for jf in json_files:
        with open(jf, 'r') as f:
            res = json.load(f)
            
        step = res.get('step', 0)
        if step == 0 and 'epoch' in res:
             step = res['epoch'] 

        for ds_name, metrics in res['datasets'].items():
            # Linear Probe
            if 'linear_probe' in metrics and 'accuracy' in metrics['linear_probe']:
                acc = metrics['linear_probe']['accuracy']
                plot_data[ds_name]['Linear Probe'].append((step, acc))
            
            # Few Shot (Mean of all k-shots)
            if 'few_shot' in metrics and isinstance(metrics['few_shot'], dict):
                accuracies = []
                for k, k_res in metrics['few_shot'].items():
                    if isinstance(k_res, dict) and 'accuracy_mean' in k_res:
                        accuracies.append(k_res['accuracy_mean'])
                
                if accuracies:
                    mean_acc = sum(accuracies) / len(accuracies)
                    plot_data[ds_name]['Few-Shot (Avg)'].append((step, mean_acc))

    # Generate Plots
    plt.style.use('ggplot')
    
    # 1. Linear Probe Plot
    plt.figure(figsize=(10, 6))
    for ds_name, metrics in plot_data.items():
        if 'Linear Probe' in metrics:
            points = sorted(metrics['Linear Probe'], key=lambda x: x[0])
            if not points: continue
            steps, accs = zip(*points)
            plt.plot(steps, accs, marker='o', label=ds_name, linewidth=2)
    
    plt.title('Linear Probe Accuracy vs Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'viz_linear_probe_trends.png', dpi=300)
    print(f"✓ Saved Linear Probe plot to {output_dir}/viz_linear_probe_trends.png")
    plt.close()

    # 2. Few-Shot Plot
    plt.figure(figsize=(10, 6))
    for ds_name, metrics in plot_data.items():
        if 'Few-Shot (Avg)' in metrics:
            points = sorted(metrics['Few-Shot (Avg)'], key=lambda x: x[0])
            if not points: continue
            steps, accs = zip(*points)
            plt.plot(steps, accs, marker='s', linestyle='--', label=ds_name, linewidth=2)
    
    plt.title('Average Few-Shot Accuracy vs Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Avg Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'viz_few_shot_trends.png', dpi=300)
    print(f"✓ Saved Few-Shot plot to {output_dir}/viz_few_shot_trends.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate & Visualize Frozen Model Checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Folder containing .pt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--config", type=str, default="configs/frozen_clip.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Setup
    config = load_config_from_yaml(args.config, 'frozen')
    config.device = args.device
    config.batch_size = 32
    config.num_workers = 4
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize Model
    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    try:
        model = get_model('frozen', config)
        model.to(config.device)
        model.eval()
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 3. Prepare Datasets
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    transform = get_eval_transforms(config.image_size)
    
    datasets = DatasetFactory.get_linear_probe_datasets(transform, config.cache_dir)
    print(f"✓ Ready to evaluate on: {list(datasets.keys())}")

    # 4. Find Checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print(f"❌ Checkpoint directory not found: {ckpt_dir}")
        return

    checkpoints = sorted(list(ckpt_dir.glob("*.pt")))
    if not checkpoints:
        print(f"❌ No .pt files found in {ckpt_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints.")

    # 5. Evaluation Loop
    evaluator = ModelEvaluator(model, config, OfflineMetrics())
    
    for ckpt_path in checkpoints:
        json_name = f"metrics_{ckpt_path.stem}.json"
        save_path = out_dir / json_name
        
        if save_path.exists():
            print(f"\nSkipping {ckpt_path.name} (Result already exists)")
            continue

        print(f"\nEvaluating: {ckpt_path.name}")
        
        # Load Weights using ROBUST LOADER
        epoch, step = load_checkpoint_robust(model, ckpt_path, config.device)
        
        if epoch is None and step is None:
             pass

        results = {
            "checkpoint": ckpt_path.name,
            "epoch": epoch,
            "step": step,
            "datasets": {}
        }
        
        # Run Benchmarks
        for ds_name, splits in datasets.items():
            print(f"  > {ds_name}...")
            ds_results = {}
            
            # Linear Probe
            try:
                acc, n = evaluator.linear_probe_evaluation(splits['train'], splits['test'])
                ds_results["linear_probe"] = {"accuracy": acc, "num_samples": n}
            except Exception as e:
                ds_results["linear_probe"] = {"error": str(e)}

            # Few Shot
            try:
                fs = evaluator.few_shot_evaluation(
                    splits['train'], splits['test'], 
                    k_shots=[1, 2, 4, 8, 16], # Added 16 as requested
                    num_trials=1 
                )
                ds_results["few_shot"] = fs
            except Exception as e:
                ds_results["few_shot"] = {"error": str(e)}
            
            results["datasets"][ds_name] = ds_results
        
        # Save Result
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"  ✓ Saved to {save_path.name}")

    # 6. Visualization Phase
    visualize_results(args.output_dir)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE ✅")
    print("="*50)

if __name__ == "__main__":
    main()