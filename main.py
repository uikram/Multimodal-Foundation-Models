"""
Main entry point for training and evaluating multimodal foundation models.

Usage:
    python main.py --models clip --mode train
    python main.py --models clip clip_lora frozen --mode evaluate
    python main.py --models all --mode full_pipeline
"""

import argparse
import warnings
import torch
from pathlib import Path
from utils.plotting import plot_training_curves
# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="The channel dimension is ambiguous")
warnings.filterwarnings("ignore", category=UserWarning)

from models import get_model
from utils.config import CLIPConfig, CLIPLoRAConfig, FrozenConfig, load_config_from_yaml
from utils.helpers import seed_everything
from training.train import ModelTrainer
from evaluation.evaluate import ModelEvaluator
from evaluation.metrics import MetricsTracker


# Model configurations mapping
CONFIG_MAP = {
    'clip': CLIPConfig,
    'clip_lora': CLIPLoRAConfig,
    'frozen': FrozenConfig
}

# Benchmark datasets for evaluation
BENCHMARK_DATASETS = ['cifar100', 'food101', 'flowers102', 'dtd', 'eurosat']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multimodal Foundation Models Training & Evaluation"
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['clip', 'clip_lora', 'frozen', 'all'],
        default=['clip'],
        help='Models to train/evaluate (default: clip)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'full_pipeline'],
        default='full_pipeline',
        help='Execution mode (default: full_pipeline)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config YAML file'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=BENCHMARK_DATASETS,
        help='Datasets to evaluate on (default: all)'
    )

    return parser.parse_args()


def get_models_list(models_arg):
    """Convert models argument to list."""
    if 'all' in models_arg:
        return ['clip', 'clip_lora', 'frozen']
    return models_arg


def load_model_checkpoint(model, model_name, config):
    """Load trained checkpoint for evaluation."""
    import torch  
    from pathlib import Path
    if model_name == 'clip_lora':
        # LoRA saves adapter checkpoints per epoch (epoch_1, epoch_2, epoch_3)
        checkpoint_dir = Path(config.output_dir)
        
        # Find the latest epoch checkpoint
        epoch_dirs = sorted(checkpoint_dir.glob("epoch_*"))
        
        if not epoch_dirs:
            print(f"⚠️  No checkpoints found in {checkpoint_dir}")
            return model
        
        latest_checkpoint = epoch_dirs[-1]  # Get last epoch (epoch_3)
        print(f"✓ Loading CLIP-LoRA checkpoint from: {latest_checkpoint}")
        
        # FIX: Load adapter weights directly onto existing PeftModel
        # model.model is the PeftModel inside CLIPLoRA wrapper
        from peft import set_peft_model_state_dict
        import torch
        
        # Check for both .safetensors and .bin formats
        adapter_file_safetensors = latest_checkpoint / "adapter_model.safetensors"
        adapter_file_bin = latest_checkpoint / "adapter_model.bin"
        
        # Load adapter weights (prefer .safetensors if both exist)
        if adapter_file_safetensors.exists():
            print("✓ Loading from .safetensors format")
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_file_safetensors))
        elif adapter_file_bin.exists():
            print("✓ Loading from .bin format")
            adapter_weights = torch.load(adapter_file_bin, map_location=config.device)
        else:
            print(f"❌ No adapter_model file found in {latest_checkpoint}")
            print(f"   Expected: adapter_model.safetensors or adapter_model.bin")
            return model
        
        # Apply to existing model
        set_peft_model_state_dict(model.model, adapter_weights)
        print("✓ Loaded LoRA adapter weights")
        
    elif model_name == 'frozen':
        checkpoint_path = config.checkpoint_dir / "best_model.pt"
        
        if not checkpoint_path.exists():
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
            return model
        
        print(f"✓ Loading Frozen checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # Direct load - architectures match perfectly now!
        model.vision_encoder.load_state_dict(checkpoint['model_state'])
        print(f"✓ Loaded vision encoder with {len(checkpoint['model_state'])} parameters")

    
    return model

def initialize_model(model_name: str, config_path: str = None):
    """Initialize model with configuration."""
    print(f"\n{'='*60}")
    print(f"Initializing {model_name.upper()}")
    print(f"{'='*60}")
    
    # AUTO-LOAD YAML if not provided
    if config_path is None:
        config_path = f"configs/{model_name}.yaml"
        if model_name == 'clip':
            config_path = "configs/clip_baseline.yaml"
        elif model_name == 'clip_lora':
            config_path = "configs/clip_lora.yaml"
        elif model_name == 'frozen':
            config_path = "configs/frozen_clip.yaml"
    
    # Load configuration (auto-loads YAML or uses defaults)
    config = load_config_from_yaml(config_path, model_name)
    
    # Initialize model
    model = get_model(model_name, config)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        model_name=model_name.upper(),
        results_dir=Path(config.results_dir)  
    )
    
    # Track parameters
    metrics_tracker.track_parameters(model)
    
    return model, config, metrics_tracker



def train_model(model, config, metrics_tracker):
    """Train a single model."""
    print(f"\n{'-'*60}")
    print("Starting Training")
    print(f"{'-'*60}")
    
    if hasattr(config, 'vision_encoder_name'):
        from training.trainer_frozen import FrozenTrainer
        trainer = FrozenTrainer(model, config, metrics_tracker)
    else:
        from training.train import ModelTrainer
        trainer = ModelTrainer(model, config, metrics_tracker)
    
    trainer.train()
    print("\nTraining completed successfully!")
    metrics_tracker.save_metrics()
    
    # Generate training curves immediately after training
    try:
        from utils.plotting import plot_training_curves
        model_name = metrics_tracker.model_name  # Already uppercase
        plot_training_curves(model_name, config.results_dir, Path('plots'))
    except Exception as e:
        print(f"⚠️  Training curve generation failed: {e}")

def evaluate_model(model, config, metrics_tracker, datasets_to_eval=None):
    """
    Evaluate a single model on benchmark datasets.
    Runs zero-shot, linear probe, and few-shot evaluation.
    """
    print(f"\n{'-'*60}")
    print("Starting Evaluation")
    print(f"{'-'*60}")

    # Use all datasets if not specified
    if datasets_to_eval is None:
        datasets_to_eval = BENCHMARK_DATASETS

    # Import required modules
    import open_clip
    from torch.utils.data import DataLoader
    from datasets.benchmark_datasets import BenchmarkDatasets
    from utils.templates import get_templates

    # Move model to device and set to eval mode
    model.to(config.device)
    model.eval()

    # Initialize evaluator
    evaluator = ModelEvaluator(model, config, metrics_tracker)

    # Storage for all results
    all_results = {
        'zero_shot': {},
        'linear_probe': {},
        'few_shot': {}
    }

    # Get preprocessing transform
    if hasattr(model, 'preprocess'):
        transform = model.preprocess
    else:
        # Default CLIP preprocessing
        _, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name if hasattr(config, 'model_name') else 'ViT-B-32',
            pretrained=config.pretrained_tag if hasattr(config, 'pretrained_tag') else 'openai'
        )
        transform = preprocess

    # Evaluate on each dataset
    for dataset_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name.upper()}")
        print(f"{'='*60}")

        try:
            # Get class names and templates
            classnames = get_classnames(dataset_name)
            templates = get_templates(dataset_name)

            # Load datasets
            print(f"Loading {dataset_name} datasets...")
            train_dataset = load_benchmark_dataset(dataset_name, 'train', transform, config)  # ← Pass config
            test_dataset = load_benchmark_dataset(dataset_name, 'test', transform, config)  
                        
            # Zero-Shot Evaluation (only for CLIP models)
            if hasattr(model, 'encode_image') and hasattr(model, 'encode_text') and 'frozen' not in config.__class__.__name__.lower():
                print(f"\n[1/3] Zero-Shot Evaluation")
                try:
                    text_classifier = create_text_classifier(model, classnames, templates, config.device)
                    zs_acc, zs_samples = evaluator.zero_shot_evaluation(test_dataset, text_classifier)
                    all_results['zero_shot'][dataset_name] = {'accuracy': zs_acc, 'num_samples': zs_samples}
                    print(f"   {dataset_name} Zero-Shot: {zs_acc:.2f}%")
                    metrics_tracker.track_performance(accuracy=zs_acc, loss=0.0)
                except Exception as e:
                    print(f"✗ Zero-shot failed: {e}")
            else:
                print("⚠️  Zero-Shot Evaluation skipped (not applicable for Frozen architecture)")


            # 2. Linear Probe Evaluation
            print(f"\n[2/3] Linear Probe Evaluation")
            try:
                lp_acc, lp_samples = evaluator.linear_probe_evaluation(train_dataset, test_dataset)
                all_results['linear_probe'][dataset_name] = {
                    'accuracy': lp_acc,
                    'num_samples': lp_samples
                }
                print(f"✓ {dataset_name} Linear Probe: {lp_acc:.2f}%")

                # FIX 3: Track performance metrics
                metrics_tracker.track_performance(accuracy=lp_acc, loss=0.0)

            except Exception as e:
                print(f"✗ Linear probe failed: {e}")

            # 3. Few-Shot Evaluation
            print(f"\n[3/3] Few-Shot Evaluation")
            try:
                k_shots = config.k_shots if hasattr(config, 'k_shots') else [1, 2, 4, 8, 16]
                fs_results = evaluator.few_shot_evaluation(train_dataset, test_dataset, k_shots)
                all_results['few_shot'][dataset_name] = fs_results
                print(f"✓ {dataset_name} Few-Shot: {fs_results}")
            except Exception as e:
                print(f"✗ Few-shot failed: {e}")

        except Exception as e:
            print(f"✗ Failed to evaluate {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # === FIX 2: Track detailed latency metrics (once, after all datasets) ===
    print(f"\n{'='*60}")
    print("Measuring Detailed Inference Latency")
    print(f"{'='*60}")
    try:
        # Use last test_dataset for latency measurement
        latency_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        metrics_tracker.track_inference_latency(model, latency_loader, num_samples=100)
        print("✓ Latency metrics recorded")
    except Exception as e:
        print(f"⚠️  Latency tracking failed: {e}")

    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = metrics_tracker.results_dir / f"evaluation_results_{timestamp}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n✓ Saved evaluation results to {results_file}")
    except Exception as e:
        print(f"⚠️  Failed to save evaluation results: {e}")

    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)
    return all_results


def get_classnames(dataset_name):
    """Get class names for a dataset - ONLY from templates.py"""
    from utils.templates import (
        CIFAR100_CLASS_NAMES, FOOD101_CLASS_NAMES, FLOWERS102_CLASS_NAMES,
        DESCRIBABLETEXTURES_CLASS_NAMES, EUROSAT_CLASS_NAMES
    )
    
    dataset_name = dataset_name.lower()
    
    classnames_map = {
        'cifar100': CIFAR100_CLASS_NAMES,
        'food101': FOOD101_CLASS_NAMES,
        'flowers102': FLOWERS102_CLASS_NAMES,
        'dtd': DESCRIBABLETEXTURES_CLASS_NAMES,
        'eurosat': EUROSAT_CLASS_NAMES,
    }
    
    return classnames_map.get(dataset_name)


def load_benchmark_dataset(dataset_name, split, transform, config):
    """Load a specific benchmark dataset."""
    from datasets.benchmark_datasets import BenchmarkDatasets
    
    dataset_name_lower = dataset_name.lower()
    
    # FIX: Use cache_dir instead of data_root for benchmarks
    cache_dir = config.cache_dir
    
    if dataset_name_lower == 'cifar100':
        return BenchmarkDatasets.get_cifar100(cache_dir, transform, split)
    elif dataset_name_lower == 'food101':
        return BenchmarkDatasets.get_food101(cache_dir, transform, split)
    elif dataset_name_lower == 'flowers102':
        return BenchmarkDatasets.get_flowers102(cache_dir, transform, split)
    elif dataset_name_lower == 'dtd':
        return BenchmarkDatasets.get_dtd(cache_dir, transform, split)
    elif dataset_name_lower == 'eurosat':
        return BenchmarkDatasets.get_eurosat(cache_dir, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_text_classifier(model, classnames, templates, device):
    """Create text classifier by encoding class names with templates."""
    import open_clip
    
    text_features = []
    
    with torch.no_grad():
        for classname in classnames:
            # Generate prompts from templates
            texts = [template.format(classname) for template in templates]
            
            # Tokenize
            texts_tokenized = open_clip.tokenize(texts).to(device)
            
            # Encode - returns [num_templates, embedding_dim]
            class_embeddings = model.encode_text(texts_tokenized)
            
            # Normalize each embedding
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            
            # Average over templates - returns [embedding_dim]
            class_embedding = class_embeddings.mean(dim=0)
            
            # Normalize the averaged embedding
            class_embedding = class_embedding / class_embedding.norm()
            
            text_features.append(class_embedding)
    
    # Stack all class embeddings: [num_classes, embedding_dim]
    text_classifier = torch.stack(text_features, dim=0).to(device)
    
    # Transpose for matrix multiplication: [embedding_dim, num_classes]
    text_classifier = text_classifier.T
    
    # print(f"✓ Text classifier shape: {text_classifier.shape}")  # Debug
    
    return text_classifier




def run_full_pipeline(model_name: str, config_path: str = None, datasets_to_eval=None):
    """Run complete training and evaluation pipeline."""
    # Initialize
    model, config, metrics_tracker = initialize_model(model_name, config_path)
    
    # Train (now saves metrics internally)
    train_model(model, config, metrics_tracker)
    
    # Load best checkpoint for evaluation
    model = load_model_checkpoint(model, model_name, config)
    
    # Evaluate
    evaluate_model(model, config, metrics_tracker, datasets_to_eval)
    
    # Save final evaluation metrics
    metrics_tracker.save_metrics() 
    
    # Print summary
    metrics_tracker.print_summary()


def main():
    """Main execution function."""
    args = parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Get models list
    models = get_models_list(args.models)

    print(f"\n{'='*60}")
    print(f"MULTIMODAL FOUNDATION MODELS - {args.mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Models: {', '.join([m.upper() for m in models])}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Execute for each model
    for model_name in models:
        try:
            if args.mode == 'train':
                model, config, metrics_tracker = initialize_model(model_name, args.config)
                train_model(model, config, metrics_tracker)
                metrics_tracker.save_metrics()

            elif args.mode == 'evaluate':
                model, config, metrics_tracker = initialize_model(model_name, args.config)
                # Load trained checkpoint
                model = load_model_checkpoint(model, model_name, config)
                evaluate_model(model, config, metrics_tracker, args.datasets)
                metrics_tracker.save_metrics()

            elif args.mode == 'full_pipeline':
                run_full_pipeline(model_name, args.config, args.datasets)

        except Exception as e:
            print(f"\n❌ ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison plots if multiple models and not disabled
    if len(models) > 1 and not args.no_plots:
        print(f"\n{'='*60}")
        print("Generating Comparison Plots")
        print(f"{'='*60}")
        try:
            from utils.plotting import generate_comparison_plots
            generate_comparison_plots(models, Path('results_attained'), Path('plots'))
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")

    print(f"\n{'='*60}")
    print("All tasks completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
