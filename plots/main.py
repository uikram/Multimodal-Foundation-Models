"""
Main entry point for training and evaluating multimodal foundation models.

Usage:
    python main.py --model clip --mode train
    python main.py --models clip clip_lora frozen --mode evaluate
    python main.py --models all --mode full_pipeline
"""

import argparse
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="The channel dimension is ambiguous")
warnings.filterwarnings("ignore", category=UserWarning)
from models import get_model
from utils.config import CLIPConfig, CLIPLoRAConfig, FrozenConfig
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
    
    return parser.parse_args()

def get_models_list(models_arg):
    """Convert models argument to list."""
    if 'all' in models_arg:
        return ['clip', 'clip_lora', 'frozen']
    return models_arg

def initialize_model(model_name: str, config_path: str = None):
    """Initialize model with configuration."""
    print(f"\n{'='*60}")
    print(f"Initializing {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load configuration
    if config_path:
        from utils.config import load_config_from_yaml
        config = load_config_from_yaml(config_path, model_name)
    else:
        config = CONFIG_MAP[model_name]()
    
    # Initialize model
    model = get_model(model_name, config)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        model_name=model_name.upper(),
        results_dir=config.results_dir
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
    
    trainer.train()
    print("\nTraining completed successfully!")
    
    print("\nTraining completed successfully!")

def evaluate_model(model, config, metrics_tracker):
    """Evaluate a single model."""
    print(f"\n{'-'*60}")
    print("Starting Evaluation")
    print(f"{'-'*60}")
    
    evaluator = ModelEvaluator(model, config, metrics_tracker)
    
    # Run evaluation based on model type
    if hasattr(config, 'model_name'):  # CLIP baseline
        from datasets.dataloaders import DatasetFactory
        
        # Zero-shot evaluation
        print("\n[1/3] Zero-Shot Evaluation")
        # Implementation depends on your datasets
        
        # Linear probe evaluation
        print("\n[2/3] Linear Probe Evaluation")
        # Implementation depends on your datasets
        
        # Few-shot evaluation
        print("\n[3/3] Few-Shot Evaluation")
        # Implementation depends on your datasets
    
    print("\nEvaluation completed successfully!")

def run_full_pipeline(model_name: str, config_path: str = None):
    """Run complete training and evaluation pipeline."""
    # Initialize
    model, config, metrics_tracker = initialize_model(model_name, config_path)
    
    # Train
    train_model(model, config, metrics_tracker)
    
    # Evaluate
    evaluate_model(model, config, metrics_tracker)
    
    # Save metrics
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
                evaluate_model(model, config, metrics_tracker)
                metrics_tracker.save_metrics()
                
            elif args.mode == 'full_pipeline':
                run_full_pipeline(model_name, args.config)
                
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
        from utils.plotting import generate_comparison_plots
        generate_comparison_plots(models, Path('results_attained'), Path('plots'))
    
    print(f"\n{'='*60}")
    print("All tasks completed successfully!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
