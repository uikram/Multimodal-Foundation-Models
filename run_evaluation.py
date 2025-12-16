"""
Complete Evaluation Script
Runs zero-shot, linear probe, and few-shot evaluation on all benchmark datasets.

Usage:
    python run_evaluation.py --model frozen
    python run_evaluation.py --model clip_lora
    python run_evaluation.py --model clip
    python run_evaluation.py --model all
"""

import argparse
import torch
import json
from pathlib import Path
from datasets.benchmark_datasets import get_benchmark_dataset
from evaluation.evaluate import ModelEvaluator
from evaluation.metrics import MetricsTracker
from models import get_model
from utils.config import CLIPConfig, CLIPLoRAConfig, FrozenConfig
from utils.helpers import seed_everything
import warnings
warnings.filterwarnings("ignore")

# Benchmark datasets to evaluate on
BENCHMARK_DATASETS = [
    "cifar100",
    "food101", 
    "flowers102",
    "dtd",
    "eurosat"
]

def load_trained_model(model_name, config):
    """Load a trained model from checkpoint."""
    print(f"\nLoading {model_name.upper()} model...")

    # Initialize model
    model = get_model(model_name, config)

    # Load checkpoint if it exists
    if model_name == "frozen":
        checkpoint_path = Path("frozen_checkpoints/best_model.pt")
    elif model_name == "clip_lora":
        checkpoint_path = Path("clip_lora_checkpoints/best_model.pt")
    else:  # CLIP baseline
        print("Using pretrained CLIP (no checkpoint needed)")
        return model

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("✓ Checkpoint loaded successfully")
    else:
        print(f"⚠️  WARNING: No checkpoint found at {checkpoint_path}")
        print("   Using randomly initialized weights")

    model.to(config.device)
    model.eval()
    return model


def run_zero_shot_evaluation(model, evaluator, dataset_name, config):
    """Run zero-shot evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"Zero-Shot Evaluation: {dataset_name.upper()}")
    print(f"{'='*60}")

    try:
        # Get dataset
        from datasets.benchmark_datasets import get_benchmark_dataset
        dataset, classnames = get_benchmark_dataset(dataset_name, split='test', transform=None)

        # Create text classifier (encode class names)
        import open_clip
        from utils.templates import get_templates

        templates = get_templates(dataset_name)
        text_classifier = []

        with torch.no_grad():
            for classname in classnames:
                # Create prompts from templates
                texts = [template.format(classname) for template in templates]

                # Encode texts
                texts_tokenized = open_clip.tokenize(texts).to(config.device)
                class_embeddings = model.encode_text(texts_tokenized)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                # Average over templates
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                text_classifier.append(class_embedding)

        text_classifier = torch.stack(text_classifier, dim=1).to(config.device)

        # Run evaluation
        accuracy, num_samples = evaluator.zero_shot_evaluation(dataset, text_classifier)

        print(f"\n✓ {dataset_name.upper()} Zero-Shot Accuracy: {accuracy:.2f}%")
        print(f"  Samples evaluated: {num_samples}")

        return {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "num_samples": num_samples
        }

    except Exception as e:
        print(f"\n❌ Failed to evaluate {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_linear_probe_evaluation(model, evaluator, dataset_name, config):
    """Run linear probe evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"Linear Probe Evaluation: {dataset_name.upper()}")
    print(f"{'='*60}")

    try:
        # Get train and test datasets
        from datasets.benchmark_datasets import get_benchmark_dataset
        train_dataset, _ = get_benchmark_dataset(dataset_name, split='train', transform=None)
        test_dataset, _ = get_benchmark_dataset(dataset_name, split='test', transform=None)

        # Run evaluation
        accuracy, num_samples = evaluator.linear_probe_evaluation(train_dataset, test_dataset)

        print(f"\n✓ {dataset_name.upper()} Linear Probe Accuracy: {accuracy:.2f}%")
        print(f"  Samples evaluated: {num_samples}")

        return {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "num_samples": num_samples
        }

    except Exception as e:
        print(f"\n❌ Failed to evaluate {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_few_shot_evaluation(model, evaluator, dataset_name, config, k_shots=[1, 2, 4, 8, 16]):
    """Run few-shot evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"Few-Shot Evaluation: {dataset_name.upper()}")
    print(f"{'='*60}")

    try:
        # Get train and test datasets
        from datasets.benchmark_datasets import get_benchmark_dataset
        train_dataset, _ = get_benchmark_dataset(dataset_name, split='train', transform=None)
        test_dataset, _ = get_benchmark_dataset(dataset_name, split='test', transform=None)

        # Run evaluation
        results = evaluator.few_shot_evaluation(train_dataset, test_dataset, k_shots)

        print(f"\n✓ {dataset_name.upper()} Few-Shot Results:")
        for k, acc in results.items():
            print(f"  {k}: {acc:.2f}%")

        return {
            "dataset": dataset_name,
            "results": results
        }

    except Exception as e:
        print(f"\n❌ Failed to evaluate {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(model_name, config):
    """Complete evaluation pipeline for a single model."""
    print(f"\n{'#'*60}")
    print(f"# EVALUATING {model_name.upper()}")
    print(f"{'#'*60}")

    # Load model
    model = load_trained_model(model_name, config)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        model_name=model_name.upper(),
        results_dir=Path("results_attained")
    )
    metrics_tracker.track_parameters(model)

    # Initialize evaluator
    evaluator = ModelEvaluator(model, config, metrics_tracker)

    # Results storage
    all_results = {
        "model": model_name,
        "zero_shot": {},
        "linear_probe": {},
        "few_shot": {}
    }

    # Run evaluations on each dataset
    for dataset_name in BENCHMARK_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Zero-shot
        print("\n[1/3] Running Zero-Shot Evaluation...")
        zs_result = run_zero_shot_evaluation(model, evaluator, dataset_name, config)
        if zs_result:
            all_results["zero_shot"][dataset_name] = zs_result

        # Linear probe
        print("\n[2/3] Running Linear Probe Evaluation...")
        lp_result = run_linear_probe_evaluation(model, evaluator, dataset_name, config)
        if lp_result:
            all_results["linear_probe"][dataset_name] = lp_result

        # Few-shot
        print("\n[3/3] Running Few-Shot Evaluation...")
        fs_result = run_few_shot_evaluation(model, evaluator, dataset_name, config)
        if fs_result:
            all_results["few_shot"][dataset_name] = fs_result

    # Save results
    results_file = Path("results_attained") / f"{model_name}_evaluation_results.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")

    # Print summary
    print_evaluation_summary(all_results)

    return all_results


def print_evaluation_summary(results):
    """Print a summary of all evaluation results."""
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    model_name = results["model"].upper()

    # Zero-shot summary
    print(f"\n{model_name} - Zero-Shot Accuracy:")
    print(f"{'Dataset':<15} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 40)
    for dataset, res in results["zero_shot"].items():
        if res:
            print(f"{dataset:<15} {res['accuracy']:>6.2f}%   {res['num_samples']:>8}")

    # Linear probe summary
    print(f"\n{model_name} - Linear Probe Accuracy:")
    print(f"{'Dataset':<15} {'Accuracy':<10} {'Samples':<10}")
    print("-" * 40)
    for dataset, res in results["linear_probe"].items():
        if res:
            print(f"{dataset:<15} {res['accuracy']:>6.2f}%   {res['num_samples']:>8}")

    # Few-shot summary
    print(f"\n{model_name} - Few-Shot Accuracy:")
    print(f"{'Dataset':<15} {'1-shot':<10} {'16-shot':<10}")
    print("-" * 40)
    for dataset, res in results["few_shot"].items():
        if res and res.get("results"):
            one_shot = res["results"].get("1-shot", 0)
            sixteen_shot = res["results"].get("16-shot", 0)
            print(f"{dataset:<15} {one_shot:>6.2f}%   {sixteen_shot:>6.2f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "clip_lora", "frozen", "all"],
        required=True,
        help="Model to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    # Get models to evaluate
    if args.model == "all":
        models = ["clip", "clip_lora", "frozen"]
    else:
        models = [args.model]

    # Configs
    CONFIG_MAP = {
        'clip': CLIPConfig,
        'clip_lora': CLIPLoRAConfig,
        'frozen': FrozenConfig
    }

    print(f"\n{'#'*60}")
    print(f"# MULTIMODAL MODELS EVALUATION")
    print(f"# Models: {', '.join([m.upper() for m in models])}")
    print(f"# Device: {args.device}")
    print(f"# Seed: {args.seed}")
    print(f"{'#'*60}\n")

    # Evaluate each model
    all_model_results = {}
    for model_name in models:
        config = CONFIG_MAP[model_name]()
        config.device = args.device

        try:
            results = evaluate_model(model_name, config)
            all_model_results[model_name] = results
        except Exception as e:
            print(f"\n❌ ERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if len(models) > 1:
        print(f"\n{'#'*60}")
        print("# COMPARISON ACROSS ALL MODELS")
        print(f"{'#'*60}\n")

        # TODO: Add comparison plots here
        print("Run plotting script to generate comparison plots")

    print(f"\n✅ Evaluation complete! Check results_attained/ for detailed JSON results\n")


if __name__ == "__main__":
    main()