"""
Visualization utilities for generating comparison plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(model_name: str, results_dir: Path, metric_type: str):
    """Load metrics from JSON file."""
    model_dir = results_dir / model_name
    
    # Find the most recent metric file
    metric_files = list(model_dir.glob(f"{metric_type}_*.json"))
    
    if not metric_files:
        return None
    
    # Sort by timestamp and get most recent
    latest_file = sorted(metric_files)[-1]
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def generate_comparison_plots(models, results_dir: Path, plots_dir: Path):
    """Generate comparison plots for multiple models."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all models
    model_data = {}
    for model in models:
        model_name = model.upper()
        model_data[model_name] = {
            'parameters': load_metrics(model_name, results_dir, 'parameters'),
            'memory': load_metrics(model_name, results_dir, 'memory'),
            'latency': load_metrics(model_name, results_dir, 'latency'),
            'performance': load_metrics(model_name, results_dir, 'performance')
        }
    
    # 1. Parameter Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(model_data.keys())
    total_params = [model_data[m]['parameters']['total_parameters'] / 1e6 
                    for m in model_names if model_data[m]['parameters']]
    trainable_params = [model_data[m]['parameters']['trainable_parameters'] / 1e6 
                       for m in model_names if model_data[m]['parameters']]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, total_params, width, label='Total', alpha=0.8)
    ax.bar(x + width/2, trainable_params, width, label='Trainable', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Parameter Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'parameter_comparison.png', dpi=300)
    plt.close()
    
    # 2. Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    memory_values = []
    for m in model_names:
        if model_data[m]['memory']:
            mem = model_data[m]['memory'].get('training_peak_gpu_mb', 0)
            if mem != 'N/A':
                memory_values.append(float(mem))
            else:
                memory_values.append(0)
        else:
            memory_values.append(0)
    
    ax.bar(model_names, memory_values, alpha=0.8, color='coral')
    ax.set_xlabel('Model')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title('Training Memory Usage Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'memory_usage_comparison.png', dpi=300)
    plt.close()
    
    # 3. Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    latency_values = []
    for m in model_names:
        if model_data[m]['latency']:
            latency_values.append(model_data[m]['latency']['average_ms'])
        else:
            latency_values.append(0)
    
    ax.bar(model_names, latency_values, alpha=0.8, color='skyblue')
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Inference Latency (ms)')
    ax.set_title('Inference Latency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'latency_comparison.png', dpi=300)
    plt.close()
    
    # 4. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_values = []
    for m in model_names:
        if model_data[m]['performance']:
            accuracy_values.append(model_data[m]['performance']['accuracy'])
        else:
            accuracy_values.append(0)
    
    ax.bar(model_names, accuracy_values, alpha=0.8, color='lightgreen')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Comparison plots saved to {plots_dir}")
