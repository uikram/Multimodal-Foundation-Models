# Multimodal Foundation Models: Training & Evaluation Framework

A comprehensive PyTorch framework for training and evaluating vision-language models with CLIP-based architectures. Supports full fine-tuning, LoRA adaptation, and frozen encoder approaches with automated benchmarking, metrics tracking, and visualization.

## 🚀 Quick Start

### Train and Evaluate All Models
```bash
python main.py --models all --mode full_pipeline
```

### Train Single Model
```bash
python main.py --models clip_lora --mode train
```

### Evaluate Pretrained Model
```bash
python main.py --models clip --mode evaluate
```

## 📦 Installation

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch
pip install numpy pandas matplotlib scikit-learn psutil pyyaml tqdm
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import open_clip; print(f'OpenCLIP: {open_clip.__version__}')"
```

## 📋 Requirements

- **Python**: >= 3.8
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **CUDA**: >= 11.3 for GPU acceleration

## 🎯 Usage Examples

### Basic Commands

```bash
# Full pipeline (train + evaluate + visualize)
python main.py --models all --mode full_pipeline

# Train specific model
python main.py --models clip_lora --mode train

# Evaluate on specific datasets
python main.py --models frozen --mode evaluate --datasets cifar100 food101

# Train with custom configuration
python main.py --models clip_lora --mode train --config configs/my_config.yaml

# Use specific GPU
python main.py --models frozen --mode train --gpu 1

# Set random seed
python main.py --models all --mode train --seed 42

# Disable plots
python main.py --models all --mode full_pipeline --no-plots
```

## 🏗️ Project Structure

```
multimodal-foundation-models/
├── main.py                      # Main entry point
├── README.md                    # This file
│
├── configs/                     # Model configurations
│   ├── clip_baseline.yaml
│   ├── clip_lora.yaml
│   └── frozen_clip.yaml
│
├── models/                      # Model architectures
│   ├── clip_baseline.py         # Pretrained CLIP
│   ├── clip_lora.py             # CLIP with LoRA
│   └── frozen_clip.py           # Custom vision + frozen CLIP
│
├── training/                    # Training modules
│   ├── train.py
│   ├── trainer_clip.py
│   ├── trainer_lora.py
│   └── trainer_frozen.py
│
├── evaluation/                  # Evaluation modules
│   ├── evaluate.py
│   └── metrics.py
│
├── datasets/                    # Dataset loaders
│   ├── benchmark_datasets.py
│   └── dataloaders.py
│
├── utils/                       # Utilities
│   ├── config.py
│   ├── helpers.py
│   ├── plotting.py
│   └── templates.py
│
├── results_attained/            # Results (auto-created)
│   ├── CLIP/
│   ├── CLIP_LORA/
│   └── FROZEN/
│
├── plots/                       # Visualizations (auto-created)
└── checkpoints/                 # Model weights (auto-created)
```

## 🔧 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--models` | list | `clip` | Models: `clip`, `clip_lora`, `frozen`, `all` |
| `--mode` | str | `full_pipeline` | Mode: `train`, `evaluate`, `full_pipeline` |
| `--config` | str | None | Path to custom YAML config |
| `--seed` | int | `42` | Random seed |
| `--gpu` | int | `0` | GPU device ID |
| `--datasets` | list | all | Datasets: `cifar100`, `food101`, `flowers102`, `dtd`, `eurosat` |
| `--no-plots` | flag | False | Disable plot generation |

## 🎨 Supported Models

### 1. CLIP Baseline
Pretrained OpenAI CLIP for zero-shot evaluation
```bash
python main.py --models clip --mode evaluate
```

### 2. CLIP-LoRA
Parameter-efficient fine-tuning with Low-Rank Adaptation
```bash
python main.py --models clip_lora --mode full_pipeline
```

### 3. Frozen CLIP
Custom vision encoder with frozen CLIP text encoder
```bash
python main.py --models frozen --mode full_pipeline
```

## 📊 Evaluation Methods

### Zero-Shot Classification
Text-based classification without training data
```bash
python main.py --models clip --mode evaluate
```

### Linear Probe
Train linear classifier on frozen features
```bash
# Automatically included in full_pipeline mode
python main.py --models clip_lora --mode full_pipeline
```

### Few-Shot Learning
Evaluate with limited samples (1, 2, 4, 8, 16-shot)
```bash
# Automatically included in full_pipeline mode
python main.py --models frozen --mode full_pipeline
```

## 🗂️ Benchmark Datasets

Automatically downloaded on first use:

- **CIFAR-100**: 100 classes, 50K train, 10K test
- **Food-101**: 101 food categories, 75K train, 25K test
- **Flowers-102**: 102 flower species, 2K train, 6K test
- **DTD**: 47 texture categories
- **EuroSAT**: 10 satellite image classes

## 📈 Output Files

### Metrics (JSON files in `results_attained/`)
- `parameters_*.json` - Model size and trainable params
- `training_history_*.json` - Loss/accuracy per epoch
- `memory_*.json` - GPU/CPU memory usage
- `latency_*.json` - Inference speed metrics
- `performance_*.json` - Final accuracy and loss
- `classification_report_*.json` - Per-class metrics

### Visualizations (PNG files in `plots/`)
- `{model}_loss_curves.png` - Training/validation loss
- `{model}_accuracy_curves.png` - Training/validation accuracy
- `{model}_combined_curves.png` - Loss + accuracy subplots
- `parameter_comparison.png` - Model size comparison
- `memory_usage_comparison.png` - Memory comparison
- `latency_comparison.png` - Speed comparison
- `accuracy_comparison.png` - Accuracy comparison

## ⚙️ Configuration

### Edit Config Files

```yaml
# configs/clip_lora.yaml
num_epochs: 10              # Number of training epochs
learning_rate: 1e-4         # Learning rate
batch_size: 64              # Batch size
lora_r: 8                   # LoRA rank
lora_alpha: 16              # LoRA scaling
```

### Common Settings

```yaml
# configs/frozen_clip.yaml
vision_encoder_name: "resnet50"    # Vision backbone
num_epochs: 20                      # Training epochs
learning_rate: 1e-3                 # Learning rate
batch_size: 128                     # Batch size
checkpoint_dir: "./frozen_checkpoints"
save_every_n_epochs: 5
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config file
# configs/clip_lora.yaml
batch_size: 32  # Reduced from 64
```

### Dataset Download Issues
```bash
# Manually download datasets
python -c "from torchvision import datasets; datasets.CIFAR100('./data', download=True)"
```

### Slow Training
- Increase `num_workers` in config
- Use SSD storage for datasets
- Enable mixed precision training
- Use `pin_memory=True` in dataloaders

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade torch torchvision open-clip-torch
```

## 📊 Expected Performance

### Training Time (NVIDIA A100)
- CLIP-LoRA (10 epochs): ~2-3 hours
- Frozen CLIP (20 epochs): ~4-6 hours

### Typical Accuracy (CIFAR-100)
- CLIP Baseline (zero-shot): 65-70%
- CLIP-LoRA (fine-tuned): 75-80%
- Frozen CLIP: 70-75%

## 🔄 Workflow Examples

### Complete Training Pipeline
```bash
# 1. Train all models
python main.py --models all --mode train

# 2. Evaluate all models
python main.py --models all --mode evaluate

# Or do both in one command
python main.py --models all --mode full_pipeline
```

### Compare Two Models
```bash
python main.py --models clip_lora frozen --mode full_pipeline
```

### Quick Testing
```bash
# Test on single dataset
python main.py --models clip --mode evaluate --datasets cifar100
```

## 📝 Key Features

✅ **Automated Training** - Epoch tracking, checkpointing, early stopping  
✅ **Comprehensive Metrics** - Loss, accuracy, memory, latency  
✅ **Beautiful Visualizations** - Training curves, comparison plots  
✅ **Multi-Dataset Evaluation** - Zero-shot, linear probe, few-shot  
✅ **Flexible Configuration** - YAML configs, command-line args  
✅ **Reproducible** - Seed control, deterministic training  

## 🎓 Citation

```bibtex
@software{multimodal_foundation_models,
  title={Multimodal Foundation Models: Training and Evaluation Framework},
  year={2025}
}
```

## 📄 License

MIT License

## 🤝 Support

For issues and questions, please open a GitHub issue or contact the maintainers.

---

**Last Updated**: December 16, 2025
