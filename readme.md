# Multimodal Foundation Models: CLIP, CLIP+LoRA, and Frozen

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A unified, research-grade implementation of three multimodal foundation model architectures for vision-language tasks.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Metrics](#metrics)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This repository provides a **unified, production-ready implementation** of three state-of-the-art multimodal foundation models:

1. **CLIP Baseline** - OpenAI's pretrained vision-language model
2. **CLIP + LoRA** - Parameter-efficient fine-tuning with Low-Rank Adaptation
3. **Frozen** - Trainable vision encoder with frozen large language model

### Why This Repository?

- ✅ **Single Unified Codebase** - No separate folders, consistent APIs
- ✅ **Comprehensive Metrics** - Track parameters, memory, latency, accuracy automatically
- ✅ **Flexible Execution** - Train/evaluate one or multiple models simultaneously
- ✅ **Research-Ready** - Reproducible experiments with proper documentation
- ✅ **Production-Quality** - Clean code, extensive error handling, modular design

---

## ✨ Features

### Training & Evaluation
- Zero-shot classification on 5 benchmark datasets
- Linear probe evaluation with logistic regression
- Few-shot learning (1, 2, 4, 8, 16-shot)
- Automatic checkpoint saving and resuming
- Mixed-precision training (FP16)

### Metrics Tracking
- **Parameters**: Total, trainable, frozen counts
- **Memory**: Peak GPU/CPU usage during training and inference
- **Latency**: Per-sample inference time with statistics
- **Performance**: Accuracy, loss, classification reports
- **Training Time**: Complete wall-clock time tracking

### Datasets
- CIFAR-100 (100 classes, 60K images)
- Food-101 (101 classes, 101K images)
- Flowers-102 (102 classes, 8K images)
- DTD (47 textures, 5.6K images)
- EuroSAT (10 land use classes, 27K images)
- Conceptual Captions (3.3M image-text pairs)

---

## 🤖 Models

### 1. CLIP Baseline

**Architecture**: Vision Transformer (ViT-B/32) + Text Transformer

**Parameters**: ~151M (all pretrained)

**Use Cases**:
- Zero-shot image classification
- Cross-modal retrieval
- Transfer learning via linear probes

**Training**: Not required (uses pretrained OpenAI weights)

---

### 2. CLIP + LoRA

**Architecture**: CLIP with Low-Rank Adaptation adapters

**Parameters**:
- Total: ~151M
- Trainable: ~1.2M (0.8%)
- Frozen: ~150M

**LoRA Configuration**:
- Rank (r): 16
- Alpha: 32
- Target modules: `q_proj`, `v_proj`
- Dropout: 0.05

**Use Cases**:
- Domain adaptation with limited compute
- Fine-tuning on custom image-text datasets
- Efficient transfer learning

**Training**: Required (Conceptual Captions or custom data)

---

### 3. Frozen

**Architecture**: Trainable Vision Encoder (ResNet-50) + Frozen LLM (GPT-2 Large)

**Parameters**:
- Total: ~800M
- Trainable: ~25M (3.1%, vision encoder only)
- Frozen: ~775M (GPT-2 Large)

**Design**:
- Vision features projected to 2 visual tokens
- Visual tokens prepended to text sequence
- Language model frozen during training

**Use Cases**:
- Image captioning
- Visual question answering
- Vision-language understanding tasks

**Training**: Required (Conceptual Captions or custom data)

---

## 🔧 Installation

### Prerequisites
