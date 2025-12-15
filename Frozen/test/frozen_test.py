import sys
import os

# --- CRITICAL: Add parent directory to system path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

# Local Imports (using the files we just created in test/)
import config
from data_loader import DatasetFactory
from config import FrozenConfig
from model import FrozenModel  # Imports from root/model.py

# --- Configuration ---
# Update this if your checkpoint name differs
FROZEN_CHECKPOINT_PATH = "../saved_epochs/checkpoint_epoch_2.pt" 

LINEAR_PROBE_FILE = config.RESULTS_FOLDER / "frozen_linear_probe_results.json"
FEW_SHOT_FILE = config.RESULTS_FOLDER / "frozen_few_shot_results.json"

# --- Constants for ResNet (Frozen Model) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Helper Classes & Functions ---

class FrozenFeatureWrapper(torch.nn.Module):
    """
    Wraps FrozenModel to output flat features for evaluation.
    We extract the 'visual prefix' and flatten it to [B, Features].
    """
    def __init__(self, frozen_model):
        super().__init__()
        self.vision_encoder = frozen_model.vision_encoder
        
    def forward(self, images):
        # 1. Get features from the encoder
        features = self.vision_encoder(images)
        
        # 2. Flatten if necessary
        # The encoder likely returns [Batch, Prefix_Len, Hidden_Dim] (3D)
        # We need [Batch, Prefix_Len * Hidden_Dim] (2D) for Logistic Regression
        if features.dim() == 3:
            features = features.flatten(start_dim=1)
            
        return features

def extract_features(dataset, model, device, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, dict):
                images = batch['images'].to(device)
                if 'labels' in batch: labels = batch['labels'].to(device)
                elif 'label' in batch: labels = batch['label'].to(device)
                else: labels = torch.zeros(images.shape[0], device=device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            
            features = model(images)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def train_log_reg(train_features, train_labels, C=config.LOGISTIC_REGRESSION_C):
    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=0, n_jobs=-1)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_classifier(classifier, test_features, test_labels):
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
    return accuracy, len(test_labels)

def save_results(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filepath}")

# --- Main Logic ---

def run_evaluation():
    # reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"--- Setting up Frozen Model ---")
    
    # 1. Initialize Model
    f_config = FrozenConfig() 
    f_config.device = config.DEVICE
    raw_model = FrozenModel(f_config)
    
    # 2. Load Checkpoint
    if os.path.exists(FROZEN_CHECKPOINT_PATH):
        print(f"Loading weights from: {FROZEN_CHECKPOINT_PATH}")
        checkpoint = torch.load(FROZEN_CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=True)
        
        # Safe state_dict loading (handling prefixes)
        state_dict = {}
        for k, v in checkpoint.items():
            if k == 'model_state_dict': 
                for sub_k, sub_v in v.items():
                    state_dict[sub_k.replace('module.', '')] = sub_v
                break
            state_dict[k.replace('model.', '').replace('module.', '')] = v
            
        keys = raw_model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected): {len(keys.missing_keys)}")
    else:
        print(f"WARNING: Checkpoint not found at {FROZEN_CHECKPOINT_PATH}. Using RANDOM weights.")
    
    raw_model.to(config.DEVICE)
    model = FrozenFeatureWrapper(raw_model)
    model.eval()

    # 3. Define Transform (Resize -> Crop, using ImageNet stats)
    frozen_transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    print("\nPreparing datasets...")
    datasets = DatasetFactory.get_linear_probe_datasets(frozen_transform)

    # --- PART 1: LINEAR PROBE ---
    print("\n=== Starting Linear Probe Analysis (Frozen) ===")
    lp_results = {"scores": {}}
    cached_features = {}

    for name, data in datasets.items():
        try:
            print(f"Processing {name}...")
            train_feats, train_labels = extract_features(data['train'], model, config.DEVICE)
            test_feats, test_labels = extract_features(data['test'], model, config.DEVICE)
            
            cached_features[name] = (train_feats, train_labels, test_feats, test_labels)
            
            classifier = train_log_reg(train_feats, train_labels)
            acc, total = evaluate_classifier(classifier, test_feats, test_labels)
            
            lp_results["scores"][name] = {"accuracy": acc, "total": total}
            print(f"  > {name} Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"  > Failed {name}: {e}")

    save_results(lp_results, LINEAR_PROBE_FILE)

    # --- PART 2: FEW-SHOT ANALYSIS ---
    print("\n=== Starting Few-Shot Analysis (Frozen) ===")
    fs_results = {"scores": {}}
    
    for name, (tr_f, tr_l, te_f, te_l) in cached_features.items():
        print(f"Evaluating {name} Few-Shot...")
        fs_results["scores"][name] = {}
        unique_classes = np.unique(tr_l)
        
        for k in config.K_SHOTS:
            indices = []
            for c in unique_classes:
                c_indices = np.where(tr_l == c)[0]
                n_samples = min(len(c_indices), k)
                if n_samples > 0:
                    chosen = np.random.choice(c_indices, n_samples, replace=False)
                    indices.extend(chosen)
            
            if len(indices) == 0: continue
            
            k_feats = tr_f[indices]
            k_labels = tr_l[indices]
            
            try:
                clf = train_log_reg(k_feats, k_labels)
                acc, _ = evaluate_classifier(clf, te_f, te_l)
                fs_results["scores"][name][f"{k}-shot"] = acc
                print(f"  > {k}-shot Accuracy: {acc:.2f}%")
            except Exception as e:
                print(f"  > {k}-shot Failed: {e}")

    save_results(fs_results, FEW_SHOT_FILE)
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    run_evaluation()