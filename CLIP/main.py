import time
import open_clip
import numpy as np
import config
import core
from data_loader import DatasetFactory
import ssl
import random 
import os     
import torch   

# Bypass SSL verification for datasets downloads (common issue in universities/offices)
ssl._create_default_https_context = ssl._create_unverified_context

# --- Reproducibility ---
def seed_everything(seed=42):
    """
    Sets all random seeds to ensure 100% reproducible results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model():
    print(f"Loading {config.MODEL_NAME}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, 
        pretrained=config.PRETRAINED_TAG, 
        device=config.DEVICE
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    return model, preprocess, tokenizer

def run_zero_shot(model, tokenizer, preprocess):
    print("\n--- Starting Zero-Shot Analysis ---")
    results = {"scores": {}}
    
    configs = DatasetFactory.get_zeroshot_config(preprocess)
    
    for name, conf in configs.items():
        try:
            print(f"Evaluating {name}...")
            class_names = conf["class_getter"](conf["dataset"])
            
            text_classifier = core.get_zeroshot_classifier(
                model, tokenizer, conf["templates"], class_names
            )
            
            acc, corr, total = core.run_zeroshot_eval(model, text_classifier, conf["dataset"])
            
            results["scores"][name] = {
                "accuracy": acc, "correct": corr, "total": total
            }
            print(f"{name} Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback
            traceback.print_exc()
            
    core.save_results(results, config.ZERO_SHOT_RESULTS_FILE)

def run_linear_probe(model, preprocess):
    print("\n--- Starting Linear Probe Analysis ---")
    results = {"scores": {}}
    
    datasets = DatasetFactory.get_linear_probe_datasets(preprocess)
    
    for name, data in datasets.items():
        try:
            print(f"Processing {name}...")
            # Note: We extract features for the FULL training set here
            train_feats, train_labels = core.extract_features(data['train'], model)
            test_feats, test_labels = core.extract_features(data['test'], model)
            
            classifier = core.train_log_reg(train_feats, train_labels)
            acc, total = core.evaluate_classifier(classifier, test_feats, test_labels)
            
            results["scores"][name] = {"accuracy": acc, "total": total}
            print(f"{name} Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    core.save_results(results, config.LINEAR_PROBE_RESULTS_FILE)

def run_few_shot(model, preprocess):
    print("\n--- Starting Few-Shot Analysis ---")
    results = {"scores": {}}
    datasets = DatasetFactory.get_linear_probe_datasets(preprocess)
    
    # 1. Pre-calculate features to avoid re-running backbone for every k-shot
    cached_data = {}
    print("Pre-loading features for few-shot...")
    for name, data in datasets.items():
        try:
            print(f"  Caching {name}...")
            tr_f, tr_l = core.extract_features(data['train'], model)
            te_f, te_l = core.extract_features(data['test'], model)
            cached_data[name] = (tr_f, tr_l, te_f, te_l)
        except Exception as e:
            print(f"  Failed to cache {name}: {e}")

    # 2. Run K-Shot loops
    for name, (tr_f, tr_l, te_f, te_l) in cached_data.items():
        print(f"\nEvaluating {name} Few-Shot...")
        results["scores"][name] = {}
        unique_classes = np.unique(tr_l)
        
        for k in config.K_SHOTS:
            # --- Logic Fix for Reproducibility & Robustness ---
            indices = []
            for c in unique_classes:
                c_indices = np.where(tr_l == c)[0]
                
                # FIX: If a class has fewer samples than k (e.g. Flowers102 has 10, k=16),
                # take ALL available samples instead of skipping the class or crashing.
                n_samples = min(len(c_indices), k)
                
                if n_samples > 0:
                    # We use random choice with seed to ensure we pick a reproducible subset
                    chosen = np.random.choice(c_indices, n_samples, replace=False)
                    indices.extend(chosen)
            
            # Guard against completely empty datasets
            if len(indices) == 0:
                print(f"  Skipping {k}-shot: No samples found.")
                continue
                
            k_feats = tr_f[indices]
            k_labels = tr_l[indices]
            
            try:
                clf = core.train_log_reg(k_feats, k_labels, verbose=0)
                acc, _ = core.evaluate_classifier(clf, te_f, te_l)
                results["scores"][name][f"{k}-shot"] = acc
                print(f"  {k}-shot Accuracy: {acc:.2f}%")
            except Exception as e:
                print(f"  {k}-shot Failed: {e}")
            
    core.save_results(results, config.FEW_SHOT_RESULTS_FILE)

if __name__ == "__main__":
    # 1. Set seed immediately for reproducibility
    seed_everything(42) 
    
    # 2. Setup Model
    model, preprocess, tokenizer = setup_model()
    
    # 3. Run Experiments
    # Comment out lines if you only want to run specific parts
    run_zero_shot(model, tokenizer, preprocess)
    run_linear_probe(model, preprocess)
    run_few_shot(model, preprocess)