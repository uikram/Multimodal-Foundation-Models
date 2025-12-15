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
import glob
import json

# --- NEW IMPORTS for LoRA Support ---
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel, PeftConfig

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuration ---
LORA_CHECKPOINT_DIR = "../clip_lora_checkpoints/epoch-3"


# --- Wrapper Class ---
# This makes the Hugging Face model "look like" an OpenCLIP model
# so your existing test functions (run_zero_shot, etc.) work without changes.
class HFToOpenCLIPWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        # Map logit_scale so open_clip code can access it
        self.logit_scale = hf_model.logit_scale

    def encode_image(self, image, normalize=True):
        # OpenCLIP passes a [Batch, 3, H, W] tensor.
        # HF expects 'pixel_values'.
        features = self.hf_model.get_image_features(pixel_values=image)
        if normalize:
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def encode_text(self, text, normalize=True):
        # OpenCLIP passes a [Batch, SeqLen] tensor.
        # HF expects 'input_ids'.
        features = self.hf_model.get_text_features(input_ids=text)
        if normalize:
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features
    
    # Forward allows direct calls if your code uses model(img, txt)
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()

# --- Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model():
    print(f"--- Setting up LoRA Model ---")
    
    # 1. Find the Adapter Config & Weights
    # We look for adapter_config.json to find the base model name
    config_files = glob.glob(os.path.join(LORA_CHECKPOINT_DIR, "**", "adapter_config.json"), recursive=True)
    
    if not config_files:
        raise FileNotFoundError(f"Could not find adapter_config.json in {LORA_CHECKPOINT_DIR}")
    
    # Use the most recent config found
    adapter_config_path = sorted(config_files)[-1]
    adapter_path = os.path.dirname(adapter_config_path)
    print(f"Found Adapter at: {adapter_path}")

    # 2. Determine Base Model from Config
    with open(adapter_config_path, 'r') as f:
        adapter_conf = json.load(f)
        # Default fallback if not in config
        base_model_id = adapter_conf.get("base_model_name_or_path", "openai/clip-vit-base-patch32")
    
    print(f"Loading Base Model (Hugging Face): {base_model_id}")
    
    # 3. Load Base Model & Processor
    try:
        base_model = CLIPModel.from_pretrained(base_model_id)
        # Load the adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("SUCCESS: LoRA Adapter loaded via PEFT.")
        
        # Merge weights for faster inference (optional but recommended)
        model = model.merge_and_unload()
        model.to(config.DEVICE)
        model.eval()
        
    except Exception as e:
        print(f"CRITICAL ERROR loading PEFT model: {e}")
        raise e

    # 4. Wrap the model to be compatible with your test script
    wrapped_model = HFToOpenCLIPWrapper(model)

    # 5. Setup Preprocessing & Tokenizer
    # We use open_clip tools to maintain compatibility with your DataLoaders
    # (Hugging Face and OpenCLIP usually share the same normalization stats)
    print("Setting up transforms...")
    _, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, 
        pretrained=config.PRETRAINED_TAG, 
        device=config.DEVICE
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    
    return wrapped_model, preprocess, tokenizer

# --- EXISTING TEST FUNCTIONS (Unchanged) ---
def run_zero_shot(model, tokenizer, preprocess):
    print("\n--- Starting Zero-Shot Analysis ---")
    results = {"scores": {}}
    configs = DatasetFactory.get_zeroshot_config(preprocess)
    
    for name, conf in configs.items():
        try:
            print(f"Evaluating {name}...")
            class_names = conf["class_getter"](conf["dataset"])
            
            # This calls model.encode_text internally
            text_classifier = core.get_zeroshot_classifier(
                model, tokenizer, conf["templates"], class_names
            )
            
            # This calls model.encode_image internally
            acc, corr, total = core.run_zeroshot_eval(model, text_classifier, conf["dataset"])
            
            results["scores"][name] = {"accuracy": acc, "correct": corr, "total": total}
            print(f"{name} Accuracy: {acc:.2f}%")
        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback; traceback.print_exc()
            
    core.save_results(results, config.ZERO_SHOT_RESULTS_FILE)

def run_linear_probe(model, preprocess):
    print("\n--- Starting Linear Probe Analysis ---")
    results = {"scores": {}}
    datasets = DatasetFactory.get_linear_probe_datasets(preprocess)
    
    for name, data in datasets.items():
        try:
            print(f"Processing {name}...")
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

    for name, (tr_f, tr_l, te_f, te_l) in cached_data.items():
        print(f"\nEvaluating {name} Few-Shot...")
        results["scores"][name] = {}
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
                clf = core.train_log_reg(k_feats, k_labels, verbose=0)
                acc, _ = core.evaluate_classifier(clf, te_f, te_l)
                results["scores"][name][f"{k}-shot"] = acc
                print(f"  {k}-shot Accuracy: {acc:.2f}%")
            except Exception as e:
                print(f"  {k}-shot Failed: {e}")
            
    core.save_results(results, config.FEW_SHOT_RESULTS_FILE)

if __name__ == "__main__":
    seed_everything(42) 
    
    # 1. Setup Wrapper Model
    model, preprocess, tokenizer = setup_model()
    
    # 2. Run Experiments
    run_zero_shot(model, tokenizer, preprocess)
    run_linear_probe(model, preprocess)
    run_few_shot(model, preprocess)