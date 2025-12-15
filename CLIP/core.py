import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

import config

def extract_features(dataset, model):
    """
    Extracts CLIP features, normalizes them, and returns as numpy arrays.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(config.DEVICE)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def train_log_reg(train_features, train_labels, verbose=1):
    """
    Trains a Logistic Regression classifier (Linear Probe).
    """
    classifier = LogisticRegression(
        random_state=0, 
        C=config.LOGISTIC_REGRESSION_C, 
        max_iter=1000, 
        verbose=verbose, 
        n_jobs=-1,
        solver='lbfgs'
    )
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_classifier(classifier, test_features, test_labels):
    """
    Evaluates a scikit-learn classifier.
    """
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    return accuracy, len(test_labels)

def get_zeroshot_classifier(model, tokenizer, templates, class_names):
    """
    Creates zero-shot classifier weights by encoding text prompts.
    """
    all_text_features = []
    model.eval()
    
    with torch.no_grad():
        for class_name in tqdm(class_names, desc="Encoding prompts"):
            texts = [template.format(class_name) for template in templates]
            text_tokens = tokenizer(texts).to(config.DEVICE)
            
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Ensemble (mean) over templates
            text_features = text_features.mean(dim=0)
            text_features /= text_features.norm()
            
            all_text_features.append(text_features)
            
    return torch.stack(all_text_features, dim=0).to(config.DEVICE)

def run_zeroshot_eval(model, text_classifier, dataset):
    """
    Runs Zero-Shot evaluation loop.
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    total_correct = 0
    total_samples = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Zero-Shot Eval"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity -> Logits
            logits = (100.0 * image_features @ text_classifier.T)
            _, predictions = logits.max(1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)
            
    return (total_correct / total_samples) * 100.0, total_correct, total_samples

def save_results(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    if not filepath.exists():
        return {}
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("scores", {})