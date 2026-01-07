"""
Model evaluation module with zero-shot, few-shot, and linear probe evaluation.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ModelEvaluator:
    """Evaluator for multimodal foundation models."""
    
    def __init__(self, model, config, metrics):
        self.model = model
        self.config = config
        self.metrics = metrics
        
    def _extract_features(self, dataset):
        """
        Extract image features from a dataset.
        
        Args:
            dataset: PyTorch dataset
            
        Returns:
            features: numpy array of shape [N, feature_dim]
            labels: numpy array of shape [N]
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        all_features = []
        all_labels = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images, labels = batch
                elif isinstance(batch, dict):
                    images = batch.get('images', batch.get('pixel_values'))
                    labels = batch.get('labels', batch.get('label'))
                    
                    if images is None:
                        raise KeyError(f"Batch missing images/pixel_values. Keys: {batch.keys()}")
                else:
                    images = batch
                    labels = None
                
                images = images.to(self.config.device)
                
                # Extract features
                if hasattr(self.model, 'encode_image'):
                    features = self.model.encode_image(images)
                else:
                    features = self.model(images)
                
                all_features.append(features.cpu().numpy())
                
                if labels is not None:
                    all_labels.append(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels) if all_labels else None
        
        return features, labels
    
    def zero_shot_evaluation(self, test_dataset, text_classifier):
        """
        Zero-shot evaluation using pre-computed text classifier.
        """
        # [FIX] Skip Frozen model
        from models.frozen_clip import FrozenCLIP
        if isinstance(self.model, FrozenCLIP):
            print("Skipping Zero-Shot for Frozen (Generative model incompatible with Contrastive eval)")
            return {'top1': 'N/A', 'top5': 'N/A'}

        self.metrics.start_evaluation_timer()
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        all_top5_predictions = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Zero-shot evaluation"):
                if isinstance(batch, (list, tuple)):
                    images, labels = batch
                else:
                    images = batch['images']
                    labels = batch['labels']
                
                images = images.to(self.config.device)
                
                # Get image features
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity with text classifier
                logits = image_features @ text_classifier  # [batch_size, num_classes]
                
                # Top-1 predictions
                predictions = logits.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                
                # Top-5 predictions
                top5_pred = torch.topk(logits, k=5, dim=1)[1].cpu().numpy()
                all_top5_predictions.extend(top5_pred)
                
                all_labels.extend(labels.numpy())
        
        self.metrics.end_evaluation_timer()
        
        # Calculate accuracies
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        top1_acc = accuracy_score(all_labels, all_predictions) * 100
        
        # Top-5 accuracy
        top5_correct = sum([1 for i, label in enumerate(all_labels) 
                           if label in all_top5_predictions[i]])
        top5_acc = (top5_correct / len(all_labels)) * 100
        
        results = {
            'top1': top1_acc,
            'top5': top5_acc
        }
        
        return results
    
    def linear_probe_evaluation(self, train_dataset, test_dataset):
        """
        Linear probe evaluation using entire training dataset.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            
        Returns:
            accuracy: Test accuracy
            num_samples: Number of test samples
        """
        print("Extracting training features...")
        train_features, train_labels = self._extract_features(train_dataset)
        
        print("Extracting test features...")
        test_features, test_labels = self._extract_features(test_dataset)
        
        self.metrics.start_inference_timer()
        
        # Train logistic regression
        import os
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='lbfgs',
            n_jobs=min(8, os.cpu_count() or 1),
            verbose=0
        )
        
        print("Training linear classifier...")
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        predictions = classifier.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions) * 100
        
        self.metrics.end_inference_timer()
        
        return accuracy, len(test_dataset)
    
    def few_shot_evaluation(self, train_dataset, test_dataset, k_shots=[1, 2, 4, 8, 16], num_trials=3):
        """
        Few-shot evaluation with multiple random trials for robust results.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            k_shots: List of k values to evaluate
            num_trials: Number of random trials per k-shot (default: 3)
            
        Returns:
            Dictionary with mean accuracy, std, and all trial results
        """
        print(f"Caching features for few-shot ({num_trials} trials per k)...")
        
        # Extract features once (reuse across all trials)
        train_features, train_labels = self._extract_features(train_dataset)
        test_features, test_labels = self._extract_features(test_dataset)
        
        results = {}
        
        for k in k_shots:
            print(f"Evaluating {k}-shot...")
            trial_accuracies = []
            
            # Run multiple trials with different seeds
            for trial_idx in range(num_trials):
                seed = 42 + trial_idx  # Seeds: 42, 43, 44
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                try:
                    # Sample k examples per class with current seed
                    selected_indices = []
                    unique_labels = np.unique(train_labels)
                    
                    for label in unique_labels:
                        label_indices = np.where(train_labels == label)[0]
                        
                        if len(label_indices) >= k:
                            # Random sample with current seed
                            sampled = np.random.choice(label_indices, k, replace=False)
                            selected_indices.extend(sampled)
                        else:
                            # [NEW] Oversample with replacement to get exactly k samples
                            sampled = np.random.choice(label_indices, k, replace=True)
                            selected_indices.extend(sampled) 
                    
                    selected_indices = np.array(selected_indices)
                    
                    # Prepare training data for this trial
                    X_train = train_features[selected_indices]
                    y_train = train_labels[selected_indices]
                    
                    import os
                    # Train logistic regression classifier
                    classifier = LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        C=1.0,
                        solver='lbfgs',
                        n_jobs=min(8, os.cpu_count() or 1)
                    )
                    
                    classifier.fit(X_train, y_train)
                    
                    # Evaluate on test set
                    predictions = classifier.predict(test_features)
                    accuracy = accuracy_score(test_labels, predictions) * 100
                    
                    trial_accuracies.append(accuracy)
                    
                except Exception as e:
                    print(f"  Trial {trial_idx + 1}/{num_trials} failed: {e}")
                    continue
            
            # Compute statistics across trials
            if len(trial_accuracies) > 0:
                mean_acc = np.mean(trial_accuracies)
                std_acc = np.std(trial_accuracies)
                min_acc = np.min(trial_accuracies)
                max_acc = np.max(trial_accuracies)
                
                results[f'{k}-shot'] = {
                    'accuracy_mean': round(float(mean_acc), 2),
                    'accuracy_std': round(float(std_acc), 2),
                    'accuracy_min': round(float(min_acc), 2),
                    'accuracy_max': round(float(max_acc), 2),
                    'num_trials': len(trial_accuracies),
                    'all_trials': [round(float(acc), 2) for acc in trial_accuracies]
                }
                
                print(f"  {k}-shot: {mean_acc:.2f}% ± {std_acc:.2f}% (range: {min_acc:.2f}%-{max_acc:.2f}%)")
            else:
                results[f'{k}-shot'] = {
                    'accuracy_mean': 0.0,
                    'accuracy_std': 0.0,
                    'num_trials': 0,
                    'error': 'All trials failed'
                }
                print(f"  {k}-shot: All trials failed!")
        
        return results
