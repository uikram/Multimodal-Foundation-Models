"""
Unified evaluation pipeline for all models.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from .metrics import MetricsTracker
import time


class ModelEvaluator:
    """Unified evaluator for all model types."""

    def __init__(self, model, config, metrics_tracker: MetricsTracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device

    def extract_features(self, dataset):
        """
        Extract features from a dataset using the model's image encoder.
        Returns features and labels as numpy arrays.
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
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)

                # Encode images
                features = self.model.encode_image(images)

                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)

                all_features.append(features.cpu())
                all_labels.append(labels)

        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        return all_features, all_labels

    def linear_probe_evaluation(self, train_dataset, test_dataset):
        """Perform linear probe evaluation with timing."""
        # Start overall evaluation timer
        self.metrics.start_evaluation_timer()

        print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_dataset)

        print("Extracting test features...")
        # Start inference timer specifically for test set
        self.metrics.start_inference_timer()
        test_features, test_labels = self.extract_features(test_dataset)
        self.metrics.end_inference_timer()

        print("Training logistic regression...")
        # Get logistic regression C parameter
        C = getattr(self.config, 'logistic_regression_c', 0.316)

        classifier = LogisticRegression(
            random_state=0,
            C=C,
            max_iter=1000,
            verbose=1,
            n_jobs=-1,
            solver='lbfgs'
        )
        classifier.fit(train_features, train_labels)

        print("Evaluating...")
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0

        # End overall evaluation timer
        self.metrics.end_evaluation_timer()

        # Track classification report
        self.metrics.track_classification_report(test_labels, predictions)

        return accuracy, len(test_labels)

    def calculate_top_k_accuracy(self, logits, labels, k=5):
        """
        Calculate Top-K accuracy.
        
        Args:
            logits: torch.Tensor [batch_size, num_classes] prediction scores
            labels: torch.Tensor [batch_size] ground truth labels
            k: Top-K to calculate (default: 5)
        
        Returns:
            Top-K accuracy as percentage
        """
        # Get top-k predictions
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        
        # Check if true label is in top-k
        labels = labels.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == labels).any(dim=1).float()
        
        return correct.mean().item() * 100.0

    def zero_shot_evaluation(self, dataset, text_classifier):
        """Perform zero-shot evaluation with Top-1 and Top-5 accuracy."""
        # Start overall evaluation timer
        self.metrics.start_evaluation_timer()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        all_logits = []  
        all_predictions = []
        all_labels = []

        # Track pure inference time
        inference_times = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Zero-Shot Eval"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Time pure inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_start = time.time()

                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = (100.0 * image_features @ text_classifier)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - inference_start)

                # Store logits for Top-5 calculation
                all_logits.append(logits.cpu())
                
                # Get Top-1 predictions
                _, predictions = logits.max(1)
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # End overall evaluation timer
        self.metrics.end_evaluation_timer()

        # Record inference time
        total_inference_time = sum(inference_times)
        self.metrics.metrics['inference_time'] = {
            'total_seconds': total_inference_time,
            'total_minutes': total_inference_time / 60,
            'total_hours': total_inference_time / 3600,
            'average_per_batch_ms': (total_inference_time / len(inference_times)) * 1000
        }

        # Concatenate all batches
        all_logits = torch.cat(all_logits)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Calculate Top-1 Accuracy
        top1_accuracy = np.mean((all_labels.numpy() == all_predictions.numpy()).astype(float)) * 100.0
        
        # Calculate Top-5 Accuracy
        num_classes = all_logits.size(1)
        if num_classes >= 5:
            top5_accuracy = self.calculate_top_k_accuracy(all_logits, all_labels, k=5)
        else:
            # If fewer than 5 classes, Top-5 = Top-1
            top5_accuracy = top1_accuracy
            print(f"⚠️  Dataset has only {num_classes} classes. Top-5 = Top-1")

        # Print results
        print(f"✓ Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"✓ Top-5 Accuracy: {top5_accuracy:.2f}%")

        # Track classification report (using Top-1 predictions)
        self.metrics.track_classification_report(all_labels.numpy(), all_predictions.numpy())

        # Return both accuracies
        return {
            'top1': top1_accuracy,
            'top5': top5_accuracy,
            'num_samples': len(all_labels)
        }


    def few_shot_evaluation(self, train_dataset, test_dataset, k_shots):
        """Perform few-shot evaluation."""
        print("Caching features for few-shot...")
        train_features, train_labels = self.extract_features(train_dataset)
        test_features, test_labels = self.extract_features(test_dataset)

        results = {}
        unique_classes = np.unique(train_labels)

        # Get logistic regression C parameter
        C = getattr(self.config, 'logistic_regression_c', 0.316)

        for k in k_shots:
            print(f"Evaluating {k}-shot...")

            indices = []
            for c in unique_classes:
                c_indices = np.where(train_labels == c)[0]
                n_samples = min(len(c_indices), k)

                if n_samples > 0:
                    chosen = np.random.choice(c_indices, n_samples, replace=False)
                    indices.extend(chosen)

            if len(indices) == 0:
                continue

            k_features = train_features[indices]
            k_labels = train_labels[indices]

            try:
                clf = LogisticRegression(
                    random_state=0,
                    C=C,
                    max_iter=1000,
                    verbose=0,
                    n_jobs=-1,
                    solver='lbfgs'
                )
                clf.fit(k_features, k_labels)
                predictions = clf.predict(test_features)
                accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
                
                # Track classification report for the final k-shot value
                if k == k_shots[-1]:  # Only for last k (e.g., 16-shot)
                    self.metrics.track_classification_report(test_labels, predictions)

                results[f"{k}-shot"] = accuracy
                print(f"  {k}-shot Accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"  {k}-shot Failed: {e}")

        return results
