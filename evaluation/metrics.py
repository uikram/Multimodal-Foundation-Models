"""
Comprehensive metrics computation and storage.
"""

import json
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

class MetricsTracker:
    """Unified metrics tracking for all models."""
        
    def __init__(self, model_name: str, results_dir: Path):
        """Initialize metrics tracker."""
        self.model_name = model_name
        
        # FIX: Ensure results_dir is a Path object
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        
        self.results_dir = results_dir / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = {
            'parameters': {},
            'memory': {},
            'latency': {},
            'training_time': {},
            'inference_time': {},
            'evaluation_time': {},
            'performance': {},
            'classification_report': {},
            'training_history': {}
        }
        
        # Per-epoch training history
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Timing trackers
        self.train_start_time = None
        self.train_end_time = None
        self.inference_start_time = None
        self.inference_end_time = None
        self.eval_start_time = None
        self.eval_end_time = None


        
    def track_parameters(self, model):
        """Track model parameters."""
        param_counts = model.count_parameters()
        self.metrics['parameters'] = {
            'total_parameters': param_counts['total'],
            'trainable_parameters': param_counts['trainable'],
            'frozen_parameters': param_counts['total'] - param_counts['trainable']
        }
        
    def track_epoch_metrics(self, epoch: int, train_loss: float = None, 
                        val_loss: float = None, train_acc: float = None, 
                        val_acc: float = None):
        """
        Track per-epoch training metrics for plotting curves.
        
        Args:
            epoch: Current epoch number (1-indexed)
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch (optional)
            train_acc: Training accuracy for this epoch (optional)
            val_acc: Validation accuracy for this epoch (optional)
        """
        self.training_history['epochs'].append(epoch)
        
        if train_loss is not None:
            self.training_history['train_loss'].append(float(train_loss))
        
        if val_loss is not None:
            self.training_history['val_loss'].append(float(val_loss))
        
        if train_acc is not None:
            self.training_history['train_accuracy'].append(float(train_acc))
        
        if val_acc is not None:
            self.training_history['val_accuracy'].append(float(val_acc))
        
        # Update metrics dict for saving
        self.metrics['training_history'] = self.training_history
 
            
    def start_training_timer(self):
        """Start training timer."""
        self.train_start_time = time.time()
        
    def end_training_timer(self):
        """End training timer and record."""
        self.train_end_time = time.time()
        self.metrics['training_time'] = {
            'total_seconds': self.train_end_time - self.train_start_time,
            'total_hours': (self.train_end_time - self.train_start_time) / 3600
        }
        
    def track_gpu_memory(self, stage: str = 'training'):
        """Track GPU memory usage."""
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
            self.metrics['memory'][f'{stage}_peak_gpu_mb'] = peak_memory_mb
            self.metrics['memory'][f'{stage}_current_gpu_mb'] = current_memory_mb
            
            # Reset peak stats for next stage
            torch.cuda.reset_peak_memory_stats()
        else:
            self.metrics['memory'][f'{stage}_peak_gpu_mb'] = 'N/A'
            self.metrics['memory'][f'{stage}_current_gpu_mb'] = 'N/A'
            
    def track_cpu_memory(self):
        """Track CPU memory usage."""
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics['memory']['cpu_memory_mb'] = cpu_memory_mb
        
    def track_performance(self, accuracy: float = None, top5_accuracy: float = None, loss: float = 0.0):
        """
        Track performance metrics.
        
        Args:
            accuracy: Top-1 accuracy (primary metric)
            top5_accuracy: Top-5 accuracy (optional)
            loss: Final loss value
        """
        self.metrics['performance'] = {
            'final_loss': float(loss)
        }
        
        # Track Top-1 if provided
        if accuracy is not None:
            self.metrics['performance']['top1_accuracy'] = float(accuracy)
        
        # Track Top-5 if provided
        if top5_accuracy is not None:
            self.metrics['performance']['top5_accuracy'] = float(top5_accuracy)

        
    def track_classification_report(self, y_true, y_pred, class_names=None):
        """Generate and track classification report."""
        report_dict = classification_report(
            y_true, 
            y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        self.metrics['classification_report'] = report_dict
        
    def save_metrics(self, run_id: str = None):
        """Save all metrics to JSON files."""
        # Remove timestamp - use simple filenames
        # if run_id is None:
        #     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for metric_type, data in self.metrics.items():
            if data:  # Only save if there's data
                # Use simple filename without timestamp
                filename = f"{metric_type}.json"  # ← CHANGED: No run_id suffix
                filepath = self.results_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Saved {metric_type} to {filepath}")
        
    def load_metrics(self, run_id: str):
        """Load metrics from JSON files."""
        for metric_type in self.metrics.keys():
            filename = f"{metric_type}_{run_id}.json"
            filepath = self.results_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.metrics[metric_type] = json.load(f)
                    
    def print_summary(self):
        """Print metrics summary."""
        print(f"\n{'='*60}")
        print(f"Metrics Summary for {self.model_name}")
        print(f"{'='*60}")
        
        for metric_type, data in self.metrics.items():
            if data:
                print(f"\n{metric_type.upper()}:")
                print(json.dumps(data, indent=2))

    def start_inference_timer(self):
        """Start inference timer."""
        self.inference_start_time = time.time()
        
    def end_inference_timer(self):
        """End inference timer and record."""
        self.inference_end_time = time.time()
        total_time = self.inference_end_time - self.inference_start_time
        
        self.metrics['inference_time'] = {
            'total_seconds': total_time,
            'total_minutes': total_time / 60,
            'total_hours': total_time / 3600
        }
    
    def start_evaluation_timer(self):
        """Start complete evaluation timer (includes data loading, preprocessing, inference)."""
        self.eval_start_time = time.time()
        
    def end_evaluation_timer(self):
        """End evaluation timer and record."""
        self.eval_end_time = time.time()
        total_time = self.eval_end_time - self.eval_start_time
        
        self.metrics['evaluation_time'] = {
            'total_seconds': total_time,
            'total_minutes': total_time / 60,
            'total_hours': total_time / 3600
        }
    
    def track_inference_latency(self, model, dataloader, num_samples: int = 100):
        """
        Track detailed inference latency metrics.
        This measures pure model inference time (forward pass only),
        excluding data loading and preprocessing.
        """
        model.eval()
        latencies = []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(model.device) if isinstance(b, torch.Tensor) else b 
                            for b in batch]
                
                # Synchronize GPU before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                
                # Pure inference - use appropriate method based on batch type
                try:
                    if isinstance(batch, dict):
                        # For CLIP/Frozen models with dict batches
                        if 'images' in batch:
                            _ = model.encode_image(batch['images'])
                        else:
                            _ = model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        # For tuple/list batches (image, label)
                        images = batch[0] if isinstance(batch, (list, tuple)) else batch
                        _ = model.encode_image(images)
                    else:
                        # Fallback: direct tensor
                        _ = model.encode_image(batch)
                except AttributeError:
                    # Model doesn't have encode_image, try forward
                    try:
                        if isinstance(batch, dict):
                            _ = model(**batch)
                        else:
                            _ = model(batch)
                    except:
                        # Skip this batch if can't process
                        continue
                
                # Synchronize GPU after inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
        
        # Calculate statistics
        latencies_array = np.array(latencies)
        self.metrics['latency'] = {
            'average_ms': float(np.mean(latencies_array)),
            'median_ms': float(np.median(latencies_array)),
            'std_ms': float(np.std(latencies_array)),
            'min_ms': float(np.min(latencies_array)),
            'max_ms': float(np.max(latencies_array)),
            'p95_ms': float(np.percentile(latencies_array, 95)),
            'p99_ms': float(np.percentile(latencies_array, 99)),
            'total_inference_time_s': float(np.sum(latencies_array) / 1000),
            'samples_measured': len(latencies),
            'throughput_samples_per_second': len(latencies) / (np.sum(latencies_array) / 1000)
        }
        
        # Track inference memory
        self.track_gpu_memory('inference')