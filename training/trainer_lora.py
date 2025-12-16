"""
Trainer for CLIP + LoRA model.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


class CLIPLoRATrainer:
    """Trainer for CLIP with LoRA adapters."""
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train(self):
        """Execute full training pipeline."""
        print("\n" + "="*60)
        print("CLIP + LoRA TRAINING")
        print("="*60)
        
        from datasets.dataloaders import get_conceptual_captions_loader
        
        # Load dataset
        print("Loading Conceptual Captions dataset...")
        train_loader = get_conceptual_captions_loader(self.config, self.model.processor)
        
        print(f"✓ Loaded {len(train_loader)} batches")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Total samples: {len(train_loader.dataset)}")
        
        # Track memory before training
        self.metrics.track_gpu_memory('pre_training')
        self.metrics.track_cpu_memory()
        
        # Start timer
        self.metrics.start_training_timer()
        
        # Training loop
        print(f"\nTraining for {self.config.num_epochs} epochs...")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = self.train_epoch(epoch, train_loader)
            self.metrics.track_epoch_metrics(epoch+1, train_loss=epoch_loss)
            # Save checkpoint
            self.save_checkpoint(epoch, epoch_loss)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Complete")
            print(f"Average Loss: {epoch_loss:.4f}")
            print(f"{'='*60}\n")
        
        # End timer
        self.metrics.end_training_timer()
        
        # Track memory after training
        self.metrics.track_gpu_memory('post_training')
        
        # Track final performance
        self.metrics.track_performance(accuracy=0.0, loss=epoch_loss)
        
        print("\n✅ Training Complete!")
        print(f"Best Loss: {self.best_loss:.4f}")
    
    def train_epoch(self, epoch, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            dynamic_ncols=True
        )
        
        for batch in pbar:
            loss = self.train_step(batch)
            
            epoch_loss += loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}'
            })
        
        return epoch_loss / batch_count if batch_count > 0 else 0.0
    
    def train_step(self, batch):
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model.forward(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        checkpoint_path = self.config.output_dir / f"epoch_{epoch+1}"
        self.model.save_pretrained(checkpoint_path)
        
        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.config.output_dir / "best_model"
            self.model.save_pretrained(best_path)
            print(f"  ✓ Saved best model to {best_path}")
        
        print(f"  ✓ Saved checkpoint to {checkpoint_path}")
