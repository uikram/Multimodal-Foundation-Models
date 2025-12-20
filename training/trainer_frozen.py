"""
Trainer for Frozen model (trainable vision encoder + frozen LLM).
"""

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import GPT2Tokenizer
import math


class FrozenTrainer:
    """Trainer for Frozen architecture."""
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Optimizer (explicitly use vision_encoder parameters)
        self.optimizer = torch.optim.Adam(
            self.model.vision_encoder.parameters(),  # ← FIXED: Explicit targeting
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Training state - track multiple metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_perplexity = float('inf')
        self.best_epoch_by_perplexity = -1
    
    def train(self):
        """Execute full training pipeline."""
        print("\n" + "="*60)
        print("FROZEN MODEL TRAINING")
        print("="*60)
        
        # Load datasets (with optional debug mode)
        train_loader, val_loader = self.get_dataloaders()
        
        print(f"✓ Loaded {len(train_loader)} training batches")
        print(f"✓ Loaded {len(val_loader)} validation batches")
        print(f"  Batch size: {self.config.batch_size}")
        
        # Debug mode info
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            print(f"\n⚠️  DEBUG MODE ENABLED")
            print(f"  Training on {self.config.debug_samples} samples only")
            print(f"  Validating on {self.config.debug_val_samples} samples only")
        
        # Track memory before training
        self.metrics.track_gpu_memory('pre_training')
        self.metrics.track_cpu_memory()
        
        # Start timer
        self.metrics.start_training_timer()
        
        # Training loop
        print(f"\nTraining for {self.config.num_epochs} epochs...")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss, val_perplexity = self.validate(val_loader)
            
            self.metrics.track_epoch_metrics(
                epoch+1, 
                train_loss=train_loss, 
                val_loss=val_loss,
                perplexity=val_perplexity
            )
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Complete")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_perplexity:.2f}")
            print(f"{'='*60}\n")
            
            # Save checkpoint (now considers perplexity)
            self.save_checkpoint(epoch, val_loss, val_perplexity)
        
        # End timer
        self.metrics.end_training_timer()

        # Track memory after training
        self.metrics.track_gpu_memory('post_training')

        # Track final training performance
        self.metrics.track_performance(
            accuracy=0.0, 
            loss=self.best_val_loss,
            perplexity=self.best_perplexity
        ) 

        print("\n✅ Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Perplexity: {self.best_perplexity:.2f} (Epoch {self.best_epoch_by_perplexity + 1})")
        print(f"💡 Using perplexity-based checkpoint for downstream tasks")
    
    def get_dataloaders(self):
        """Get training and validation dataloaders with optional debug mode."""
        from datasets.dataloaders import FrozenConceptualCaptionsDataset
        from pathlib import Path
        
        # Create full datasets
        train_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.train_image_dir),
            Path(self.config.train_file),
            self.tokenizer,
            self.config
        )
        
        val_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.val_image_dir),
            Path(self.config.val_file),
            self.tokenizer,
            self.config
        )
        
        # Apply debug mode if enabled
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            debug_samples = getattr(self.config, 'debug_samples', 1000)
            debug_val_samples = getattr(self.config, 'debug_val_samples', 200)
            
            # Create subset of data
            train_indices = list(range(min(debug_samples, len(train_ds))))
            val_indices = list(range(min(debug_val_samples, len(val_ds))))
            
            train_ds = Subset(train_ds, train_indices)
            val_ds = Subset(val_ds, val_indices)
            
            print(f"\n🔍 Debug Mode Active:")
            print(f"  Training samples: {len(train_ds)}")
            print(f"  Validation samples: {len(val_ds)}")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            dynamic_ncols=True
        )
        
        for batch in pbar:
            loss = self.train_step(batch)
            epoch_loss += loss
            
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        return epoch_loss / len(train_loader)
    
    def train_step(self, batch):
        """Single training step."""
        images = batch['images'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            logits, loss = self.model.forward(
                images, input_ids, attention_mask, labels
            )
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimization step (fixed increment logic)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1  # ← FIXED: Only increment after accumulation
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self, val_loader):
        """
        Validation loop with perplexity calculation.
        Returns: (avg_loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    _, loss = self.model.forward(
                        images, input_ids, attention_mask, labels
                    )
                
                # Count non-padding tokens for accurate perplexity
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, val_loss, perplexity):
        """
        Save model checkpoint.
        Saves both:
        - best_model.pt: Based on lowest validation loss (backward compatibility)
        - best_model_perplexity.pt: Based on lowest perplexity (better for generation tasks)
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.vision_encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'perplexity': perplexity
        }
        
        # Save regular checkpoint every epoch
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Saved checkpoint to {checkpoint_path}")
        
        # Save best model by validation loss (backward compatibility)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (by val loss) to {best_path}")
        
        # Save best model by perplexity (BETTER for downstream tasks!)
        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity
            self.best_epoch_by_perplexity = epoch
            best_perplexity_path = self.config.checkpoint_dir / "best_model_perplexity.pt"
            torch.save(checkpoint, best_perplexity_path)
            print(f"  🌟 Saved BEST model (by perplexity) to {best_perplexity_path}")
            print(f"     Perplexity: {perplexity:.2f} (lower is better for generation)")
