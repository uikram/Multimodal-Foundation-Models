"""
Trainer for Frozen model (trainable vision encoder + frozen LLM).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer


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
        
        # Optimizer (Vision Encoder only)
        self.optimizer = torch.optim.Adam(
                                        self.model.vision_encoder.parameters(),
                                        lr=config.learning_rate,
                                        weight_decay=config.weight_decay
                                        )

        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Execute full training pipeline."""
        print("\n" + "="*60)
        print("FROZEN MODEL TRAINING")
        print("="*60)
        
        # Load datasets
        train_loader, val_loader = self.get_dataloaders()
        
        print(f"✓ Loaded {len(train_loader)} training batches")
        print(f"✓ Loaded {len(val_loader)} validation batches")
        print(f"  Batch size: {self.config.batch_size}")
        
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
            val_loss = self.validate(val_loader)
            self.metrics.track_epoch_metrics(epoch+1, train_loss=train_loss, val_loss=val_loss)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Complete")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
        
            # End timer
            self.metrics.end_training_timer()

            # Track memory after training
            self.metrics.track_gpu_memory('post_training')

            # Track final training performance
            self.metrics.track_performance(accuracy=0.0, loss=self.best_val_loss) 

            print("\n✅ Training Complete!")
            print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
    def get_dataloaders(self):
        """Get training and validation dataloaders."""
        from datasets.dataloaders import FrozenConceptualCaptionsDataset
        from pathlib import Path
        from torch.utils.data import DataLoader
        
        # Check if debug mode is enabled
        debug_mode = getattr(self.config, 'debug_mode', False)
        max_samples = 500 if debug_mode else None
        val_max_samples = 100 if debug_mode else None
        
        if debug_mode:
            print("🔧 DEBUG MODE ENABLED: Training on limited samples")
        
        # Create datasets
        train_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.train_image_dir),
            Path(self.config.train_file),
            self.tokenizer,
            self.config,
            debug_mode=debug_mode,
            max_samples=max_samples
        )
        
        val_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.val_image_dir),
            Path(self.config.val_file),
            self.tokenizer,
            self.config,
            debug_mode=debug_mode,
            max_samples=val_max_samples
        )
        
        # Create dataloaders (disable multiprocessing in debug mode)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0 if debug_mode else self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0 if debug_mode else self.config.num_workers,
            pin_memory=True
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
        
        # Optimization step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        
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
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.vision_encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model to {best_path}")
        
        print(f"  ✓ Saved checkpoint to {checkpoint_path}")
