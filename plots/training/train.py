"""
Unified training pipeline for all model types.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelTrainer:
    """Unified trainer for all model architectures."""
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
        
        # Determine model type and setup accordingly
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self):
        """Detect model type from config."""
        if hasattr(self.config, 'model_name'):
            return 'clip_baseline'
        elif hasattr(self.config, 'lora_r'):
            return 'clip_lora'
        elif hasattr(self.config, 'vision_encoder_name'):
            return 'frozen'
        else:
            raise ValueError("Unknown model type")
    
    def train(self):
        """Execute training based on model type."""
        if self.model_type == 'clip_baseline':
            print("CLIP Baseline is pretrained - skipping training")
            return
        elif self.model_type == 'clip_lora':
            self._train_clip_lora()
        elif self.model_type == 'frozen':
            self._train_frozen()
    
    def _train_clip_lora(self):
        """Training loop for CLIP + LoRA."""
        from datasets.dataloaders import get_conceptual_captions_loader
        import warnings
        warnings.filterwarnings("ignore")        
        # Load dataset
        print("Loading Conceptual Captions dataset...")
        train_loader = get_conceptual_captions_loader(self.config, self.model.processor)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        # Track memory before training
        self.metrics.track_gpu_memory('pre_training')
        self.metrics.track_cpu_memory()
        
        # Start training timer
        self.metrics.start_training_timer()
        
        print(f"\nStarting CLIP+LoRA training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model.forward(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                current_loss = loss.item()
                epoch_loss += current_loss
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.config.output_dir / f"epoch_{epoch+1}"
            self.model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # End training timer
        self.metrics.end_training_timer()
        
        # Track memory after training
        self.metrics.track_gpu_memory('post_training')
        
        # Track final performance
        self.metrics.track_performance(accuracy=0.0, loss=avg_loss)
    
    def _train_frozen(self):
        """Training loop for Frozen model."""
        from datasets.dataloaders import get_frozen_dataset_loader
        
        # Load datasets
        print("Loading Conceptual Captions dataset...")
        train_loader, val_loader = get_frozen_dataset_loader(self.config)
        
        # Optimizer (only vision encoder parameters)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if self.config.fp16 else None
        
        # Track memory before training
        self.metrics.track_gpu_memory('pre_training')
        self.metrics.track_cpu_memory()
        
        # Start training timer
        self.metrics.start_training_timer()
        
        print(f"\nStarting Frozen training for {self.config.num_epochs} epochs...")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )
            
            for batch_idx, batch in enumerate(pbar):
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
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimization step
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Update metrics
                current_loss = loss.item() * self.config.gradient_accumulation_steps
                epoch_loss += current_loss
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                global_step += 1
            
            # Validation
            val_loss = self._validate_frozen(val_loader)
            avg_train_loss = epoch_loss / len(train_loader)
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.config.checkpoint_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.vision_encoder.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")
        
        # End training timer
        self.metrics.end_training_timer()
        
        # Track memory after training
        self.metrics.track_gpu_memory('post_training')
        
        # Track final performance
        self.metrics.track_performance(accuracy=0.0, loss=best_val_loss)
    
    def _validate_frozen(self, val_loader):
        """Validation loop for Frozen model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
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
