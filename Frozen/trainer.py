import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import FrozenConfig
from model import FrozenModel
from dataset import ConceptualCaptionsDataset

class FrozenTrainer:
    def __init__(self, config: FrozenConfig):
        self.config = config
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model
        self.model = FrozenModel(config).to(config.device)
        
        # Optimizer (Vision Encoder only)
        self.optimizer = torch.optim.Adam(
            self.model.vision_encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed Precision
        self.scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')

    def get_dataloaders(self):
        # Paths based on config.data_root
        train_path = self.config.data_root / "train.jsonl"
        val_path = self.config.data_root / "validation.jsonl"
        
        train_ds = ConceptualCaptionsDataset(train_path, self.config.data_root, self.tokenizer, self.config)
        val_ds = ConceptualCaptionsDataset(val_path, self.config.data_root, self.tokenizer, self.config)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True
        )
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self.get_dataloaders()
        print(f"Starting training on {self.config.device}...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(pbar):
                loss = self._train_step(batch)
                
                # Update progress bar
                epoch_loss += loss
                pbar.set_postfix({'loss': loss})
                
                # Validation interval
                if self.global_step % 1000 == 0 and self.global_step > 0:
                    self._validate_and_save(val_loader)
                    self.model.train()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            self._validate_and_save(val_loader, epoch=epoch)

    def _train_step(self, batch):
        images = batch['images'].to(self.config.device)
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        
        # Automatic Mixed Precision context
        with torch.amp.autocast('cuda', enabled=self.config.fp16):
            logits, loss = self.model(images, input_ids, attention_mask, labels)
            loss = loss / self.config.gradient_accumulation_steps
            
        # Backward
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Optimization Step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            
        return loss.item() * self.config.gradient_accumulation_steps

    def _validate_and_save(self, val_loader, epoch=None):
        val_loss = self.validate(val_loader)
        print(f" -> Val Loss: {val_loss:.4f}")
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint(val_loss, epoch, is_best=True)
        elif epoch is not None:
             self._save_checkpoint(val_loss, epoch, is_best=False)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch['images'].to(self.config.device)
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                with torch.amp.autocast('cuda', enabled=self.config.fp16):
                    _, loss = self.model(images, input_ids, attention_mask, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def _save_checkpoint(self, val_loss, epoch, is_best=False):
        name = f"best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.config.checkpoint_dir / name
        torch.save({
            'epoch': epoch,
            'step': self.global_step,
            'model_state': self.model.vision_encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
        print(f"Saved checkpoint: {path}")