import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.config = config

    def train_epoch(self, epoch_index):
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_index+1}/{self.config.num_epochs}")
        
        for batch in loop:
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            # CLIP Forward Pass (automatically computes Contrastive Loss)
            outputs = self.model(**batch, return_loss=True)
            loss = outputs.loss
            
            loss.backward()
            self.optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            batch_count += 1
            
            loop.set_postfix(loss=current_loss)
            
        return total_loss / batch_count if batch_count > 0 else 0.0

    def run(self):
        print("\nStarting Training...")
        
        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            save_path = self.config.output_dir / f"epoch-{epoch+1}"
            self.model.save_pretrained(save_path)
            print(f"Saved adapter to {save_path}")
            
        print("Training Complete!")