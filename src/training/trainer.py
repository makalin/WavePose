"""
WavePose Trainer

Main training loop for WiFi-based pose estimation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json
import time
from pathlib import Path

from ..models import WiFiDensePoseRCNN
from ..data import PoseDataset
from .losses import PoseLoss
from .metrics import PoseMetrics
from .optimizers import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)


class WavePoseTrainer:
    """
    Trainer for WavePose WiFi-based pose estimation model.
    
    Handles training, validation, and model checkpointing.
    """
    
    def __init__(self,
                 model: WiFiDensePoseRCNN,
                 train_dataset: PoseDataset,
                 val_dataset: Optional[PoseDataset] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize WavePose trainer.
        
        Args:
            model: WavePose model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Default configuration
        self.config = {
            'batch_size': 16,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'save_dir': './checkpoints',
            'log_dir': './logs',
            'save_freq': 10,
            'val_freq': 5,
            'gradient_clip': 1.0,
            'mixed_precision': True,
            'early_stopping_patience': 20
        }
        
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
        
        # Setup loss function
        self.criterion = PoseLoss()
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            self.model.parameters(),
            self.config['optimizer'],
            self.config['learning_rate'],
            self.config['weight_decay']
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            self.config['scheduler'],
            len(self.train_loader) * self.config['num_epochs']
        )
        
        # Setup metrics
        self.metrics = PoseMetrics()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
    
    def setup_logging(self):
        """Setup logging and tensorboard."""
        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch)
                    loss = self.criterion(predictions, batch)
            else:
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record losses and metrics
            epoch_losses.append({k: v.item() for k, v in loss.items()})
            
            # Calculate metrics
            batch_metrics = self.metrics.calculate_batch(predictions, batch)
            epoch_metrics.append(batch_metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Aggregate epoch results
        epoch_loss = self._aggregate_losses(epoch_losses)
        epoch_metrics = self._aggregate_metrics(epoch_metrics)
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate for one epoch."""
        if not self.val_loader:
            return {}, {}
        
        self.model.eval()
        
        epoch_losses = []
        epoch_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch)
                
                # Record losses and metrics
                epoch_losses.append({k: v.item() for k, v in loss.items()})
                
                # Calculate metrics
                batch_metrics = self.metrics.calculate_batch(predictions, batch)
                epoch_metrics.append(batch_metrics)
        
        # Aggregate epoch results
        epoch_loss = self._aggregate_losses(epoch_losses)
        epoch_metrics = self._aggregate_metrics(epoch_metrics)
        
        return epoch_loss, epoch_metrics
    
    def _aggregate_losses(self, losses: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate losses across batches."""
        if not losses:
            return {}
        
        aggregated = {}
        for key in losses[0].keys():
            aggregated[key] = np.mean([loss[key] for loss in losses])
        
        return aggregated
    
    def _aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across batches."""
        if not metrics:
            return {}
        
        aggregated = {}
        for key in metrics[0].keys():
            aggregated[key] = np.mean([metric[key] for metric in metrics])
        
        return aggregated
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Number of training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Number of validation samples: {len(self.val_dataset)}")
        
        # Training history
        history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self.train_epoch()
            
            # Validation phase
            if self.val_dataset and (epoch + 1) % self.config['val_freq'] == 0:
                val_loss, val_metrics = self.validate_epoch()
            else:
                val_loss, val_metrics = {}, {}
            
            # Log results
            self._log_epoch_results(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            if val_loss:
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Early stopping check
            if val_loss and self._should_stop_early(val_loss['total']):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(self.current_epoch, train_loss, val_loss, is_final=True)
        
        # Training summary
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def _log_epoch_results(self, epoch: int, train_loss: Dict[str, float], 
                          train_metrics: Dict[str, float], val_loss: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """Log epoch results to tensorboard and console."""
        # Log to tensorboard
        for key, value in train_loss.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        if val_loss:
            for key, value in val_loss.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        if val_metrics:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log learning rate
        self.writer.add_scalar('train/learning_rate', 
                              self.optimizer.param_groups[0]['lr'], epoch)
        
        # Console logging
        log_msg = f"Epoch {epoch + 1:3d} - "
        log_msg += f"Train Loss: {train_loss.get('total', 0):.4f} "
        
        if val_loss:
            log_msg += f"Val Loss: {val_loss.get('total', 0):.4f} "
        
        logger.info(log_msg)
    
    def save_checkpoint(self, epoch: int, train_loss: Dict[str, float], 
                       val_loss: Dict[str, float], is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss and val_loss['total'] < self.best_val_loss:
            self.best_val_loss = val_loss['total']
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        if is_final:
            final_path = os.path.join(self.config['save_dir'], 'final_model.pth')
            torch.save(checkpoint, final_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch + 1}")
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config['early_stopping_patience']
    
    def close(self):
        """Clean up resources."""
        self.writer.close()
