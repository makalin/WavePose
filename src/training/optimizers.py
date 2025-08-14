"""
Optimizers and learning rate schedulers for training.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts, OneCycleLR
)
from typing import Dict, Any, Optional


def get_optimizer(parameters, optimizer_type: str = 'adam', 
                  learning_rate: float = 1e-4, weight_decay: float = 1e-4,
                  **kwargs) -> torch.optim.Optimizer:
    """
    Get optimizer for training.
    
    Args:
        parameters: Model parameters
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw', 'rmsprop')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            **kwargs
        )
    
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            **kwargs
        )
    
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
            **kwargs
        )
    
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
            eps=1e-8,
            **kwargs
        )
    
    elif optimizer_type.lower() == 'adagrad':
        return optim.Adagrad(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = 'cosine',
                 total_steps: Optional[int] = None, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'cosine_warm_restarts', 'onecycle')
        total_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler instance
    """
    if scheduler_type.lower() == 'cosine':
        if total_steps:
            return CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
        else:
            return CosineAnnealingLR(optimizer, T_max=100, **kwargs)
    
    elif scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type.lower() == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        verbose = kwargs.get('verbose', True)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
        )
    
    elif scheduler_type.lower() == 'cosine_warm_restarts':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    
    elif scheduler_type.lower() == 'onecycle':
        if not total_steps:
            raise ValueError("total_steps is required for OneCycleLR scheduler")
        
        max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
        pct_start = kwargs.get('pct_start', 0.3)
        anneal_strategy = kwargs.get('anneal_strategy', 'cos')
        
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            **kwargs
        )
    
    elif scheduler_type.lower() == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupScheduler:
    """Learning rate scheduler with warmup."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 base_lr: float, target_lr: float):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            base_lr: Base learning rate (starting from this)
            target_lr: Target learning rate (warmup to this)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_step = 0
        
        # Set initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
    
    def step(self):
        """Update learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr + (self.target_lr - self.base_lr) * (self.current_step / self.warmup_steps)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_step += 1
    
    def state_dict(self):
        """Get scheduler state."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr,
            'target_lr': self.target_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lr = state_dict['base_lr']
        self.target_lr = state_dict['target_lr']


class CosineAnnealingWarmupRestarts:
    """Cosine annealing scheduler with warmup and restarts."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, first_cycle_steps: int,
                 cycle_mult: float = 1., max_lr: float = 10., min_lr: float = 1e-6,
                 warmup_steps: int = 0, gamma: float = 1., last_epoch: int = -1):
        """
        Initialize cosine annealing scheduler with warmup and restarts.
        
        Args:
            optimizer: Optimizer instance
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Factor to multiply cycle length after each restart
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: Factor to multiply learning rate after each restart
            last_epoch: Last epoch number
        """
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step()
        
        self.last_epoch = last_epoch
    
    def step(self, epoch=None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_epoch()
        else:
            self.step_epoch(epoch)
    
    def step_epoch(self, epoch=None):
        """Update learning rate for a specific epoch."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.step()
    
    def step(self):
        """Update learning rate for current step."""
        self.last_epoch += 1
        
        if self.last_epoch >= self.cur_cycle_steps:
            self.cycle += 1
            self.last_epoch = 0
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.last_epoch / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def create_optimizer_and_scheduler(model: torch.nn.Module, config: Dict[str, Any]):
    """
    Create optimizer and scheduler from configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Extract configuration
    optimizer_config = config.get('optimizer', {})
    scheduler_config = config.get('scheduler', {})
    
    # Create optimizer
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type=optimizer_config.get('type', 'adam'),
        learning_rate=optimizer_config.get('learning_rate', 1e-4),
        weight_decay=optimizer_config.get('weight_decay', 1e-4),
        **optimizer_config.get('kwargs', {})
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=scheduler_config.get('type', 'cosine'),
        total_steps=scheduler_config.get('total_steps'),
        **scheduler_config.get('kwargs', {})
    )
    
    return optimizer, scheduler
