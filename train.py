#!/usr/bin/env python3
"""
WavePose Training Script

Main script for training the WiFi-based pose estimation model.
"""

import argparse
import os
import sys
import logging
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import WiFiDensePoseRCNN
from src.data import PoseDataset
from src.training import WavePoseTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train WavePose model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing CSI data files')
    parser.add_argument('--annotations_dir', type=str, required=True,
                       help='Directory containing pose annotations')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--num_keypoints', type=int, default=17,
                       help='Number of keypoints to predict')
    parser.add_argument('--num_body_regions', type=int, default=24,
                       help='Number of body regions for DensePose')
    parser.add_argument('--use_fpn', action='store_true', default=True,
                       help='Use Feature Pyramid Network')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--config_file', type=str,
                       help='Path to configuration file')
    
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_datasets(data_dir: str, annotations_dir: str, val_split: float = 0.2):
    """Create training and validation datasets."""
    # Create full dataset
    full_dataset = PoseDataset(
        csi_data_dir=data_dir,
        pose_annotations_dir=annotations_dir,
        target_size=(256, 256),
        num_keypoints=17,
        num_body_regions=24
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    
    return train_dataset, val_dataset


def create_model(args) -> WiFiDensePoseRCNN:
    """Create WavePose model."""
    model = WiFiDensePoseRCNN(
        num_classes=2,  # background + person
        num_keypoints=args.num_keypoints,
        num_body_regions=args.num_body_regions,
        use_fpn=args.use_fpn,
        pretrained=args.pretrained
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = {}
    if args.config_file:
        config = load_config(args.config_file)
        logger.info(f"Loaded configuration from {args.config_file}")
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'device': args.device,
        'num_workers': args.num_workers,
        'mixed_precision': args.mixed_precision,
        'save_dir': args.save_dir,
        'log_dir': args.log_dir
    })
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        args.data_dir, args.annotations_dir, args.val_split
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = WavePoseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        history = trainer.train()
        logger.info("Training completed successfully!")
        
        # Save training history
        history_file = os.path.join(args.save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_file}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
