#!/usr/bin/env python3
"""
WavePose Inference Script

Script for running inference with trained WavePose models.
"""

import argparse
import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import WiFiDensePoseRCNN
from src.data import CSIDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run WavePose inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to model configuration file')
    
    # Input arguments
    parser.add_argument('--csi_file', type=str, required=True,
                       help='Path to CSI data file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    # Inference arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save inference results')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = 'auto') -> WiFiDensePoseRCNN:
    """Load trained WavePose model from checkpoint."""
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = WiFiDensePoseRCNN(
        num_classes=2,
        num_keypoints=17,
        num_body_regions=24
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model


def load_csi_data(csi_file: str, device: torch.device) -> dict:
    """Load and preprocess CSI data."""
    # Create CSI loader
    csi_loader = CSIDataLoader(
        data_dir=os.path.dirname(csi_file),
        normalize=True
    )
    
    # Load CSI data
    csi_data = csi_loader.load_csi_data(csi_file)
    csi_data = csi_loader.preprocess_csi(csi_data)
    
    # Convert to tensors and move to device
    csi_tensors = {}
    for key, value in csi_data.items():
        if isinstance(value, np.ndarray):
            csi_tensors[key] = torch.from_numpy(value).float().to(device)
    
    return csi_tensors


def run_inference(model: WiFiDensePoseRCNN, csi_data: dict) -> dict:
    """Run inference with the model."""
    with torch.no_grad():
        predictions = model.inference(csi_data)
    
    return predictions


def visualize_results(predictions: dict, output_dir: str):
    """Generate visualizations of inference results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize keypoints
    if 'keypoints' in predictions:
        keypoints = predictions['keypoints']
        if 'coordinates' in keypoints:
            coords = keypoints['coordinates'][0].cpu().numpy()  # First batch
            confidences = keypoints['confidences'][0].cpu().numpy()
            
            # Create keypoint visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot keypoints
            for i, (coord, conf) in enumerate(zip(coords, confidences)):
                if conf > 0.1:  # Only show confident keypoints
                    x, y = coord
                    ax.scatter(x, y, c='red', s=100, alpha=0.7)
                    ax.annotate(f'KP{i}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title('Keypoint Predictions')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            
            # Save plot
            keypoint_plot_path = os.path.join(output_dir, 'keypoints.png')
            plt.savefig(keypoint_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Keypoint visualization saved to {keypoint_plot_path}")
    
    # Visualize DensePose
    if 'densepose' in predictions:
        densepose = predictions['densepose']
        
        if 'uv_coords' in densepose:
            uv_coords = densepose['uv_coords'][0].cpu().numpy()  # First batch
            
            # Create UV coordinate visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            # Show first 8 body regions
            for i in range(min(8, uv_coords.shape[0])):
                u_coords = uv_coords[i, 0]  # U coordinates
                v_coords = uv_coords[i, 1]  # V coordinates
                
                # Create RGB visualization from UV coordinates
                uv_vis = np.stack([u_coords, v_coords, np.zeros_like(u_coords)], axis=2)
                uv_vis = np.clip(uv_vis, 0, 1)
                
                axes[i].imshow(uv_vis)
                axes[i].set_title(f'Region {i}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(8, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            densepose_plot_path = os.path.join(output_dir, 'densepose_uv.png')
            plt.savefig(densepose_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"DensePose visualization saved to {densepose_plot_path}")
    
    # Visualize detections
    if 'detections' in predictions:
        detections = predictions['detections']
        if 'bboxes' in detections and len(detections['bboxes']) > 0:
            bboxes = detections['bboxes'][0].cpu().numpy()  # First batch
            scores = detections['scores'][0].cpu().numpy()
            
            # Create detection visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            for bbox, score in zip(bboxes, scores):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = plt.Rectangle((x1, y1), width, height, 
                                   fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1, f'Person: {score:.3f}', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title('Person Detections')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            
            # Save plot
            detection_plot_path = os.path.join(output_dir, 'detections.png')
            plt.savefig(detection_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Detection visualization saved to {detection_plot_path}")


def save_results(predictions: dict, output_dir: str):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save keypoints
    if 'keypoints' in predictions:
        keypoints = predictions['keypoints']
        keypoint_file = os.path.join(output_dir, 'keypoints.json')
        
        keypoint_data = {
            'coordinates': keypoints['coordinates'].cpu().numpy().tolist(),
            'confidences': keypoints['confidences'].cpu().numpy().tolist()
        }
        
        with open(keypoint_file, 'w') as f:
            json.dump(keypoint_data, f, indent=2)
        print(f"Keypoint results saved to {keypoint_file}")
    
    # Save DensePose
    if 'densepose' in predictions:
        densepose = predictions['densepose']
        densepose_file = os.path.join(output_dir, 'densepose.json')
        
        densepose_data = {
            'uv_coords': densepose['uv_coords'].cpu().numpy().tolist()
        }
        
        if 'body_regions' in densepose:
            densepose_data['body_regions'] = densepose['body_regions'].cpu().numpy().tolist()
        
        with open(densepose_file, 'w') as f:
            json.dump(densepose_data, f, indent=2)
        print(f"DensePose results saved to {densepose_file}")
    
    # Save detections
    if 'detections' in predictions:
        detections = predictions['detections']
        detection_file = os.path.join(output_dir, 'detections.json')
        
        detection_data = {
            'bboxes': detections['bboxes'].cpu().numpy().tolist(),
            'scores': detections['scores'].cpu().numpy().tolist(),
            'labels': detections['labels'].cpu().numpy().tolist()
        }
        
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f, indent=2)
        print(f"Detection results saved to {detection_file}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.checkpoint, device)
    
    # Load CSI data
    logger.info("Loading CSI data...")
    csi_data = load_csi_data(args.csi_file, device)
    
    # Run inference
    logger.info("Running inference...")
    predictions = run_inference(model, csi_data)
    
    # Save results
    if args.save_results:
        logger.info("Saving results...")
        save_results(predictions, args.output_dir)
    
    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_results(predictions, args.output_dir)
    
    logger.info("Inference completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    
    if 'keypoints' in predictions:
        keypoints = predictions['keypoints']
        num_detected = (keypoints['confidences'] > 0.1).sum().item()
        print(f"Keypoints detected: {num_detected}/17")
    
    if 'detections' in predictions:
        detections = predictions['detections']
        num_persons = len(detections['bboxes'])
        print(f"Persons detected: {num_persons}")
    
    if 'densepose' in predictions:
        print("DensePose UV maps generated")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
