#!/usr/bin/env python3
"""
WavePose Demo Script

Demonstrates the WiFi-based pose estimation model with sample data.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import WiFiDensePoseRCNN
from src.utils.visualization import visualize_pose, visualize_csi
from src.utils.data_utils import create_sample_data


def create_demo_model():
    """Create a demo WavePose model."""
    print("Creating WavePose model...")
    
    model = WiFiDensePoseRCNN(
        num_classes=2,
        num_keypoints=17,
        num_body_regions=24,
        use_fpn=True,
        pretrained=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_demo_data():
    """Create sample CSI and pose data for demonstration."""
    print("Creating sample data...")
    
    # Create sample CSI data
    batch_size = 2
    num_antennas = 3
    num_subcarriers = 64
    height, width = 256, 256
    
    # Generate realistic CSI data
    csi_data = {
        'amplitude': torch.randn(batch_size, 1, num_antennas, num_subcarriers),
        'phase': torch.randn(batch_size, 1, num_antennas, num_subcarriers) * np.pi
    }
    
    # Create sample pose annotations
    pose_data = {
        'target_keypoints': torch.randn(batch_size, 17, height, width),
        'target_densepose': torch.randn(batch_size, 24, 2, height, width),
        'target_segmentation': torch.randint(0, 2, (batch_size, height, width)).float()
    }
    
    print(f"Sample data created:")
    print(f"  CSI data: {csi_data['amplitude'].shape}")
    print(f"  Keypoint targets: {pose_data['target_keypoints'].shape}")
    print(f"  DensePose targets: {pose_data['target_densepose'].shape}")
    
    return csi_data, pose_data


def run_demo_inference(model, csi_data):
    """Run inference with the demo model."""
    print("Running inference...")
    
    model.eval()
    with torch.no_grad():
        predictions = model.inference(csi_data)
    
    print("Inference completed!")
    print(f"  Detections: {len(predictions['detections']['bboxes'])} persons detected")
    print(f"  Keypoints: {predictions['keypoints']['coordinates'].shape}")
    print(f"  DensePose: {predictions['densepose']['uv_coords'].shape}")
    
    return predictions


def visualize_demo_results(csi_data, predictions, output_dir='./demo_results'):
    """Create visualizations of the demo results."""
    print("Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize CSI data
    csi_fig = visualize_csi(
        {k: v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v 
         for k, v in csi_data.items()},
        save_path=os.path.join(output_dir, 'csi_visualization.png')
    )
    
    # Visualize pose results
    keypoints = predictions['keypoints']['coordinates'][0].cpu().numpy()
    densepose = predictions['densepose']['uv_coords'][0].cpu().numpy()
    detections = predictions['detections']['bboxes'][0].cpu().numpy()
    
    pose_fig = visualize_pose(
        keypoints=keypoints,
        densepose=densepose,
        detections=detections,
        image_size=(256, 256),
        save_path=os.path.join(output_dir, 'pose_visualization.png')
    )
    
    print(f"Visualizations saved to {output_dir}")
    
    return csi_fig, pose_fig


def main():
    """Main demo function."""
    print("=" * 60)
    print("WavePose Demo - WiFi-based Human Pose Estimation")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model
        model = create_demo_model()
        model.to(device)
        
        # Create sample data
        csi_data, pose_data = create_demo_data()
        
        # Move data to device
        csi_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in csi_data.items()}
        
        # Run inference
        predictions = run_demo_inference(model, csi_data)
        
        # Create visualizations
        csi_fig, pose_fig = visualize_demo_results(csi_data, predictions)
        
        # Show results
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("  - ./demo_results/csi_visualization.png")
        print("  - ./demo_results/pose_visualization.png")
        print("\nYou can now:")
        print("  1. Train the model with real data using: python train.py")
        print("  2. Run inference with: python infer.py")
        print("  3. Explore the source code in the src/ directory")
        
        # Display plots
        plt.show()
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
