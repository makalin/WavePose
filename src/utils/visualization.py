"""
Visualization utilities for WavePose.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import cv2


def visualize_pose(keypoints: np.ndarray, densepose: Optional[np.ndarray] = None,
                   detections: Optional[np.ndarray] = None, 
                   image_size: Tuple[int, int] = (256, 256),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize pose estimation results.
    
    Args:
        keypoints: Keypoint coordinates (N, 2) or (N, 3) with confidence
        densepose: DensePose UV maps (optional)
        detections: Bounding box detections (optional)
        image_size: Size of the visualization image
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2 if densepose is not None else 1, 
                             figsize=(12, 6) if densepose is not None else (8, 6))
    
    if densepose is not None and not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Plot keypoints
    ax = axes[0] if isinstance(axes, list) else axes
    ax.set_xlim(0, image_size[1])
    ax.set_ylim(0, image_size[0])
    ax.set_aspect('equal')
    ax.set_title('Pose Estimation Results')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Plot keypoints
    if keypoints is not None and len(keypoints) > 0:
        if keypoints.shape[1] == 3:  # With confidence
            coords = keypoints[:, :2]
            confidences = keypoints[:, 2]
            
            # Plot keypoints with confidence-based colors
            for i, (coord, conf) in enumerate(zip(coords, confidences)):
                if conf > 0.1:  # Only show confident keypoints
                    color = plt.cm.Reds(conf)
                    ax.scatter(coord[0], coord[1], c=[color], s=100, alpha=0.7)
                    ax.annotate(f'KP{i}', (coord[0], coord[1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:  # Just coordinates
            ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=100, alpha=0.7)
            
            # Add keypoint labels
            for i, coord in enumerate(keypoints):
                ax.annotate(f'KP{i}', (coord[0], coord[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot detections
    if detections is not None and len(detections) > 0:
        for i, bbox in enumerate(detections):
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                width = x2 - x1
                height = y2 - y1
                
                rect = plt.Rectangle((x1, y1), width, height, 
                                   fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1, f'Person {i}', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot DensePose if available
    if densepose is not None and len(axes) > 1:
        ax_dp = axes[1]
        ax_dp.set_title('DensePose UV Maps')
        
        if len(densepose.shape) == 4:  # (batch, regions, 2, H, W)
            # Show first few body regions
            num_regions = min(4, densepose.shape[1])
            
            for i in range(num_regions):
                u_coords = densepose[0, i, 0]  # U coordinates
                v_coords = densepose[0, i, 1]  # V coordinates
                
                # Create RGB visualization from UV coordinates
                uv_vis = np.stack([u_coords, v_coords, np.zeros_like(u_coords)], axis=2)
                uv_vis = np.clip(uv_vis, 0, 1)
                
                ax_dp.imshow(uv_vis)
                ax_dp.set_title(f'Region {i}')
                ax_dp.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_csi(csi_data: Dict[str, np.ndarray], 
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize CSI data.
    
    Args:
        csi_data: Dictionary containing CSI data
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot amplitude
    if 'amplitude' in csi_data:
        amplitude = csi_data['amplitude']
        if len(amplitude.shape) == 3:  # (samples, antennas, subcarriers)
            # Average across antennas
            avg_amplitude = np.mean(amplitude, axis=1)
            
            im1 = axes[0].imshow(avg_amplitude.T, aspect='auto', cmap='viridis')
            axes[0].set_title('CSI Amplitude (Averaged across antennas)')
            axes[0].set_xlabel('Time samples')
            axes[0].set_ylabel('Subcarriers')
            plt.colorbar(im1, ax=axes[0])
    
    # Plot phase
    if 'phase' in csi_data:
        phase = csi_data['phase']
        if len(phase.shape) == 3:
            # Average across antennas
            avg_phase = np.mean(phase, axis=1)
            
            im2 = axes[1].imshow(avg_phase.T, aspect='auto', cmap='twilight')
            axes[1].set_title('CSI Phase (Averaged across antennas)')
            axes[1].set_xlabel('Time samples')
            axes[1].set_ylabel('Subcarriers')
            plt.colorbar(im2, ax=axes[1])
    
    # Plot amplitude over time for specific subcarrier
    if 'amplitude' in csi_data:
        amplitude = csi_data['amplitude']
        if len(amplitude.shape) == 3:
            # Plot middle subcarrier over time
            mid_subcarrier = amplitude.shape[2] // 2
            
            for ant in range(min(3, amplitude.shape[1])):  # Plot first 3 antennas
                axes[2].plot(amplitude[:, ant, mid_subcarrier], 
                            label=f'Antenna {ant}', alpha=0.7)
            
            axes[2].set_title(f'Amplitude over time (Subcarrier {mid_subcarrier})')
            axes[2].set_xlabel('Time samples')
            axes[2].set_ylabel('Amplitude')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    
    # Plot phase over time for specific subcarrier
    if 'phase' in csi_data:
        phase = csi_data['phase']
        if len(phase.shape) == 3:
            # Plot middle subcarrier over time
            mid_subcarrier = phase.shape[2] // 2
            
            for ant in range(min(3, phase.shape[1])):  # Plot first 3 antennas
                axes[3].plot(phase[:, ant, mid_subcarrier], 
                            label=f'Antenna {ant}', alpha=0.7)
            
            axes[3].set_title(f'Phase over time (Subcarrier {mid_subcarrier})')
            axes[3].set_xlabel('Time samples')
            axes[3].set_ylabel('Phase (radians)')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CSI visualization saved to {save_path}")
    
    return fig


def create_pose_video(keypoints_sequence: List[np.ndarray], 
                      densepose_sequence: Optional[List[np.ndarray]] = None,
                      output_path: str = 'pose_video.mp4',
                      fps: int = 30) -> None:
    """
    Create a video from pose estimation results.
    
    Args:
        keypoints_sequence: List of keypoint arrays for each frame
        densepose_sequence: List of DensePose arrays for each frame (optional)
        output_path: Path to save the output video
        fps: Frames per second
    """
    if not keypoints_sequence:
        print("No keypoints provided for video creation")
        return
    
    # Get dimensions from first frame
    first_keypoints = keypoints_sequence[0]
    if len(first_keypoints.shape) == 2 and first_keypoints.shape[1] >= 2:
        # Assume keypoints are in [0, 1] range, scale to video dimensions
        video_width, video_height = 640, 480
    else:
        video_width, video_height = 256, 256
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    for frame_idx, keypoints in enumerate(keypoints_sequence):
        # Create frame
        frame = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255
        
        if len(keypoints.shape) == 2 and keypoints.shape[1] >= 2:
            # Scale keypoints to video dimensions
            scaled_keypoints = keypoints[:, :2] * np.array([video_width, video_height])
            
            # Draw keypoints
            for i, coord in enumerate(scaled_keypoints):
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'KP{i}', (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add frame number
        cv2.putText(frame, f'Frame {frame_idx}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Pose video saved to {output_path}")


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot losses
    if 'train_loss' in history and history['train_loss']:
        train_losses = [loss.get('total', 0) if isinstance(loss, dict) else loss 
                       for loss in history['train_loss']]
        axes[0].plot(train_losses, label='Train Loss', color='blue')
        
        if 'val_loss' in history and history['val_loss']:
            val_losses = [loss.get('total', 0) if isinstance(loss, dict) else loss 
                         for loss in history['val_loss']]
            axes[0].plot(val_losses, label='Validation Loss', color='red')
        
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'train_metrics' in history and history['train_metrics']:
        # Extract learning rate if available
        lr_values = []
        for metrics in history['train_metrics']:
            if 'learning_rate' in metrics:
                lr_values.append(metrics['learning_rate'])
        
        if lr_values:
            axes[1].plot(lr_values, color='green')
            axes[1].set_title('Learning Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].grid(True, alpha=0.3)
    
    # Plot keypoint metrics
    if 'train_metrics' in history and history['train_metrics']:
        pck_values = []
        for metrics in history['train_metrics']:
            if 'pck' in metrics:
                pck_values.append(metrics['pck'])
        
        if pck_values:
            axes[2].plot(pck_values, label='Train PCK', color='blue')
            
            if 'val_metrics' in history and history['val_metrics']:
                val_pck_values = []
                for metrics in history['val_metrics']:
                    if 'pck' in metrics:
                        val_pck_values.append(metrics['pck'])
                
                if val_pck_values:
                    axes[2].plot(val_pck_values, label='Validation PCK', color='red')
            
            axes[2].set_title('Keypoint PCK')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('PCK')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    
    # Plot DensePose metrics
    if 'train_metrics' in history and history['train_metrics']:
        uv_acc_values = []
        for metrics in history['train_metrics']:
            if 'uv_accuracy' in metrics:
                uv_acc_values.append(metrics['uv_accuracy'])
        
        if uv_acc_values:
            axes[3].plot(uv_acc_values, label='Train UV Accuracy', color='blue')
            
            if 'val_metrics' in history and history['val_metrics']:
                val_uv_acc_values = []
                for metrics in history['val_metrics']:
                    if 'uv_accuracy' in metrics:
                        val_uv_acc_values.append(metrics['uv_accuracy'])
                
                if val_uv_acc_values:
                    axes[3].plot(val_uv_acc_values, label='Validation UV Accuracy', color='red')
            
            axes[3].set_title('DensePose UV Accuracy')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('UV Accuracy')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig
