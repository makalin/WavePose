"""
Data utility functions for WavePose.
"""

import numpy as np
import torch
import h5py
import json
import os
from typing import Dict, List, Tuple, Optional, Union


def create_sample_data(output_dir: str = './sample_data', 
                      num_samples: int = 100,
                      num_antennas: int = 3,
                      num_subcarriers: int = 64) -> None:
    """
    Create sample CSI and pose data for testing.
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of samples to create
        num_antennas: Number of WiFi antennas
        num_subcarriers: Number of OFDM subcarriers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_samples} sample data files...")
    
    for i in range(num_samples):
        # Create sample CSI data
        csi_data = create_sample_csi_data(num_antennas, num_subcarriers)
        
        # Create sample pose annotations
        pose_annotations = create_sample_pose_annotations()
        
        # Save CSI data
        csi_file = os.path.join(output_dir, f'sample_csi_{i:04d}.h5')
        save_csi_data(csi_data, csi_file)
        
        # Save pose annotations
        pose_file = os.path.join(output_dir, f'sample_pose_{i:04d}.json')
        save_pose_annotations(pose_annotations, pose_file, csi_file)
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{num_samples} samples")
    
    print(f"Sample data created in {output_dir}")


def create_sample_csi_data(num_antennas: int = 3, 
                          num_subcarriers: int = 64) -> Dict[str, np.ndarray]:
    """
    Create sample CSI data.
    
    Args:
        num_antennas: Number of WiFi antennas
        num_subcarriers: Number of OFDM subcarriers
        
    Returns:
        Dictionary containing sample CSI data
    """
    # Generate realistic CSI data
    num_samples = np.random.randint(50, 200)
    
    # Amplitude data (log-normal distribution for realistic WiFi signals)
    amplitude = np.random.lognormal(mean=0, sigma=1, 
                                   size=(num_samples, num_antennas, num_subcarriers))
    
    # Phase data (uniform distribution with some correlation)
    phase = np.random.uniform(-np.pi, np.pi, 
                             size=(num_samples, num_antennas, num_subcarriers))
    
    # Add some temporal correlation
    for ant in range(num_antennas):
        for sub in range(num_subcarriers):
            # Add low-pass filter effect
            phase[:, ant, sub] = np.convolve(phase[:, ant, sub], 
                                            np.ones(5)/5, mode='same')
    
    # Timestamps
    timestamps = np.arange(num_samples) / 1000.0  # 1kHz sampling rate
    
    # Metadata
    metadata = {
        'source': 'sample_data',
        'num_antennas': num_antennas,
        'num_subcarriers': num_subcarriers,
        'sample_rate': 1000,
        'frequency_band': '5GHz',
        'bandwidth': 80
    }
    
    return {
        'amplitude': amplitude,
        'phase': phase,
        'timestamp': timestamps,
        'metadata': metadata
    }


def create_sample_pose_annotations() -> Dict:
    """
    Create sample pose annotations.
    
    Returns:
        Dictionary containing sample pose annotations
    """
    # Generate random keypoints (17 COCO keypoints)
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    num_keypoints = len(keypoint_names)
    keypoints = []
    
    for i in range(num_keypoints):
        # Generate random coordinates in [0, 1] range
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        
        # Generate random visibility (0: invisible, 1: visible, 2: occluded)
        visibility = np.random.choice([0, 1, 2], p=[0.1, 0.7, 0.2])
        
        keypoints.extend([x, y, visibility])
    
    # Generate random bounding box
    x1 = np.random.uniform(0.1, 0.6)
    y1 = np.random.uniform(0.1, 0.6)
    x2 = x1 + np.random.uniform(0.2, 0.4)
    y2 = y1 + np.random.uniform(0.3, 0.6)
    
    # Generate DensePose annotations (simplified)
    densepose = {
        'uv_coords': np.random.rand(24, 2, 256, 256),  # 24 body regions
        'body_regions': np.random.randint(0, 24, size=(256, 256))
    }
    
    # Generate segmentation mask
    segmentation = np.random.randint(0, 2, size=(256, 256))
    
    return {
        'keypoints': keypoints,
        'bbox': [x1, y1, x2, y2],
        'densepose': densepose,
        'segmentation': 'sample_segmentation.png',
        'person_id': 0,
        'timestamp': np.random.uniform(0, 100),
        'csi_file': 'sample_csi_0000.h5'
    }


def save_csi_data(csi_data: Dict[str, np.ndarray], file_path: str) -> None:
    """
    Save CSI data to HDF5 file.
    
    Args:
        csi_data: CSI data dictionary
        file_path: Path to save the file
    """
    with h5py.File(file_path, 'w') as f:
        # Save main data
        for key, value in csi_data.items():
            if key == 'metadata':
                # Save metadata as attributes
                for meta_key, meta_value in value.items():
                    f.attrs[meta_key] = meta_value
            else:
                f.create_dataset(key, data=value, compression='gzip')


def save_pose_annotations(annotations: Dict, file_path: str, 
                         csi_file: str) -> None:
    """
    Save pose annotations to JSON file.
    
    Args:
        annotations: Pose annotations dictionary
        file_path: Path to save the file
        csi_file: Associated CSI file path
    """
    # Update CSI file reference
    annotations['csi_file'] = os.path.basename(csi_file)
    
    with open(file_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def load_sample_data(data_dir: str = './sample_data') -> Tuple[List[str], List[str]]:
    """
    Load sample data files.
    
    Args:
        data_dir: Directory containing sample data
        
    Returns:
        Tuple of (csi_files, pose_files)
    """
    csi_files = []
    pose_files = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            csi_files.append(os.path.join(data_dir, file))
        elif file.endswith('.json'):
            pose_files.append(os.path.join(data_dir, file))
    
    csi_files.sort()
    pose_files.sort()
    
    return csi_files, pose_files


def create_training_dataset(csi_files: List[str], 
                           pose_files: List[str],
                           output_dir: str = './training_data') -> None:
    """
    Create a training dataset from sample data.
    
    Args:
        csi_files: List of CSI file paths
        pose_files: List of pose annotation file paths
        output_dir: Output directory for training data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train/val split
    num_samples = len(csi_files)
    val_split = 0.2
    val_size = int(num_samples * val_split)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create training data
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy training data
    for idx in train_indices:
        csi_src = csi_files[idx]
        pose_src = pose_files[idx]
        
        csi_dst = os.path.join(train_dir, os.path.basename(csi_src))
        pose_dst = os.path.join(train_dir, os.path.basename(pose_src))
        
        # Copy files
        import shutil
        shutil.copy2(csi_src, csi_dst)
        shutil.copy2(pose_src, pose_dst)
    
    # Copy validation data
    for idx in val_indices:
        csi_src = csi_files[idx]
        pose_src = pose_files[idx]
        
        csi_dst = os.path.join(val_dir, os.path.basename(csi_src))
        pose_dst = os.path.join(val_dir, os.path.basename(pose_src))
        
        # Copy files
        import shutil
        shutil.copy2(csi_src, csi_dst)
        shutil.copy2(pose_src, pose_dst)
    
    print(f"Training dataset created:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Output directory: {output_dir}")


def validate_data_format(data_dir: str) -> Dict[str, Union[bool, str]]:
    """
    Validate the format of data files.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'file_count': 0
    }
    
    csi_files = []
    pose_files = []
    
    # Count files
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            csi_files.append(file)
        elif file.endswith('.json'):
            pose_files.append(file)
    
    results['file_count'] = len(csi_files) + len(pose_files)
    
    # Validate CSI files
    for csi_file in csi_files[:5]:  # Check first 5 files
        try:
            with h5py.File(os.path.join(data_dir, csi_file), 'r') as f:
                required_keys = ['amplitude', 'phase']
                for key in required_keys:
                    if key not in f:
                        results['errors'].append(f"Missing {key} in {csi_file}")
                        results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Error reading {csi_file}: {e}")
            results['valid'] = False
    
    # Validate pose files
    for pose_file in pose_files[:5]:  # Check first 5 files
        try:
            with open(os.path.join(data_dir, pose_file), 'r') as f:
                data = json.load(f)
                required_keys = ['keypoints', 'bbox']
                for key in required_keys:
                    if key not in data:
                        results['errors'].append(f"Missing {key} in {pose_file}")
                        results['valid'] = False
        except Exception as e:
            results['errors'].append(f"Error reading {pose_file}: {e}")
            results['valid'] = False
    
    # Check for matching files
    if len(csi_files) != len(pose_files):
        results['warnings'].append("Number of CSI and pose files don't match")
    
    return results


def print_data_summary(data_dir: str) -> None:
    """
    Print a summary of the data directory.
    
    Args:
        data_dir: Directory to summarize
    """
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return
    
    csi_files = []
    pose_files = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.h5'):
            csi_files.append(file)
        elif file.endswith('.json'):
            pose_files.append(file)
    
    print(f"Data Directory Summary: {data_dir}")
    print(f"  CSI files: {len(csi_files)}")
    print(f"  Pose files: {len(pose_files)}")
    print(f"  Total files: {len(csi_files) + len(pose_files)}")
    
    if csi_files:
        print(f"  CSI file example: {csi_files[0]}")
    if pose_files:
        print(f"  Pose file example: {pose_files[0]}")
    
    # Validate data format
    validation = validate_data_format(data_dir)
    if validation['valid']:
        print("  Data format: ✓ Valid")
    else:
        print("  Data format: ✗ Invalid")
        for error in validation['errors']:
            print(f"    Error: {error}")
    
    for warning in validation['warnings']:
        print(f"    Warning: {warning}")
