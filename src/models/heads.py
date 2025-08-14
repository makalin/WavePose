"""
Pose Estimation Heads

Keypoint and DensePose heads for WiFi-based pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class KeypointHead(nn.Module):
    """
    Keypoint detection head for human pose estimation.
    
    Predicts keypoint locations using heatmap regression.
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_keypoints: int = 17,
                 heatmap_size: Tuple[int, int] = (64, 64),
                 use_attention: bool = True,
                 use_offset: bool = True):
        """
        Initialize keypoint head.
        
        Args:
            in_channels: Number of input channels
            num_keypoints: Number of keypoints to predict
            heatmap_size: Size of output heatmaps (H, W)
            use_attention: Whether to use attention mechanism
            use_offset: Whether to predict offset maps
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.use_attention = use_attention
        self.use_offset = use_offset
        
        # Keypoint heatmap prediction
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_keypoints, 1)
        )
        
        # Offset prediction (for sub-pixel accuracy)
        if use_offset:
            self.offset_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, num_keypoints * 2, 1)  # x, y offsets
            )
        
        # Attention mechanism
        if use_attention:
            self.keypoint_attention = KeypointAttention(in_channels, num_keypoints)
        
        # Upsampling to target heatmap size
        self.upsample = nn.Upsample(size=heatmap_size, mode='bilinear', align_corners=False)
        
        # Final activation for heatmaps
        self.heatmap_activation = nn.Sigmoid()
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through keypoint head.
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            Dictionary containing keypoint predictions:
                - 'heatmaps': Keypoint heatmaps (B, num_keypoints, H, W)
                - 'offsets': Offset maps (B, num_keypoints*2, H, W) if enabled
        """
        batch_size = features.shape[0]
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.keypoint_attention(features)
        
        # Predict heatmaps
        heatmaps = self.heatmap_conv(features)
        heatmaps = self.upsample(heatmaps)
        heatmaps = self.heatmap_activation(heatmaps)
        
        # Predict offsets if enabled
        offsets = None
        if self.use_offset:
            offsets = self.offset_conv(features)
            offsets = self.upsample(offsets)
            # Reshape to (B, num_keypoints, 2, H, W)
            offsets = offsets.view(batch_size, self.num_keypoints, 2, 
                                 self.heatmap_size[0], self.heatmap_size[1])
        
        outputs = {'heatmaps': heatmaps}
        if offsets is not None:
            outputs['offsets'] = offsets
        
        return outputs
    
    def get_keypoint_locations(self, heatmaps: torch.Tensor, 
                              offsets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract keypoint locations from heatmaps and offsets.
        
        Args:
            heatmaps: Keypoint heatmaps (B, num_keypoints, H, W)
            offsets: Offset maps (B, num_keypoints, 2, H, W)
            
        Returns:
            Dictionary containing keypoint locations and confidences
        """
        batch_size, num_keypoints, height, width = heatmaps.shape
        
        # Find maximum locations in heatmaps
        heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
        
        # Convert to 2D coordinates
        y_coords = max_indices // width
        x_coords = max_indices % width
        
        # Convert to float coordinates
        x_coords = x_coords.float()
        y_coords = y_coords.float()
        
        # Apply offsets if available
        if offsets is not None:
            # Extract offsets at maximum locations
            batch_indices = torch.arange(batch_size, device=heatmaps.device)
            keypoint_indices = torch.arange(num_keypoints, device=heatmaps.device)
            
            # Get offsets at max locations
            x_offsets = offsets[batch_indices, keypoint_indices, 0, y_coords, x_coords]
            y_offsets = offsets[batch_indices, keypoint_indices, 1, y_coords, x_coords]
            
            # Apply offsets
            x_coords = x_coords + x_offsets
            y_coords = y_coords + y_offsets
        
        # Normalize coordinates to [0, 1]
        x_coords = x_coords / (width - 1)
        y_coords = y_coords / (height - 1)
        
        # Stack coordinates
        coordinates = torch.stack([x_coords, y_coords], dim=2)
        
        return {
            'coordinates': coordinates,  # (B, num_keypoints, 2)
            'confidences': max_vals,    # (B, num_keypoints)
            'heatmaps': heatmaps        # (B, num_keypoints, H, W)
        }


class DensePoseHead(nn.Module):
    """
    DensePose head for dense human pose estimation.
    
    Predicts UV coordinates and body region segmentation for each pixel.
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_body_regions: int = 24,
                 uv_size: Tuple[int, int] = (256, 256),
                 use_attention: bool = True,
                 use_region_classification: bool = True):
        """
        Initialize DensePose head.
        
        Args:
            in_channels: Number of input channels
            num_body_regions: Number of body regions
            uv_size: Size of UV coordinate maps (H, W)
            use_attention: Whether to use attention mechanism
            use_region_classification: Whether to predict body regions
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_body_regions = num_body_regions
        self.uv_size = uv_size
        self.use_attention = use_attention
        self.use_region_classification = use_region_classification
        
        # UV coordinate prediction (U and V for each body region)
        self.uv_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_body_regions * 2, 1)  # U and V for each region
        )
        
        # Body region classification
        if use_region_classification:
            self.region_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, num_body_regions, 1)
            )
        
        # Attention mechanism
        if use_attention:
            self.densepose_attention = DensePoseAttention(in_channels, num_body_regions)
        
        # Upsampling to target UV size
        self.upsample = nn.Upsample(size=uv_size, mode='bilinear', align_corners=False)
        
        # Final activation for UV coordinates
        self.uv_activation = nn.Sigmoid()
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DensePose head.
        
        Args:
            features: Input features (B, C, H, W)
            
        Returns:
            Dictionary containing DensePose predictions:
                - 'uv_coords': UV coordinate maps (B, num_regions*2, H, W)
                - 'body_regions': Body region classification (B, num_regions, H, W)
        """
        batch_size = features.shape[0]
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.densepose_attention(features)
        
        # Predict UV coordinates
        uv_coords = self.uv_conv(features)
        uv_coords = self.upsample(uv_coords)
        uv_coords = self.uv_activation(uv_coords)
        
        # Reshape UV coordinates to (B, num_regions, 2, H, W)
        uv_coords = uv_coords.view(batch_size, self.num_body_regions, 2, 
                                  self.uv_size[0], self.uv_size[1])
        
        # Predict body regions if enabled
        body_regions = None
        if self.use_region_classification:
            body_regions = self.region_conv(features)
            body_regions = self.upsample(body_regions)
            # Apply softmax across body regions
            body_regions = F.softmax(body_regions, dim=1)
        
        outputs = {'uv_coords': uv_coords}
        if body_regions is not None:
            outputs['body_regions'] = body_regions
        
        return outputs
    
    def get_densepose_maps(self, uv_coords: torch.Tensor, 
                           body_regions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process DensePose outputs for visualization and analysis.
        
        Args:
            uv_coords: UV coordinate maps (B, num_regions, 2, H, W)
            body_regions: Body region classification (B, num_regions, H, W)
            
        Returns:
            Dictionary containing processed DensePose maps
        """
        batch_size, num_regions, _, height, width = uv_coords.shape
        
        # Extract U and V coordinates
        u_coords = uv_coords[:, :, 0, :, :]  # (B, num_regions, H, W)
        v_coords = uv_coords[:, :, 1, :, :]  # (B, num_regions, H, W)
        
        # Find dominant body region for each pixel
        if body_regions is not None:
            # Get region with highest probability
            region_indices = torch.argmax(body_regions, dim=1)  # (B, H, W)
            
            # Create region-specific UV maps
            region_uv_maps = torch.zeros(batch_size, 2, height, width, device=uv_coords.device)
            
            for b in range(batch_size):
                for r in range(num_regions):
                    region_mask = (region_indices[b] == r)
                    region_uv_maps[b, 0][region_mask] = u_coords[b, r][region_mask]
                    region_uv_maps[b, 1][region_mask] = v_coords[b, r][region_mask]
        else:
            # Use first region as default
            region_uv_maps = uv_coords[:, 0, :, :, :]  # (B, 2, H, W)
            region_indices = torch.zeros(batch_size, height, width, dtype=torch.long, device=uv_coords.device)
        
        return {
            'uv_maps': region_uv_maps,           # (B, 2, H, W)
            'u_coords': u_coords,                # (B, num_regions, H, W)
            'v_coords': v_coords,                # (B, num_regions, H, W)
            'body_regions': body_regions,        # (B, num_regions, H, W)
            'region_indices': region_indices     # (B, H, W)
        }


class KeypointAttention(nn.Module):
    """Attention mechanism for keypoint detection."""
    
    def __init__(self, channels: int, num_keypoints: int):
        super().__init__()
        
        self.channels = channels
        self.num_keypoints = num_keypoints
        
        # Keypoint-specific attention
        self.keypoint_conv = nn.Conv2d(channels, num_keypoints, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply keypoint attention."""
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Keypoint-specific attention
        keypoint_weights = torch.sigmoid(self.keypoint_conv(x))
        keypoint_weights = keypoint_weights.mean(dim=1, keepdim=True)
        
        return x * keypoint_weights


class DensePoseAttention(nn.Module):
    """Attention mechanism for DensePose estimation."""
    
    def __init__(self, channels: int, num_body_regions: int):
        super().__init__()
        
        self.channels = channels
        self.num_body_regions = num_body_regions
        
        # Body region-specific attention
        self.region_conv = nn.Conv2d(channels, num_body_regions, 1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DensePose attention."""
        # Spatial attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        # Region-specific attention
        region_weights = torch.sigmoid(self.region_conv(x))
        region_weights = region_weights.mean(dim=1, keepdim=True)
        
        return x * region_weights


class MultiScaleKeypointHead(nn.Module):
    """
    Multi-scale keypoint head for better accuracy.
    
    Predicts keypoints at multiple scales and combines them.
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_keypoints: int = 17,
                 scales: List[float] = [1.0, 0.5, 0.25],
                 use_attention: bool = True):
        """
        Initialize multi-scale keypoint head.
        
        Args:
            in_channels: Number of input channels
            num_keypoints: Number of keypoints to predict
            scales: List of scales for multi-scale prediction
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.scales = scales
        self.use_attention = use_attention
        
        # Create keypoint heads for each scale
        self.scale_heads = nn.ModuleList([
            KeypointHead(in_channels, num_keypoints, use_attention=use_attention)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Conv2d(len(scales) * num_keypoints, num_keypoints, 1)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-scale keypoint head."""
        scale_outputs = []
        
        # Get predictions at each scale
        for i, (scale, head) in enumerate(zip(self.scales, self.scale_heads)):
            if scale != 1.0:
                # Resize features to scale
                scaled_features = F.interpolate(features, scale_factor=scale, mode='bilinear', align_corners=False)
                scale_output = head(scaled_features)
            else:
                scale_output = head(features)
            
            scale_outputs.append(scale_output['heatmaps'])
        
        # Concatenate scale outputs
        multi_scale_heatmaps = torch.cat(scale_outputs, dim=1)
        
        # Fuse scales
        fused_heatmaps = self.scale_fusion(multi_scale_heatmaps)
        
        return {'heatmaps': fused_heatmaps}
