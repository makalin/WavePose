"""
Modality Translation Network

Converts 1D WiFi CSI signals into 2D image-like features for pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class ModalityTranslationNetwork(nn.Module):
    """
    Modality Translation Network for WiFi CSI to image-like features.
    
    This network converts 1D WiFi Channel State Information (CSI) into
    2D image-like representations that can be processed by standard
    computer vision models.
    """
    
    def __init__(self,
                 input_channels: int = 2,  # amplitude and phase
                 num_antennas: int = 3,
                 num_subcarriers: int = 64,
                 hidden_dims: Tuple[int, ...] = (128, 256, 512),
                 output_size: Tuple[int, int] = (256, 256),
                 use_attention: bool = True,
                 dropout: float = 0.1):
        """
        Initialize the modality translation network.
        
        Args:
            input_channels: Number of input channels (amplitude, phase)
            num_antennas: Number of WiFi antennas
            num_subcarriers: Number of OFDM subcarriers
            hidden_dims: Hidden dimensions for MLP layers
            output_size: Target output size (height, width)
            use_attention: Whether to use attention mechanism
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.hidden_dims = hidden_dims
        self.output_size = output_size
        self.use_attention = use_attention
        
        # Calculate input size
        input_size = input_channels * num_antennas * num_subcarriers
        
        # MLP encoder
        self.mlp_layers = self._build_mlp(input_size, hidden_dims)
        
        # Reshape and upsampling layers
        self.reshape_layer = nn.Linear(hidden_dims[-1], 
                                     output_size[0] * output_size[1])
        
        # Convolutional upsampling
        self.conv_upsample = self._build_conv_upsample(output_size)
        
        # Attention mechanism
        if use_attention:
            self.attention = CrossAttention(hidden_dims[-1], output_size)
        
        # Output projection
        self.output_proj = nn.Conv2d(1, 3, kernel_size=1)  # 3 channels for RGB-like output
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
    
    def _build_mlp(self, input_size: int, hidden_dims: Tuple[int, ...]) -> nn.ModuleList:
        """Build MLP layers."""
        layers = nn.ModuleList()
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        return layers
    
    def _build_conv_upsample(self, target_size: Tuple[int, int]) -> nn.ModuleList:
        """Build convolutional upsampling layers."""
        layers = nn.ModuleList()
        
        # Start with a reasonable intermediate size
        current_size = (16, 16)  # Start with 16x16
        
        while current_size[0] < target_size[0] or current_size[1] < target_size[1]:
            # Upsample by factor of 2
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(1, 1, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(1))
            
            current_size = (current_size[0] * 2, current_size[1] * 2)
        
        # Final adjustment to exact target size
        if current_size != target_size:
            layers.append(nn.Upsample(size=target_size, mode='bilinear', align_corners=False))
        
        return layers
    
    def forward(self, csi_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the modality translation network.
        
        Args:
            csi_data: Dictionary containing CSI data
                - 'amplitude': CSI amplitude data (B, C, H, W)
                - 'phase': CSI phase data (B, C, H, W)
        
        Returns:
            Image-like features (B, 3, H, W)
        """
        batch_size = csi_data['amplitude'].shape[0]
        
        # Extract amplitude and phase
        amplitude = csi_data['amplitude']  # (B, 1, num_antennas, num_subcarriers)
        phase = csi_data['phase']          # (B, 1, num_antennas, num_subcarriers)
        
        # Concatenate along channel dimension
        combined = torch.cat([amplitude, phase], dim=1)  # (B, 2, num_antennas, num_subcarriers)
        
        # Flatten to 1D
        flattened = combined.view(batch_size, -1)  # (B, 2 * num_antennas * num_subcarriers)
        
        # Pass through MLP
        x = flattened
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Reshape to 2D
        x = self.reshape_layer(x)  # (B, H * W)
        x = x.view(batch_size, 1, self.output_size[0], self.output_size[1])
        
        # Apply dropout
        x = self.dropout(x)
        
        # Convolutional upsampling
        for layer in self.conv_upsample:
            x = layer(x)
        
        # Output projection to 3 channels
        output = self.output_proj(x)
        
        return output
    
    def get_feature_maps(self, csi_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            csi_data: Input CSI data
            
        Returns:
            Dictionary containing intermediate features
        """
        batch_size = csi_data['amplitude'].shape[0]
        
        # Extract and combine CSI data
        amplitude = csi_data['amplitude']
        phase = csi_data['phase']
        combined = torch.cat([amplitude, phase], dim=1)
        flattened = combined.view(batch_size, -1)
        
        # MLP features
        mlp_features = []
        x = flattened
        for layer in self.mlp_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                mlp_features.append(x)
        
        # Reshape features
        x = self.reshape_layer(x)
        x = x.view(batch_size, 1, self.output_size[0], self.output_size[1])
        
        # Upsampled features
        upsampled_features = [x]
        for layer in self.conv_upsample:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                upsampled_features.append(x)
        
        return {
            'mlp_features': mlp_features,
            'upsampled_features': upsampled_features,
            'final_output': x
        }


class CrossAttention(nn.Module):
    """Cross-attention mechanism for modality translation."""
    
    def __init__(self, feature_dim: int, spatial_size: Tuple[int, int]):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        
        # Attention parameters
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.scale = feature_dim ** -0.5
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention."""
        # Create spatial queries (learnable spatial positions)
        batch_size = x.shape[0]
        h, w = self.spatial_size
        
        # Generate spatial queries
        spatial_queries = torch.randn(batch_size, h * w, self.feature_dim, device=x.device)
        spatial_queries = spatial_queries.requires_grad_(True)
        
        # Apply attention
        q = self.query(spatial_queries)  # (B, H*W, D)
        k = self.key(x).unsqueeze(1)    # (B, 1, D)
        v = self.value(x).unsqueeze(1)  # (B, 1, D)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_probs, v).squeeze(1)  # (B, H*W, D)
        
        # Output projection and residual connection
        output = self.output_proj(attended)
        output = self.layer_norm(output + spatial_queries)
        
        return output.mean(dim=1)  # (B, D)
