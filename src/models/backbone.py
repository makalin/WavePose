"""
WiFi Backbone Network

Feature extraction backbone for WiFi-based pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class WiFiBackbone(nn.Module):
    """
    WiFi Backbone Network for feature extraction.
    
    Processes the output of the modality translation network
    to extract high-level features for pose estimation.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 num_layers: int = 4,
                 use_residual: bool = True,
                 use_attention: bool = True):
        """
        Initialize WiFi backbone.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Build backbone layers
        self.layers = self._build_layers()
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(out_channels)
        
        # Output projection
        self.output_proj = nn.Conv2d(out_channels, out_channels, 1)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
    
    def _build_layers(self) -> nn.ModuleList:
        """Build backbone layers."""
        layers = nn.ModuleList()
        
        # Input layer
        layers.append(nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # Hidden layers
        current_channels = 64
        for i in range(self.num_layers):
            if i == 0:
                # First hidden layer
                layers.append(ResidualBlock(current_channels, 128, stride=2))
                current_channels = 128
            elif i == 1:
                # Second hidden layer
                layers.append(ResidualBlock(current_channels, 256, stride=2))
                current_channels = 256
            elif i == 2:
                # Third hidden layer
                layers.append(ResidualBlock(current_channels, 512, stride=2))
                current_channels = 512
            else:
                # Additional layers
                layers.append(ResidualBlock(current_channels, current_channels))
        
        # Final layer to match output channels
        if current_channels != self.out_channels:
            layers.append(nn.Conv2d(current_channels, self.out_channels, 1))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, out_channels, H', W')
        """
        # Apply backbone layers
        for i, layer in enumerate(self.layers):
            if self.use_residual and isinstance(layer, ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Output projection and normalization
        x = self.output_proj(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps at different layers
        """
        feature_maps = []
        
        # Apply layers and collect features
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.Conv2d, ResidualBlock)):
                x = layer(x)
                feature_maps.append(x)
            else:
                x = layer(x)
        
        return feature_maps


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.shortcut(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature refinement."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = ChannelAttention(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial and channel attention."""
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        spatial_weights = self.spatial_conv(x)
        x = x * spatial_weights
        
        return x


class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        # Average pooling branch
        avg_out = self.shared_mlp(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.shared_mlp(self.max_pool(x))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class WiFiBackboneV2(nn.Module):
    """
    Alternative WiFi backbone with different architecture.
    
    Uses a more lightweight design suitable for real-time applications.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 width_multiplier: float = 1.0):
        """
        Initialize WiFi backbone V2.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            width_multiplier: Width multiplier for model scaling
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width_multiplier = width_multiplier
        
        # Calculate channel dimensions
        base_channels = int(32 * width_multiplier)
        
        # Build lightweight backbone
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions
            DepthwiseSeparableConv(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparableConv(base_channels * 2, base_channels * 4, stride=2),
            DepthwiseSeparableConv(base_channels * 4, base_channels * 8, stride=2),
            
            # Final convolution
            nn.Conv2d(base_channels * 8, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through lightweight backbone."""
        return self.features(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise separable convolution."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
