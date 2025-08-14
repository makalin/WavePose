"""
Loss functions for WiFi-based pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PoseLoss(nn.Module):
    """
    Combined loss function for pose estimation.
    
    Combines detection, keypoint, and DensePose losses.
    """
    
    def __init__(self,
                 detection_weight: float = 1.0,
                 keypoint_weight: float = 1.0,
                 densepose_weight: float = 1.0,
                 keypoint_loss_type: str = 'mse',
                 densepose_loss_type: str = 'mse'):
        """
        Initialize pose loss.
        
        Args:
            detection_weight: Weight for detection loss
            keypoint_weight: Weight for keypoint loss
            densepose_weight: Weight for DensePose loss
            keypoint_loss_type: Type of keypoint loss ('mse', 'focal', 'wing')
            densepose_loss_type: Type of DensePose loss ('mse', 'smooth_l1')
        """
        super().__init__()
        
        self.detection_weight = detection_weight
        self.keypoint_weight = keypoint_weight
        self.densepose_weight = densepose_weight
        
        # Create individual loss functions
        self.detection_loss = DetectionLoss()
        self.keypoint_loss = KeypointLoss(loss_type=keypoint_loss_type)
        self.densepose_loss = DensePoseLoss(loss_type=densepose_loss_type)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Detection loss
        if 'detections' in predictions and 'target_detections' in targets:
            detection_loss = self.detection_loss(predictions['detections'], targets['target_detections'])
            losses['detection'] = detection_loss
        
        # Keypoint loss
        if 'keypoints' in predictions and 'target_keypoints' in targets:
            keypoint_loss = self.keypoint_loss(predictions['keypoints'], targets['target_keypoints'])
            losses['keypoint'] = keypoint_loss
        
        # DensePose loss
        if 'densepose' in predictions and 'target_densepose' in targets:
            densepose_loss = self.densepose_loss(predictions['densepose'], targets['target_densepose'])
            losses['densepose'] = densepose_loss
        
        # Calculate total loss
        total_loss = 0.0
        if 'detection' in losses:
            total_loss += self.detection_weight * losses['detection']
        if 'keypoint' in losses:
            total_loss += self.keypoint_weight * losses['keypoint']
        if 'densepose' in losses:
            total_loss += self.densepose_weight * losses['densepose']
        
        losses['total'] = total_loss
        
        return losses


class DetectionLoss(nn.Module):
    """Loss function for person detection."""
    
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate detection loss."""
        # Classification loss
        cls_loss = self.cls_loss(predictions['classification'], targets['labels'])
        
        # Regression loss (only for positive samples)
        pos_mask = targets['labels'] > 0
        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss(
                predictions['bbox'][pos_mask],
                targets['bbox'][pos_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=predictions['classification'].device)
        
        # Combined loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return total_loss


class KeypointLoss(nn.Module):
    """Loss function for keypoint detection."""
    
    def __init__(self, loss_type: str = 'mse', use_offset: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.use_offset = use_offset
        
        if loss_type == 'mse':
            self.heatmap_loss = nn.MSELoss()
        elif loss_type == 'focal':
            self.heatmap_loss = FocalLoss()
        elif loss_type == 'wing':
            self.heatmap_loss = WingLoss()
        else:
            raise ValueError(f"Unknown keypoint loss type: {loss_type}")
        
        if use_offset:
            self.offset_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> torch.Tensor:
        """Calculate keypoint loss."""
        # Heatmap loss
        heatmap_loss = self.heatmap_loss(predictions['heatmaps'], targets)
        
        # Offset loss (if enabled and available)
        offset_loss = torch.tensor(0.0, device=heatmap_loss.device)
        if self.use_offset and 'offsets' in predictions:
            # Only calculate offset loss for visible keypoints
            visible_mask = targets.sum(dim=(2, 3)) > 0  # (B, num_keypoints)
            
            if visible_mask.sum() > 0:
                # Extract offsets for visible keypoints
                visible_offsets = predictions['offsets'][visible_mask]
                # This is a simplified implementation - you'd need proper offset targets
                offset_loss = self.offset_loss(visible_offsets, torch.zeros_like(visible_offsets))
        
        return heatmap_loss + 0.1 * offset_loss


class DensePoseLoss(nn.Module):
    """Loss function for DensePose estimation."""
    
    def __init__(self, loss_type: str = 'mse', use_region_classification: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.use_region_classification = use_region_classification
        
        if loss_type == 'mse':
            self.uv_loss = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.uv_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown DensePose loss type: {loss_type}")
        
        if use_region_classification:
            self.region_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate DensePose loss."""
        # UV coordinate loss
        uv_loss = self.uv_loss(predictions['uv_coords'], targets['uv_coords'])
        
        # Body region classification loss (if enabled and available)
        region_loss = torch.tensor(0.0, device=uv_loss.device)
        if self.use_region_classification and 'body_regions' in predictions:
            # This is a simplified implementation - you'd need proper region targets
            # For now, we'll skip this loss
            pass
        
        return uv_loss + 0.1 * region_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss."""
        # Convert targets to binary (0 or 1)
        targets_binary = (targets > 0).float()
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets_binary, reduction='none')
        
        # Apply focal loss formula
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class WingLoss(nn.Module):
    """Wing loss for keypoint regression."""
    
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate wing loss."""
        delta = torch.abs(predictions - targets)
        
        # Apply wing loss formula
        delta_abs = delta.abs()
        wing_loss = torch.where(
            delta_abs < self.omega,
            self.omega * torch.log(1 + delta_abs / self.epsilon),
            delta_abs - self.omega / 2
        )
        
        return wing_loss.mean()


class IoULoss(nn.Module):
    """Intersection over Union loss for bounding box regression."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate IoU loss."""
        # Calculate intersection
        x1 = torch.max(predictions[:, 0], targets[:, 0])
        y1 = torch.max(predictions[:, 1], targets[:, 1])
        x2 = torch.min(predictions[:, 2], targets[:, 2])
        y2 = torch.min(predictions[:, 3], targets[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        pred_area = (predictions[:, 2] - predictions[:, 0]) * (predictions[:, 3] - predictions[:, 1])
        target_area = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Convert to loss (1 - IoU)
        loss = 1 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ConsistencyLoss(nn.Module):
    """Consistency loss for temporal coherence in video sequences."""
    
    def __init__(self, temporal_weight: float = 0.1):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, current_pred: torch.Tensor, 
                previous_pred: torch.Tensor) -> torch.Tensor:
        """Calculate temporal consistency loss."""
        if previous_pred is None:
            return torch.tensor(0.0, device=current_pred.device)
        
        # Calculate difference between consecutive predictions
        consistency_loss = self.mse_loss(current_pred, previous_pred)
        
        return self.temporal_weight * consistency_loss
