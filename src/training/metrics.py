"""
Metrics for evaluating WiFi-based pose estimation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PoseMetrics:
    """
    Metrics calculator for pose estimation.
    
    Calculates various metrics for keypoints, DensePose, and detection.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.keypoint_metrics = KeypointMetrics()
        self.densepose_metrics = DensePoseMetrics()
        self.detection_metrics = DetectionMetrics()
    
    def calculate_batch(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate metrics for a batch.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing batch metrics
        """
        metrics = {}
        
        # Keypoint metrics
        if 'keypoints' in predictions and 'target_keypoints' in targets:
            keypoint_metrics = self.keypoint_metrics.calculate_batch(
                predictions['keypoints'], targets['target_keypoints']
            )
            metrics.update(keypoint_metrics)
        
        # DensePose metrics
        if 'densepose' in predictions and 'target_densepose' in targets:
            densepose_metrics = self.densepose_metrics.calculate_batch(
                predictions['densepose'], targets['target_densepose']
            )
            metrics.update(densepose_metrics)
        
        # Detection metrics
        if 'detections' in predictions and 'target_detections' in targets:
            detection_metrics = self.detection_metrics.calculate_batch(
                predictions['detections'], targets['target_detections']
            )
            metrics.update(detection_metrics)
        
        return metrics
    
    def calculate_epoch(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate epoch-level metrics.
        
        Args:
            all_metrics: List of batch metrics
            
        Returns:
            Dictionary containing epoch metrics
        """
        if not all_metrics:
            return {}
        
        epoch_metrics = {}
        
        # Aggregate metrics across batches
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics if key in metrics]
            if values:
                epoch_metrics[key] = np.mean(values)
        
        return epoch_metrics


class KeypointMetrics:
    """Metrics for keypoint detection."""
    
    def __init__(self, pck_threshold: float = 0.1):
        """
        Initialize keypoint metrics.
        
        Args:
            pck_threshold: Threshold for PCK calculation
        """
        self.pck_threshold = pck_threshold
    
    def calculate_batch(self, predictions: Dict[str, torch.Tensor], 
                       targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate keypoint metrics for a batch.
        
        Args:
            predictions: Keypoint predictions
            targets: Ground truth keypoint heatmaps
            
        Returns:
            Dictionary containing keypoint metrics
        """
        metrics = {}
        
        # Extract keypoint locations
        if 'coordinates' in predictions:
            pred_coords = predictions['coordinates']
            pred_confidences = predictions.get('confidences', torch.ones_like(pred_coords[:, :, 0]))
            
            # Calculate PCK (Percentage of Correct Keypoints)
            pck = self.calculate_pck(pred_coords, targets, pred_confidences)
            metrics['pck'] = pck
            
            # Calculate mean distance
            mean_dist = self.calculate_mean_distance(pred_coords, targets, pred_confidences)
            metrics['mean_distance'] = mean_dist
        
        # Calculate heatmap accuracy
        if 'heatmaps' in predictions:
            heatmap_acc = self.calculate_heatmap_accuracy(predictions['heatmaps'], targets)
            metrics['heatmap_accuracy'] = heatmap_acc
        
        return metrics
    
    def calculate_pck(self, pred_coords: torch.Tensor, targets: torch.Tensor, 
                      confidences: torch.Tensor) -> float:
        """Calculate PCK (Percentage of Correct Keypoints)."""
        batch_size, num_keypoints, _ = pred_coords.shape
        
        correct_keypoints = 0
        total_keypoints = 0
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                if confidences[b, k] > 0.1:  # Only consider confident predictions
                    # Find ground truth keypoint location
                    gt_heatmap = targets[b, k]
                    gt_coord = self._heatmap_to_coord(gt_heatmap)
                    
                    if gt_coord is not None:
                        # Calculate distance
                        pred_coord = pred_coords[b, k]
                        distance = torch.norm(pred_coord - gt_coord)
                        
                        if distance < self.pck_threshold:
                            correct_keypoints += 1
                        total_keypoints += 1
        
        return correct_keypoints / max(total_keypoints, 1)
    
    def calculate_mean_distance(self, pred_coords: torch.Tensor, targets: torch.Tensor, 
                               confidences: torch.Tensor) -> float:
        """Calculate mean distance between predicted and ground truth keypoints."""
        batch_size, num_keypoints, _ = pred_coords.shape
        
        distances = []
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                if confidences[b, k] > 0.1:
                    gt_heatmap = targets[b, k]
                    gt_coord = self._heatmap_to_coord(gt_heatmap)
                    
                    if gt_coord is not None:
                        pred_coord = pred_coords[b, k]
                        distance = torch.norm(pred_coord - gt_coord)
                        distances.append(distance.item())
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_heatmap_accuracy(self, pred_heatmaps: torch.Tensor, 
                                  target_heatmaps: torch.Tensor) -> float:
        """Calculate heatmap accuracy using MSE."""
        return torch.nn.functional.mse_loss(pred_heatmaps, target_heatmaps).item()
    
    def _heatmap_to_coord(self, heatmap: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert heatmap to coordinate."""
        if heatmap.max() < 0.1:  # No keypoint
            return None
        
        # Find maximum location
        max_idx = torch.argmax(heatmap)
        height, width = heatmap.shape
        y = max_idx // width
        x = max_idx % width
        
        # Normalize to [0, 1]
        coord = torch.tensor([x.float() / (width - 1), y.float() / (height - 1)])
        
        return coord


class DensePoseMetrics:
    """Metrics for DensePose estimation."""
    
    def __init__(self):
        """Initialize DensePose metrics."""
        pass
    
    def calculate_batch(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate DensePose metrics for a batch.
        
        Args:
            predictions: DensePose predictions
            targets: Ground truth DensePose targets
            
        Returns:
            Dictionary containing DensePose metrics
        """
        metrics = {}
        
        # UV coordinate accuracy
        if 'uv_coords' in predictions and 'uv_coords' in targets:
            uv_acc = self.calculate_uv_accuracy(
                predictions['uv_coords'], targets['uv_coords']
            )
            metrics['uv_accuracy'] = uv_acc
        
        # Body region accuracy
        if 'body_regions' in predictions and 'body_regions' in targets:
            region_acc = self.calculate_region_accuracy(
                predictions['body_regions'], targets['body_regions']
            )
            metrics['region_accuracy'] = region_acc
        
        return metrics
    
    def calculate_uv_accuracy(self, pred_uv: torch.Tensor, 
                             target_uv: torch.Tensor) -> float:
        """Calculate UV coordinate accuracy."""
        return torch.nn.functional.mse_loss(pred_uv, target_uv).item()
    
    def calculate_region_accuracy(self, pred_regions: torch.Tensor, 
                                 target_regions: torch.Tensor) -> float:
        """Calculate body region classification accuracy."""
        pred_labels = torch.argmax(pred_regions, dim=1)
        target_labels = torch.argmax(target_regions, dim=1)
        
        correct = (pred_labels == target_labels).float().mean()
        return correct.item()


class DetectionMetrics:
    """Metrics for person detection."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detection metrics.
        
        Args:
            iou_threshold: IoU threshold for positive detection
        """
        self.iou_threshold = iou_threshold
    
    def calculate_batch(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate detection metrics for a batch.
        
        Args:
            predictions: Detection predictions
            targets: Ground truth detection targets
            
        Returns:
            Dictionary containing detection metrics
        """
        metrics = {}
        
        # Classification accuracy
        if 'classification' in predictions and 'labels' in targets:
            cls_acc = self.calculate_classification_accuracy(
                predictions['classification'], targets['labels']
            )
            metrics['classification_accuracy'] = cls_acc
        
        # Bounding box IoU
        if 'bbox' in predictions and 'bbox' in targets:
            bbox_iou = self.calculate_bbox_iou(
                predictions['bbox'], targets['bbox']
            )
            metrics['bbox_iou'] = bbox_iou
        
        return metrics
    
    def calculate_classification_accuracy(self, pred_logits: torch.Tensor, 
                                        target_labels: torch.Tensor) -> float:
        """Calculate classification accuracy."""
        pred_labels = torch.argmax(pred_logits, dim=1)
        correct = (pred_labels == target_labels).float().mean()
        return correct.item()
    
    def calculate_bbox_iou(self, pred_bbox: torch.Tensor, 
                           target_bbox: torch.Tensor) -> float:
        """Calculate bounding box IoU."""
        batch_size = pred_bbox.shape[0]
        ious = []
        
        for i in range(batch_size):
            iou = self._calculate_single_iou(pred_bbox[i], target_bbox[i])
            ious.append(iou)
        
        return np.mean(ious)
    
    def _calculate_single_iou(self, pred_bbox: torch.Tensor, 
                              target_bbox: torch.Tensor) -> float:
        """Calculate IoU for a single bounding box pair."""
        # Extract coordinates
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
        x1_target, y1_target, x2_target, y2_target = target_bbox
        
        # Calculate intersection
        x1_inter = max(x1_pred, x1_target)
        y1_inter = max(y1_pred, y1_target)
        x2_inter = min(x2_pred, x2_target)
        y2_inter = min(y2_pred, y2_target)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        target_area = (x2_target - x1_target) * (y2_target - y1_target)
        union = pred_area + target_area - intersection
        
        return intersection / union


class TemporalMetrics:
    """Metrics for temporal consistency in video sequences."""
    
    def __init__(self):
        """Initialize temporal metrics."""
        pass
    
    def calculate_temporal_consistency(self, predictions: List[torch.Tensor]) -> float:
        """
        Calculate temporal consistency across frames.
        
        Args:
            predictions: List of predictions for consecutive frames
            
        Returns:
            Temporal consistency score
        """
        if len(predictions) < 2:
            return 0.0
        
        consistency_scores = []
        
        for i in range(1, len(predictions)):
            # Calculate difference between consecutive predictions
            diff = torch.nn.functional.mse_loss(predictions[i], predictions[i-1])
            consistency_scores.append(diff.item())
        
        # Lower score means higher consistency
        return np.mean(consistency_scores)
    
    def calculate_temporal_smoothness(self, predictions: List[torch.Tensor]) -> float:
        """
        Calculate temporal smoothness (second-order consistency).
        
        Args:
            predictions: List of predictions for consecutive frames
            
        Returns:
            Temporal smoothness score
        """
        if len(predictions) < 3:
            return 0.0
        
        smoothness_scores = []
        
        for i in range(2, len(predictions)):
            # Calculate second-order difference
            second_diff = predictions[i] - 2 * predictions[i-1] + predictions[i-2]
            smoothness = torch.norm(second_diff).item()
            smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores)
