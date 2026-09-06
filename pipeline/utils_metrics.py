# -*- coding: utf-8 -*-
"""
Utility module for XAI and clinical evaluation metrics.
Computes Dice, IoU, Pointing Game Accuracy, and clinical metrics like Sensitivity, Specificity.
"""

import numpy as np
import logging
from typing import Dict
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Compute Dice coefficient between predicted and ground truth masks.
    
    Args:
        pred_mask: Predicted binary mask [0, 1]
        gt_mask: Ground truth binary mask [0, 1]
        smooth: Smoothing constant to avoid division by zero
    
    Returns:
        Dice coefficient in range [0, 1]
    """
    # Ensure binary
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth masks.
    
    Args:
        pred_mask: Predicted binary mask [0, 1]
        gt_mask: Ground truth binary mask [0, 1]
        smooth: Smoothing constant
    
    Returns:
        IoU score in range [0, 1]
    """
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def pointing_game_accuracy(saliency_map: np.ndarray, gt_mask: np.ndarray, 
                          threshold: float = 0.5) -> float:
    """
    Compute Pointing Game Accuracy: whether the argmax of saliency overlaps with any positive pixel in GT.
    
    Args:
        saliency_map: Saliency/attention map (e.g., from Grad-CAM), shape (H, W)
        gt_mask: Ground truth binary mask, shape (H, W)
        threshold: Threshold for saliency binarization
    
    Returns:
        Boolean accuracy (1 if argmax overlaps with GT, 0 otherwise)
    """
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Find argmax of saliency
    max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    
    # Check if it overlaps with any positive pixel in GT
    if gt_mask[max_idx] > 0.5:
        return 1.0
    else:
        return 0.0


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute sensitivity (True Positive Rate / Recall).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels or probabilities
    
    Returns:
        Sensitivity in range [0, 1]
    """
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 else y_pred
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    if tp + fn == 0:
        return 0.0
    
    return float(tp / (tp + fn))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity (True Negative Rate).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels or probabilities
    
    Returns:
        Specificity in range [0, 1]
    """
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 else y_pred
    
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    
    if tn + fp == 0:
        return 0.0
    
    return float(tn / (tn + fp))


def compute_roc_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Compute ROC AUC score.
    
    Args:
        y_true: Ground truth binary labels
        y_pred_prob: Predicted probabilities (not binary)
    
    Returns:
        AUC-ROC in range [0, 1]
    """
    try:
        return float(roc_auc_score(y_true, y_pred_prob))
    except Exception as e:
        logger.warning(f"Could not compute AUC-ROC: {e}")
        return 0.0


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision (TP / (TP + FP)).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels or probabilities

    Returns:
        Precision in range [0, 1]
    """
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 else y_pred

    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))

    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels or probabilities
    
    Returns:
        F1 score in range [0, 1]
    """
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 else y_pred
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    if tp + fp + fn == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary accuracy.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels or probabilities
    
    Returns:
        Accuracy in range [0, 1]
    """
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == np.float32 else y_pred
    return float(np.mean(y_true == y_pred_binary))


def compute_all_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                        y_pred_binary: np.ndarray = None, 
                        saliency_map: np.ndarray = None,
                        gt_spatial_mask: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all clinical and XAI metrics in one pass.
    
    Args:
        y_true: Ground truth binary labels (N,)
        y_pred_prob: Predicted probabilities (N,) or (N, 1)
        y_pred_binary: Predicted binary labels (optional, computed from y_pred_prob if None)
        saliency_map: Saliency maps (optional) for XAI metrics, shape (N, H, W)
        gt_spatial_mask: Ground truth spatial masks (optional), shape (N, H, W)
    
    Returns:
        Dictionary of all computed metrics
    """
    # Ensure shapes are (N,)
    y_true = np.squeeze(y_true)
    y_pred_prob = np.squeeze(y_pred_prob)
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_binary = np.squeeze(y_pred_binary)
    
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred_prob),
        "sensitivity": sensitivity(y_true, y_pred_prob),
        "specificity": specificity(y_true, y_pred_prob),
        "f1_score": compute_f1_score(y_true, y_pred_prob),
        "auc_roc": compute_roc_auc(y_true, y_pred_prob),
    }
    
    # XAI metrics if saliency and ground truth spatial masks provided
    if saliency_map is not None and gt_spatial_mask is not None:
        saliency_map = np.squeeze(saliency_map)
        gt_spatial_mask = np.squeeze(gt_spatial_mask)
        
        if saliency_map.ndim == 3 and gt_spatial_mask.ndim == 3:
            # Compute per-sample metrics
            dice_scores = []
            iou_scores = []
            pointing_accs = []
            
            for i in range(min(len(saliency_map), len(gt_spatial_mask))):
                dice_scores.append(dice_coefficient(saliency_map[i], gt_spatial_mask[i]))
                iou_scores.append(iou_score(saliency_map[i], gt_spatial_mask[i]))
                pointing_accs.append(pointing_game_accuracy(saliency_map[i], gt_spatial_mask[i]))
            
            metrics["dice_mean"] = float(np.mean(dice_scores))
            metrics["dice_std"] = float(np.std(dice_scores))
            metrics["iou_mean"] = float(np.mean(iou_scores))
            metrics["iou_std"] = float(np.std(iou_scores))
            metrics["pointing_game_accuracy"] = float(np.mean(pointing_accs))
        else:
            # Single sample
            metrics["dice"] = dice_coefficient(saliency_map, gt_spatial_mask)
            metrics["iou"] = iou_score(saliency_map, gt_spatial_mask)
            metrics["pointing_game_accuracy"] = pointing_game_accuracy(saliency_map, gt_spatial_mask)
    
    return metrics
