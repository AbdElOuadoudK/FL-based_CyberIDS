"""
Inference module for testing neural network models.

This module provides functions for testing a neural network model and calculating performance metrics.

By @Ouadoud
"""

from torch import nn, no_grad, float32
import numpy
from typing import Tuple
from .globals import GLOBALMODEL, DEVICE, NUM_CORES
import torch

# Set the number of CPU threads to use
torch.set_num_threads(NUM_CORES)

# Define the binary cross-entropy loss function
loss_function = nn.BCELoss()

def test(test_loader) -> dict:
    """
    Test the neural network model and calculate performance metrics.
    
    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        
    Returns:
        dict: A dictionary containing test loss, accuracy, precision, recall, F1 score, false positive rate, and counts of TP, TN, FP, and FN.

    By @Ouadoud
    """
    # Initialize metrics
    loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Set the model to evaluation mode
    GLOBALMODEL.eval()
    
    with no_grad():
        for items, targets in test_loader:
            # Move items to the correct device and data type
            items = items.to(dtype=float32, device=DEVICE)
            
            # Get model outputs
            outputs = GLOBALMODEL(items)
            
            # Calculate the loss
            loss += loss_function(outputs, targets.float()).item()

            # Calculate TP, TN, FP, FN
            predictions = outputs >= 0.5
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()

        # Calculate performance metrics
        test_loss = loss / len(test_loader.dataset)
        test_accuracy = (tp + tn) / (tp + tn + fp + fn)

        log_entry = {
            'Accuracy': test_accuracy,
            'Precision': (tp) / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': (tp) / (tp + fn) if (tp + fn) > 0 else 0,
            'F1': (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
            'FPR': (fp) / (tn + fp) if (tn + fp) > 0 else 0,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Loss': test_loss,
        }
    
    # Return the log entry with all the metrics
    return log_entry
