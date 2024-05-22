"""
Inference module for testing neural network models.

This module provides functions for counting identical rows in matrices and testing a neural network model.

By @Ouadoud
"""

from torch import nn, no_grad, float32
import numpy
from typing import Tuple
from .globals import GLOBALMODEL, DEVICE

def count_identical_rows(mat1: numpy.ndarray, mat2: numpy.ndarray) -> int:
    """
    Count the number of identical rows between two matrices.
    
    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.
        
    Returns:
        int: Number of identical rows between the two matrices.

    By @Ouadoud
    """
    # Convert matrices to tuples of rows
    rows_mat1 = [tuple(row) for row in mat1]
    rows_mat2 = [tuple(row) for row in mat2]
    
    # Count identical rows
    same_rows_count = sum(row1 == row2 for row1, row2 in zip(rows_mat1, rows_mat2))
    
    return same_rows_count

# Define the loss function
loss_function = nn.CrossEntropyLoss()

def test(test_loader) -> Tuple[float, float]:
    """
    Test the neural network model.
    
    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        
    Returns:
        Tuple[float, float]: A tuple containing test loss and test accuracy.

    By @Ouadoud
    """    
    correct, total, loss = 0, 0, 0.0
    with no_grad():
        for items, targets in test_loader:
            # Move items to the correct device and data type
            items = items.to(dtype=float32, device=DEVICE)
            
            # Get model outputs
            outputs = GLOBALMODEL(items)
            
            # Calculate the loss
            loss += loss_function(outputs, targets).item()
            
            # Get the maximum values and create a mask
            max_values = numpy.max(outputs.cpu().numpy(), axis=1)
            mask = outputs.cpu().numpy() == max_values[:, numpy.newaxis]
            
            # Count correct predictions
            correct += count_identical_rows(targets.cpu().numpy().astype(int), mask.astype(int))
    
    # Return the average loss and accuracy
    return loss / len(test_loader.dataset), correct / len(test_loader.dataset)
