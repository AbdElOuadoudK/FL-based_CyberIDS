"""
Data formatting module for creating datasets and loading test data.

This module provides a custom Dataset class for handling input and target data,
and a function to load test data into a DataLoader.

By @Ouadoud
"""

from typing import Tuple
import numpy
import pandas
from torch.utils.data import Dataset, DataLoader
from .globals import NUM_CORES

class Dataset(Dataset):
    """
    Custom dataset class for handling input and target data.

    By @Ouadoud
    """
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray):
        """
        Initialize the dataset with input and target data.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Target data.

        By @Ouadoud
        """
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        By @Ouadoud
        """
        return self.y.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Get a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing input data and target data for the specified index.

        By @Ouadoud
        """
        return self.x[idx], self.y[idx]

def load_test_data(data: Tuple[pandas.DataFrame, pandas.DataFrame] = None, batch_size: int = 1024) -> DataLoader:
    """
    Load test data into a DataLoader.

    Args:
        data (Tuple[pandas.DataFrame, pandas.DataFrame]): A tuple containing input and target data as pandas DataFrames.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the test data.

    By @Ouadoud
    """
    dataset = Dataset(x=data[0].values, y=data[1].values)
    test_loader = DataLoader(dataset, batch_size=batch_size)
    return test_loader
