
"""
dataformat.py

This module defines a PyTorch Dataset class for formatting input and target data, and a function to load training data
and create data loaders for training and validation sets.

By @Ouadoud.
"""

from typing import Tuple, List
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
#from .globals import NUM_CORES

class TrainDataset(Dataset):
    """
    A PyTorch Dataset class to format the data, with separate train and valid splits.
    """
    def __init__(self,
                 x: ndarray,
                 y: ndarray,
                 y_columns: List[str],
                 indices: List[int],
                 valid_rate: float):
        x = DataFrame(x.copy()).iloc[indices]
        y = DataFrame(y.copy(), columns=y_columns).iloc[indices]

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            x, y, test_size=valid_rate, stratify=y)

        self.is_train = True

    def __len__(self) -> int:
        return len(self.y_train) if self.is_train else len(self.y_valid)

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        if self.is_train:
            return self.x_train.values[idx].copy(), self.y_train.values[idx].copy()
        else:
            return self.x_valid.values[idx].copy(), self.y_valid.values[idx].copy()

def load_train_data(data: Tuple[DataFrame, DataFrame],
                    num_clients: int,
                    size: float = 1.0,
                    valid_rate: float = 0.5,
                    batch_size: int = 1024,
                    random_state: int = None
                   ) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Load the training data and create separate train and validation loaders for each client.
    
    Args:
        data (tuple): Tuple containing input (X) and target (Y) data.
        num_clients (int): Number of clients (and loaders) to create.
        size (float): Proportion of data to use.
        valid_rate (float): Validation set ratio.
        batch_size (int): Batch size for the loaders.

    Returns:
        Tuple: Two lists of DataLoader instances for training and validation, one for each client.
    """
    
    x_data, y_data = data
    y_data = y_data.sample(frac=size, random_state=random_state) # if not dist...
    x_data = x_data.reindex(y_data.index)
    
    total_indices = list(range(len(y_data)))
    client_indices = [total_indices[i::num_clients] for i in range(num_clients)]

    train_loaders, valid_loaders = [], []

    for indices in client_indices:
        dataset = TrainDataset(
            x=x_data.values, 
            y=y_data.values, 
            y_columns=y_data.columns, 
            indices=indices, 
            valid_rate=valid_rate)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset, batch_size=batch_size)

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)

    return train_loaders, valid_loaders

"""
Data formatting module for creating datasets and loading test data.

This module provides a custom Dataset class for handling input and target data,
and a function to load test data into a DataLoader.

By @Ouadoud
"""

class TestDataset(Dataset):
    """
    Custom dataset class for handling input and target data.

    By @Ouadoud
    """
    def __init__(self, x: ndarray, y: ndarray):
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
    
    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        """
        Get a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing input data and target data for the specified index.

        By @Ouadoud
        """
        return self.x[idx], self.y[idx]

def load_test_data(data: Tuple[DataFrame, DataFrame] = None, batch_size: int = 1024) -> DataLoader:
    """
    Load test data into a DataLoader.

    Args:
        data (Tuple[pandas.DataFrame, pandas.DataFrame]): A tuple containing input and target data as pandas DataFrames.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the test data.

    By @Ouadoud
    """
    dataset = TestDataset(x=data[0].values, y=data[1].values)
    test_loader = DataLoader(dataset, batch_size=batch_size)
    return test_loader
