"""
dataformat.py

This module defines a PyTorch Dataset class for formatting input and target data, and a function to load training data
and create data loaders for training and validation sets.

By @Ouadoud.
"""

from typing import Tuple, List
import numpy, pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .globals import NUM_CORES

class Dataset(Dataset):
    """
    A PyTorch Dataset class to format the data.

    Args:
        x (pandas.DataFrame): Input data.
        y (pandas.DataFrame): Target data.
        y_columns (List[str]): Column names for target data.
        size (float): Size of subset to be retrieved.
        valid_rate (float): Validation set ratio.
        distribution (dict, optional): Distribution for sampling. Defaults to None.
        sparse_y (bool, optional): Whether target data is sparse. Defaults to True.

    Returns:
        tuple: A tuple containing both input and target data, after being formatted.

    Note: According to whether distribution has been defined, and due to the order of dispatching, the valid_rate may not 
    always be satisfied. As the priority goes to the distribution criteria, possible cases where the sum of the label 
    proportions does not equal to the valid_rate (and, eventually the train rate) may occur.
    
    By @Ouadoud.
    """
    def __init__(self,
                 x: numpy.ndarray,
                 y: numpy.ndarray,
                 y_columns: List[str],
                 size: float,
                 valid_rate: float,
                 distribution: dict = None,
                 sparse_y: bool = True
                ):
        x = pandas.DataFrame(x)
        y = pandas.DataFrame(y, columns=y_columns)
        self.is_train = True
        
        if distribution is None:
            indexes = y.sample(frac=size).index
        else:
            indexes = list()
            if sparse_y:
                for k, v in distribution:
                    tmp = y[y[k] == 1]
                    tmp = tmp.sample(n=v if v <= len(tmp) else len(tmp))
                    indexes += list(tmp.index)
            else:
                for k, v in distribution:
                    tmp = y[y[y.columns] == k]
                    tmp = tmp.sample(n=v if v <= len(tmp) else len(tmp))
                    indexes += list(tmp.index)
        
        x = x.reindex(indexes)
        y = y.reindex(indexes)

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x, y, test_size=valid_rate, stratify=y)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
            
        By @Ouadoud.
        """
        if self.is_train:
            return self.y_train.shape[0]
        else:
            return self.y_valid.shape[0]

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Get a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing input data and target data for the specified index.
            
        By @Ouadoud.
        """
        if self.is_train:
            return self.x_train.values[idx], self.y_train.values[idx]
        else:
            return self.x_valid.values[idx], self.y_valid.values[idx]

def load_train_data(data: Tuple[pandas.DataFrame, pandas.DataFrame] = None,
                    size: float = 1.0,
                    valid_rate: float = 0.5,
                    batch_size: int = 1024,
                    distribution: dict = None,
                    sparse_y: bool = True
                   ) -> Tuple[DataLoader, DataLoader]:
    """
    Load training data and create train/valid loaders.
    
    Args:
        data (tuple, optional): Tuple containing input and target data. Defaults to None.
        size (float, optional): Size of the dataset. Defaults to 1.0.
        valid_rate (float, optional): Validation set ratio. Defaults to 0.5.
        batch_size (int, optional): Batch size. Defaults to 1024.
        distribution (dict, optional): Distribution for sampling. Defaults to None.
        sparse_y (bool, optional): Whether target data is sparse. Defaults to True.

    Returns:
        tuple: A tuple containing training and validation data loaders.
        
    By @Ouadoud.
    """
    data = Dataset(x=data[0].values, y=data[1].values, y_columns=data[1].columns,
                   size=size, valid_rate=valid_rate, 
                   distribution=distribution, sparse_y=sparse_y)

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data, batch_size=batch_size)
    
    return train_loader, valid_loader
