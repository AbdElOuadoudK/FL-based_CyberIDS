from typing import Tuple, List
import numpy
import random
from sklearn.model_selection import train_test_split
import pandas
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class Dataset(Dataset):
    """
    A PyTorch Dataset class to format the data.

    Args:
        x (pandas.DataFrame): Input data.
        y (pandas.DataFrame): Target data.
        size (float): Size of subset to be retrieved.
        distribution (dict, optional): Distribution for sampling. Defaults to None.
        sparse_y (bool, optional): Whether target data is sparse. Defaults to True.

    Returns:
        tuple: A tuple containing both input and target data, after being formatted.

    Note: accroding to whether distribtuion has been defined, and due to 
    order of dispatching, the valide_rate may not be always satisfied. 
    As the priority goes to the distribution criteria, possible cases where the sum of the label proportions 
    does not equal to the valid_rate (and, eventually the train rate) may occure.
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
                for k, v in distribution :
                    tmp = y[y[k] == 1]
                    tmp = tmp.sample(n=v if v <= len(tmp) else len(tmp))
                    indexes += list(tmp.index)
            
            else:
                for k, v in distribution :
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
        """
        if self.is_train :
            return self.y_train.shape[0]
        else :
            return self.y_valid.shape[0]

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Get a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing input data and target data for the specified index.
        """
         
        if self.is_train :
            return self.x_train.values[idx], self.y_train.values[idx]
        else :
            return self.x_valid.values[idx], self.y_valid.values[idx]

def load_train_data(data: Tuple[pandas.DataFrame, pandas.DataFrame] = None,
              size: float = 1.,
              valid_rate: float = .5,
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
    """
    data = Dataset(x=data[0].values, y=data[1].values, y_columns=data[1].columns,
                   size=size, valid_rate=valid_rate, 
                   distribution=distribution, sparse_y=sparse_y)

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data, batch_size=batch_size)
    
    return train_loader, valid_loader
