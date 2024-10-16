
"""
Neural network model definition for classification.

This module defines a simple neural network model for classification tasks,
including methods to get and set model parameters.

By @Ouadoud
"""
from collections import OrderedDict
from typing import List
from torch import set_num_threads, Tensor, tensor
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU, Dropout, Sigmoid
from numpy import ndarray

class NeuralNet(Module):
    """
    A simple neural network model for classification.

    By @Ouadoud.
    """
    def __init__(self, num_cores: int = 4):
        """
        Initialize the neural network model.

        Args:
            num_cores (int, optional): Number of CPU cores to use. Defaults to 4.
        
        By @Ouadoud.
        """
        super(NeuralNet, self).__init__()
        self.sequential = Sequential(
            Linear(50, 64),
            BatchNorm1d(64),
            ReLU(),
            Dropout(p=0.25),
            Linear(64, 32),
            BatchNorm1d(32),
            ReLU(),
            Dropout(p=0.25),
            Linear(32, 16),
            BatchNorm1d(16),
            ReLU(),
            Dropout(p=0.25),
            Linear(16, 1),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        
        By @Ouadoud.
        """
        return self.sequential(x)

    def get_parameters(self) -> List[ndarray]:
        """
        Get the parameters of the network as a list of numpy arrays.

        Returns:
            List[numpy.ndarray]: List of network parameters.
        
        By @Ouadoud.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
    
    def set_parameters(self, parameters: List[ndarray]) -> None:
        """
        Set the parameters of the network from a list of numpy arrays.

        Args:
            parameters (List[numpy.ndarray]): List of network parameters.
        
        By @Ouadoud.
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def _save_model(self):
        pass