from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    A simple neural network model for classification.

    By @Ouadoud
    """

    def __init__(self):
        """
        Initialize the neural network model.
        
        Layers:
            - Linear: Fully connected layer with 122 input features and 64 output features.
            - ReLU: Activation function.
            - Linear: Fully connected layer with 64 input features and 32 output features.
            - ReLU: Activation function.
            - Linear: Fully connected layer with 32 input features and 16 output features.
            - ReLU: Activation function.
            - Linear: Fully connected layer with 16 input features and 5 output features.
            - Dropout: Regularization layer with 0.5 dropout probability.
        """
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.sequential = nn.Sequential(
            nn.Linear(122, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.sequential(self.dropout(x))

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.
        
        Returns:
            List[np.ndarray]: List of model parameters as NumPy arrays.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the parameters of the model.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as NumPy arrays.
        """
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
