from collections import OrderedDict
from typing import List, Optional
import torch
import numpy as np
from flwr.client import Client, start_client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from .train import _main, test
from .globals import MODEL

class FederatedClient(Client):
    """
    Client class for federated learning.
    
    By @Ouadoud
    """

    def __init__(self, client_id: str, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, distribution: Optional[dict], epochs: int):
        """
        Initialize the client with necessary data loaders and parameters.

        Args:
            client_id (str): Unique identifier for the client.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            distribution (Optional[dict]): Distribution configuration for the client.
            epochs (int): Number of epochs for training.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.distribution = distribution
        self.epochs = epochs
        self.__round = 0

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Retrieve the current parameters of the model.

        Args:
            ins (GetParametersIns): Instructions for getting parameters.

        Returns:
            GetParametersRes: Response containing model parameters and status.
        """
        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(MODEL.get_parameters())

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def set_parameters(self, ins: FitIns | EvaluateIns):
        """
        Set the parameters of the model.

        Args:
            ins (FitIns | EvaluateIns): Instructions containing parameters to set.
        """
        # Deserialize parameters to NumPy ndarray's
        parameters = parameters_to_ndarrays(ins.parameters)
        MODEL.set_parameters(parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """
        Train the model using the provided parameters and configuration.

        Args:
            ins (FitIns): Instructions for fitting the model.

        Returns:
            FitRes: Response containing updated parameters, number of examples, and training metrics.
        """
        print(f"Fit, config: {ins.config}")

        # Update local model, train, get updated parameters
        self.set_parameters(ins)
        self.__round, loss, accuracy = _main(self.__round, self.client_id, self.train_loader, self.epochs)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(MODEL.get_parameters())

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_loader),
            metrics={"loss": float(loss),
                     "accuracy": float(accuracy)},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model using the provided parameters and configuration.

        Args:
            ins (EvaluateIns): Instructions for evaluating the model.

        Returns:
            EvaluateRes: Response containing evaluation metrics and status.
        """
        print(f"Evaluate, config: {ins.config}")

        self.set_parameters(ins)
        loss, accuracy = test(self.valid_loader)
        print(f"Evaluation >> Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valid_loader),
            metrics={"loss": float(loss),
                     "accuracy": float(accuracy)},
        )


class AgentAlgorithm:
    """
    Agent algorithm for starting the federated learning client.

    By @Ouadoud
    """

    def __init__(self, client_id: str, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, distribution: Optional[dict] = None, epochs: int = 1):
        """
        Initialize the agent with a federated client.

        Args:
            client_id (str): Unique identifier for the client.
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            distribution (Optional[dict]): Distribution configuration for the client.
            epochs (int): Number of epochs for training.
        """
        self.client = FederatedClient(client_id, train_loader, valid_loader, distribution, epochs)

    def __call__(self, server_address: str = '127.0.0.1', port: int = 1234):
        """
        Start the communication with the server.

        Args:
            server_address (str): The server address to connect to.
            port (int): The port to use for the connection.
        """
        print("\nStarting communication...")
        start_client(server_address=f"{server_address}:{port}", client=self.client)
        print("Ending communication...\n")
