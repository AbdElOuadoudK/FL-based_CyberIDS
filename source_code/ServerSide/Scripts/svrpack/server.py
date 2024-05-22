"""
Server module for federated learning setup and execution.

This module defines the server algorithm for federated learning,
initializing the server, and starting the training rounds.

By @Ouadoud
"""

import numpy
from flwr.server import start_server, ServerConfig
from .globals import WEIGHTS, NUM_CLIENTS
from .utils import global_evaluate, log_fit_metrics, log_eval_metrics
from .aggregation import Aggregation

class Server_Algorithm:
    """
    Federated learning server algorithm.

    This class initializes and starts the federated learning server,
    conducting the specified number of training rounds.

    By @Ouadoud
    """
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, num_rounds: int):
        """
        Initialize the server algorithm with data and configuration.

        Args:
            X (numpy.ndarray): Features for evaluation.
            y (numpy.ndarray): Labels for evaluation.
            num_rounds (int): Number of federated learning rounds.

        By @Ouadoud
        """
        self.num_rounds = num_rounds
        self.X = X
        self.y = y
        
    def __call__(self, server_address: str = '127.0.0.1', port: int = 1234) -> dict:
        """
        Start the federated learning server and begin training rounds.

        Args:
            server_address (str): Address of the server.
            port (int): Port for the server.

        Returns:
            dict: History of the federated learning process.

        By @Ouadoud
        """
        print("\nStarting communication...")
        history = start_server(
            server_address=f"{server_address}:{port}",
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=Aggregation(
                min_fit_clients=NUM_CLIENTS,
                min_evaluate_clients=NUM_CLIENTS,
                min_available_clients=NUM_CLIENTS,
                evaluate_fn=global_evaluate(X=self.X, y=self.y),
                initial_parameters=WEIGHTS,
                fit_metrics_aggregation_fn=log_fit_metrics,
                evaluate_metrics_aggregation_fn=log_eval_metrics,
            )
        )
        print("\nEnding communication...")
        return history
