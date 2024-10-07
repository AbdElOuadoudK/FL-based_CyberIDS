
"""
Server module for federated learning setup and execution.

This module defines the server algorithm for federated learning,
initializing the server, and starting the training rounds.

By @Ouadoud
"""

from numpy import ndarray
from flwr.server import start_server, ServerConfig
from flwr.simulation import start_simulation
from ..globals import WEIGHTS, NUM_CLIENTS, LOGS_PATH, NUM_CORES
from .aggregation import Aggregation
from torch import set_num_threads

set_num_threads(NUM_CORES)

class Server_Algorithm:
    """
    Federated learning server algorithm.

    This class initializes and starts the federated learning server,
    conducting the specified number of training rounds.

    By @Ouadoud
    """
    
    def __init__(self, X: ndarray, y: ndarray, num_rounds: int):
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
        
    def __call__(self) -> dict: #, server_address: str = '127.0.0.1', port: int = 1234
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
            #server_address=f"{server_address}:{port}",
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=Aggregation(num_clients=NUM_CLIENTS, 
                                 weights=WEIGHTS, 
                                 X=self.X, y=self.y, 
                                 num_rounds=self.num_rounds)
        )




















        
        print("\nEnding communication...")

        return history
