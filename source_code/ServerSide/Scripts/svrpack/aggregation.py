"""
Aggregation module for federated learning.

This module contains the Aggregation class which extends FedAvg to handle
the aggregation of client updates in federated learning.

By @Ouadoud
"""

from numpy import savez
from os import path
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes
from .globals import LOGS_PATH

class Aggregation(FedAvg):
    """
    Aggregation class extending FedAvg to handle the aggregation of client updates.

    By @Ouadoud
    """

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the results from multiple clients after training.

        Args:
            server_round (int): The current round of federated learning.
            results (List[Tuple[ClientProxy, FitRes]]): List of results from clients.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): List of failures during the round.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: Aggregated parameters and metrics.

        By @Ouadoud
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Save aggregated_ndarrays
            logs_path = path.join(LOGS_PATH, "global_weights.npz")
            savez(logs_path, *parameters_to_ndarrays(aggregated_parameters))
        
        return aggregated_parameters, aggregated_metrics

    def initialize_parameters(self,
                              client_manager: ClientManager
                             ) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        Args:
            client_manager (ClientManager): The client manager for handling client connections.

        Returns:
            Optional[Parameters]: The initialized global parameters.

        By @Ouadoud
        """
        initial_parameters = ndarrays_to_parameters(self.initial_parameters)
        del self.initial_parameters
        return initial_parameters
