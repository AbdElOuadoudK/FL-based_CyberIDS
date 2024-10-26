
"""
Aggregation module for federated learning.

This module contains the Aggregation class which extends FedAvg to handle
the aggregation of client updates in federated learning.

By @Ouadoud
"""

from numpy import savez
from os.path import join
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import FitRes
from torch import set_num_threads
from .server_utils import global_evaluate, log_fit_metrics, log_eval_metrics, fit_config
from ..globals import WEIGHTS, NUM_CLIENTS, CHKPTS_PATH, NUM_CORES


class Aggregation(FedAvg):
    """
    Aggregation class extending FedAvg to handle the aggregation of client updates.

    By @Ouadoud
    """

    def __init__(self, 
                 num_clients, 
                 weights,
                 X, y,
                 num_rounds
                ):
        
        super().__init__(
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=global_evaluate(X, y),
            on_fit_config_fn=fit_config(num_rounds),
            initial_parameters=weights,
            fit_metrics_aggregation_fn=log_fit_metrics,
            evaluate_metrics_aggregation_fn=log_eval_metrics,
        )
    
    def aggregate_fit(
            self,
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
            # Save aggregated parameters as numpy arrays
            chkpts_path = join(CHKPTS_PATH, "global_weights.npz")
            savez(chkpts_path, *parameters_to_ndarrays(aggregated_parameters))
        
        return aggregated_parameters, aggregated_metrics

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        Args:
            client_manager (ClientManager): The client manager for handling client connections.

        Returns:
            Optional[Parameters]: The initialized global parameters.

        By @Ouadoud
        """
        initial_parameters = ndarrays_to_parameters(self.initial_parameters)
        del self.initial_parameters  # Remove initial parameters after initialization
        return initial_parameters
