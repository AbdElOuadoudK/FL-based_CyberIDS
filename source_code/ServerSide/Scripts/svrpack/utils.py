"""
Utility functions for federated learning logging and evaluation.

This module provides functions to log training and validation metrics and
evaluate the global model on test data.

By @Ouadoud
"""

from os import path
from typing import List, Dict, Optional, Tuple
from pandas import DataFrame
from flwr.common import NDArrays, Scalar
from .globals import LOGS_PATH, BATCH_SIZE, GLOBALMODEL
from .inference import test
from .dataformat import load_test_data

def log_fit_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Log and save training metrics from multiple clients.

    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): List of metrics with client counts.

    Returns:
        Dict[str, float]: Averaged training metrics.

    By @Ouadoud
    """
    accuracy = sum(n * m['accuracy'] for n, m in metrics) / sum(n for n, _ in metrics)
    loss = sum(n * m['loss'] for n, m in metrics) / sum(n for n, _ in metrics)
    
    # Save training logs as a CSV file containing columns: [accuracy, loss]
    DataFrame([[accuracy, loss]], columns=["accuracy", "loss"]).to_csv(
        path.join(LOGS_PATH, "train_logs.csv"), mode='a', index=False, header=False
    )
    
    # Print distributed training metrics
    print(f"Training performance of local models (Averaged): Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    return {
        "loss": loss,
        "accuracy": accuracy,
    }

def log_eval_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Log and save validation metrics from multiple clients.

    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): List of metrics with client counts.

    Returns:
        Dict[str, float]: Averaged validation metrics.

    By @Ouadoud
    """
    accuracy = sum(n * m['accuracy'] for n, m in metrics) / sum(n for n, _ in metrics)
    loss = sum(n * m['loss'] for n, m in metrics) / sum(n for n, _ in metrics)
    
    # Save validation logs as a CSV file containing columns: [accuracy, loss]
    DataFrame([[accuracy, loss]], columns=["accuracy", "loss"]).to_csv(
        path.join(LOGS_PATH, "valid_logs.csv"), mode='a', index=False, header=False
    )
    
    # Print distributed validation metrics
    print(f"Validating performance of local models (Averaged): Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    return {
        "loss": loss,
        "accuracy": accuracy,
    }

def global_evaluate(X: NDArrays, y: NDArrays):
    """
    Evaluate the global model on the provided test data.

    Args:
        X (NDArrays): Test features.
        y (NDArrays): Test labels.

    Returns:
        Callable: Function to evaluate the global model with the given parameters.

    By @Ouadoud
    """
    def evaluate(server_round: int, 
                 parameters: NDArrays, 
                 config: Dict[str, Scalar]
                ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nonlocal X, y
        test_loader = load_test_data(data=(X, y), batch_size=BATCH_SIZE)
        GLOBALMODEL.set_parameters(parameters)
        loss, accuracy = test(test_loader)
        msg = f"Round: {server_round}" if server_round > 0 else "Initial state"
        print("\n" + msg)

        # Print centralized testing metrics
        print(f"Testing performance of global model : Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate
