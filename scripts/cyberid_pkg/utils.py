"""
Utility functions for federated learning logging and evaluation.

This module provides functions to log training and validation metrics and
evaluate the global model on test data.

By @Ouadoud
"""

from os import path, listdir
from typing import List, Dict, Optional, Tuple
from pandas import DataFrame
from flwr.common import NDArrays, Scalar
from .globals import LOGS_PATH, BATCH_SIZE, GLOBALMODEL
from .inference import test_model
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
    avg_len = sum(n for n, _ in metrics)
    train_accuracy = sum(n * m['Accuracy'] for n, m in metrics) / avg_len
    train_precision = sum(n * m['Precision'] for n, m in metrics) / avg_len
    train_recall = sum(n * m['Recall'] for n, m in metrics) / avg_len
    train_f1 = sum(n * m['F1'] for n, m in metrics) / avg_len
    train_fpr = sum(n * m['FPR'] for n, m in metrics) / avg_len
    train_loss = sum(n * m['Loss'] for n, m in metrics) / avg_len

    # Save training logs as a CSV file
    metrics_list = [train_accuracy, train_precision, train_recall, train_f1, train_fpr, train_loss]
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "Loss"]



    logs_path = path.join(LOGS_PATH, "train_logs.csv")
    if "train_logs.csv" in listdir(LOGS_PATH):
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, index=False, mode='w')
        
    loss = "{:.3e}".format(train_loss)
    accuracy = "{:.4f} %".format(train_accuracy * 100)
    
    # Print distributed training metrics
    print(f"Train performance of local models (Averaged): Loss: {loss} | Accuracy: {accuracy}")
    return {
        "Accuracy": train_accuracy,
        "Precision": train_precision,
        "Recall": train_recall,
        "F1": train_f1,
        "FPR": train_fpr,
        "Loss": train_loss,
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
    avg_len = sum(n for n, _ in metrics)
    valid_accuracy = sum(n * m['Accuracy'] for n, m in metrics) / avg_len
    valid_precision = sum(n * m['Precision'] for n, m in metrics) / avg_len
    valid_recall = sum(n * m['Recall'] for n, m in metrics) / avg_len
    valid_f1 = sum(n * m['F1'] for n, m in metrics) / avg_len
    valid_fpr = sum(n * m['FPR'] for n, m in metrics) / avg_len
    valid_loss = sum(n * m['Loss'] for n, m in metrics) / avg_len

    # Save validation logs as a CSV file
    metrics_list = [valid_accuracy, valid_precision, valid_recall, valid_f1, valid_fpr, valid_loss]
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "Loss"]
    
    logs_path = path.join(LOGS_PATH, "valid_logs.csv")
    if "valid_logs.csv" in listdir(LOGS_PATH):
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, header=False, index=False, mode='a')
    else:
        DataFrame([metrics_list], columns=metrics_cols).to_csv(logs_path, index=False, mode='w')
    
    loss = "{:.3e}".format(valid_loss)
    accuracy = "{:.4f} %".format(valid_accuracy * 100)
    
    # Print distributed validation metrics
    print(f"Validation performance of local models (Averaged): Loss: {loss} | Accuracy: {accuracy}")
    return {
        "Accuracy": valid_accuracy,
        "Precision": valid_precision,
        "Recall": valid_recall,
        "F1": valid_f1,
        "FPR": valid_fpr,
        "Loss": valid_loss,
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
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nonlocal X, y
        test_loader = load_test_data(data=(X, y), batch_size=BATCH_SIZE)
        GLOBALMODEL.set_parameters(parameters)
        log_entry = test_model(test_loader)
        
        msg = f"Round: {server_round}" if server_round > 0 else "Initial state"
        print("\n" + msg)
        
        loss = "{:.3e}".format(log_entry['Loss'])
        accuracy = "{:.4f} %".format(log_entry['Accuracy'] * 100)
        
        # Print centralized testing metrics
        print(f"Test performance of global model: Loss: {loss} | Accuracy: {accuracy}")
            
        logs_path = path.join(LOGS_PATH, "test_logs.csv")
        if "test_logs.csv" in listdir(LOGS_PATH):
            DataFrame([log_entry]).to_csv(logs_path, header=False, index=False, mode='a')
        else:
            DataFrame([log_entry]).to_csv(logs_path, index=False, mode='w') 
                
        return log_entry['Loss'], {"Accuracy": log_entry['Accuracy']}
    
    return evaluate

def fit_config(rounds: int):
    def _fit_config(server_round: int):
        """Return training configuration dict for each round."""
        nonlocal rounds
        config = {"round": server_round}
        
        if server_round == 1:
            config.update({"public_key": 123, "rounds": rounds})
        
        return config
    
    return _fit_config
