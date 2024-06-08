# Server Side

This directory contains scripts and data for the server-side implementation of a federated learning project.

## Data

This directory stores the data used for training and evaluation.

- **test.feather**: Feather file containing test data.

## Logs

This directory contains log files generated during the federated learning process.

- **global_weights.npz**: Numpy zip file containing the global model weights.
- **server_logs.txt**: Text file containing logs related to the server operation.
- **train_logs.csv**: CSV file containing training metrics logs.
- **valid_logs.csv**: CSV file containing validation metrics logs.

## Scripts

This directory contains the main scripts and packages for the server-side implementation.

### svrpack (package)

This package contains scripts and modules related to the federated learning server.

- **aggregation.py**: Module for aggregating client updates in federated learning.
- **inference.py**: Module for testing neural network models.
- **server.py**: Module defining the server algorithm for federated learning.
- **utils.py**: Utility functions for logging and evaluation in federated learning.

### MainServer.ipynb

Python notebook implementing the server-side federated learning process.

---
*By @Ouadoud*

This file contains descriptions about the files, packages, and folders of the current directory - Server Side.

