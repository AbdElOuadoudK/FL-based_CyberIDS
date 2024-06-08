# Client Side

This directory contains scripts, data, logs, and packages related to the client side of a Federated Learning project.

## Directory Structure

- **Data/**: Directory for storing data used in the client-side scripts.
- **Logs/**: Directory for storing logs generated during the execution of the client-side scripts.
- **Scripts/**: Directory containing the main client-side scripts and packages.

  - **cltpack/**: Package containing the client-side implementation for the Federated Learning project.
  - **data_processing/**: Package containing scripts and utilities for processing data used in the project.
  - **MainClient.ipynb**: Jupyter notebook implementing the main client-side scripts and demonstrating the usage of packages.

## cltpack Package

The `cltpack` package contains the following modules:

- **dataformat.py**: Defines a custom dataset class and functions for loading test data.
- **neuralnet.py**: Defines a simple neural network model for classification.
- **inference.py**: Provides functions for testing the neural network model.
- **client.py**: Contains client-side classes and functions for interacting with the server.
- **globals.py**: Sets up global constants and initializes the global neural network model with weights.
- **utils.py**: Provides utility functions for logging and evaluating training and validation metrics.

## data_processing Package

The `data_processing` package contains scripts and utilities for processing data:

- **data_process.py**: Script for preprocessing data before it is used in the Federated Learning process. Contains functions for transforming and cleaning data used in the project.

## MainClient.ipynb

The `MainClient.ipynb` notebook serves as the main entry point to demonstrate the usage of the `cltpack` and `data_processing` packages. It provides examples of how to preprocess data, train models, and interact with the server for Federated Learning.

### How to Use <sup>*</sup>

1. **Data**: Store the data to be used in the `Data/` directory.
2. **Logs**: Logs generated during execution will be stored in the `Logs/` directory.
3. **Scripts**: Execute the `MainClient.ipynb` notebook to see an example of the client-side implementation.

---
*By @Ouadoud*

This project is part of a Federated Learning experiment for intrusion detection, where training agents (clients) train a global model collaboratively while keeping data decentralized. For more details on how to run the client-side scripts and notebooks, refer to the respective scripts and notebooks.

<sup>*</sup> If you want to simulate the execution, **do not** change any of the file/dir orders, **neither** their specified names, as it may cause confusion at the path level. However, If your purpose tend to do so, make sure to manage that correctly all along the execution, by following and maintaining the same settings been applied.

**P.S.** For server-side implementation and setup, please refer to the corresponding README.md file in the `Server Side` directory.
