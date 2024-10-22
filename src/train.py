"""
This script contains functions to train the LISTA model and save its state.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .architecture import LISTA


def train_model(
    Y: np.ndarray,
    dictionary: np.ndarray,
    alpha: float,
    lipschitz_const: float,
    learning_rate: float,
    batch_size: int,
    max_iterations: int,
    n_epochs: int,
    *,
    device=torch.device("cpu"),
    optimizer_name: str = "SGD",
):
    """
    Trains a LISTA model.

    Args:
        Y (np.ndarray): Input measurement data of shape (n_samples, n)
        dictionary (np.ndarray): Dictionary matrix of shape (n, m)
        alpha (float): Regularisation parameter
        lipschitz_const (float): Lipschitz constant for gradient step
        learning_rate (float): Learning rate for the chosen optimiser
        batch_size (int): Size of training batches
        max_iterations (int): Maximum number of internal iterations for LISTA
        n_epochs (int): Number of epochs
    """
    n, m = dictionary.shape

    # Convert to tensors
    Y = torch.from_numpy(Y).float().to(device)
    w_d = torch.from_numpy(dictionary).float().to(device)

    # Use Torch dataset for batch processing
    dataset = TensorDataset(Y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialise LISTA model
    net = (
        LISTA(
            n,
            m,
            w_d,
            max_iterations=max_iterations,
            lipschitz_const=lipschitz_const,
            theta=alpha / lipschitz_const,
            device=device,
        )
        .float()
        .to(device)
    )
    net.weights_init()

    # Build optimiser and criterion
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=0.9
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimiser not recognised.")

    # Training loop
    loss_list = []
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{n_epochs}")

        for (Y_batch,) in data_loader:
            optimizer.zero_grad()

            # Get the model's output
            estimated_sparse_code = net(Y_batch)
            reconstructed_Y = torch.mm(estimated_sparse_code, w_d.T)

            # Calculate losses
            data_fidelity_loss = criterion1(Y_batch, reconstructed_Y)
            sparsity_loss = alpha * criterion2(
                estimated_sparse_code, torch.zeros_like(estimated_sparse_code)
            )
            total_loss = data_fidelity_loss + sparsity_loss

            # Backpropagation and optimisation step
            total_loss.backward()
            optimizer.step()

            # Track loss
            with torch.no_grad():
                loss_list.append(total_loss.item())

    return net, loss_list


def save_model(checkpoint_path, model, *, verbose=False):
    """
    Save the model (and any additional data) to `checkpoint_path` using pickle.

    Args:
        checkpoint_path (str): The file path where the model should be saved.
        model (object): The model or data to be saved.
        verbose (bool): If True, prints success or error messages.
                        Defaults to True.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)
        if verbose:
            print(
                f"Model and error list successfully saved to "
                f"'{checkpoint_path}'"
            )
    except Exception as e:
        if verbose:
            print(f"Error while saving model and error list: {e}")
