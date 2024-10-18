"""
This script contains functions to train the LISTA model and save its state.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from .architecture import LISTA


def train_model(
    Y: np.ndarray,
    dictionary: np.ndarray,
    alpha: float,
    L: float,
    learning_rate: float,
    max_iterations: int,
    n_epochs: int,
    *,
    device=torch.device("cpu"),
):
    n, m = dictionary.shape
    n_samples = Y.shape[0]
    batch_size = 128
    steps_per_epoch = n_samples // batch_size

    # Convert to tensors
    Y = torch.from_numpy(Y)
    Y = Y.float().to(device)
    w_d = torch.from_numpy(dictionary)
    w_d = w_d.float().to(device)

    # Initialise LISTA model
    net = LISTA(
        n, m, w_d, max_iterations=max_iterations, L=L, theta=alpha / L
    )
    net = net.float().to(device)
    net.weights_init()

    # Build optimiser and criterion
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    all_zeros = torch.zeros(batch_size, m).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9
    )

    loss_list = []
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")

        indices = np.random.choice(
            a=n_samples, size=n_samples, replace=False, p=None
        )
        Y_shuffled = Y[indices]

        for step in range(steps_per_epoch):
            Y_batch = Y_shuffled[step * batch_size : (step + 1) * batch_size]
            optimizer.zero_grad()

            # Get the outputs
            x_h = net(Y_batch)
            y_h = torch.mm(x_h, w_d.T)

            # Compute the loss
            loss1 = criterion1(Y_batch.float(), y_h.float())
            loss2 = alpha * criterion2(x_h.float(), all_zeros.float())
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_list.append(loss.item())

    return net, loss_list


def save_model(checkpoint_path, model, *, verbose=False):
    """
    Save the model (and any additional data) to `checkpoint_path` using pickle.

    Parameters:
    - checkpoint_path (str): The file path where the model should be saved.
    - model (object): The model or data to be saved.
    - verbose (bool): If True, prints success or error messages.
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
