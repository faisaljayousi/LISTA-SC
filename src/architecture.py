"""
This module implements the LISTA (Learned Iterative Shrinkage-Thresholding
Algorithm) as a PyTorch neural network module for sparse recovery.
"""

import numpy as np
import torch
import torch.nn as nn


class LISTA(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        W_d: np.ndarray,
        max_iterations: int,
        lipschitz_const: float,
        theta: float,
        device: torch.device = torch.device("cpu"),
    ):

        """
        Initialises the LISTA model.

        Args:
            n (int): dimension of the measurement vector
            m (int): dimension of the sparse signal
            W_d (np.ndarray): dictionary
            max_iterations (int): max number of internal iterations
            lipschitz_const (float): Lipschitz constant for the gradient step
            theta (float): Threshold value for the shrinkage function
            device (torch.device): ...
        """

        super(LISTA, self).__init__()

        self.W_d = W_d  # Dictionary
        self.max_iterations = max_iterations
        self.lipschitz_const = lipschitz_const
        self.device = device

        self._W = nn.Linear(in_features=n, out_features=m, bias=False)
        self._S = nn.Linear(in_features=m, out_features=m, bias=False)

        self.shrinkage = nn.Softshrink(theta)
        self.theta = theta

        self.weights_init()

    def weights_init(self):
        """
        Initialises the weights for the LISTA model based on the provided
        dictionary and Lipschitz constant.
        """
        # Use dictionary to initialise weights
        A = self.W_d.cpu().numpy()

        # Create MI and filter matrices
        S = (
            torch.from_numpy(
                np.eye(A.shape[1])
                - (1 / self.lipschitz_const) * np.matmul(A.T, A)
            )
            .float()
            .to(self.device)
        )  # mutual inhibition matrix
        W = (
            torch.from_numpy((1 / self.lipschitz_const) * A.T)
            .float()
            .to(self.device)
        )  # filter matrix

        # Assign initialised weights to layers
        self._S.weight = nn.Parameter(S, requires_grad=True)
        self._W.weight = nn.Parameter(W, requires_grad=True)

    def forward(self, Y: torch.Tensor):
        """
        Forward pass through the LISTA model.

        Args:
            Y (torch.Tensor): Input measurement data.

        Returns:
            torch.Tensor: Output tensor representing the estimated sparse
            signal.
        """
        X = self.shrinkage(self._W(Y))  # Sparse code

        if self.max_iterations == 1:
            return X

        for _ in range(self.max_iterations):
            X = self.shrinkage(self._W(Y) + self._S(X))

        return X
