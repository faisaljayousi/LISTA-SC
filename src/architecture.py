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
        L: float,
        theta: float,
        device: torch.device = torch.device("cpu"),
    ):

        """
        Initialises the LISTA model.

        Args:
            n: int, dimensions of the measurement
            m: int, dimensions of the sparse signal
            W_d: array, dictionary
            max_iter:int, max number of internal iteration
            L: Lipschitz const
            theta: Thresholding
        """

        super(LISTA, self).__init__()

        self._W = nn.Linear(in_features=n, out_features=m, bias=False)
        self._S = nn.Linear(in_features=m, out_features=m, bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.theta = theta
        self.max_iter = max_iterations
        self.A = W_d
        self.L = L

        self.device = device

    def weights_init(self):
        """
        Initialises the weights for the LISTA model based on the provided
        dictionary and Lipschitz constant.
        """
        A = self.A.cpu().numpy()
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1 / L) * np.matmul(A.T, A))
        S = S.float().to(self.device)
        W = torch.from_numpy((1 / L) * A.T)
        W = W.float().to(self.device)
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(W)

    def forward(self, Y: torch.Tensor):
        """
        Forward pass through the LISTA model.

        Args:
            Y (torch.Tensor): Input tensor representing measurements.

        Returns:
            torch.Tensor: Output tensor representing the estimated sparse
            signal.
        """
        X = self.shrinkage(self._W(Y))

        if self.max_iter == 1:
            return X

        for _ in range(self.max_iter):
            X = self.shrinkage(self._W(Y) + self._S(X))
        return X
