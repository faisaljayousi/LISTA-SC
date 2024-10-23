"""
This script implements the Iterative Shrinkage-Thresholding Algorithm (ISTA)
for sparse recovery, along with the soft-thresholding proximal operator.
"""

import numpy as np


def prox(x, theta):
    """
    Soft-thresholding function (proximal operator of $|| . ||_1$).
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))


def ista(X, W_d, alpha, L, max_iter, eps):
    """
    Perform Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse
    recovery.
    """
    eig, _ = np.linalg.eig(W_d.T @ W_d)
    assert L > np.max(
        eig
    ), f"`L` must be greater than than maximum eigenvalue: {np.max(eig)}"

    recon_errors = []
    Z_previous = np.zeros((W_d.shape[1], 1))

    for _ in range(max_iter):
        # Compute gradient
        grad = W_d.T @ (W_d @ Z_previous - X)

        # Update estimate
        Z_current = prox(Z_previous - 1 / L * grad, alpha / L)

        # Check for convergence
        if np.sum(np.abs(Z_current - Z_previous)) <= eps:
            break

        # Update estimate
        Z_previous = Z_current

        # Calculate reconstruction error
        recon_error = np.linalg.norm(X - W_d @ Z_current, ord=2) ** 2
        recon_errors.append(recon_error)

    return Z_current, recon_errors
