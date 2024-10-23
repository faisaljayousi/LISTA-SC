"""
This script simulates sparse signals and their corresponding measurements.
"""

import numpy as np
from scipy.linalg import orth


def generate(
    m: int, n: int, k: int, N: int, w_d: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate sparse signals and their measurements.

    Args:
        m (int):
        n (int):
        k (int): sparsity level
        N (int):

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - x (np.ndarray): Measurements of shape (N, n).
            - z (np.ndarray): Sparse signals of shape (N, m).
            - w_d (np.ndarray): Dictionary of shape (n, m).
    """
    if w_d is None:
        # Generate dictionary
        w_d = np.random.randn(n, m)
        w_d = np.transpose(orth(np.transpose(w_d)))

    # Generate sparse signals (z) and measurements (x)
    z = np.zeros((N, m))
    x = np.zeros((N, n))

    for i in range(N):
        # Randomly select indices for non-zero elements in the sparse signal
        index_k = np.random.choice(a=m, size=k, replace=False)
        z[i, index_k] = 0.5 * np.random.randn(k, 1).ravel()
        x[i] = np.dot(w_d, z[i, :])

    return x, z, w_d
