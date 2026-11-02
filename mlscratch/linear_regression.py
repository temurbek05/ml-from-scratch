"""Linear regression from scratch (NumPy only). YOU implement every function.

Rules: no scikit-learn, no np.polyfit, no np.linalg.lstsq. Write the math yourself.
Then write docs/01-linear-regression.md: derive the normal equation and the gradient.
"""
import numpy as np


def add_bias(X):
    """Return X with a leading column of ones. (n, d) -> (n, d + 1)."""
    raise NotImplementedError


def fit_closed_form(X, y, l2=0.0):
    """Least-squares weights, shape (d + 1,), bias at index 0.

    Solve (Xb^T Xb + l2 * P) w = Xb^T y, where P is the identity with P[0, 0] = 0
    (the bias is NOT regularized). Use np.linalg.solve, not np.linalg.inv.
    """
    raise NotImplementedError


def predict(X, w):
    """Predictions for X (without a bias column) given weights w."""
    raise NotImplementedError


def mse(y, y_hat):
    """Mean squared error."""
    raise NotImplementedError


def fit_gradient_descent(X, y, lr=0.1, n_iter=2000):
    """Batch gradient descent on the MSE, starting from w = 0.

    Return (w, losses) where losses[i] is the MSE measured BEFORE update i
    (so len(losses) == n_iter). Derive the gradient by hand first.
    """
    raise NotImplementedError
