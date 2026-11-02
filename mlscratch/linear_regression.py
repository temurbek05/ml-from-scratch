"""Linear regression from scratch (NumPy only). YOU implement every function.

Rules: no scikit-learn, no np.polyfit, no np.linalg.lstsq. Write the math yourself.
Then write docs/01-linear-regression.md: derive the normal equation and the gradient.
"""
import numpy as np


def add_bias(X):
    """Return X with a leading column of ones. (n, d) -> (n, d + 1)."""
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_closed_form(X, y, l2=0.0):
    """Least-squares weights, shape (d + 1,), bias at index 0.

    Solve (Xb^T Xb + l2 * P) w = Xb^T y, where P is the identity with P[0, 0] = 0
    (the bias is NOT regularized). Use np.linalg.solve, not np.linalg.inv.
    """
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    P = np.eye(Xb.shape[1])
    P[0, 0] = 0.0  # do not penalize the bias
    A = Xb.T @ Xb + l2 * P
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def predict(X, w):
    """Predictions for X (without a bias column) given weights w."""
    return add_bias(X) @ w


def mse(y, y_hat):
    """Mean squared error."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.mean((y - y_hat) ** 2))


def fit_gradient_descent(X, y, lr=0.1, n_iter=2000):
    """Batch gradient descent on the MSE, starting from w = 0.

    Return (w, losses) where losses[i] is the MSE measured BEFORE update i
    (so len(losses) == n_iter). Derive the gradient by hand first.
    """
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    n = Xb.shape[0]
    w = np.zeros(Xb.shape[1])
    losses = np.empty(n_iter)
    for i in range(n_iter):
        r = Xb @ w - y                  # residuals
        losses[i] = np.mean(r ** 2)     # loss before this update
        w = w - lr * (2.0 / n) * (Xb.T @ r)  # grad of the MSE = (2/n) Xb^T r
    return w, losses
