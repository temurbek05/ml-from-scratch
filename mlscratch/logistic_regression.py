"""Logistic regression from scratch (NumPy only). YOU implement every function.

Then write docs/02-logistic-regression.md: derive the gradient of the log-loss and
explain why the sigmoid must be implemented in a numerically stable way.
"""
import numpy as np

from .linear_regression import add_bias


def sigmoid(z):
    """Numerically stable sigmoid. Must not overflow for z = -1000 or z = 1000.

    Hint: treat z >= 0 and z < 0 separately.
    """
    z = np.asarray(z, dtype=float)
    # exp is only ever taken of a non-positive number, so it cannot overflow
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_loss(y, p, eps=1e-12):
    """Mean binary cross-entropy. Clip p into [eps, 1 - eps] before taking logs."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def loss_and_grad(w, Xb, y, l2=0.0):
    """Return (loss, gradient) for weights w on Xb (bias column ALREADY included).

    loss = log_loss(y, sigmoid(Xb @ w)) + 0.5 * l2 * ||w[1:]||^2   (bias not penalized)
    The test checks your gradient against finite differences.
    """
    y = np.asarray(y, dtype=float)
    n = Xb.shape[0]
    p = sigmoid(Xb @ w)
    loss = log_loss(y, p) + 0.5 * l2 * np.sum(w[1:] ** 2)
    grad = Xb.T @ (p - y) / n           # (1/n) Xb^T (p - y)
    grad[1:] += l2 * w[1:]              # derivative of the penalty; bias is skipped
    return float(loss), grad


def fit(X, y, lr=0.5, n_iter=2000, l2=0.0):
    """Train by gradient descent from w = 0. X has NO bias column; add it here.

    Return weights of shape (d + 1,), bias first.
    """
    Xb = add_bias(X)
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        _, grad = loss_and_grad(w, Xb, y, l2=l2)
        w = w - lr * grad
    return w


def predict_proba(X, w):
    """P(y = 1 | x) for X without a bias column."""
    return sigmoid(add_bias(X) @ w)


def predict(X, w, threshold=0.5):
    """Hard labels (0.0 or 1.0)."""
    return (predict_proba(X, w) >= threshold).astype(float)
