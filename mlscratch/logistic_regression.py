"""Logistic regression from scratch (NumPy only). YOU implement every function.

Then write docs/02-logistic-regression.md: derive the gradient of the log-loss and
explain why the sigmoid must be implemented in a numerically stable way.
"""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid. Must not overflow for z = -1000 or z = 1000.

    Hint: treat z >= 0 and z < 0 separately.
    """
    raise NotImplementedError


def log_loss(y, p, eps=1e-12):
    """Mean binary cross-entropy. Clip p into [eps, 1 - eps] before taking logs."""
    raise NotImplementedError


def loss_and_grad(w, Xb, y, l2=0.0):
    """Return (loss, gradient) for weights w on Xb (bias column ALREADY included).

    loss = log_loss(y, sigmoid(Xb @ w)) + 0.5 * l2 * ||w[1:]||^2   (bias not penalized)
    The test checks your gradient against finite differences.
    """
    raise NotImplementedError


def fit(X, y, lr=0.5, n_iter=2000, l2=0.0):
    """Train by gradient descent from w = 0. X has NO bias column; add it here.

    Return weights of shape (d + 1,), bias first.
    """
    raise NotImplementedError


def predict_proba(X, w):
    """P(y = 1 | x) for X without a bias column."""
    raise NotImplementedError


def predict(X, w, threshold=0.5):
    """Hard labels (0.0 or 1.0)."""
    raise NotImplementedError
