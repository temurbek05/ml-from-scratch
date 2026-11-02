# Logistic regression: derivation

> Status: TODO template. To be written by the AI assistant (Claude) at the repository owner's direction, then checked against the tests.

## Problem and notation
TODO: binary labels, sigmoid, parameters (define every symbol and its shape).

## 1. Likelihood and log-loss
TODO: from the Bernoulli likelihood to the (mean) negative log-likelihood.

## 2. Gradient
TODO: show the gradient is $X^\top (p - y) / n$. Use the chain rule and the derivative of the sigmoid.

## 3. Numerical stability
TODO: what does the naive `1 / (1 + exp(-z))` do at $z = -1000$ and why? How does the stable version work? Why clip $p$ to $[\varepsilon, 1-\varepsilon]$ before the log?

## 4. Checking the gradient
TODO: finite differences: the formula, the step size, what a matching result proves and what it does not.

## 5. Fitting
TODO: the optimizer used, learning rate, stopping rule. What happens on perfectly separable data?

## Notes
TODO: short paragraph. Anything surprising, plus the repository owner's own comment if they want to add one.
