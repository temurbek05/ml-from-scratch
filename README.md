# ML from scratch

Linear regression, logistic regression and PCA, implemented with NumPy only, with the mathematical derivation written next to each one.

> Status: scaffold only. Skeleton and tests were provided by an AI assistant (Claude). The implementations and derivations are being written by the AI assistant, directed and reviewed by the repository owner. Update this note honestly as the work progresses.

## Why this repo exists
To show how each model works inside, not only how to call a library: each result is derived step by step, implemented, and checked against tests (including a finite-difference gradient check).

## Setup
```
python -m pip install numpy
python -m unittest discover -s tests -v
```
Every test fails at first (`NotImplementedError`). The goal is to make all 17 pass.

## Order of work
1. `mlscratch/linear_regression.py` then `docs/01-linear-regression.md`
2. `mlscratch/logistic_regression.py` then `docs/02-logistic-regression.md`
3. `mlscratch/pca.py` then `docs/03-pca.md`

## Rules
- No scikit-learn and no ready-made solvers for the parts the docstrings say to write.
- Derive first (matrix calculus, linear algebra), then implement, then type the math into `docs/`.
- Tests in `tests/` are not edited to make code pass.
- Commit often with real messages, on the days the work actually happens.

## What each derivation must contain
- **Linear regression:** the loss, its gradient, the normal equation, why `np.linalg.solve` is preferred over an explicit inverse, what ridge changes, when gradient descent matches the closed form.
- **Logistic regression:** the likelihood, the log-loss, the gradient `X^T (p - y) / n`, why the naive sigmoid overflows, what a finite-difference check proves.
- **PCA:** centering, covariance, SVD, explained variance, reconstruction error.

## Extensions (after the tests pass)
A notebook that compares this implementation with scikit-learn on a real dataset (same predictions to numerical precision), plus the loss curve, a logistic decision boundary and the PCA scree plot. Keep the results table here up to date.

## Who did what
- Provided as scaffold (AI): function stubs and the 17 tests.
- Implementation, derivations, notebook: TODO (fill in honestly when done).
