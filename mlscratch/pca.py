"""PCA via the SVD (NumPy only). YOU implement every function.

Then write docs/03-pca.md: show why the right singular vectors of the centered
data are the eigenvectors of the covariance matrix, and how explained variance
relates to the singular values.
"""
import numpy as np


def fit(X, k):
    """Return (mean, components, explained_variance).

    mean: (d,)   components: (k, d), rows orthonormal   explained_variance: (k,)
    Center the data, use np.linalg.svd(..., full_matrices=False), and use the
    sample variance (divide by n - 1) so it matches np.cov.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if not 1 <= k <= min(n, d):
        raise ValueError(f"k must be in [1, {min(n, d)}], got {k}")
    mean = X.mean(axis=0)
    Xc = X - mean
    # Xc = U diag(S) Vt; rows of Vt are the principal directions, S is sorted descending
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    explained_variance = S[:k] ** 2 / (n - 1)
    return mean, components, explained_variance


def transform(X, mean, components):
    """Project X onto the components. (n, d) -> (n, k)."""
    return (np.asarray(X, dtype=float) - mean) @ components.T


def inverse_transform(Z, mean, components):
    """Map scores back to the original space. (n, k) -> (n, d)."""
    return Z @ components + mean
