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
    raise NotImplementedError


def transform(X, mean, components):
    """Project X onto the components. (n, d) -> (n, k)."""
    raise NotImplementedError


def inverse_transform(Z, mean, components):
    """Map scores back to the original space. (n, k) -> (n, d)."""
    raise NotImplementedError
