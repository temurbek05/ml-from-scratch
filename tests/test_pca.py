import unittest

import numpy as np

from mlscratch import pca


def make_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    scales = np.array([5.0, 3.0, 1.0, 0.5])
    Z = rng.normal(size=(n, 4)) * scales
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))  # random rotation
    return Z @ Q.T + np.array([1.0, -2.0, 0.5, 3.0])


class TestPCA(unittest.TestCase):
    def test_components_are_orthonormal(self):
        X = make_data()
        mean, comps, var = pca.fit(X, 3)
        self.assertEqual(comps.shape, (3, 4))
        np.testing.assert_allclose(comps @ comps.T, np.eye(3), atol=1e-10)

    def test_matches_covariance_eigendecomposition(self):
        X = make_data()
        mean, comps, var = pca.fit(X, 4)
        np.testing.assert_allclose(mean, X.mean(axis=0), atol=1e-12)
        vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        np.testing.assert_allclose(var, vals, rtol=1e-6)
        for i in range(4):
            self.assertAlmostEqual(abs(float(comps[i] @ vecs[:, i])), 1.0, places=6)

    def test_full_rank_reconstruction_is_exact(self):
        X = make_data()
        mean, comps, _ = pca.fit(X, 4)
        Z = pca.transform(X, mean, comps)
        self.assertEqual(Z.shape, (500, 4))
        X_rec = pca.inverse_transform(Z, mean, comps)
        np.testing.assert_allclose(X_rec, X, atol=1e-9)

    def test_reconstruction_error_decreases_with_k(self):
        X = make_data()
        errors = []
        for k in (1, 2, 3, 4):
            mean, comps, _ = pca.fit(X, k)
            X_rec = pca.inverse_transform(pca.transform(X, mean, comps), mean, comps)
            errors.append(np.mean((X - X_rec) ** 2))
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-12)


if __name__ == "__main__":
    unittest.main()
