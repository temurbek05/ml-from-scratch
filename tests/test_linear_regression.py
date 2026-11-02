import unittest

import numpy as np

from mlscratch import linear_regression as lr


def make_data(n=200, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 3.0 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + noise * rng.normal(size=n)
    return X, y


class TestLinearRegression(unittest.TestCase):
    def test_add_bias(self):
        X = np.arange(6.0).reshape(3, 2)
        Xb = lr.add_bias(X)
        self.assertEqual(Xb.shape, (3, 3))
        np.testing.assert_allclose(Xb[:, 0], 1.0)
        np.testing.assert_allclose(Xb[:, 1:], X)

    def test_closed_form_recovers_weights(self):
        X, y = make_data()
        w = lr.fit_closed_form(X, y)
        self.assertEqual(w.shape, (3,))
        np.testing.assert_allclose(w, [3.0, 2.0, -1.0], atol=0.01)

    def test_predict_shape_and_values(self):
        X, y = make_data()
        w = lr.fit_closed_form(X, y)
        y_hat = lr.predict(X, w)
        self.assertEqual(y_hat.shape, y.shape)
        self.assertLess(lr.mse(y, y_hat), 1e-3)

    def test_mse_value(self):
        self.assertAlmostEqual(lr.mse(np.array([1.0, 2.0]), np.array([2.0, 4.0])), 2.5)

    def test_ridge_shrinks_slopes_but_not_bias(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 2))
        X = X - X.mean(axis=0)  # centered features
        y = 5.0 + 2.0 * X[:, 0] + 0.5 * rng.normal(size=300)
        w_ols = lr.fit_closed_form(X, y, l2=0.0)
        w_ridge = lr.fit_closed_form(X, y, l2=100.0)
        self.assertLess(np.linalg.norm(w_ridge[1:]), np.linalg.norm(w_ols[1:]))
        # with centered features and an unpenalized bias, the intercept is mean(y)
        self.assertAlmostEqual(w_ridge[0], y.mean(), places=8)

    def test_gradient_descent_matches_closed_form(self):
        X, y = make_data(noise=0.1)
        w_cf = lr.fit_closed_form(X, y)
        w_gd, losses = lr.fit_gradient_descent(X, y, lr=0.1, n_iter=2000)
        np.testing.assert_allclose(w_gd, w_cf, atol=1e-2)
        self.assertEqual(len(losses), 2000)

    def test_gradient_descent_loss_never_increases(self):
        X, y = make_data(noise=0.1)
        _, losses = lr.fit_gradient_descent(X, y, lr=0.05, n_iter=300)
        diffs = np.diff(np.asarray(losses))
        self.assertTrue(np.all(diffs <= 1e-12))
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
