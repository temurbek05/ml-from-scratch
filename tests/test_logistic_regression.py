import unittest

import numpy as np

from mlscratch import logistic_regression as lg


def make_clusters(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-2.0, scale=1.0, size=(n, 2))
    X1 = rng.normal(loc=2.0, scale=1.0, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


class TestLogisticRegression(unittest.TestCase):
    def test_sigmoid_basic(self):
        self.assertAlmostEqual(float(lg.sigmoid(np.array(0.0))), 0.5)
        np.testing.assert_allclose(lg.sigmoid(np.array([-1.0, 1.0])).sum(), 1.0)

    def test_sigmoid_is_numerically_stable(self):
        with np.errstate(over="raise", invalid="raise"):
            out = lg.sigmoid(np.array([-1000.0, 1000.0]))
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)

    def test_log_loss_value(self):
        y = np.array([1.0, 0.0])
        p = np.array([0.9, 0.2])
        expected = -(np.log(0.9) + np.log(0.8)) / 2
        self.assertAlmostEqual(lg.log_loss(y, p), expected, places=10)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        Xb = np.hstack([np.ones((50, 1)), rng.normal(size=(50, 3))])
        y = (rng.random(50) > 0.5).astype(float)
        w = rng.normal(size=4)
        for l2 in (0.0, 0.3):
            _, grad = lg.loss_and_grad(w, Xb, y, l2=l2)
            num = np.zeros_like(w)
            h = 1e-6
            for i in range(len(w)):
                wp, wm = w.copy(), w.copy()
                wp[i] += h
                wm[i] -= h
                num[i] = (lg.loss_and_grad(wp, Xb, y, l2=l2)[0]
                          - lg.loss_and_grad(wm, Xb, y, l2=l2)[0]) / (2 * h)
            np.testing.assert_allclose(grad, num, atol=1e-6)

    def test_fit_separates_clusters(self):
        X, y = make_clusters()
        w = lg.fit(X, y, lr=0.5, n_iter=1000)
        p = lg.predict_proba(X, w)
        self.assertTrue(np.all((p > 0) & (p < 1)))
        acc = (lg.predict(X, w) == y).mean()
        self.assertGreaterEqual(acc, 0.95)

    def test_l2_shrinks_weights(self):
        X, y = make_clusters()
        w0 = lg.fit(X, y, lr=0.5, n_iter=1000, l2=0.0)
        w1 = lg.fit(X, y, lr=0.5, n_iter=1000, l2=1.0)
        self.assertLess(np.linalg.norm(w1[1:]), np.linalg.norm(w0[1:]))


if __name__ == "__main__":
    unittest.main()
