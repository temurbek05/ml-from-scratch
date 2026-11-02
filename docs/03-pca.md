# PCA: derivation

> Written by the AI assistant (Claude) at the repository owner's direction. Every claim below was checked numerically against `mlscratch/pca.py` (see "Numerical checks"). Math in GitHub Markdown: `$...$` inline, `$$...$$` display.

## Problem and notation

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $n \times d$ | data matrix, one sample per row |
| $\mu = \frac1n X^\top \mathbf 1$ | $d$ | column means (`mean`) |
| $X_c = X - \mathbf 1 \mu^\top$ | $n \times d$ | centered data |
| $C = \dfrac{1}{n-1} X_c^\top X_c$ | $d \times d$ | sample covariance matrix (same as `np.cov`) |
| $k$ | scalar | number of components kept |
| $V_k$ | $d \times k$ | the first $k$ principal directions as columns (`components` is $V_k^\top$) |

PCA looks for $k$ orthonormal directions along which the data varies the most, and describes each sample by its coordinates along them. Scores are $Z = X_c V_k \in \mathbb R^{n \times k}$.

## 1. Centering

The variance of the data along a unit direction $u$ is the variance of the numbers $x_i^\top u$, which is measured **around their mean** $\mu^\top u$. If we skip centering, we are computing the second moment about the origin instead:

$$\frac1n X^\top X = \frac1n X_c^\top X_c + \mu \mu^\top .$$

The extra term $\mu\mu^\top$ is a rank-one matrix that depends only on where the data cloud sits, not on how it is spread. It adds the squared mean to the "variance" along $\mu$ and tilts the leading direction toward $\mu$ (a number is given in the checks). Centering removes the location, so what remains is purely the shape of the cloud. The mean is stored in `fit` and added back in `inverse_transform`.

## 2. Covariance and eigenvectors

The variance of the projections onto a unit vector $u$ ($u^\top u = 1$) is

$$\operatorname{Var}(X_c u) = \frac{1}{n-1} \lVert X_c u \rVert^2 = u^\top C u .$$

**First component.** Maximize $u^\top C u$ subject to $u^\top u = 1$.

$C$ is symmetric and positive semi-definite, so by the spectral theorem $C = \sum_{i=1}^d \lambda_i v_i v_i^\top$ with orthonormal eigenvectors $v_i$ and eigenvalues $\lambda_1 \ge \dots \ge \lambda_d \ge 0$. Write $u = \sum_i a_i v_i$, with $\sum_i a_i^2 = u^\top u = 1$. Then

$$u^\top C u = \sum_i \lambda_i a_i^2 \;\le\; \lambda_1 \sum_i a_i^2 = \lambda_1 ,$$

with equality when $u = v_1$ (up to sign; and the maximum is attained only inside the top eigenspace). So **the direction of maximal variance is the top eigenvector of $C$, and the variance along it is the top eigenvalue.** (The same result from Lagrange multipliers: the gradient of $u^\top C u - \lambda(u^\top u - 1)$ is $2Cu - 2\lambda u$, which vanishes exactly when $Cu = \lambda u$; the variance there is $u^\top C u = \lambda$.)

**Next components.** The second direction must be orthogonal to the first. Restricting to $u \perp v_1$ forces $a_1 = 0$, and the same bound gives $\lambda_2$, attained at $v_2$; and so on. The first $k$ principal directions are $v_1, \dots, v_k$. (That this greedy choice also maximizes the *total* variance of all $k$ directions together is the Ky Fan theorem; it is equivalent to the Eckart–Young statement in section 5. Both are quoted, not re-proved here.)

Directions are determined only up to sign (and up to rotation inside an eigenspace if eigenvalues tie), so tests compare $|\cos|$, not the vectors themselves.

## 3. SVD connection

Take the thin SVD of the centered data, $X_c = U \Sigma V^\top$, with $U \in \mathbb R^{n \times r}$ and $V \in \mathbb R^{d \times r}$ having orthonormal columns, $\Sigma = \operatorname{diag}(\sigma_1 \ge \dots \ge \sigma_r > 0)$, $r = \operatorname{rank} X_c$. Then, using $U^\top U = I$,

$$X_c^\top X_c = V \Sigma U^\top U \Sigma V^\top = V \Sigma^2 V^\top \quad\Rightarrow\quad C = V \,\frac{\Sigma^2}{n-1}\, V^\top .$$

This is an eigendecomposition of $C$. So:

- the **right singular vectors** (columns of $V$, the rows of `Vt` in NumPy) are the eigenvectors of $C$, i.e. the principal directions;
- the eigenvalues are $\lambda_i = \dfrac{\sigma_i^2}{n-1}$.

**Why SVD is preferred numerically.** Forming $C$ (or $X_c^\top X_c$) **squares the condition number**: $\kappa(X_c^\top X_c) = \kappa(X_c)^2$. An eigensolver on $C$ has an absolute error of roughly $\varepsilon \lambda_1$, so any eigenvalue below about $\varepsilon\lambda_1$, i.e. any singular value below about $\sqrt{\varepsilon}\,\sigma_1 \approx 10^{-8}\sigma_1$, is lost in rounding noise. The SVD works on $X_c$ directly and keeps resolving much smaller singular values. The test below shows this on a constructed example. For well-conditioned data, such as the test data, the two routes agree to about $10^{-14}$.

## 4. Explained variance

From section 3, the variance along the $i$-th principal direction is

$$\lambda_i = \frac{\sigma_i^2}{n-1}, \qquad \text{total variance} = \operatorname{tr}(C) = \sum_{i=1}^{d} \lambda_i = \frac{\lVert X_c \rVert_F^2}{n-1},$$

(the trace of a symmetric matrix is the sum of its eigenvalues). The share of variance kept by $k$ components is $\sum_{i \le k} \lambda_i \big/ \sum_{i} \lambda_i = \sum_{i \le k} \sigma_i^2 \big/ \sum_i \sigma_i^2$; the factor $n-1$ cancels. The scores $Z$ have covariance $\operatorname{diag}(\lambda_1, \dots, \lambda_k)$: they are uncorrelated, with these variances.

**Why $n - 1$ and not $n$.** The mean $\mu$ was estimated from the same data, which uses up one degree of freedom: the centered residuals sum to zero, so only $n-1$ of them are free. For i.i.d. samples with true variance $\sigma^2$,

$$\mathbb E \sum_{i=1}^n (x_i - \bar x)^2 = (n-1)\,\sigma^2 ,$$

so dividing by $n$ underestimates the variance by the factor $(n-1)/n$, and dividing by $n-1$ is unbiased (Bessel's correction; simulation below). It matches `np.cov`'s default. The choice scales every $\lambda_i$ by the same factor, so the **components themselves do not depend on it**; only the reported variances do.

## 5. Reconstruction error

Keeping $k$ components and mapping back gives $\hat X = X_c V_k V_k^\top + \mathbf 1 \mu^\top$, which is what `inverse_transform(transform(X))` returns. The error is the part of $X_c$ outside the span of $V_k$. Since $X_c V = U \Sigma$ and $V$ is orthogonal in the full-rank case,

$$\lVert X_c - X_c V_k V_k^\top \rVert_F^2 = \lVert X_c V_{\perp} \rVert_F^2 = \sum_{i > k} \sigma_i^2 ,$$

where $V_\perp$ holds the discarded directions. In the mean-squared form that the test uses (mean over all $n \cdot d$ entries):

$$\text{MSE}(k) = \frac{1}{n d} \sum_{i > k} \sigma_i^2 = \frac{n-1}{n d} \sum_{i > k} \lambda_i .$$

Consequences:
- It **decreases** with $k$: going from $k$ to $k+1$ removes the term $\sigma_{k+1}^2 / (nd)$, which is strictly positive whenever $\sigma_{k+1} > 0$. That is exactly `test_reconstruction_error_decreases_with_k`.
- It is **zero at $k = \operatorname{rank}(X_c)$** (up to rounding), which is `test_full_rank_reconstruction_is_exact`.
- By the Eckart–Young theorem (quoted, not proved here), no other approximation of $X_c$ with rank $k$ has a smaller Frobenius error. The error is large when the spectrum decays slowly and small when a few directions carry most of the variance.

## Numerical checks (all run against the implementation)

Data: the test recipe with $n = 500$, $d = 4$ and seed 0. The true scales are $5, 3, 1, 0.5$ (rotated) and the mean is $(1, -2, 0.5, 3)$. Singular values of $X_c$: $114.13,\ 62.85,\ 22.72,\ 11.37$, so $\lambda \approx 26.10,\ 7.92,\ 1.03,\ 0.26$.

| Check | Result |
|---|---|
| Variance of $X_c u_1$ vs $\lambda_1$ | $26.102787191995386$ vs $26.102787191995372$ |
| Largest variance among 100 000 random unit vectors | $26.063 \le \lambda_1 = 26.103$ |
| $\max_i \lvert \sigma_i^2/(n-1) - \lambda_i(\text{np.cov}) \rvert$ | $1.8 \cdot 10^{-14}$ |
| $\max_i \big\lvert\, \lvert\cos(v_i^{\text{SVD}}, v_i^{\text{eigh}})\rvert - 1 \big\rvert$ | $4.4 \cdot 10^{-16}$ |
| Covariance of scores vs $\operatorname{diag}(\lambda)$ | max abs difference $1.1 \cdot 10^{-14}$ |

**Centering.** On this data $\lVert \mu \rVert = 3.94$. Without centering, the top singular direction has $\lvert\cos\rvert = 0.934$ with the true first principal component (about $21^\circ$ off) and $\lvert\cos\rvert = 0.616$ with the mean direction; its "variance" $\sigma_1^2/n = 29.09$ differs from $\lambda_1 = 26.10$. So the mean mixes into the leading direction and inflates the variance; it does not take over completely here, because the spread (variance 26) is large compared with $\lVert\mu\rVert^2 = 15.5$.

**SVD vs covariance for tiny variances.** Constructed data ($n = 1000$, $d = 5$) with known singular values $\sigma = 1,\ 10^{-2},\ 10^{-5},\ 10^{-8},\ 10^{-10}$ ($\kappa(X) \approx 10^{10}$):

| true $\sigma_i$ | $1$ | $10^{-2}$ | $10^{-5}$ | $10^{-8}$ | $10^{-10}$ |
|---|---|---|---|---|---|
| relative error of $\lambda_i$ via SVD | $1.3\cdot10^{-15}$ | $7.9\cdot10^{-16}$ | $6.3\cdot10^{-14}$ | $1.5\cdot10^{-10}$ | $1.5\cdot10^{-8}$ |
| relative error of $\lambda_i$ via `eigh(np.cov)` | $0$ | $3.8\cdot10^{-13}$ | $2.5\cdot10^{-7}$ | $0.93$ | $2008$ |

The covariance route loses everything below $\sigma \approx 10^{-8}$, as predicted by $\sqrt\varepsilon$; the SVD route still resolves $\sigma = 10^{-10}$ to about 8 digits in $\sigma$.

**$n$ vs $n - 1$.** $n = 5$ samples of a unit-variance normal, $200\,000$ repetitions: mean of $\sum (x-\bar x)^2 / n$ is $0.799$ (theory: $4/5 = 0.8$); mean of $\sum(x-\bar x)^2/(n-1)$ is $0.998$ (theory: $1$; the gap is Monte-Carlo noise).

**Reconstruction error** (mean over all $n d$ entries):

| $k$ | measured MSE | $\frac{1}{nd}\sum_{i>k}\sigma_i^2$ | cumulative explained variance |
|---|---|---|---|
| 1 | $2.297670$ | $2.297670$ | $0.7392$ |
| 2 | $0.3226521$ | $0.3226521$ | $0.9634$ |
| 3 | $0.06465535$ | $0.06465535$ | $0.9927$ |
| 4 | $3.0 \cdot 10^{-30}$ | $0$ | $1.0000$ |

## Notes

- The decomposition does not depend on which of $n$ or $n-1$ is used, but the reported `explained_variance` does. `fit` uses $n - 1$ so the values match `np.cov` and the test.
- Two things in these checks were less dramatic than a textbook sentence suggests: skipping the centering tilts the leading direction by about $21^\circ$ on this data rather than pointing it at the mean, and the covariance route is only a problem once singular values are below about $10^{-8}$ times the largest; for ordinary data both routes give the same answer.
- `fit` raises `ValueError` if $k$ is outside $[1, \min(n, d)]$ (more components than the rank allows cannot exist).
