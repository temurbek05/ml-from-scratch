# Linear regression: derivation

> Written by the AI assistant (Claude) at the repository owner's direction. Every claim below was checked numerically against `mlscratch/linear_regression.py` (see "Numerical checks"). Math in GitHub Markdown: `$...$` inline, `$$...$$` display.

## Problem and notation

| Symbol | Shape | Meaning |
|---|---|---|
| $n$, $d$ | scalars | number of samples, number of features |
| $X$ | $n \times d$ | feature matrix (no bias column) |
| $X_b = [\mathbf 1 \;\; X]$ | $n \times (d+1)$ | $X$ with a leading column of ones (`add_bias`) |
| $y$ | $n$ | targets |
| $w = (w_0, w_1, \dots, w_d)^\top$ | $d+1$ | weights; $w_0$ is the bias (intercept) |
| $r = X_b w - y$ | $n$ | residual vector |

The model predicts $\hat y = X_b w$. We look for the $w$ that makes $\hat y$ closest to $y$ in mean squared error.

## 1. Loss and its gradient

$$L(w) = \frac1n \lVert X_b w - y \rVert^2 = \frac1n\, r^\top r .$$

This is exactly `mse(y, X_b @ w)` (mean of squared errors, no extra factor $\tfrac12$).

Expand the product:

$$L(w) = \frac1n\left( w^\top X_b^\top X_b\, w \;-\; 2\, y^\top X_b\, w \;+\; y^\top y \right).$$

Two standard matrix-calculus rules are needed. For a vector $c$ and a **symmetric** matrix $A$:

$$\nabla_w (c^\top w) = c, \qquad \nabla_w (w^\top A w) = (A + A^\top) w = 2 A w .$$

Here $A = X_b^\top X_b$ is symmetric and $c = X_b^\top y$ (because $y^\top X_b w = (X_b^\top y)^\top w$). Therefore

$$\boxed{\nabla L(w) = \frac2n \left( X_b^\top X_b\, w - X_b^\top y \right) = \frac2n\, X_b^\top (X_b w - y)} .$$

The Hessian is $H = \nabla^2 L = \frac2n X_b^\top X_b$. For any vector $v$, $v^\top H v = \frac2n \lVert X_b v \rVert^2 \ge 0$, so $H$ is positive semi-definite and $L$ is **convex**: every point with zero gradient is a global minimum.

## 2. Normal equation

Set $\nabla L(w) = 0$:

$$\boxed{X_b^\top X_b \; w = X_b^\top y} .$$

When does a solution exist, and when is it unique?

- **Existence: always.** The range of $X_b^\top X_b$ equals the range of $X_b^\top$, and $X_b^\top y$ lies in that range, so the system is consistent.
- **Uniqueness:** the solution is unique if and only if $X_b^\top X_b$ is invertible, i.e. $X_b$ has full column rank: $n \ge d+1$ and no feature is an exact linear combination of the others (or of the constant column). Then $w = (X_b^\top X_b)^{-1} X_b^\top y$.
- If $X_b$ is rank-deficient, there are infinitely many solutions. They all give the same predictions $X_b w$ but different weights. Ridge (section 4) removes this ambiguity.

## 3. Why `solve` and not `inv`

Mathematically $w = A^{-1} b$ with $A = X_b^\top X_b$, $b = X_b^\top y$. Numerically the two ways of computing it differ:

- **Cost.** `np.linalg.solve(A, b)` does one LU factorization with partial pivoting, about $\tfrac23 m^3$ flops for an $m \times m$ matrix ($m = d+1$). `np.linalg.inv(A)` effectively solves $m$ systems (one per column of the identity), about $2 m^3$ flops, and then still needs a matrix-vector product.
- **Backward stability.** The computed $\hat w$ from `solve` satisfies $(A + \delta A)\hat w = b$ with $\lVert \delta A \rVert \approx \varepsilon \lVert A \rVert$ ($\varepsilon$ = machine precision), so the residual $\lVert A \hat w - b \rVert$ is tiny. Computing $A^{-1}$ first and multiplying afterwards gives no such guarantee.
- **What neither can fix: conditioning.** The error in $w$ is limited by $\kappa(A) \cdot \varepsilon$, whichever method is used. Note also that forming $X_b^\top X_b$ **squares** the condition number: $\kappa(X_b^\top X_b) = \kappa(X_b)^2$. (QR or SVD on $X_b$ avoids this; here we follow the normal equations, as the task specifies.)

Honest result of the experiment below: on a near-collinear problem the weight error was **about the same** for both methods ($0.0147$ vs $0.0137$; the problem is ill-posed, so neither can recover the true weights). The clear difference was the residual $\lVert A w - b \rVert$: $2\cdot10^{-14}$ for `solve` against $1.38$ for `inv`. So the argument for `solve` is cost and a small residual, not better weights.

## 4. Ridge

Add an $\ell_2$ penalty on the slopes only:

$$J(w) = \lVert X_b w - y \rVert^2 + \lambda \lVert w_{1:} \rVert^2 = \lVert X_b w - y \rVert^2 + \lambda\, w^\top P w, \qquad P = \operatorname{diag}(0, 1, \dots, 1).$$

(Here the data term is the **sum** of squares, not the mean, because this is what `fit_closed_form` solves. With the mean-form loss $L$, $\lambda$ would be divided by $n$.)

Gradient: $\nabla J = 2\left(X_b^\top X_b w - X_b^\top y + \lambda P w\right)$. Setting it to zero:

$$\boxed{(X_b^\top X_b + \lambda P)\, w = X_b^\top y} .$$

What changes:

- **The matrix becomes invertible for any $\lambda > 0$**, even for collinear features. If $v^\top (X_b^\top X_b + \lambda P) v = \lVert X_b v \rVert^2 + \lambda \lVert v_{1:} \rVert^2 = 0$, then $v_{1:} = 0$ and $X_b v = v_0 \mathbf 1 = 0$, so $v_0 = 0$. Hence the matrix is positive definite.
- **Slopes shrink toward zero**, more strongly for larger $\lambda$ (experiment below).

**Why the bias is not penalized.** The intercept only sets the overall level of $y$. If we shift every target by a constant, $y \to y + c$, the right fit is the same slopes and an intercept that moves by $c$. Penalizing $w_0$ would break that: the slopes would change with the origin of $y$. Also, the first row of the (unpenalized) normal equation reads $n w_0 + (\mathbf 1^\top X) w_{1:} = \mathbf 1^\top y$, i.e.

$$w_0 = \bar y - \bar x^\top w_{1:},$$

so with centered features ($\bar x = 0$) the intercept is exactly $\bar y$, for every $\lambda$. This is what `test_ridge_shrinks_slopes_but_not_bias` checks.

## 5. Gradient descent

Update rule with learning rate $\eta$ (`lr`), starting from $w_0 = 0$:

$$w_{k+1} = w_k - \eta \nabla L(w_k) = w_k - \eta\, \frac2n X_b^\top (X_b w_k - y).$$

**When does it match the closed form?** Let $w^\star$ solve the normal equation, so $H w^\star = \frac2n X_b^\top y$ and therefore $\nabla L(w) = H (w - w^\star)$. Then

$$w_{k+1} - w^\star = (I - \eta H)(w_k - w^\star), \qquad w_k - w^\star = (I - \eta H)^k (w_0 - w^\star).$$

$H$ is symmetric, so write $H = Q \Lambda Q^\top$ with eigenvalues $\lambda_1 \ge \dots \ge \lambda_{d+1}$. In the eigenbasis each component of the error is multiplied by $(1 - \eta \lambda_i)$ at every step. Hence:

- **Convergence** to $w^\star$ happens iff $|1 - \eta \lambda_i| < 1$ for all $i$, i.e.

$$\boxed{0 < \eta < \frac{2}{\lambda_{\max}(H)}, \qquad \lambda_{\max}(H) = \frac2n\, \lambda_{\max}(X_b^\top X_b)} .$$

  This needs all $\lambda_i > 0$ (full column rank). The speed is set by the slowest direction, $\max_i |1 - \eta\lambda_i|$; the ratio $\lambda_{\max}/\lambda_{\min}$ again decides how hard the problem is. If $X_b$ is rank-deficient, GD still converges, but to the solution closest to the start $w_0 = 0$ (components in the null space never move).
- **Too large a step.** If $\eta > 2/\lambda_{\max}$, the top eigen-direction is multiplied by a factor with magnitude above 1 at every step, so the error grows geometrically and the loss diverges.
- **Monotone loss.** Because $L(w) = L(w^\star) + \tfrac12 (w - w^\star)^\top H (w - w^\star)$, the loss along eigen-direction $i$ is proportional to $\lambda_i (1 - \eta \lambda_i)^{2k}$. It never increases iff $|1 - \eta\lambda_i| \le 1$ for all $i$, i.e. $0 < \eta \le 2/\lambda_{\max}$. This is what `test_gradient_descent_loss_never_increases` relies on.

## Numerical checks (all run against the implementation)

Data: $n = 200$, $d = 2$, $y = 3 + 2x_1 - x_2 + 0.1\,\varepsilon$, seed 0 (the same recipe as the tests).

| Check | Result |
|---|---|
| Analytic gradient vs central finite differences (step $10^{-6}$) | max abs difference $1.9 \cdot 10^{-9}$ |
| $\lVert \nabla L \rVert$ at the closed-form solution | $2.1 \cdot 10^{-15}$ |
| Gradient descent ($\eta = 0.1$, 2000 steps) vs closed form | max abs difference $1.6 \cdot 10^{-15}$ |
| Eigenvalues of $H$ | $1.73,\; 2.06,\; 2.17$, so $2/\lambda_{\max} = 0.920$ |
| $\eta = 0.9 \cdot 0.920$ (200 steps) | loss $12.3 \to 0.0098$ |
| $\eta = 1.0 \cdot 0.920$ | loss $12.3 \to 1.04$ (bounded, not converging: top mode factor is $-1$) |
| $\eta = 1.1 \cdot 0.920$ | loss $12.3 \to 3.4\cdot10^{31}$ (diverges) |
| Near-collinear data: $\kappa(X_b) = 1.7\cdot10^{7}$, $\kappa(X_b^\top X_b) = 3.0\cdot10^{14}$ | weight error: `solve` 0.0147, `inv` 0.0137; residual $\lVert Aw-b\rVert$: `solve` $2.0\cdot10^{-14}$, `inv` $1.38$ |

Ridge on the same data (slopes shrink; the intercept moves toward $\bar y = 2.78$):

| $\lambda$ | $w$ | $\lVert w_{1:} \rVert$ |
|---|---|---|
| 0 | $(2.9997,\; 1.995,\; -0.9881)$ | 2.226 |
| 10 | $(2.9885,\; 1.8934,\; -0.9366)$ | 2.112 |
| 100 | $(2.9228,\; 1.2986,\; -0.6374)$ | 1.447 |
| 1000 | $(2.8143,\; 0.3137,\; -0.1518)$ | 0.349 |

Here the features are not exactly centered: the mean of the features is $\bar x = (-0.098, 0.025)$, so $w_0$ is not fixed at $\bar y$: it follows $w_0 = \bar y - \bar x^\top w_{1:}$ exactly (checked: the two sides agree to all printed digits for every $\lambda$) and drifts toward $\bar y$ as the slopes shrink. With centered features $w_0 = \bar y$ for every $\lambda$, which is what the test checks.

## Notes

- The learning-rate threshold is tied to the data: it is $2/\lambda_{\max}(H)$, so rescaling the features changes which `lr` is safe. The default `lr=0.1` is safe for standardized features ($H \approx 2I$, threshold about 1) but would not be for features on a scale of 100.
- The surprise in these checks was section 3: I expected `solve` to give visibly better weights than `inv`, and it did not. The advantage showed up only in the residual.
