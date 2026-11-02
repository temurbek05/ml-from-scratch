# Logistic regression: derivation

> Written by the AI assistant (Claude) at the repository owner's direction. Every claim below was checked numerically against `mlscratch/logistic_regression.py` (see "Numerical checks"). Math in GitHub Markdown: `$...$` inline, `$$...$$` display.

## Problem and notation

| Symbol | Shape | Meaning |
|---|---|---|
| $n$, $d$ | scalars | number of samples, number of features |
| $X_b$ | $n \times (d+1)$ | features with a leading column of ones (the bias column) |
| $x_i^\top$ | $d+1$ | row $i$ of $X_b$ |
| $y$ | $n$, entries in $\{0, 1\}$ | binary labels |
| $w$ | $d+1$ | weights, bias first |
| $z = X_b w$, $z_i = x_i^\top w$ | $n$ | logits (scores) |
| $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | | sigmoid, maps $\mathbb R \to (0,1)$ |
| $p = \sigma(z)$ | $n$ | $p_i = P(y_i = 1 \mid x_i)$ |

## 1. Likelihood and log-loss

Model: $y_i \sim \mathrm{Bernoulli}(p_i)$ independently, so $P(y_i \mid x_i) = p_i^{\,y_i} (1 - p_i)^{1 - y_i}$ (this equals $p_i$ when $y_i = 1$ and $1 - p_i$ when $y_i = 0$). The likelihood of the whole data set is the product

$$\mathcal L(w) = \prod_{i=1}^n p_i^{\,y_i} (1 - p_i)^{1 - y_i}.$$

Maximizing $\mathcal L$ is the same as minimizing $-\frac1n \log \mathcal L$ (the log is increasing, the minus sign flips max to min, and $\frac1n$ only rescales). Using $\log$ of a product $=$ sum of logs:

$$\boxed{L(w) = -\frac1n \sum_{i=1}^n \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big]} .$$

This is the mean binary cross-entropy (`log_loss`). With an $\ell_2$ penalty on the slopes (bias excluded), the code minimizes

$$L_\lambda(w) = L(w) + \frac{\lambda}{2} \lVert w_{1:} \rVert^2 .$$

## 2. Gradient

Take one sample and write $\ell(p, y) = -[\,y \log p + (1-y)\log(1-p)\,]$. The chain is $w \to z \to p \to \ell$.

1. Derivative of the sigmoid. With $\sigma(z) = (1 + e^{-z})^{-1}$,
$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)\,\big(1 - \sigma(z)\big).$$
So $\dfrac{\partial p}{\partial z} = p (1 - p)$.
2. Derivative of the loss with respect to $p$:
$$\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p} = \frac{-y(1-p) + (1-y)p}{p(1-p)} = \frac{p - y}{p (1 - p)}.$$
3. Combine. The factor $p(1-p)$ cancels:
$$\frac{\partial \ell}{\partial z} = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y .$$
4. Since $z_i = x_i^\top w$, $\dfrac{\partial z_i}{\partial w} = x_i$, hence $\dfrac{\partial \ell_i}{\partial w} = (p_i - y_i)\, x_i$.

Averaging over the samples and stacking the rows $x_i^\top$ into $X_b$:

$$\boxed{\nabla L(w) = \frac1n \sum_i (p_i - y_i)\, x_i = \frac1n X_b^\top (p - y)} .$$

With the penalty, $\nabla L_\lambda = \frac1n X_b^\top (p - y) + \lambda\, (0, w_1, \dots, w_d)^\top$: the bias entry gets no penalty term.

The cancellation in step 3 is why the gradient is so clean: the cross-entropy is exactly the loss whose derivative with respect to the logit is "prediction minus label", the same form as in linear regression.

**Convexity.** Differentiating once more, $\partial p_i / \partial w = p_i(1 - p_i)\, x_i$, so the Hessian is

$$H = \frac1n X_b^\top S X_b, \qquad S = \operatorname{diag}\big(p_i (1 - p_i)\big) \succeq 0,$$

which is positive semi-definite: $L$ is convex and a point with zero gradient is a global minimum. Because $p(1-p) \le \tfrac14$, the gradient is Lipschitz with constant $\mathsf L = \lambda_{\max}(X_b^\top X_b) / (4n)$ (and $\mathsf L + \lambda$ with the penalty). A standard result then says gradient descent with $\eta < 2 / \mathsf L$ never increases the loss. This is a *sufficient* condition (see the checks).

## 3. Numerical stability

**Sigmoid.** The textbook formula $1 / (1 + e^{-z})$ evaluates $e^{-z}$, which is $e^{1000}$ for $z = -1000$. That exceeds the largest float64 (about $1.8 \cdot 10^{308}$), so NumPy returns `inf` and raises an `overflow` warning. In this particular formula the final value is still right ($1/\infty = 0$), but the overflow flag is raised; `test_sigmoid_is_numerically_stable` turns it into an error. Other algebraically equal forms are worse: $e^z / (1 + e^z)$ at $z = +1000$ computes $\infty / \infty$, which is `nan`.

The stable version never exponentiates a positive number. For $z \ge 0$ use $\sigma(z) = \dfrac{1}{1 + e^{-z}}$; for $z < 0$ multiply top and bottom by $e^{z}$:

$$\sigma(z) = \frac{e^{z}}{1 + e^{z}} .$$

Both are the same function, and in both the exponent is $-|z| \le 0$, so $e^{-|z|} \in (0, 1]$ and cannot overflow (it may underflow to 0, which is harmless). The code computes `e = exp(-abs(z))` once and picks the branch with `np.where`.

**Clipping before the log.** In float64, $\sigma(z)$ rounds to exactly `1.0` from $z = 37$ on, because $1 - \sigma(z) < 2^{-53}$. Then $\log(1 - p) = \log 0 = -\infty$, and a single confident-and-wrong sample makes the loss `inf` (NumPy: `divide by zero encountered in log`). Clipping $p$ into $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon = 10^{-12}$ caps each sample's loss at $-\log \varepsilon \approx 27.6$. The negative side is different: $\sigma(-40) = 4 \cdot 10^{-18}$ is still representable (it only underflows near $z \approx -745$), but the loss formula needs $\log p$ and $\log(1-p)$ on both sides, so it is clipped symmetrically.

**Side effect.** Clipping makes the loss *flat* once $p$ leaves $[\varepsilon, 1-\varepsilon]$, while the implemented gradient $\frac1n X_b^\top (p - y)$ is the gradient of the *unclipped* loss. They agree whenever $|z| \lesssim 27.6$. Outside that range the code's gradient is the mathematically correct one and the clipped loss value is only a safeguard (see the check below).

## 4. Checking the gradient

Central finite differences. For each coordinate $j$ with unit vector $e_j$ and step $h$:

$$\frac{\partial L}{\partial w_j} \approx \frac{L(w + h e_j) - L(w - h e_j)}{2h} .$$

Taylor expansion shows the truncation error is $O(h^2)$ (the odd terms cancel), while rounding error is about $\varepsilon_{\text{mach}} / h$. So the total error is smallest at an intermediate $h$, around $\varepsilon_{\text{mach}}^{1/3} \approx 10^{-5}$; too small an $h$ makes things *worse* (table below). The test uses $h = 10^{-6}$ with tolerance $10^{-6}$.

**What a match proves:** that, at the points tested, the analytic gradient is consistent with the loss *as implemented*. This catches sign errors, missing $1/n$, a forgotten penalty term, or a transposed matrix.

**What it does not prove:**
- That the loss itself is the right one. $2L$ with gradient $2\nabla L$ passes the check too (verified below).
- Anything at points not tested, including saturated regions (the clipping effect above).
- That the optimizer converges, or that the model generalizes.

## 5. Fitting

`fit` runs plain batch gradient descent from $w = 0$: $w \leftarrow w - \eta \nabla L_\lambda(w)$ for a fixed number of iterations (default $\eta = 0.5$, 2000 iterations). There is no stopping rule other than the iteration cap, and no line search. With the convex loss and $\eta < 2/\mathsf L$ the loss cannot increase.

**Perfectly separable data.** If some $w^\circ$ classifies every sample correctly, i.e. $y_i = 1 \Rightarrow x_i^\top w^\circ > 0$ and $y_i = 0 \Rightarrow x_i^\top w^\circ < 0$, then scaling $w^\circ \to c\, w^\circ$ with $c \to \infty$ pushes every $p_i$ toward its label and $L \to 0$. The infimum $0$ is never reached: the maximum-likelihood estimate does not exist, the gradient is never zero, and gradient descent keeps increasing $\lVert w \rVert$ forever, slowly (see the table: $\lVert w \rVert$ grows like $\log t$ and the loss like $1/t$). Which iteration you stop at decides the weights. An $\ell_2$ penalty fixes this: it makes the problem strongly convex, so a finite minimizer exists.

## Numerical checks (all run against the implementation)

**Sigmoid at extreme inputs**

| Expression | $z = -1000$ | $z = +1000$ |
|---|---|---|
| naive $1/(1+e^{-z})$ | $0.0$ plus `overflow` warning (error under `errstate(over="raise")`) | $1.0$, no warning |
| naive $e^{z}/(1+e^{z})$ | $0.0$ | `nan` plus `overflow` and `invalid` warnings |
| stable (`mlscratch`) | $0.0$, no warning | $1.0$, no warning |

Stable $\sigma(-50) = 1.9\cdot10^{-22}$ (small values keep their precision); $\sigma(z) = 1.0$ exactly from $z = 37$; unclipped $-\log(1 - 1.0) = \infty$; clipped loss for $y = 0, p = 1$ is $27.63$.

**Finite-difference error against step size** (50 random points, 3 features, $\lambda = 0$)

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ |
|---|---|---|---|---|---|---|---|---|---|
| max abs error | $1.2\cdot10^{-4}$ | $1.2\cdot10^{-6}$ | $1.2\cdot10^{-8}$ | $1.2\cdot10^{-10}$ | $1.1\cdot10^{-11}$ | $4.8\cdot10^{-11}$ | $1.2\cdot10^{-8}$ | $2.1\cdot10^{-6}$ | $1.8\cdot10^{-4}$ |

The error falls by $100\times$ per $10\times$ smaller $h$ (the $O(h^2)$ term), bottoms out near $h = 10^{-5}$, and rises again from rounding. Also: the pair ($2L$, $2\nabla L$) matches finite differences with error $9.5\cdot10^{-11}$, so the check cannot tell a wrong-by-a-constant-factor loss apart. A saturated example ($z = 40$, $y = 0$): analytic $\partial L / \partial w_1 = 1.0$, finite difference of the clipped loss $= 0.0$.

**Gradient descent on the test clusters** (`make_clusters`: 200 + 200 points around $\mp 2$, seed 0)

- $\mathsf L = 2.27$, so the sufficient bound is $2/\mathsf L = 0.88$.
- After 1000 iterations ($\eta = 0.5$): accuracy $1.0$, loss $0.0041$, $\lVert \nabla L \rVert = 0.0021$, not zero.
- These test clusters turn out to be **linearly separable** (accuracy $1.0$ and the loss is still decreasing), so the fit stops at the iteration cap, not at a minimum. This does not break the test, which only asks for accuracy $\ge 0.95$ and probabilities in $(0, 1)$.
- Step sizes $0.5,\ 0.92,\ 1.0,\ 1.5,\ 4.0$ (several above $2/\mathsf L$) ran 300 iterations each with the loss never increasing. So $2/\mathsf L$ is sufficient, not necessary: it is a worst-case bound ($p(1-p) \le \tfrac14$), and here large steps along a separating direction only make the loss smaller.

**Separable 1-D data** ($x = -2, -1, 1, 2$, labels $0, 0, 1, 1$; the bias stays $\approx 0$ by symmetry), $\eta = 0.5$

| iteration | $w_1$ | loss | $\lVert \nabla L \rVert$ |
|---|---|---|---|
| 10 | 1.557 | $1.2\cdot10^{-1}$ | $1.3\cdot10^{-1}$ |
| 100 | 3.376 | $1.7\cdot10^{-2}$ | $1.8\cdot10^{-2}$ |
| 1000 | 5.548 | $2.0\cdot10^{-3}$ | $2.0\cdot10^{-3}$ |
| 10 000 | 7.828 | $2.0\cdot10^{-4}$ | $2.0\cdot10^{-4}$ |
| 100 000 | 10.127 | $2.0\cdot10^{-5}$ | $2.0\cdot10^{-5}$ |

Each $10\times$ more iterations adds about $2.3 = \ln 10$ to $w_1$ (logarithmic growth) and divides the loss by $10$. With the penalty, after 100 000 iterations: $\lambda = 0.01 \to w_1 = 2.90$; $\lambda = 0.1 \to 1.46$; $\lambda = 1 \to 0.47$.

## Notes

- Two results differed from the usual textbook story. The naive sigmoid does **not** return a wrong value at $z = -1000$ (it only raises the overflow flag; the wrong value `nan` appears in the other algebraic form at $z = +1000$). And the finite-difference check is blind to the clipping inconsistency in saturated regions.
- On separable data the weights are an arbitrary snapshot of a process that never ends. Compare models only with a fixed iteration budget or with $\lambda > 0$. (It is known that the *direction* of $w$ approaches the hard-margin SVM direction; that was not checked in this repo.)
