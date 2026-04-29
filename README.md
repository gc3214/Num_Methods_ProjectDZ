# Basket Option Pricing — AI Equity Basket

**European basket call on NVDA · PLTR · MSFT · STX · GOOGL**

---

## Motivation

Sell-side structured products desks have seen strong demand from hedge funds and long-only managers for basket options on high-profile AI equities. These instruments offer leveraged, defined-risk thematic exposure without requiring the buyer to manage delta across five separate single-stock books. Pricing them correctly is non-trivial: each constituent has a distinct implied volatility surface, and the joint payoff is sensitive to both the shape of the individual risk-neutral distributions and the assumed dependence structure across assets.

This project prices an at-the-money European basket call on five AI-sector equities using a three-stage framework, then wraps the best-performing method into an interactive pricing calculator.

---

## Framework at a Glance

| Stage | Method | Output |
|-------|--------|--------|
| Part 1 | SABR calibration + Breeden–Litzenberger | Risk-neutral PDF per asset |
| Part 2 | Gaussian & Student-*t* copulas | Joint terminal distribution |
| Part 3 | Monte Carlo + geometric control variate | Basket call price & SE |
| Part 4 | Interactive calculator (Student-*t* + CV) | Live pricer *(in progress)* |

---

## Part 1 — Risk-Neutral Marginal Distributions

### Objective

Extract the market-implied risk-neutral PDF for each asset directly from the observed option chain — no lognormality assumption imposed on the terminal distribution.

### Pipeline

**1. Option chain ingestion.** OTM option prices are fetched live from Yahoo Finance. The most liquid expiry in the 14–90 day window is selected per asset, ensuring adequate strike coverage across the smile.

**2. SABR calibration.** The SABR model (Hagan et al. 2002) describes the joint dynamics of the forward price and instantaneous volatility:

$$dF_t = \sigma_t F_t^\beta\, dW_t^1, \qquad d\sigma_t = \nu\,\sigma_t\, dW_t^2, \qquad d\langle W^1, W^2\rangle_t = \rho\, dt$$

Parameters $(\alpha, \rho, \nu)$ are calibrated by minimising RMSE between the Hagan approximation and market implied vols. $\beta = 0.5$ is fixed as the standard equity convention, leaving three free parameters.

**3. Dense repricing.** Calibrated SABR vols are evaluated on a 500-point strike grid spanning $[0.6S,\ 1.6S]$ and converted back to call prices via Black–Scholes.

**4. Breeden–Litzenberger extraction.** The risk-neutral density is recovered from the second derivative of the call surface:

$$q(K) = e^{rT}\,\frac{\partial^2 C}{\partial K^2}$$

Monotone prices are enforced via isotonic regression before differentiation. The second derivative is computed with a Savitzky–Golay filter, which fits local polynomials and is significantly more stable than finite differences. A light Gaussian smoothing is applied and the PDF is normalised to integrate to one.

**5. Cross-validation.** Each SABR density is benchmarked against a numerical extraction (SVI or smoothing spline). The SABR density is adopted as the primary marginal because it produces well-behaved, monotone CDFs required by the copula in Part 2.

### Results

The SABR model achieves a good fit across all five names, tracking both the put-side skew and the OTM call wing closely.

- **NVDA and GOOGL** calibrate to negative $\rho$, consistent with the equity leverage effect — negative spot-vol correlation produces the standard downside skew.
- **PLTR and STX** calibrate to near-zero $\rho$, producing the more symmetric smiles visible in the data.
- **STX** stands out with a markedly higher $\alpha$ and lower $\nu$ than the other four names, reflecting a steep but less curved surface driven by structurally elevated implied volatility rather than active stochastic volatility dynamics.

The extracted risk-neutral densities display the right-skewed, log-normal-like shape expected under risk-neutral pricing, with the mode sitting slightly left of spot in each case. **NVDA and PLTR** produce broad, fat-tailed densities consistent with their high IV levels; **MSFT and GOOGL** yield tighter, more concentrated distributions; **STX** is the widest in relative terms with non-negligible left-tail mass.

Cross-validation shows close agreement between SABR and numerical densities for NVDA, PLTR, and GOOGL, validating the extraction pipeline. The largest discrepancies appear for MSFT and STX, where the numerical spline picks up localised features in the raw option chain that the parametric SABR surface smooths over. This reinforces the preference for SABR as the primary marginal: it is smoother, monotone, and more appropriate as input to the copula.

---

## Part 2 — Copula-Based Dependence Modelling

### Objective

Model the joint terminal distribution of the five assets while exactly preserving the market-implied marginals from Part 1. Sklar's theorem guarantees that any multivariate distribution can be decomposed into its marginals and a copula encoding their dependence structure.

### Inverse CDF Construction

For each asset, the discrete $(K, \text{CDF})$ table from Part 1 is interpolated into a callable inverse CDF $F_k^{-1}$. Any uniform sample $U_k \sim \mathcal{U}(0,1)$ maps to a terminal price $S_k^T = F_k^{-1}(U_k)$, preserving the full shape of the market-implied distribution.

### Pearson Correlation vs Copula Parameter

The pairwise Pearson correlation matrix $R^{\text{Pearson}}$ is estimated from one year of daily log-returns. A critical subtlety: the Gaussian copula parameter $\rho_{ij}$ is **not** equal to the Pearson correlation $r_{ij}$ when marginals are non-Gaussian. Plugging $R^{\text{Pearson}}$ directly into the copula would produce simulated prices with systematically incorrect pairwise correlations.

The correct copula parameter is solved for each pair $(i, j)$ via Brent's root-finding:

$$\text{find}\ \rho_{ij}\ \text{s.t.}\ \text{Corr}\!\left(F_i^{-1}(\Phi(Z_i)),\, F_j^{-1}(\Phi(Z_j))\right) = r_{ij}$$

where $(Z_i, Z_j)$ are standard normal with correlation $\rho_{ij}$. The objective is evaluated on a fixed set of 20,000 standard normal pairs to ensure numerical stability across Brent iterations. Assets with heavier tails or more skewed distributions require a higher copula $\rho$ to produce the same linear correlation in price space.

### Gaussian Copula

$n = 10{,}000$ correlated standard normals are drawn via Cholesky decomposition of $R^{\text{copula}}$, transformed to uniforms $U_k = \Phi(Z_k)$, and mapped to terminal prices via $F_k^{-1}$. In the uniform-space scatter plots, samples form elliptical clouds confirming **zero tail dependence**: asymptotically, joint extreme moves are independent under this model.

### Student-*t* Copula

The Gaussian copula's zero-tail-dependence property is often unrealistic for equity baskets, where crashes tend to cluster across names. The Student-*t* copula introduces **symmetric tail dependence** via a shared chi-squared mixing variable. If $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, R)$ and $W \sim \chi^2_\nu / \nu$ independently, then $\mathbf{X} = \mathbf{Z} / \sqrt{W}$ follows a multivariate $t$ with $\nu$ degrees of freedom. The pairwise tail dependence coefficient is:

$$\lambda = 2\, t_{\nu+1}\!\left(-\sqrt{(\nu+1)\,\frac{1-\rho}{1+\rho}}\right) > 0$$

for any finite $\nu$ and $\rho > -1$. $\nu$ is estimated by maximum likelihood on the historical log-return data. In the uniform-space scatter plots, Student-*t* samples show visible corner clustering — more mass near $(0,0)$ and $(1,1)$ relative to the Gaussian — confirming higher probability assigned to joint extreme events.

**Diagnostic:** Marginal recovery checks confirm that histogramming the simulated prices under both copulas closely reproduces the SABR density from Part 1, validating the inverse CDF interpolation and sampling routines end-to-end.

---

## Part 3 — Monte Carlo Pricing with Control Variate

### Contract Specification

| Parameter | Value |
|-----------|-------|
| Basket | Equal-weighted arithmetic mean, 5 assets |
| Strike $K$ | ATM — average of five spot prices |
| Maturity $T$ | From Part 1 option chain expiry |
| Risk-free rate $r$ | 5% |
| Simulations | 10,000 |

### Payoff and Estimator

$$V_T = \max\!\left(\frac{1}{5}\sum_{k=1}^{5} S_k^T - K,\; 0\right), \qquad \hat{V}_0 = e^{-rT} \cdot \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} V_T^{(i)}$$

### Control Variate: Geometric Basket

The geometric basket call has a known closed-form price under log-normal geometry. Since the arithmetic and geometric basket payoffs are driven by the same simulated paths, they are highly correlated. The control-variate estimator:

$$\hat{V}^{CV} = \hat{V}^A - \beta\!\left(\hat{V}^G_{MC} - V^G_{\text{exact}}\right), \qquad \beta = \frac{\text{Cov}(\hat{V}^A,\, \hat{V}^G)}{\text{Var}(\hat{V}^G)}$$

uses the known geometric price to cancel correlated Monte Carlo noise from the arithmetic estimate. $V^G_{\text{exact}}$ is derived from the simulated paths by estimating per-asset vols and the effective basket vol from the log-price covariance matrix.

### Results

| Method | Copula | Price ($) | Var. Reduction |
|--------|--------|-----------|----------------|
| Plain MC | Gaussian | 19.95 | — |
| Plain MC | Student-*t* | 19.11 | — |
| Control Variate | Gaussian | — | **59.3%** |
| Control Variate | Student-*t* | 19.11 | **57.1%** |

### Interpretation

**Control variate effectiveness.** The geometric basket reduces the standard error by 59.3% under the Gaussian copula and 57.1% under the Student-*t* copula at negligible additional runtime. This substantially exceeds the theoretical ~33% expected for a standard lognormal basket, a direct consequence of the high pairwise correlation structure within this AI equity basket. The CV estimates match the plain MC estimates in expectation — variance reduction only, no systematic correction — which is exactly what the method should deliver.

**Copula pricing gap.** The Student-*t* copula prices the basket \$0.84 lower than the Gaussian (\$19.11 vs \$19.95 under plain MC). This is theoretically expected: the shared chi-squared mixing variable in the Student-*t* copula concentrates more probability mass in joint extreme scenarios. For a basket call, joint crashes reduce the expected payoff relative to the Gaussian, which treats extreme co-movements as asymptotically independent. The Student-*t* CV and plain MC prices coincide at \$19.11, confirming the control variate is correcting sampling noise rather than a structural bias.

**Model risk.** The \$0.84 at-the-money price gap between copulas quantifies the model risk a structuring desk faces if it misspecifies the dependence structure. A desk pricing with a Gaussian copula would overprice the basket by \$0.84 relative to a Student-*t* pricer — a meaningful edge for a counterparty who correctly models tail dependence.

---

## Part 4 — Interactive Pricing Calculator *(In Progress)*

Part 4 implements an interactive pricing calculator wrapping the most efficient and accurate method identified in Part 3: **Student-*t* copula with geometric basket control variate**. This method dominates on both dimensions — lowest standard error (via variance reduction) and most realistic dependence structure (via tail dependence). The calculator will allow users to adjust contract parameters (strike, maturity, notional) and reprice the basket call in real time, with the price and confidence interval updating live.

---

## Project Structure

```
.
├── basket_pricing_workbook.ipynb           # Main notebook (Parts 1–4)
├── functions_marginal_distributions.py     # SABR calibration, Breeden–Litzenberger, smile fitting
├── copula.py                               # Correlation estimation, copula sampling, Student-t MLE
└── README.md
```

---

## Dependencies

```
numpy · pandas · scipy · matplotlib · yfinance · scikit-learn
```

---

## References

- Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). *Managing smile risk.* Wilmott Magazine.
- Breeden, D. T., & Litzenberger, R. H. (1978). *Prices of state-contingent claims implicit in option prices.* Journal of Business.
- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges.* Publications de l'Institut de Statistique de l'Université de Paris.
