# =============================================================================
# basket_pricer.py
# =============================================================================
#
# PURPOSE
# -------
# A self-contained, production-ready pricer for European basket call options
# on any set of N equity assets, with arbitrary portfolio weights.
#
# This file is the deliverable for Part 4 of the project. It wraps the entire
# three-part research pipeline (marginal distributions → copula → Monte Carlo)
# into a single public function that any user can call without knowledge of
# the underlying methodology.
#
# PUBLIC INTERFACE
# ----------------
# The only function a user needs:
#
#   result = price_basket_call(
#       tickers  = ["NVDA", "MSFT", "GOOGL"],
#       weights  = [0.5, 0.3, 0.2],   # must sum to 1
#       K        = None,               # None → at-the-money
#       r        = 0.05,
#       n_sim    = 10_000,
#       copula   = "student",          # "gaussian" or "student"
#       use_cv   = True
#   )
#
# THEORETICAL FOUNDATIONS
# -----------------------
# The pricer rests on three pillars developed in Parts 1–3:
#
#   Part 1 — Risk-neutral marginals
#     The SABR stochastic volatility model is calibrated to each asset's
#     observed implied volatility smile. The Breeden–Litzenberger formula
#     then extracts the risk-neutral PDF from the second derivative of the
#     call price curve with respect to strike. Integration gives the CDF,
#     whose inverse F_k^{-1} maps uniform samples to terminal asset prices.
#
#   Part 2 — Copula-based dependence
#     A Gaussian or Student-t copula encodes the joint dependence between
#     assets, completely separated from their individual distributions
#     (Sklar's theorem). The copula parameter ρ_{ij} is calibrated so that
#     the simulated asset prices reproduce the observed Pearson correlations
#     exactly — this is non-trivial when marginals are non-Gaussian. For the
#     Student-t copula, the degrees-of-freedom ν are estimated by maximum
#     likelihood on historical pseudo-observations.
#
#   Part 3 — Monte Carlo pricing with control variate
#     10,000 joint terminal price paths are drawn from the copula. The basket
#     call price is the discounted average payoff. A geometric basket control
#     variate (which has a known closed-form price) is used to reduce Monte
#     Carlo variance by ~60%.
#
# DEPENDENCIES
# ------------
#   copula.py                        (from this project)
#   functions_marginal_distributions.py  (from this project)
#   numpy, pandas, scipy, sklearn, yfinance
#
# =============================================================================

import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

# --- imports from the project's own modules ---------------------------------
# copula.py supplies the building blocks for dependence modelling
from copula import (
    estimate_correlation,   # Pearson R from historical log-returns
    build_inverse_cdf,      # interpolated F_k^{-1} from (K_grid, cdf) table
    sample_gaussian_copula, # Cholesky-based Gaussian copula sampler
    sample_student_copula,  # chi-squared-scaled Student-t copula sampler
    estimate_nu,            # MLE for Student-t degrees of freedom
)

# functions_marginal_distributions.py supplies the smile and density pipeline
from functions_marginal_distributions import (
    fetch_option_chain,     # download OTM option chain from Yahoo Finance
    build_smile,            # convert raw chain to IV smile DataFrame
    calibrate_sabr,         # fit SABR (alpha, rho, nu) to the IV smile
    sabr_vol,               # SABR implied vol at a single strike
    bs_call_price,          # Black-Scholes call price
    breeden_litzenberger,   # second-derivative density extraction (numerical)
    extract_cdf,            # CDF by trapezoidal integration of PDF
)


# =============================================================================
# STAGE 1 — MARGINAL DISTRIBUTIONS
# =============================================================================

def _build_marginals(tickers, r=0.05):
    """
    Runs the full Part 1 pipeline for an arbitrary list of tickers.

    For each asset this function:
      1. Downloads the most liquid OTM option chain (14–90 day expiry window)
         from Yahoo Finance using fetch_option_chain().
      2. Constructs the implied volatility smile from mid-quote prices via
         build_smile(), keeping only OTM options in the 60–160% moneyness range.
      3. Calibrates the SABR stochastic volatility model to the smile using
         calibrate_sabr(), which fits (alpha, rho, nu) by least squares with
         beta fixed at 0.5 (standard equity convention).
      4. Generates implied vols on a dense 500-point strike grid from the
         calibrated SABR model and converts them back to call prices.
      5. Extracts the risk-neutral PDF via the Breeden–Litzenberger formula
         (second derivative of call prices w.r.t. strike). Monotonicity is
         enforced by isotonic regression; differentiation uses a Savitzky–Golay
         filter for stability; a light Gaussian smoothing is applied.
      6. Integrates the PDF to obtain the CDF via cumulative_trapezoid().

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols, e.g. ["NVDA", "MSFT"].
    r : float
        Continuously compounded risk-free rate used for BS repricing and
        Breeden–Litzenberger. Default 0.05 (5%).

    Returns
    -------
    marginals : dict
        Keys are ticker strings. Each value is a dict with:
          "S"           : float  — current spot price
          "T"           : float  — option maturity in years
          "K_grid"      : ndarray (500,) — strike grid
          "pdf"         : ndarray (500,) — risk-neutral PDF values
          "cdf"         : ndarray (500,) — risk-neutral CDF values
          "sabr_params" : dict   — calibrated SABR parameters
    spots : ndarray (N,)
        Spot price for each ticker in the same order as tickers.
    T : float
        Common maturity (taken from the first ticker; all use the same expiry
        window selection logic so they are typically close).
    """
    marginals = {}

    for ticker in tickers:
        print(f"  [{ticker}] downloading option chain...")
        asset_data = fetch_option_chain(ticker)

        S = asset_data["spot"]
        smile = build_smile(asset_data, r=r)
        T = smile["T"].iloc[0]
        F = S * np.exp(r * T)

        print(f"  [{ticker}] calibrating SABR (S={S:.2f}, T={T:.4f}yr)...")
        sabr_params = calibrate_sabr(smile, r=r)

        # --- dense strike grid for SABR repricing ----------------------------
        # We use ±2 SABR-alpha standard deviations around the forward,
        # clipped to 60–160% of spot to avoid extreme extrapolation.
        alpha = sabr_params["alpha"]
        K_min = max(S * 0.60, F * np.exp(-2.0 * alpha * np.sqrt(T)))
        K_max = min(S * 1.60, F * np.exp( 2.0 * alpha * np.sqrt(T)))
        K_grid = np.linspace(K_min, K_max, 500)

        # --- SABR implied vols on the dense grid -----------------------------
        iv_grid = np.array([
            sabr_vol(F, K, T,
                     sabr_params["alpha"], sabr_params["beta"],
                     sabr_params["rho"],   sabr_params["nu"])
            for K in K_grid
        ])
        iv_grid = np.clip(iv_grid, 0.01, 5.0)  # guard against SABR blowup

        # --- Breeden–Litzenberger density extraction -------------------------
        # Reprice calls from SABR vols, enforce monotone decreasing prices,
        # then differentiate twice w.r.t. strike.
        C = np.array([bs_call_price(S, K, T, r, iv)
                      for K, iv in zip(K_grid, iv_grid)])
        C = np.clip(C, 0, None)

        # Isotonic regression: call prices must be monotone non-increasing in K
        ir = IsotonicRegression(increasing=False)
        C  = ir.fit_transform(K_grid, C)

        # Savitzky–Golay filter gives stable second derivatives without the
        # noise amplification of raw finite differences
        dK  = K_grid[1] - K_grid[0]
        d2C = savgol_filter(C, window_length=51, polyorder=3, deriv=2, delta=dK)

        # Risk-neutral density: q(K) = e^{rT} * d²C/dK²
        pdf = np.exp(r * T) * d2C
        pdf = np.maximum(pdf, 0)            # truncate numerical negatives
        pdf = gaussian_filter1d(pdf, sigma=4)  # light smoothing pass
        pdf /= np.trapz(pdf, K_grid)        # normalise to integrate to 1

        # CDF by cumulative trapezoidal integration
        cdf  = cumulative_trapezoid(pdf, K_grid, initial=0)
        cdf /= cdf[-1]                      # ensure exactly 1 at right end

        marginals[ticker] = {
            "S":           S,
            "T":           T,
            "K_grid":      K_grid,
            "pdf":         pdf,
            "cdf":         cdf,
            "sabr_params": sabr_params,
        }
        print(f"  [{ticker}] done. alpha={sabr_params['alpha']:.3f}, "
              f"rho={sabr_params['rho']:.3f}, nu={sabr_params['nu']:.3f}")

    spots = np.array([marginals[t]["S"] for t in tickers])
    T_common = marginals[tickers[0]]["T"]   # expiry from first ticker

    return marginals, spots, T_common


# =============================================================================
# STAGE 2 — COPULA CORRELATION CALIBRATION
# =============================================================================

def _calibrate_copula_correlation(tickers, inv_cdfs, R_pearson, seed=42):
    """
    Calibrates the copula correlation matrix R_copula from the observed Pearson
    correlation matrix R_pearson.

    WHY THIS IS NECESSARY
    ---------------------
    The Gaussian copula parameter ρ_{ij} is NOT the same as the Pearson
    correlation r_{ij} when marginals are non-Gaussian.  If we naively use
    R_pearson as the copula input, the simulated asset prices will have the
    wrong pairwise correlations.

    The correction inverts the relationship numerically: for each pair (i, j)
    we find ρ_{ij} such that

        Corr( F_i^{-1}(Φ(Z_i)),  F_j^{-1}(Φ(Z_j)) ) = r_{ij}

    where (Z_i, Z_j) ~ N(0, [[1, ρ], [ρ, 1]]) and Φ is the standard normal
    CDF.  We solve this with Brent's root-finding method.

    A single large fixed set of Z samples (20,000 pairs, seeded) is reused
    across all pair searches so the objective function is deterministic — this
    is essential for the root-finder to converge reliably.

    Parameters
    ----------
    tickers   : list of str
    inv_cdfs  : list of callable — F_k^{-1} for each asset
    R_pearson : ndarray (N, N) — empirical Pearson correlation matrix
    seed      : int — random seed for the fixed Z sample

    Returns
    -------
    R_copula : ndarray (N, N) — calibrated copula correlation matrix
    """
    N = len(tickers)
    np.random.seed(seed)
    # Fixed 2D normal sample reused for every pairwise root search.
    # 20,000 draws keeps Monte Carlo noise in the objective well below the
    # root-finding tolerance of 1e-4.
    Z_fixed = np.random.standard_normal((20_000, 2))

    R_copula = np.eye(N)

    print("  Calibrating copula correlations (Pearson → copula ρ)...")

    for i in range(N):
        for j in range(i + 1, N):
            obs_corr = R_pearson[i, j]

            # Objective: given a trial ρ, simulate correlated normals,
            # transform through the actual market-implied marginals, and
            # measure the resulting Pearson correlation.  We want the residual
            # (simulated corr − target) to be zero.
            def _objective(rho):
                R2     = np.array([[1.0, rho], [rho, 1.0]])
                L      = np.linalg.cholesky(R2)
                Z_corr = Z_fixed @ L.T          # (20000, 2) correlated normals
                U      = norm.cdf(Z_corr)        # transform to uniforms
                Si     = inv_cdfs[i](U[:, 0])    # map to asset i prices
                Sj     = inv_cdfs[j](U[:, 1])    # map to asset j prices
                return np.corrcoef(Si, Sj)[0, 1] - obs_corr

            # Brent's method is safe here because the objective is monotone
            # increasing in ρ (higher copula correlation → higher price corr)
            rho_ij = brentq(_objective, a=-0.999, b=0.999,
                            xtol=1e-4, maxiter=50)

            R_copula[i, j] = rho_ij
            R_copula[j, i] = rho_ij

            print(f"    {tickers[i]}-{tickers[j]}: "
                  f"Pearson r = {obs_corr:.4f}  →  copula ρ = {rho_ij:.4f}  "
                  f"(Δ = {rho_ij - obs_corr:+.4f})")

    return R_copula


# =============================================================================
# STAGE 3 — SIMULATION
# =============================================================================

def _run_pipeline(tickers, n_sim, copula, r=0.05, seed=42):
    """
    Orchestrates Stages 1–3 for an arbitrary basket of N assets.

    This is the function that makes Part 4 a black box: a user supplies
    tickers and gets back a simulation matrix S of shape (n_sim, N) plus
    the metadata (spots, T) needed for pricing.  All the machinery of option
    chain downloading, SABR calibration, Breeden–Litzenberger density
    extraction, inverse-CDF construction, and copula calibration is hidden
    inside.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols for the basket assets.
    n_sim   : int
        Number of Monte Carlo paths to simulate.
    copula  : str
        "gaussian" or "student".
    r       : float
        Risk-free rate.
    seed    : int
        Random seed passed to the copula sampler for reproducibility.

    Returns
    -------
    S      : ndarray (n_sim, N) — simulated terminal asset prices
    spots  : ndarray (N,)       — current spot prices
    T      : float              — option maturity in years
    nu     : float or None      — estimated Student-t d.o.f. (None if Gaussian)
    """
    # ------------------------------------------------------------------
    # Stage 1: risk-neutral marginal distributions (Part 1 pipeline)
    # ------------------------------------------------------------------
    print("Stage 1: extracting risk-neutral marginals...")
    marginals, spots, T = _build_marginals(tickers, r=r)

    # ------------------------------------------------------------------
    # Stage 2a: build inverse CDFs F_k^{-1} for each asset
    # ------------------------------------------------------------------
    # These piecewise-linear interpolants map any U ~ Uniform(0,1) to an
    # asset price, preserving the full shape of the market-implied marginal.
    print("\nStage 2: building inverse CDFs and calibrating copula...")
    inv_cdfs = []
    for ticker in tickers:
        K_grid = marginals[ticker]["K_grid"]
        cdf    = marginals[ticker]["cdf"]
        inv_cdfs.append(build_inverse_cdf(K_grid, cdf))

    # ------------------------------------------------------------------
    # Stage 2b: estimate the Pearson correlation matrix from 1y of data
    # ------------------------------------------------------------------
    R_pearson = estimate_correlation(tickers, period="1y")

    # ------------------------------------------------------------------
    # Stage 2c: calibrate copula correlation matrix (Pearson ≠ copula ρ)
    # ------------------------------------------------------------------
    R_copula = _calibrate_copula_correlation(tickers, inv_cdfs, R_pearson)

    # ------------------------------------------------------------------
    # Stage 2d: estimate Student-t d.o.f. if needed
    # ------------------------------------------------------------------
    nu = None
    if copula == "student":
        print("\n  Estimating Student-t degrees of freedom (MLE)...")
        nu = estimate_nu(tickers, R_copula, period="1y")
        print(f"  ν = {nu:.4f}")

    # ------------------------------------------------------------------
    # Stage 3: draw joint terminal price scenarios
    # ------------------------------------------------------------------
    print(f"\nStage 3: simulating {n_sim:,} paths from {copula} copula...")
    np.random.seed(seed)

    if copula == "gaussian":
        S, _ = sample_gaussian_copula(R_copula, inv_cdfs, n_sim)
    elif copula == "student":
        S, _ = sample_student_copula(R_copula, nu, inv_cdfs, n_sim)
    else:
        raise ValueError(f"copula must be 'gaussian' or 'student', got '{copula}'")

    print(f"  Simulation complete. S shape: {S.shape}")
    return S, spots, T, nu


# =============================================================================
# STAGE 4 — PRICING
# =============================================================================

def _geo_basket_call_exact(S, K, r, T, w, spots):
    """
    Closed-form price of a European geometric basket call, generalised to
    arbitrary portfolio weights.

    WHY WE NEED THIS (the control variate)
    ----------------------------------------
    Plain Monte Carlo computes the arithmetic basket call price as the
    sample average of discounted payoffs.  The standard error is O(1/√n),
    so 10,000 paths give a price with SE ≈ 0.25.

    The geometric basket is nearly perfectly correlated with the arithmetic
    basket (same paths, same assets, slightly different average) yet its
    price has a known closed form.  We use it as a control variate:

        price_CV = price_arith_MC
                   − β * (price_geo_MC − price_geo_exact)

    where β = Cov(arith payoff, geo payoff) / Var(geo payoff).

    This removes the shared Monte Carlo noise, reducing variance by ~60%.

    DERIVATION OF THE CLOSED FORM
    ------------------------------
    Define the log-geometric basket:

        ln G_T = Σ_k w_k ln S_k^T

    If S_k^T were lognormal with drift μ_k and vol σ_k, then G_T would be
    lognormal with:

        σ_G² = w^T Σ w                   (Σ = covariance matrix of log-returns)
        μ_G  = Σ_k w_k (r − ½σ_k²) + ½σ_G²
        F_G  = exp( Σ_k w_k ln S_k + μ_G * T )

    These parameters are estimated from the simulation matrix S rather than
    assumed, so they automatically reflect the non-Gaussian SABR marginals
    in an approximate sense.  This makes the control variate more accurate
    than one based on assumed lognormality.

    Parameters
    ----------
    S     : ndarray (n_sim, N) — simulated terminal prices
    K     : float              — strike
    r     : float              — risk-free rate
    T     : float              — maturity in years
    w     : ndarray (N,)       — portfolio weights summing to 1
    spots : ndarray (N,)       — current spot prices

    Returns
    -------
    price : float — closed-form geometric basket call price
    """
    N      = S.shape[1]
    log_S  = np.log(S)                       # (n_sim, N) log terminal prices

    # Estimate the covariance matrix of annualised log-returns from the paths.
    # Dividing by T converts the terminal-price covariance to an annual rate,
    # consistent with the Black-Scholes parameterisation.
    cov    = np.cov(log_S.T) / T             # (N, N) annualised log-return cov
    vols   = np.sqrt(np.diag(cov))           # (N,)   per-asset vols

    # Weighted geometric basket volatility: σ_G = sqrt(w^T Σ w)
    sigma_G = np.sqrt(w @ cov @ w)

    # Weighted drift of the log geometric basket under the risk-neutral measure.
    # Each asset contributes w_k * (r - ½σ_k²); the ½σ_G² Itô correction
    # converts from the log drift to the level drift of the geometric basket.
    mu_G = w @ (r - 0.5 * vols**2) + 0.5 * sigma_G**2

    # Risk-neutral forward of the geometric basket.
    # exp( w^T ln(spots) ) is the current geometric basket level;
    # exp( mu_G * T ) grows it to the forward.
    F_G = np.exp(w @ np.log(spots) + mu_G * T)

    # Black-Scholes formula applied to the geometric basket as a single asset
    v  = sigma_G * np.sqrt(T)
    if v < 1e-10:
        # Degenerate case: zero vol → intrinsic value only
        return np.exp(-r * T) * max(F_G - K, 0.0)

    d1 = np.log(F_G / K) / v + 0.5 * v
    d2 = d1 - v

    price = np.exp(-r * T) * (F_G * norm.cdf(d1) - K * norm.cdf(d2))
    return float(price)


def _price_with_cv(S, K, r, T, w, spots):
    """
    Prices a European basket call using the geometric basket control variate.

    The weight vector w is applied consistently to both the arithmetic and
    geometric basket calculations and to the closed-form CV anchor price.
    This is critical: if the weights differ between the simulated payoff and
    the CV anchor, the variance reduction collapses.

    Parameters
    ----------
    S     : ndarray (n_sim, N)
    K     : float
    r     : float
    T     : float
    w     : ndarray (N,)  — portfolio weights
    spots : ndarray (N,)  — current spot prices

    Returns
    -------
    price : float — control-variate-adjusted basket call price
    se    : float — standard error of the adjusted estimator
    """
    # --- arithmetic basket payoff (target) -----------------------------------
    # S @ w gives the weighted sum of terminal prices for each path
    arith_basket = S @ w                          # (n_sim,)
    payoffs_a    = np.maximum(arith_basket - K, 0)

    # --- geometric basket payoff (control) -----------------------------------
    # exp( log(S) @ w ) = ∏ S_k^{w_k}, the weighted geometric mean
    geo_basket   = np.exp(np.log(S) @ w)          # (n_sim,)
    payoffs_g    = np.maximum(geo_basket - K, 0)

    # --- closed-form geometric basket price (CV anchor) ----------------------
    V_geo_exact  = _geo_basket_call_exact(S, K, r, T, w, spots)

    # --- optimal beta (variance-minimising coefficient) ----------------------
    # β = Cov(arith payoff, geo payoff) / Var(geo payoff)
    # This is the OLS coefficient of regressing arith payoffs on geo payoffs.
    cov_matrix   = np.cov(payoffs_a, payoffs_g)   # (2, 2) covariance matrix
    beta         = cov_matrix[0, 1] / cov_matrix[1, 1]

    # --- control-variate adjusted payoffs ------------------------------------
    # We subtract the product of beta and the geo MC error (geo_MC - geo_exact
    # scaled to the same units by the discount factor).
    # Note: payoffs_g is undiscounted here, so the anchor must also be
    # undiscounted → multiply V_geo_exact by e^{rT} to match units.
    payoffs_cv   = payoffs_a - beta * (payoffs_g - np.exp(r * T) * V_geo_exact)

    # --- discount and aggregate ----------------------------------------------
    price = np.exp(-r * T) * payoffs_cv.mean()
    se    = np.exp(-r * T) * payoffs_cv.std() / np.sqrt(len(payoffs_cv))

    return price, se


def _price_plain_mc(S, K, r, T, w):
    """
    Plain Monte Carlo pricer — no variance reduction.
    Retained for comparison and for cases where use_cv=False.

    Parameters
    ----------
    S : ndarray (n_sim, N)
    K : float
    r : float
    T : float
    w : ndarray (N,)

    Returns
    -------
    price : float
    se    : float
    """
    basket  = S @ w
    payoffs = np.maximum(basket - K, 0)
    price   = np.exp(-r * T) * payoffs.mean()
    se      = np.exp(-r * T) * payoffs.std() / np.sqrt(len(payoffs))
    return price, se


# =============================================================================
# PUBLIC FUNCTION
# =============================================================================

def price_basket_call(
    tickers,
    weights  = None,
    K        = None,
    r        = 0.05,
    n_sim    = 10_000,
    copula   = "student",
    use_cv   = True,
    seed     = 42,
    verbose  = True,
):
    """
    Price a European basket call option on N equity assets.

    This is the single entry point for Part 4. It runs the complete pipeline
    from raw option market data to a final option price, with a control
    variate for variance reduction. The user only needs to specify tickers
    and contract terms; all methodology is encapsulated inside.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols, e.g. ["NVDA", "MSFT", "GOOGL"].
        Any number of assets N ≥ 2 is supported.

    weights : list or ndarray of float, optional
        Portfolio weights w_k ≥ 0 summing to 1.
        If None, equal weights 1/N are used.
        Example: [0.4, 0.4, 0.2] for a 40/40/20 basket.

    K : float or None, optional
        Strike price. If None, the at-the-money strike is used:
            K = sum_k w_k * S_k
        i.e. the current weighted average spot price.

    r : float, optional
        Continuously compounded annual risk-free rate. Default 0.05.

    n_sim : int, optional
        Number of Monte Carlo simulation paths. Default 10,000.
        More paths → lower standard error but longer runtime.
        Rule of thumb: SE ∝ 1/√n_sim.

    copula : str, optional
        Dependence model. Either:
          "gaussian" — no tail dependence (faster, simpler)
          "student"  — symmetric tail dependence via shared χ² mixing
                       (recommended; better captures joint crash risk)
        Default "student".

    use_cv : bool, optional
        If True, apply the geometric basket control variate to reduce
        Monte Carlo variance by ~60%. Recommended. Default True.

    seed : int, optional
        Random seed for reproducible results. Default 42.

    verbose : bool, optional
        Print progress messages. Default True.

    Returns
    -------
    result : dict with the following keys:

        "price"          : float   — basket call price in USD
        "se"             : float   — Monte Carlo standard error
        "ci_95"          : tuple   — 95% confidence interval (low, high)
        "copula"         : str     — copula type used
        "nu"             : float   — Student-t d.o.f. (None if Gaussian)
        "n_sim"          : int     — number of paths used
        "K"              : float   — strike price
        "T"              : float   — maturity in years
        "r"              : float   — risk-free rate
        "tickers"        : list    — asset tickers
        "weights"        : list    — portfolio weights
        "spots"          : list    — spot prices at pricing time
        "use_cv"         : bool    — whether CV was applied
        "var_reduction"  : str     — variance reduction % (None if no CV)
        "runtime_s"      : float   — total wall-clock time in seconds

    Examples
    --------
    # Equal-weighted ATM basket on AI stocks, Student-t copula with CV
    result = price_basket_call(
        tickers = ["NVDA", "PLTR", "MSFT", "STX", "GOOGL"]
    )
    print(f"Price: {result['price']:.4f} ± {result['se']:.4f}")

    # Custom weights, OTM strike, Gaussian copula
    result = price_basket_call(
        tickers = ["AAPL", "MSFT", "GOOGL"],
        weights = [0.5, 0.3, 0.2],
        K       = 210.0,
        copula  = "gaussian",
        n_sim   = 20_000
    )
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if len(tickers) < 2:
        raise ValueError("Basket must contain at least 2 assets.")

    N = len(tickers)

    # Weights: default to equal weighting; normalise to sum to 1
    if weights is None:
        w = np.ones(N) / N
        if verbose:
            print(f"No weights provided — using equal weights 1/{N}.")
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != N:
            raise ValueError(
                f"weights has length {len(w)} but tickers has length {N}."
            )
        if np.any(w < 0):
            raise ValueError("All weights must be non-negative.")
        w_sum = w.sum()
        if not np.isclose(w_sum, 1.0, atol=1e-6):
            w = w / w_sum   # silently normalise
            if verbose:
                print(f"Weights normalised to sum to 1.")

    if copula not in ("gaussian", "student"):
        raise ValueError("copula must be 'gaussian' or 'student'.")

    if n_sim < 100:
        raise ValueError("n_sim must be at least 100.")

    # ------------------------------------------------------------------
    # Run the pipeline (Stages 1–3)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("BASKET CALL PRICER — PIPELINE START")
        print("="*60)
        print(f"  Tickers : {tickers}")
        print(f"  Weights : {np.round(w, 4).tolist()}")
        print(f"  Copula  : {copula}")
        print(f"  n_sim   : {n_sim:,}")
        print(f"  r       : {r}")
        print("="*60 + "\n")

    S, spots, T, nu = _run_pipeline(tickers, n_sim, copula, r=r, seed=seed)

    # ------------------------------------------------------------------
    # Strike: use ATM if not provided
    # ------------------------------------------------------------------
    if K is None:
        K = float(w @ spots)   # weighted average spot = ATM for the basket
        if verbose:
            print(f"\n  Strike not specified — using ATM: K = {K:.4f}")
    else:
        if verbose:
            print(f"\n  Strike: K = {K:.4f}")

    # ------------------------------------------------------------------
    # Stage 4: pricing
    # ------------------------------------------------------------------
    if verbose:
        print("\nStage 4: pricing...\n")

    if use_cv:
        price, se = _price_with_cv(S, K, r, T, w, spots)

        # Compute plain MC SE for variance reduction reporting
        _, se_plain = _price_plain_mc(S, K, r, T, w)
        var_red_pct = (1 - (se / se_plain) ** 2) * 100
        var_reduction_str = f"{var_red_pct:.1f}%"
    else:
        price, se = _price_plain_mc(S, K, r, T, w)
        var_reduction_str = None

    # 95% confidence interval: price ± 1.96 * SE
    ci_95 = (price - 1.96 * se, price + 1.96 * se)

    runtime = time.time() - t_start

    # ------------------------------------------------------------------
    # Build and return result dictionary
    # ------------------------------------------------------------------
    result = {
        "price":         round(price, 6),
        "se":            round(se, 6),
        "ci_95":         (round(ci_95[0], 6), round(ci_95[1], 6)),
        "copula":        copula,
        "nu":            round(nu, 4) if nu is not None else None,
        "n_sim":         n_sim,
        "K":             round(K, 4),
        "T":             round(T, 6),
        "r":             r,
        "tickers":       tickers,
        "weights":       w.tolist(),
        "spots":         spots.tolist(),
        "use_cv":        use_cv,
        "var_reduction": var_reduction_str,
        "runtime_s":     round(runtime, 2),
    }

    if verbose:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"  Price        : ${result['price']:.4f}")
        print(f"  Std Error    : {result['se']:.4f}")
        print(f"  95% CI       : (${ci_95[0]:.4f}, ${ci_95[1]:.4f})")
        print(f"  Copula       : {copula}" +
              (f" (ν = {nu:.2f})" if nu else ""))
        print(f"  Maturity T   : {T:.4f} yr")
        print(f"  Strike K     : ${K:.4f}")
        if var_reduction_str:
            print(f"  Var reduction: {var_reduction_str}")
        print(f"  Runtime      : {runtime:.1f}s")
        print("="*60 + "\n")

    return result