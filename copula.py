import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar


def estimate_correlation(tickers, period="1y"): 
    """
    Estimates the correlation matrix of log-returns for a list of tickers.
    Downloads adjusted closing prices over the past year, computes daily
    log-returns, and returns the empirical 5x5 correlation matrix.
    """
    prices = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    R = log_returns.corr().values
    return R

def build_inverse_cdf(K_grid, cdf):
    """
    Builds the inverse CDF (quantile function) for a single asset by
    interpolating the discrete (K, cdf) table from Part 1.
    Given a probability u in [0,1], returns the corresponding asset price.
    """
    # Remove duplicate cdf values to ensure invertibility
    _, unique_idx = np.unique(cdf, return_index=True)
    cdf_unique = cdf[unique_idx]
    K_unique = K_grid[unique_idx]
    
    return interp1d(cdf_unique, K_unique, 
                    kind="linear", 
                    bounds_error=False,
                    fill_value=(K_unique[0], K_unique[-1]))

def sample_gaussian_copula(R, inv_cdfs, n_sim):
    """
    Samples from the Gaussian copula with correlation matrix R and transforms
    the resulting uniform samples to asset prices using the inverse CDFs.
    
    Steps (Choi 2026, M9 slide 10):
    1. Generate independent standard normals Z of shape (n_sim x n_assets)
    2. Correlate via Cholesky: Z' = Z @ L.T where L L.T = R
    3. Transform to uniforms: U_k = N(Z'_k)
    4. Transform to asset prices: S_k = F_k^{-1}(U_k)
    
    Parameters
    - R       : (n_assets x n_assets) correlation matrix
    - inv_cdfs: list of n_assets callable inverse CDFs from build_inverse_cdf()
    - n_sim   : number of simulations
    
    Returns
    - S : (n_sim x n_assets) array of simulated asset prices
    - U : (n_sim x n_assets) array of uniform copula samples
    """
    n_assets = len(inv_cdfs)
    
    # Step 1  
    Z = np.random.standard_normal((n_sim, n_assets))
    
    # Step 2 
    L = np.linalg.cholesky(R)
    Z_corr = Z @ L.T
    
    # Step 3 
    U = norm.cdf(Z_corr)
    
    # Step 4 
    S = np.column_stack([inv_cdfs[k](U[:, k]) for k in range(n_assets)])
    
    return S, U

def estimate_nu(tickers, R, period="1y"):
    """
    Estimates the degrees of freedom parameter nu of the Student-t copula
    by maximum likelihood. Historical log-returns are transformed to 
    pseudo-observations (empirical ranks scaled to [0,1]), which isolates 
    the dependence structure from the marginals. The log-likelihood of the 
    Student-t copula is then maximized over nu using the correlation matrix 
    R already estimated in the Gaussian copula step.
    
    Parameters:
    - tickers : list of ticker strings
    - R       : (n_assets x n_assets) correlation matrix from estimate_correlation()
    - period  : historical window for log-returns (default 1y)
    
    Returns:
    - nu : estimated degrees of freedom (scalar)
    """
    # Download historical returns
    prices = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n, d = log_returns.shape

    # Transform to pseudo-observations (empirical ranks scaled to [0,1])
    pseudo = np.zeros_like(log_returns)
    for i in range(d):
        pseudo[:, i] = pd.Series(log_returns[:, i]).rank() / (n + 1)

    # Transform pseudo-observations to Student-t scores
    def neg_log_likelihood(nu):
        # Transform uniforms to t-scores
        t_scores = student_t.ppf(pseudo, df=nu)
        # Evaluate multivariate t log-likelihood
        from scipy.stats import multivariate_t
        mvt = multivariate_t(shape=R, df=nu)
        ll = mvt.logpdf(t_scores)
        # Subtract univariate t log-densities (copula likelihood)
        for i in range(d):
            ll -= student_t.logpdf(t_scores[:, i], df=nu)
        return -np.sum(ll)

    result = minimize_scalar(neg_log_likelihood, bounds=(2.1, 50), method="bounded")
    return result.x

def sample_student_copula(R, nu, inv_cdfs, n_sim):
    """
    Samples from the Student-t copula with correlation matrix R and degrees
    of freedom nu, and transforms the resulting uniform samples to asset 
    prices using the inverse CDFs.
    
    Steps:
    1. Generate independent standard normals Z of shape (n_sim x n_assets)
    2. Correlate via Cholesky: Z' = Z @ L.T where L L.T = R
    3. Draw shared chi-squared: V ~ chi2(nu), shape (n_sim,)
    4. Scale: Z'' = Z' / sqrt(V / nu)
    5. Transform to uniforms: U_k = t_nu(Z''_k)
    6. Transform to asset prices: S_k = F_k^{-1}(U_k)
    
    Parameters:
    - R       : (n_assets x n_assets) correlation matrix
    - nu      : degrees of freedom (scalar), estimated by estimate_nu()
    - inv_cdfs: list of n_assets callable inverse CDFs from build_inverse_cdf()
   -  n_sim   : number of simulations
    
    Returns:
    - S : (n_sim x n_assets) array of simulated asset prices
    - U : (n_sim x n_assets) array of uniform copula samples
    """
    n_assets = len(inv_cdfs)

    # Step 1 
    Z = np.random.standard_normal((n_sim, n_assets))

    # Step 2 
    L = np.linalg.cholesky(R)
    Z_corr = Z @ L.T

    # Step 3 
    V = np.random.chisquare(nu, size=n_sim)

    # Step 4 
    Z_t = Z_corr / np.sqrt(V / nu)[:, np.newaxis]

    # Step 5 
    U = student_t.cdf(Z_t, df=nu)

    # Step 6 
    S = np.column_stack([inv_cdfs[k](U[:, k]) for k in range(n_assets)])

    return S, U