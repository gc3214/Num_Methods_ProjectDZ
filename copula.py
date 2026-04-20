import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import interp1d

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
