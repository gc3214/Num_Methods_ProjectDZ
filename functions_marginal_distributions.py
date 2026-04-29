import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


def fetch_option_chain(ticker, min_points=15):
    """
    Downloads option chain data for a given ticker from Yahoo Finance.
    Scans all available expiry dates between 14 and 90 days from today
    and picks the one with the most liquid OTM options, with a preference
    for expiries close to 45 days out.
    Returns a dictionary with the spot price, chosen expiry and raw chain data.
    """
    tk = yf.Ticker(ticker)
    expiries = tk.options
    spot = tk.fast_info["last_price"]
    best_expiry = None
    best_df = None
    best_score = -1

    for exp in expiries:
        T_days = (pd.Timestamp(exp) - pd.Timestamp.today()).days
        if T_days < 14 or T_days > 90:
            continue
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
        calls["type"] = "call"
        puts = chain.puts.copy()
        puts["type"] = "put"
        df = pd.concat([calls, puts])
        df["mid"] = (df["bid"] + df["ask"]) / 2
        otm = df[
            ((df["type"] == "call") & (df["strike"] >= spot * 0.98) |
             (df["type"] == "put") & (df["strike"] <= spot * 1.02))
            & (df["bid"] > 0)
            & (df["openInterest"] > 0)
        ]
        score = len(otm)
        penalty = abs(T_days - 45) / 45
        adj_score = score * (1 - 0.3 * penalty)
        if adj_score > best_score:
            best_score = adj_score
            best_expiry = exp
            best_df = df

    best_df["mid"] = (best_df["bid"] + best_df["ask"]) / 2
    return {"ticker": ticker, "spot": spot, "expiry": best_expiry, "chain": best_df}


def bs_call_price(S, K, T, r, sigma):
    """
    Computes the Black-Scholes price of a European call option..
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """
    Computes the Black-Scholes price of a European put option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(option_price, S, K, T, r, option_type):
    """
    Recovers the implied volatility from a market option price by numerically
    inverting the Black-Scholes formula using Brent's method.
    Returns NaN if no solution is found within the search bounds.
    """
    if option_type == "call":
        price_func = lambda sigma: bs_call_price(S, K, T, r, sigma) - option_price
    else:
        price_func = lambda sigma: bs_put_price(S, K, T, r, sigma) - option_price
    try:
        return brentq(price_func, 1e-6, 5)
    except ValueError:
        return np.nan


def build_smile(asset_data, r=0.05):
    """
    Constructs the implied volatility smile from the raw option chain.
    Keeps only OTM options within 60-160% of spot, filters out illiquid
    quotes and computes implied volatility for each surviving strike.
    Returns a clean DataFrame with one row per strike.
    """
    S = asset_data["spot"]
    expiry = asset_data["expiry"]
    T = (pd.Timestamp(expiry) - pd.Timestamp.today()).days / 365.0
    df = asset_data["chain"].copy()
    df = df[(df["bid"] > 0) & (df["openInterest"] > 0)]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[(df["strike"] >= S * 0.60) & (df["strike"] <= S * 1.60)]
    otm_calls = df[(df["type"] == "call") & (df["strike"] >= S * 0.98)]
    otm_puts = df[(df["type"] == "put") & (df["strike"] <= S * 1.02)]
    otm = pd.concat([otm_calls, otm_puts]).sort_values("strike").reset_index(drop=True)
    otm["T"] = T
    otm["S"] = S
    otm["iv"] = otm.apply(
        lambda row: implied_volatility(row["mid"], S, row["strike"], T, r, row["type"]),
        axis=1
    )
    otm = otm.dropna(subset=["iv"])
    otm = otm[(otm["iv"] > 0.005) & (otm["iv"] < 4.0)]
    return otm[["strike", "iv", "type", "mid", "openInterest", "T", "S"]]


def svi_w(x, a, b, rho, m, sigma):
    """
    Evaluates the SVI parametric formula for total implied variance.
    Takes log-moneyness x and the five SVI parameters as inputs.
    """
    return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma ** 2))


def fit_svi(smile_df, r=0.05):
    """
    Fits the SVI model to the implied volatility smile using least squares.
    Works in total variance space (iv^2 * T) and uses the forward price
    as the at-the-money reference. Returns a dictionary of fitted parameters.
    """
    S = smile_df["S"].iloc[0]
    T = smile_df["T"].iloc[0]
    F = S * np.exp(r * T)
    x = np.log(smile_df["strike"] / F).values
    w_mkt = (smile_df["iv"].values ** 2) * T

    def objective(params):
        a, b, rho, m, sigma = params
        return np.sum((svi_w(x, a, b, rho, m, sigma) - w_mkt) ** 2)

    constraints = [
        {"type": "ineq", "fun": lambda p: p[1]},
        {"type": "ineq", "fun": lambda p: 1 - abs(p[2])},
        {"type": "ineq", "fun": lambda p: p[4]},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, [0.04, 0.1, -0.3, 0.0, 0.1],
                       constraints=constraints,
                       method="SLSQP",
                       options={"maxiter": 500})
    a, b, rho, m, sigma = res.x
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "F": F, "T": T, "S": S}


def svi_iv(K, params):
    """
    Evaluates the fitted SVI model at an array of strikes.
    Converts strikes to log-moneyness using the forward price,
    then returns implied volatilities in annualised decimal form.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    x = np.log(K / params["F"])
    w = svi_w(x, params["a"], params["b"], params["rho"], params["m"], params["sigma"])
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / params["T"])


def fit_smile(smile_df, ticker=""):
    """
    Selects the best smile fitting method based on the number of data points.
    Uses SVI for 20 or more points, a smoothing spline for 6 to 19 points
    and a flat smile as a last resort for fewer than 6 points.
    Returns the fitted model and a string identifying the method used.
    """
    n = len(smile_df)
    S = smile_df["S"].iloc[0]

    if n >= 20:
        try:
            params = fit_svi(smile_df)
            if params["b"] > 0.001 and abs(params["rho"]) < 0.99:
                return params, "svi"
        except Exception:
            pass

    if n >= 6:
        x = np.log(smile_df["strike"] / S).values
        iv = smile_df["iv"].values
        idx = np.argsort(x)
        x, iv = x[idx], iv[idx]
        _, u = np.unique(x, return_index=True)
        x, iv = x[u], iv[u]
        s = len(x) * (0.003 if n >= 12 else 0.008)
        spline = UnivariateSpline(x, iv, k=min(3, len(x) - 1), s=s, ext=3)
        return spline, "spline"

    median_iv = smile_df["iv"].median()
    return lambda x: np.full_like(np.asarray(x, dtype=float), median_iv), "flat"


def call_price_from_smile(K_arr, smile_params, S, T, r, method="svi"):
    """
    Reprices European call options across a grid of strikes using the fitted smile.
    Evaluates the smile model at each strike to get the local implied volatility,
    then plugs it into the Black-Scholes formula.
    """
    if method == "svi":
        iv_arr = svi_iv(K_arr, smile_params)
    elif method == "sabr":
        # Il SABR necessita del Forward price
        F = S * np.exp(r * T)
        iv_arr = np.array([
            sabr_vol(F, K, T, smile_params["alpha"], smile_params["beta"], 
                     smile_params["rho"], smile_params["nu"])
            for K in K_arr
        ])
    elif method == "flat":
        iv_arr = smile_params(np.zeros_like(K_arr))
    else: # Spline
        iv_arr = smile_params(np.log(K_arr / S))
        
    iv_arr = np.clip(iv_arr, 0.01, 5.0)
    return np.array([bs_call_price(S, K, T, r, iv) for K, iv in zip(K_arr, iv_arr)])

def breeden_litzenberger(smile_params, S, T, r, n_strikes=500, method="svi"):
    """
    Extracts the risk-neutral probability density function using the
    Breeden-Litzenberger formula.
    Enforces monotone decreasing call prices via isotonic regression before
    differentiating and uses a Savitzky-Golay filter for stable derivatives.
    Returns the strike grid and the normalised PDF.
    """
    if method == "svi":
        atm_iv = float(svi_iv(np.array([S]), smile_params)[0])
    elif method == "sabr":
        atm_iv = float(smile_params["alpha"]) 
    elif method == "flat":
        atm_iv = float(smile_params(np.array([0.0]))[0])
    else:
        atm_iv = float(smile_params(np.array([0.0])))
        
    atm_iv = np.clip(atm_iv, 0.05, 2.0)
    if method == "sabr":
        F = S * np.exp(r * T)
        K_min = max(S * 0.60, F * np.exp(-2.0 * atm_iv * np.sqrt(T)))
        K_max = min(S * 1.60, F * np.exp( 2.0 * atm_iv * np.sqrt(T)))
    else:
        K_min = max(S * np.exp(-2.0 * atm_iv * np.sqrt(T)), S * 0.60)
        K_max = min(S * np.exp(2.0 * atm_iv * np.sqrt(T)), S * 1.60)

    K_grid = np.linspace(K_min, K_max, n_strikes)
    dK = K_grid[1] - K_grid[0]

    C = call_price_from_smile(K_grid, smile_params, S, T, r, method)
    C = np.clip(C, 0, None)

    ir = IsotonicRegression(increasing=False)
    C = ir.fit_transform(K_grid, C)

    d2C = savgol_filter(C, window_length=51, polyorder=3, deriv=2, delta=dK)
    pdf = np.exp(r * T) * d2C
    pdf = np.maximum(pdf, 0)
    pdf = gaussian_filter1d(pdf, sigma=4)
    pdf /= np.trapz(pdf, K_grid)

    return K_grid, pdf



def extract_cdf(K_grid, pdf):
    """
    Computes the cumulative distribution function from the PDF by numerical
    integration using the trapezoidal rule. The result is normalised to
    ensure it reaches exactly 1 at the right boundary.
    """
    cdf = cumulative_trapezoid(pdf, K_grid, initial=0)
    cdf /= cdf[-1]
    return cdf

def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) SABR implied volatility approximation.
    Returns the implied vol for a given strike K, forward F, maturity T
    and SABR parameters alpha, beta, rho, nu.
    """
    if abs(F - K) < 1e-10:
        # ATM formula
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / F ** (2 - 2 * beta))
            + (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
            + ((2 - 3 * rho ** 2) / 24) * nu ** 2
        ) * T
        return term1 * term2

    log_FK = np.log(F / K)
    FK_mid = (F * K) ** ((1 - beta) / 2)

    z     = (nu / alpha) * FK_mid * log_FK
    chi_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    term_A = alpha / (
        FK_mid * (
            1
            + ((1 - beta) ** 2 / 24) * log_FK ** 2
            + ((1 - beta) ** 4 / 1920) * log_FK ** 4
        )
    )
    term_B = z / chi_z if abs(chi_z) > 1e-10 else 1.0
    term_C = 1 + (
        ((1 - beta) ** 2 / 24) * (alpha ** 2 / FK_mid ** 2)
        + (rho * beta * nu * alpha) / (4 * FK_mid)
        + ((2 - 3 * rho ** 2) / 24) * nu ** 2
    ) * T

    return term_A * term_B * term_C


def calibrate_sabr(smile_df, r=0.05, beta=0.5):
    """
    Calibrates SABR parameters (alpha, rho, nu) to the observed implied
    volatility smile for a single asset, with beta fixed at 0.5.
    Uses least squares minimisation via scipy.optimize.minimize.
    Returns a dictionary of calibrated parameters plus forward and maturity.
    """
    S = smile_df["S"].iloc[0]
    T = smile_df["T"].iloc[0]
    F = S * np.exp(r * T)

    strikes = smile_df["strike"].values
    iv_mkt  = smile_df["iv"].values

    def loss(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e6
        iv_fit = np.array([sabr_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
        return np.sum((iv_fit - iv_mkt) ** 2)

    best_result = None
    best_loss   = np.inf

    # Multiple starting points for robustness
    for alpha0 in [0.2, 0.4, 0.6]:
        for rho0 in [-0.5, 0.0]:
            for nu0 in [0.3, 0.6]:
                res = minimize(loss, [alpha0, rho0, nu0],
                               method="Nelder-Mead",
                               options={"maxiter": 2000, "xatol": 1e-6})
                if res.fun < best_loss:
                    best_loss   = res.fun
                    best_result = res

    alpha, rho, nu = best_result.x
    return {
        "alpha": alpha, "beta": beta, "rho": rho, "nu": nu,
        "F": F, "T": T, "S": S
    }