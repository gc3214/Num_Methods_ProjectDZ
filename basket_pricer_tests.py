import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PROJECT_FILE = os.path.join(HERE, "basket_pricer.py")


def load_basket_pricer():
    project_dir = PROJECT_FILE.parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    if "yfinance" not in sys.modules:
        sys.modules["yfinance"] = types.ModuleType("yfinance")

    if "copula" not in sys.modules:
        copula = types.ModuleType("copula")
        copula.estimate_correlation = lambda *args, **kwargs: None
        copula.build_inverse_cdf = lambda *args, **kwargs: None
        copula.sample_gaussian_copula = lambda *args, **kwargs: None
        copula.sample_student_copula = lambda *args, **kwargs: None
        copula.estimate_nu = lambda *args, **kwargs: None
        sys.modules["copula"] = copula

    if "functions_marginal_distributions" not in sys.modules:
        fmd = types.ModuleType("functions_marginal_distributions")
        fmd.fetch_option_chain = lambda *args, **kwargs: None
        fmd.build_smile = lambda *args, **kwargs: None
        fmd.calibrate_sabr = lambda *args, **kwargs: None
        fmd.sabr_vol = lambda *args, **kwargs: None
        fmd.bs_call_price = lambda *args, **kwargs: None
        fmd.breeden_litzenberger = lambda *args, **kwargs: None
        fmd.extract_cdf = lambda *args, **kwargs: None
        sys.modules["functions_marginal_distributions"] = fmd

    spec = importlib.util.spec_from_file_location("basket_pricer", PROJECT_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def bp():
    return load_basket_pricer()


def test_price_basket_call_validates_basic_inputs(bp, monkeypatch):
    monkeypatch.setattr(bp, "_run_pipeline", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")))

    with pytest.raises(ValueError, match="at least 2 assets"):
        bp.price_basket_call(["AAPL"], verbose=False)

    with pytest.raises(ValueError, match="length"):
        bp.price_basket_call(["AAPL", "MSFT"], weights=[1.0], verbose=False)

    with pytest.raises(ValueError, match="non-negative"):
        bp.price_basket_call(["AAPL", "MSFT"], weights=[0.5, -0.5], verbose=False)

    with pytest.raises(ValueError, match="copula must be"):
        bp.price_basket_call(["AAPL", "MSFT"], copula="bad", verbose=False)

    with pytest.raises(ValueError, match="at least 100"):
        bp.price_basket_call(["AAPL", "MSFT"], n_sim=99, verbose=False)


def test_price_basket_call_normalizes_weights_and_returns_expected_shape(bp, monkeypatch):
    S = np.array([[110.0, 90.0], [100.0, 100.0], [120.0, 80.0]])
    spots = np.array([100.0, 100.0])
    T = 1.0

    monkeypatch.setattr(bp, "_run_pipeline", lambda *args, **kwargs: (S, spots, T, None))

    captured = {}

    def fake_price_with_cv(S_in, K, r, T_in, w, spots_in):
        captured["w"] = w.copy()
        captured["K"] = K
        captured["S"] = S_in.copy()
        return 12.5, 0.75

    monkeypatch.setattr(bp, "_price_with_cv", fake_price_with_cv)
    monkeypatch.setattr(bp, "_price_plain_mc", lambda *args, **kwargs: (10.0, 1.0))

    result = bp.price_basket_call(["AAPL", "MSFT"], weights=[2.0, 1.0], K=None, use_cv=True, verbose=False)

    assert np.allclose(captured["w"], np.array([2.0, 1.0]) / 3.0)
    assert captured["K"] == pytest.approx(100.0)
    assert result["price"] == 12.5
    assert result["se"] == 0.75
    assert result["use_cv"] is True
    assert result["var_reduction"] == "43.8%"
    assert result["spots"] == [100.0, 100.0]


def test_price_basket_call_plain_mc_path(bp, monkeypatch):
    S = np.array([[110.0, 90.0], [100.0, 100.0], [120.0, 80.0]])
    spots = np.array([100.0, 100.0])
    T = 1.0

    monkeypatch.setattr(bp, "_run_pipeline", lambda *args, **kwargs: (S, spots, T, None))
    monkeypatch.setattr(bp, "_price_plain_mc", lambda *args, **kwargs: (8.5, 0.25))

    result = bp.price_basket_call(["AAPL", "MSFT"], K=105.0, use_cv=False, verbose=False)

    assert result["price"] == 8.5
    assert result["se"] == 0.25
    assert result["var_reduction"] is None
    assert result["K"] == 105.0


def test_geo_basket_exact_zero_vol_returns_intrinsic(bp):
    S = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    spots = np.array([100.0, 100.0])
    w = np.array([0.5, 0.5])
    price = bp._geo_basket_call_exact(S, K=95.0, r=0.03, T=1.0, w=w, spots=spots)
    forward = 100.0 * np.exp(0.03)
    expected = np.exp(-0.03) * max(forward - 95.0, 0.0)
    assert price == pytest.approx(expected)


def test_price_with_cv_is_finite_and_better_behaved_on_simple_case(bp):
    rng = np.random.default_rng(0)
    base = np.array([100.0, 100.0])
    shocks = rng.normal(scale=5.0, size=(5000, 2))
    S = base + shocks
    S = np.maximum(S, 1e-6)
    w = np.array([0.5, 0.5])
    spots = base

    price, se = bp._price_with_cv(S, K=100.0, r=0.02, T=1.0, w=w, spots=spots)

    assert np.isfinite(price)
    assert np.isfinite(se)
    assert se >= 0
