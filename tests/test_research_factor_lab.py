"""The research factor lab runs end to end on synthetic data (no network)."""

import numpy as np
import pandas as pd

from research import factor_lab as lab


def _frames(n_days=900, n=30, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    symbols = [f"S{i:02d}" for i in range(n)]
    rets = rng.normal(0.0004, 0.02, size=(n_days, n))
    closes = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=symbols)
    opens = closes.shift(1).fillna(100) * (1 + rng.normal(0, 0.005, size=(n_days, n)))
    volumes = pd.DataFrame(rng.lognormal(14, 0.5, size=(n_days, n)), index=dates, columns=symbols)
    market = pd.Series(rng.normal(0.0004, 0.012, size=n_days), index=dates)
    return closes, opens, volumes, market


def test_candidate_factors_have_the_panel_shape_and_finite_recent_values():
    closes, opens, volumes, market = _frames()
    returns = closes.pct_change(fill_method=None)
    resid = lab.residual_returns(returns, market, 252)
    sector_map = pd.Series(["A"] * 15 + ["B"] * 15, index=closes.columns)
    frames = {
        "resid_mom": lab.residual_momentum_factor(resid, 252, 21),
        "sharpe_mom": lab.sharpe_momentum_factor(returns, 252, 21),
        "frog": lab.momentum_consistency_factor(returns, 252, 21),
        "idio": lab.idiosyncratic_vol_factor(resid, 63),
        "max": lab.max_return_factor(returns, 21),
        "skew": lab.skewness_factor(returns, 126),
        "beta": lab.beta_factor(returns, market, 252),
        "overnight": lab.overnight_momentum_factor(opens, closes, 63),
        "intraday": lab.intraday_reversal_factor(opens, closes, 63),
        "abn_vol": lab.abnormal_volume_factor(volumes, 5, 63),
        "vol_trend": lab.volume_trend_factor(volumes, returns, 63),
        "amihud": lab.amihud_factor(returns, closes, volumes, 63),
        "turnover": lab.turnover_factor(closes, volumes),
        "low_prox": lab.low_proximity_factor(closes, 252),
        "accel": lab.acceleration_factor(closes),
        "sector_mom": lab.sector_momentum_factor(closes, sector_map),
        "seasonality": lab.seasonality_factor(returns, 2),
    }
    for name, f in frames.items():
        assert f.shape == closes.shape, name
        tail = f.iloc[-50:]
        assert np.isfinite(tail.values).mean() > 0.9, name
        # z-scored across names: mean ~0, std ~1 per day
        assert abs(tail.mean(axis=1)).max() < 1e-6, name
        assert (tail.std(axis=1) - 1).abs().max() < 1e-6, name


def test_forward_returns_and_ic_pipeline():
    closes, opens, volumes, market = _frames()
    returns = closes.pct_change(fill_method=None)
    fwd = lab.forward_returns(opens, closes, 21)
    # Row t: from the open of t+1 to the open of t+22.
    t = 100
    expected = opens.iloc[t + 22] / opens.iloc[t + 1] - 1
    pd.testing.assert_series_equal(fwd.iloc[t], expected, check_names=False)
    assert fwd.iloc[-22:].isna().all().all()

    # A factor that IS the future return has IC 1; its negative has IC -1.
    ic = lab.daily_ic(fwd, fwd).dropna()
    assert (ic.round(9) == 1).all()
    ic = lab.daily_ic(-fwd, fwd).dropna()
    assert (ic.round(9) == -1).all()

    dates = closes.index[::21]
    out = lab.summarize(fwd, fwd, dates)
    assert out["ic"] == 1 and out["hit"] == 1 and out["top12_ann"] > 0
    assert set(out) >= {"icir", "t", "n", "rank_turnover", "ic_21-26"}
