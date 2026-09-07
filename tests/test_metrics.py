"""Tests for MetricsCalculator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.metrics import MetricsCalculator


@pytest.fixture()
def calc() -> MetricsCalculator:
    return MetricsCalculator()


def make_nav(returns: pd.Series, start: float = 1_000_000) -> pd.Series:
    return (1 + returns).cumprod() * start


class TestMetricsCalculator:
    def test_total_return_positive(self, calc):
        r = pd.Series([0.01] * 252)
        nav = make_nav(r)
        m = calc.compute(r, nav)
        assert m.total_return > 0

    def test_total_return_negative(self, calc):
        r = pd.Series([-0.01] * 252)
        nav = make_nav(r)
        m = calc.compute(r, nav)
        assert m.total_return < 0

    def test_sharpe_zero_for_zero_returns(self, calc):
        r = pd.Series([0.0] * 100)
        nav = make_nav(r)
        m = calc.compute(r, nav, risk_free_rate=0.0)
        assert m.sharpe == pytest.approx(0.0, abs=1e-6)

    def test_sharpe_positive_for_consistent_gains(self, calc):
        r = pd.Series([0.002] * 252)
        nav = make_nav(r)
        m = calc.compute(r, nav, risk_free_rate=0.0)
        assert m.sharpe > 0

    def test_max_drawdown_never_positive(self, calc):
        rng = np.random.default_rng(7)
        r = pd.Series(rng.normal(0.0005, 0.015, 500))
        nav = make_nav(r)
        m = calc.compute(r, nav)
        assert m.max_drawdown <= 0

    def test_max_drawdown_finds_known_trough(self, calc):
        # Up 10%, then down 50%, then flat
        r = pd.Series([0.10, -0.50] + [0.0] * 10)
        nav = make_nav(r)
        m = calc.compute(r, nav, risk_free_rate=0.0)
        # peak = 1.1, trough = 0.55 → dd = -0.5
        assert m.max_drawdown == pytest.approx(-0.50, rel=1e-3)

    def test_win_rate_all_gains(self, calc):
        r = pd.Series([0.001] * 100)
        nav = make_nav(r)
        m = calc.compute(r, nav)
        assert m.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losses(self, calc):
        r = pd.Series([-0.001] * 100)
        nav = make_nav(r)
        m = calc.compute(r, nav)
        assert m.win_rate == pytest.approx(0.0)

    def test_calmar_ratio_sign(self, calc):
        r = pd.Series([0.005] * 252)
        nav = make_nav(r)
        m = calc.compute(r, nav)
        # positive CAGR / negative dd → positive calmar
        assert m.calmar > 0

    def test_var_cvar_ordering(self, calc):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0, 0.02, 1000))
        nav = make_nav(r)
        m = calc.compute(r, nav)
        # CVaR should be <= VaR (tail average is worse than the 5th percentile)
        assert m.cvar_95 <= m.var_95

    def test_raises_on_too_few_observations(self, calc):
        r = pd.Series([0.01])
        nav = make_nav(r)
        with pytest.raises(ValueError, match="at least 2"):
            calc.compute(r, nav)

    def test_rolling_sharpe_length_matches_input(self, calc):
        r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 200))
        rs = calc.rolling_sharpe(r, window=63)
        assert len(rs) == len(r)

    def test_drawdown_series_always_non_positive(self, calc):
        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0, 0.02, 300))
        nav = make_nav(r)
        dd = calc.drawdown_series(nav)
        assert (dd <= 0).all()
