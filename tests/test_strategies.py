"""Tests for strategy signal generation — checking no lookahead bias."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategy import MeanReversionStrategy, MomentumStrategy


@pytest.fixture()
def prices_volumes():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    symbols = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (400, 10)), axis=0)),
        index=dates, columns=symbols,
    )
    volumes = pd.DataFrame(
        rng.uniform(1e6, 5e6, (400, 10)), index=dates, columns=symbols
    )
    return prices, volumes


class TestMomentumStrategy:
    def test_output_shape_matches_input(self, prices_volumes):
        prices, volumes = prices_volumes
        w = MomentumStrategy().run(prices, volumes)
        assert w.shape == prices.shape

    def test_weights_bounded(self, prices_volumes):
        prices, volumes = prices_volumes
        w = MomentumStrategy().run(prices, volumes)
        assert (w.abs() <= 1.0 + 1e-9).all().all()

    def test_per_name_cap_respected(self, prices_volumes):
        prices, volumes = prices_volumes
        cap = 0.10
        w = MomentumStrategy(max_weight=cap).run(prices, volumes)
        assert (w.abs() <= cap + 1e-9).all().all()

    def test_no_lookahead_first_rows_zero(self, prices_volumes):
        """Weights must be 0 until enough history exists for the longest lookback."""
        prices, volumes = prices_volumes
        strategy = MomentumStrategy(lookback_long=252, lookback_short=21)
        w = strategy.run(prices, volumes)
        # First 252 rows can't have a signal (not enough history)
        assert (w.iloc[:252] == 0).all().all()

    def test_gross_exposure_bounded(self, prices_volumes):
        prices, volumes = prices_volumes
        w = MomentumStrategy().run(prices, volumes)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all()


class TestMeanReversionStrategy:
    def test_output_shape_matches_input(self, prices_volumes):
        prices, volumes = prices_volumes
        w = MeanReversionStrategy().run(prices, volumes)
        assert w.shape == prices.shape

    def test_weights_bounded(self, prices_volumes):
        prices, volumes = prices_volumes
        w = MeanReversionStrategy().run(prices, volumes)
        assert (w.abs() <= 1.0 + 1e-9).all().all()

    def test_no_lookahead_first_rows_zero(self, prices_volumes):
        prices, volumes = prices_volumes
        strategy = MeanReversionStrategy(mean_window=60, z_window=20)
        w = strategy.run(prices, volumes)
        # rolling(60).mean() first produces a value at index 59, so a signal
        # can legitimately appear there — everything before must be zero
        assert (w.iloc[:59] == 0).all().all()

    def test_flat_market_gives_flat_weights(self):
        """Perfectly constant prices → z-score is 0/NaN → no position."""
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        prices = pd.DataFrame(100.0, index=dates, columns=["X"])
        volumes = pd.DataFrame(1e6, index=dates, columns=["X"])
        w = MeanReversionStrategy().run(prices, volumes)
        assert (w == 0).all().all()
