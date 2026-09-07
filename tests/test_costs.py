"""Tests for TransactionCostModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.costs import TransactionCostModel


@pytest.fixture()
def sample_data():
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    symbols = ["AAPL", "MSFT"]
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        rng.uniform(100, 200, (30, 2)), index=dates, columns=symbols
    )
    volumes = pd.DataFrame(
        rng.uniform(1e6, 5e6, (30, 2)), index=dates, columns=symbols
    )
    return prices, volumes


class TestTransactionCostModel:
    def test_spread_cost_is_linear(self, sample_data):
        prices, volumes = sample_data
        model = TransactionCostModel(spread_bps=10.0, impact_eta=0.0)
        trades = pd.DataFrame(
            0.1, index=prices.index, columns=prices.columns
        )
        result = model.compute(trades, prices, volumes)

        expected = (trades.abs() * prices * 10.0 / 10_000).sum(axis=1)
        pd.testing.assert_series_equal(
            result.spread_cost, expected, check_names=False
        )

    def test_impact_cost_increases_with_trade_size(self, sample_data):
        prices, volumes = sample_data
        model = TransactionCostModel(spread_bps=0.0, impact_eta=0.1)

        small = pd.DataFrame(0.01, index=prices.index, columns=prices.columns)
        large = pd.DataFrame(0.50, index=prices.index, columns=prices.columns)

        small_cost = model.compute(small, prices, volumes).impact_cost
        large_cost = model.compute(large, prices, volumes).impact_cost

        # skip the ADV warmup window where both are zero
        warmup = model.adv_window
        assert (large_cost.iloc[warmup:] > small_cost.iloc[warmup:]).all()

    def test_zero_trade_means_zero_cost(self, sample_data):
        prices, volumes = sample_data
        model = TransactionCostModel()
        trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        result = model.compute(trades, prices, volumes)
        assert (result.total == 0).all()

    def test_total_is_spread_plus_impact(self, sample_data):
        prices, volumes = sample_data
        model = TransactionCostModel(spread_bps=5.0, impact_eta=0.1)
        trades = pd.DataFrame(0.1, index=prices.index, columns=prices.columns)
        result = model.compute(trades, prices, volumes)
        pd.testing.assert_series_equal(
            result.total,
            result.spread_cost + result.impact_cost,
            check_names=False,
        )

    def test_output_is_series_indexed_by_date(self, sample_data):
        prices, volumes = sample_data
        model = TransactionCostModel()
        trades = pd.DataFrame(0.05, index=prices.index, columns=prices.columns)
        result = model.compute(trades, prices, volumes)
        assert isinstance(result.total, pd.Series)
        pd.testing.assert_index_equal(result.total.index, prices.index)
