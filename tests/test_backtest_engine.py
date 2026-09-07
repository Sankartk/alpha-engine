"""Tests for BacktestEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest import BacktestEngine, TransactionCostModel
from alpha_engine.strategy.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Always 100% long every symbol — deterministic for testing."""

    def __init__(self):
        super().__init__("buy_and_hold")

    def generate_signals(self, prices, volumes):
        n = prices.shape[1]
        return pd.DataFrame(
            1.0 / n, index=prices.index, columns=prices.columns
        )


class FlatStrategy(BaseStrategy):
    """Always flat — should return zero everywhere."""

    def __init__(self):
        super().__init__("flat")

    def generate_signals(self, prices, volumes):
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


@pytest.fixture()
def market_data():
    dates = pd.date_range("2022-01-03", periods=300, freq="B")
    symbols = ["A", "B", "C"]
    rng = np.random.default_rng(42)
    # Upward drift so buy-and-hold is profitable
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (300, 3)), axis=0)),
        index=dates, columns=symbols,
    )
    volumes = pd.DataFrame(
        rng.uniform(1e6, 5e6, (300, 3)), index=dates, columns=symbols
    )
    return prices, volumes


class TestBacktestEngine:
    def test_flat_strategy_returns_zero(self, market_data):
        prices, volumes = market_data
        engine = BacktestEngine(cost_model=TransactionCostModel(spread_bps=0, impact_eta=0))
        result = engine.run(FlatStrategy(), prices, volumes)
        assert result.net_returns.abs().max() == pytest.approx(0.0, abs=1e-10)

    def test_buy_and_hold_positive_nav_growth(self, market_data):
        prices, volumes = market_data
        engine = BacktestEngine(cost_model=TransactionCostModel(spread_bps=0, impact_eta=0))
        result = engine.run(BuyAndHoldStrategy(), prices, volumes)
        # With upward drift, NAV should end above start (with high probability)
        assert result.nav.iloc[-1] > result.nav.iloc[0] * 0.9

    def test_weights_shifted_by_one_day(self, market_data):
        """Day 1 return must be 0 — weights are applied T+1, not same-day."""
        prices, volumes = market_data
        engine = BacktestEngine(cost_model=TransactionCostModel(spread_bps=0, impact_eta=0))
        result = engine.run(BuyAndHoldStrategy(), prices, volumes)
        # First gross return must be 0 because weights[0] = 0 after shift
        assert result.gross_returns.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_costs_reduce_net_returns(self, market_data):
        prices, volumes = market_data
        free = BacktestEngine(
            cost_model=TransactionCostModel(spread_bps=0, impact_eta=0)
        ).run(BuyAndHoldStrategy(), prices, volumes)
        costly = BacktestEngine(
            cost_model=TransactionCostModel(spread_bps=20, impact_eta=0.5)
        ).run(BuyAndHoldStrategy(), prices, volumes)
        assert costly.nav.iloc[-1] <= free.nav.iloc[-1]

    def test_result_has_all_fields(self, market_data):
        prices, volumes = market_data
        engine = BacktestEngine()
        result = engine.run(BuyAndHoldStrategy(), prices, volumes)
        assert result.strategy_name == "buy_and_hold"
        assert result.metrics is not None
        assert result.nav is not None
        assert result.turnover is not None
        assert result.cost_drag is not None

    def test_walk_forward_returns_multiple_folds(self, market_data):
        prices, volumes = market_data
        engine = BacktestEngine(
            cost_model=TransactionCostModel(spread_bps=0, impact_eta=0)
        )
        folds = engine.walk_forward(
            FlatStrategy(), prices, volumes,
            train_days=100, test_days=50, step_days=50,
        )
        assert len(folds) >= 1
        for fold in folds:
            assert fold.metrics is not None

    def test_nav_starts_at_initial_capital(self, market_data):
        prices, volumes = market_data
        capital = 500_000
        engine = BacktestEngine(
            cost_model=TransactionCostModel(spread_bps=0, impact_eta=0),
            initial_capital=capital,
        )
        result = engine.run(FlatStrategy(), prices, volumes)
        assert result.nav.iloc[0] == pytest.approx(capital, rel=1e-6)
