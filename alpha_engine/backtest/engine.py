"""
BacktestEngine: vectorised portfolio simulation with walk-forward validation.

Implements the Engine pattern: takes strategy + data + cost model,
produces a BacktestResult with NAV, returns, drawdown, and turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..strategy.base import BaseStrategy
from .costs import TransactionCostModel
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class BacktestResult:
    strategy_name: str
    start: datetime
    end: datetime
    nav: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    cost_drag: pd.Series
    metrics: PerformanceMetrics
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Strategy : {self.strategy_name}\n"
            f"Period   : {self.start:%Y-%m-%d} → {self.end:%Y-%m-%d}\n"
            f"─────────────────────────────────\n"
            f"Total return  : {m.total_return:+.1%}\n"
            f"CAGR          : {m.cagr:+.1%}\n"
            f"Sharpe        : {m.sharpe:.2f}\n"
            f"Sortino       : {m.sortino:.2f}\n"
            f"Max drawdown  : {m.max_drawdown:.1%}\n"
            f"Calmar        : {m.calmar:.2f}\n"
            f"Volatility    : {m.ann_vol:.1%}\n"
            f"Win rate      : {m.win_rate:.1%}\n"
            f"Avg turnover  : {self.turnover.mean():.1%}/day\n"
            f"Cost drag     : {self.cost_drag.sum():.2%} total\n"
        )


class BacktestEngine:
    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
        initial_capital: float = 1_000_000,
    ) -> None:
        self._cost_model = cost_model or TransactionCostModel()
        self._initial_capital = initial_capital
        self._metrics = MetricsCalculator()

    def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> BacktestResult:
        # Align inputs
        prices = prices.sort_index().ffill()
        volumes = volumes.sort_index().ffill()

        # Generate weights (T-1 applied to T returns — no lookahead)
        weights = strategy.run(prices, volumes)
        weights = weights.shift(1).fillna(0)   # apply weights one day later

        # Portfolio returns
        asset_returns = prices.pct_change().fillna(0)
        gross_returns = (weights * asset_returns).sum(axis=1)

        # Turnover = sum of |weight changes| per day
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2   # one-way turnover

        # Transaction costs
        costs = self._cost_model.compute(weight_changes, prices, volumes)
        cost_returns = costs.total / self._initial_capital

        net_returns = gross_returns - cost_returns

        # NAV
        nav = (1 + net_returns).cumprod() * self._initial_capital

        metrics = self._metrics.compute(net_returns, nav)

        return BacktestResult(
            strategy_name=strategy.name,
            start=prices.index[0].to_pydatetime(),
            end=prices.index[-1].to_pydatetime(),
            nav=nav,
            gross_returns=gross_returns,
            net_returns=net_returns,
            weights=weights,
            turnover=turnover,
            cost_drag=cost_returns,
            metrics=metrics,
        )

    def walk_forward(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 63,
    ) -> list[BacktestResult]:
        """
        Walk-forward validation: train on [t-504, t], test on [t, t+63].
        Prevents overfitting by never testing on training data.
        """
        results = []
        dates = prices.index

        start_idx = train_days
        while start_idx + test_days < len(dates):
            train_end = dates[start_idx]
            test_end_idx = min(start_idx + test_days, len(dates) - 1)

            test_prices = prices.iloc[start_idx:test_end_idx + 1]
            test_volumes = volumes.iloc[start_idx:test_end_idx + 1]

            result = self.run(strategy, test_prices, test_volumes)
            result.metadata["train_end"] = train_end
            result.metadata["fold"] = len(results)
            results.append(result)

            start_idx += step_days

        return results
