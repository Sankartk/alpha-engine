"""
TransactionCostModel: linear + square-root market impact cost model.

Implements the model from Almgren et al. (2005):
  cost = spread_cost + impact_cost
  spread_cost = |trade_value| * spread_bps / 10000
  impact_cost = |trade_value| * eta * sigma * sqrt(|trade_value| / ADV)

where ADV = 20-day average daily volume in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostBreakdown:
    spread_cost: pd.Series
    impact_cost: pd.Series
    total: pd.Series


class TransactionCostModel:
    def __init__(
        self,
        spread_bps: float = 5.0,      # one-way spread in basis points
        impact_eta: float = 0.1,      # market impact coefficient
        adv_window: int = 20,         # ADV lookback in days
    ) -> None:
        self.spread_bps = spread_bps
        self.impact_eta = impact_eta
        self.adv_window = adv_window

    def compute(
        self,
        trades: pd.DataFrame,       # weight changes: index=dates, columns=symbols
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> CostBreakdown:
        trade_values = trades.abs() * prices

        # Spread cost: simple linear
        spread_cost = trade_values * self.spread_bps / 10_000

        # Market impact: square-root model
        dollar_volumes = prices * volumes
        adv = dollar_volumes.rolling(self.adv_window).mean()
        sigma = prices.pct_change().rolling(20).std() * np.sqrt(252)

        participation = (trade_values / adv.replace(0, np.nan)).clip(0, 1)
        impact_cost = (
            trade_values
            * self.impact_eta
            * sigma
            * np.sqrt(participation.clip(lower=0))
        ).fillna(0)

        total = spread_cost + impact_cost
        # Sum across symbols per date → Series
        return CostBreakdown(
            spread_cost=spread_cost.sum(axis=1),
            impact_cost=impact_cost.sum(axis=1),
            total=total.sum(axis=1),
        )
