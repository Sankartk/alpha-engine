"""
MetricsCalculator: computes standard quantitative performance metrics.

All metrics are computed from net returns (after costs).
No external dependency beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    skew: float
    kurtosis: float
    var_95: float       # 1-day 95% Value at Risk
    cvar_95: float      # 1-day 95% Conditional VaR


class MetricsCalculator:
    TRADING_DAYS = 252

    def compute(
        self,
        returns: pd.Series,
        nav: pd.Series,
        risk_free_rate: float = 0.05,
    ) -> PerformanceMetrics:
        r = returns.dropna()
        if len(r) < 2:
            raise ValueError("Need at least 2 return observations")

        rf_daily = risk_free_rate / self.TRADING_DAYS
        excess = r - rf_daily

        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        n_years = len(r) / self.TRADING_DAYS
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1

        ann_vol = r.std() * np.sqrt(self.TRADING_DAYS)
        sharpe = (excess.mean() / excess.std()) * np.sqrt(self.TRADING_DAYS) \
            if excess.std() > 0 else 0.0

        downside = r[r < 0]
        sortino = (excess.mean() / downside.std()) * np.sqrt(self.TRADING_DAYS) \
            if len(downside) > 1 and downside.std() > 0 else 0.0

        # Max drawdown
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max
        max_dd = drawdown.min()

        if max_dd == 0:
            # no drawdown observed — calmar is unbounded
            calmar = float("inf") if cagr > 0 else 0.0
        else:
            calmar = cagr / abs(max_dd)

        win_rate = (r > 0).mean()

        gross_gains = r[r > 0].sum()
        gross_losses = abs(r[r < 0].sum())
        profit_factor = gross_gains / gross_losses if gross_losses > 0 else np.inf

        var_95 = np.percentile(r, 5)
        cvar_95 = r[r <= var_95].mean() if (r <= var_95).any() else var_95

        return PerformanceMetrics(
            total_return=float(total_return),
            cagr=float(cagr),
            ann_vol=float(ann_vol),
            sharpe=float(sharpe),
            sortino=float(sortino),
            max_drawdown=float(max_dd),
            calmar=float(calmar),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            skew=float(r.skew()),
            kurtosis=float(r.kurtosis()),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
        )

    def rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.05,
    ) -> pd.Series:
        rf_daily = risk_free_rate / self.TRADING_DAYS
        excess = returns - rf_daily
        return (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(self.TRADING_DAYS)
        )

    def drawdown_series(self, nav: pd.Series) -> pd.Series:
        running_max = nav.cummax()
        return (nav - running_max) / running_max
