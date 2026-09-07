"""
MomentumStrategy: cross-sectional momentum with vol-scaling and trend filter.

Signal: rank stocks by 12-1 month return (skip last month to avoid reversal),
vol-scale by inverse 20-day realised vol, apply 200-day SMA trend filter.

No lookahead: all rolling windows use only past data at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        lookback_long: int = 252,   # 12 months
        lookback_short: int = 21,   # 1 month (skipped)
        vol_window: int = 20,
        sma_window: int = 200,
        top_pct: float = 0.20,      # long top 20%
        bottom_pct: float = 0.20,   # short bottom 20%
        max_weight: float = 0.10,   # per-name cap
    ) -> None:
        super().__init__(name="momentum")
        if not 0 < top_pct <= 1.0:
            raise ValueError("top_pct must be in (0, 1]")
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.vol_window = vol_window
        self.sma_window = sma_window
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.max_weight = max_weight

    def generate_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        returns = prices.pct_change()

        # 12-1 month momentum: return from t-252 to t-21
        mom = (
            prices.shift(self.lookback_short) / prices.shift(self.lookback_long) - 1
        )

        # Inverse-vol scaling
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)
        inv_vol = 1.0 / vol

        # Trend filter: only long if price > 200 SMA
        sma200 = prices.rolling(self.sma_window).mean()
        trend_ok = prices > sma200

        # Cross-sectional rank [0, 1]
        rank = mom.rank(axis=1, pct=True)

        n = prices.shape[1]
        n_long = max(1, int(n * self.top_pct))
        n_short = max(1, int(n * self.bottom_pct))

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for date in prices.index:
            r = rank.loc[date]
            if r.isna().all():
                continue

            long_mask = (r >= (1 - self.top_pct)) & trend_ok.loc[date]
            short_mask = r <= self.bottom_pct

            long_syms = r[long_mask].nlargest(n_long).index
            short_syms = r[short_mask].nsmallest(n_short).index

            if len(long_syms) > 0:
                w = inv_vol.loc[date, long_syms]
                w = w / w.sum() * 0.5   # 50% gross long
                weights.loc[date, long_syms] = w.clip(upper=self.max_weight)

            if len(short_syms) > 0:
                w = inv_vol.loc[date, short_syms]
                w = w / w.sum() * -0.5  # 50% gross short
                weights.loc[date, short_syms] = w.clip(lower=-self.max_weight)

        return weights
