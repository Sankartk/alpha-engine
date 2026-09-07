"""
MeanReversionStrategy: z-score mean reversion with RSI confirmation.

Entry: z-score of 20-day price vs 60-day mean exceeds +/-2 standard deviations.
Filter: RSI(14) confirms oversold/overbought before entering.
Exit: z-score crosses back through 0.

No lookahead: rolling stats use only past data at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        z_window: int = 20,
        mean_window: int = 60,
        rsi_window: int = 14,
        z_entry: float = 2.0,
        z_exit: float = 0.0,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        max_weight: float = 0.15,
    ) -> None:
        super().__init__(name="mean_reversion")
        self.z_window = z_window
        self.mean_window = mean_window
        self.rsi_window = rsi_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_weight = max_weight

    def _rsi(self, prices: pd.DataFrame) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        rolling_mean = prices.rolling(self.mean_window).mean()
        rolling_std = prices.rolling(self.z_window).std()
        z_score = (prices - rolling_mean) / rolling_std.replace(0, np.nan)

        rsi = self._rsi(prices)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        in_position = pd.DataFrame(False, index=prices.index, columns=prices.columns)

        for i in range(1, len(prices)):
            date = prices.index[i]
            z = z_score.iloc[i]
            r = rsi.iloc[i]
            prev_pos = in_position.iloc[i - 1]

            long_entry = (z < -self.z_entry) & (r < self.rsi_oversold) & ~prev_pos
            short_entry = (z > self.z_entry) & (r > self.rsi_overbought) & ~prev_pos
            exit_long = (z >= self.z_exit) & prev_pos
            exit_short = (z <= self.z_exit) & prev_pos

            new_pos = prev_pos.copy()
            new_pos[long_entry] = True
            new_pos[short_entry] = True   # short positions also tracked as True
            new_pos[exit_long | exit_short] = False

            in_position.iloc[i] = new_pos

            long_mask = long_entry | (prev_pos & ~exit_long & (z < 0))
            short_mask = short_entry | (prev_pos & ~exit_short & (z > 0))

            if long_mask.any():
                n = long_mask.sum()
                weights.loc[date, long_mask[long_mask].index] = min(
                    self.max_weight, 0.5 / n
                )
            if short_mask.any():
                n = short_mask.sum()
                weights.loc[date, short_mask[short_mask].index] = -min(
                    self.max_weight, 0.5 / n
                )

        return weights
