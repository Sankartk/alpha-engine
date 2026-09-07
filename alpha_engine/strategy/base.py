"""
BaseStrategy: abstract interface for all strategies.

Implements Template Method — subclasses implement generate_signals(),
the base class handles validation and output formatting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    symbol: str
    date: pd.Timestamp
    signal: SignalType
    weight: float          # target portfolio weight [0, 1]
    confidence: float      # [0, 1]
    metadata: dict         # strategy-specific context

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0,1], got {self.weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


class BaseStrategy(ABC):
    """
    All strategies produce a DataFrame of target weights indexed by (date, symbol).

    The backtester consumes this weight matrix — strategies never see the portfolio,
    ensuring clean separation of concerns.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,   # columns = symbols, index = dates, values = close
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return weight DataFrame: index=dates, columns=symbols, values=[-1, 1].
        Positive = long, negative = short, 0 = flat.
        Must NOT use future data (no lookahead bias).
        """
        ...

    def validate(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Clip weights to [-1, 1] and fill NaN with 0."""
        return weights.clip(-1.0, 1.0).fillna(0.0)

    def run(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        raw = self.generate_signals(prices, volumes)
        return self.validate(raw)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
