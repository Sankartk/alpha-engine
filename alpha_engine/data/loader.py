"""
DataLoader: fetches and caches OHLCV data from Alpaca or yfinance.

Implements the Repository pattern — callers never touch the raw API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


class BaseDataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns [open, high, low, close, volume]."""
        ...


class YFinanceProvider(BaseDataProvider):
    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]].dropna()


class AlpacaProvider(BaseDataProvider):
    def __init__(self, api_key: str, secret_key: str) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as e:
            raise ImportError("pip install alpaca-py") from e

        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._TimeFrame = TimeFrame
        self._StockBarsRequest = StockBarsRequest

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        req = self._StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self._client.get_stock_bars(req).df
        df = bars.xs(symbol) if symbol in bars.index.get_level_values(0) else bars
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]].dropna()


class DataLoader:
    """
    Fetches OHLCV data with local parquet caching.
    Implements cache-aside pattern: check disk first, fetch on miss, save on success.
    """

    def __init__(
        self,
        provider: BaseDataProvider | None = None,
        cache_dir: str | Path = ".cache/ohlcv",
    ) -> None:
        self._provider = provider or YFinanceProvider()
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        start = start or datetime.now() - timedelta(days=730)
        end = end or datetime.now()

        cache_path = self._cache_path(symbol, start, end)
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        df = self._provider.fetch(symbol, start, end)
        df.to_parquet(cache_path)
        return df

    def get_many(
        self,
        symbols: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        return {sym: self.get(sym, start, end) for sym in symbols}

    def _cache_path(self, symbol: str, start: datetime, end: datetime) -> Path:
        key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        return self._cache_dir / key
