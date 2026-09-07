"""
PaperTrader: executes strategy signals via Alpaca paper trading API.

Implements the Adapter pattern — translates strategy weight targets
into Alpaca order calls. All orders are market-on-close.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    status: str
    filled_price: float | None
    submitted_at: datetime


class PaperTrader:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError as e:
            raise ImportError("pip install alpaca-py") from e

        key = api_key or os.environ["ALPACA_API_KEY"]
        secret = secret_key or os.environ["ALPACA_SECRET_KEY"]

        self._client = TradingClient(key, secret, paper=True)
        self._MarketOrderRequest = MarketOrderRequest
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_positions(self) -> dict[str, float]:
        positions = self._client.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def rebalance(
        self,
        target_weights: pd.Series,
        portfolio_value: float | None = None,
        dry_run: bool = False,
    ) -> list[OrderResult]:
        """
        Rebalance portfolio to target weights.
        target_weights: Series indexed by symbol, values in [-1, 1].
        """
        if portfolio_value is None:
            portfolio_value = self.get_account()["portfolio_value"]

        current = self.get_positions()
        prices = self._get_latest_prices(list(target_weights.index))

        results = []
        for symbol, weight in target_weights.items():
            if symbol not in prices:
                continue

            target_qty = (weight * portfolio_value) / prices[symbol]
            current_qty = current.get(symbol, 0.0)
            diff = target_qty - current_qty

            if abs(diff) < 1:
                continue

            side = self._OrderSide.BUY if diff > 0 else self._OrderSide.SELL
            qty = abs(round(diff, 2))

            if dry_run:
                results.append(OrderResult(
                    symbol=symbol, side=str(side), qty=qty,
                    status="dry_run", filled_price=None,
                    submitted_at=datetime.utcnow(),
                ))
                continue

            req = self._MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=self._TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            results.append(OrderResult(
                symbol=symbol, side=str(side), qty=qty,
                status=order.status.value,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=datetime.utcnow(),
            ))
            time.sleep(0.1)   # rate limit

        return results

    def liquidate_all(self) -> None:
        self._client.close_all_positions(cancel_orders=True)

    def _get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        data_client = StockHistoricalDataClient(key, secret)

        req = StockLatestTradeRequest(symbol_or_symbols=symbols)
        trades = data_client.get_stock_latest_trade(req)
        return {sym: float(t.price) for sym, t in trades.items()}
