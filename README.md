# alpha-engine

**Everyone has a trading strategy that "would have worked." Almost nobody can prove it.**

alpha-engine is a backtesting framework that tries its best to prove your strategy *doesn't* work before the market does it for you — with real transaction costs, walk-forward validation, and a paper-trading loop that runs the same code against live prices.

---

## Why I built it

I've seen too many backtests that were fiction. The pattern is always the same: someone writes a loop over historical prices, forgets that trades cost money, accidentally lets tomorrow's data leak into today's decision, and ends up with a beautiful equity curve that would have lost money in the real world.

The three most common ways backtests lie:

1. **Lookahead bias** — your signal at time *t* uses information that only exists at *t+1*. Sometimes it's as subtle as applying today's weight to today's return.
2. **Ignored costs** — a strategy with 50% daily turnover and a Sharpe of 2.0 on paper is often a Sharpe of 0.3 after spread and market impact.
3. **Overfitting** — tune parameters on the whole dataset and you've built a model of the past, not the future.

This engine is built around not making those mistakes.

## How it stops the lies

**No lookahead, structurally.** Strategies produce target weights for day *t*. The engine shifts them by one day before multiplying by returns. You literally cannot earn tomorrow's return on today's signal — the code won't let you.

**Real costs.** Every rebalance pays the bid-ask spread (linear) plus market impact (Almgren's square-root model: the bigger your order relative to average daily volume, the worse your fill). High-turnover strategies get punished the way they should.

**Walk-forward validation.** Instead of one backtest over the full history, `walk_forward()` trains on 504 days, tests on the next 63, then steps forward and repeats. If your strategy only works on the data it was tuned on, this exposes it.

## What's inside

Two strategies, both deliberately simple:

| Strategy | Idea | Filters |
|---|---|---|
| **Momentum** | Stocks that beat their peers over the last 12 months (skipping the most recent month) tend to keep winning | Volatility scaling, 200-day trend filter, 10% per-name cap |
| **Mean reversion** | Prices stretched 2+ standard deviations from their 60-day mean tend to snap back | RSI confirmation so you don't catch falling knives |

Both are vectorized pandas — no per-day Python loops — and both are fully tested, including tests that verify no future data leaks into signals.

## The dashboard

```bash
pip install -e ".[dev]"
streamlit run alpha_engine/dashboard/app.py
```

NAV curve, drawdown chart, rolling 63-day Sharpe, a weight heatmap so you can see what the strategy actually held, and 13 performance metrics including VaR and CVaR.

## Live paper trading

The same weight targets the backtester produces can be sent to Alpaca's paper trading API:

```bash
export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
python scripts/run_backtest.py --strategy momentum --walk-forward
```

This is where the honest accounting happens: live paper fills include real slippage and partial fills that no backtest perfectly models. The gap between backtest and paper performance is itself a measurement.

## Metrics reported

Total return, CAGR, annualized volatility, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, skew, kurtosis, 95% VaR, 95% CVaR — plus daily turnover and total cost drag, because those two numbers tell you whether the strategy survives contact with a broker.

## Project layout

```
alpha_engine/
  data/loader.py        — OHLCV fetching with parquet caching (yfinance or Alpaca)
  strategy/             — BaseStrategy interface + momentum + mean reversion
  backtest/engine.py    — the vectorized simulator + walk-forward
  backtest/costs.py     — spread + square-root impact model
  backtest/metrics.py   — the 13 metrics above
  live/paper_trader.py  — Alpaca adapter for live paper execution
  dashboard/app.py      — Streamlit UI
tests/                  — 34 tests, including lookahead-bias guards
```

## What I'd add next

- Short-borrow cost model (right now shorts are free, which is another small lie)
- Portfolio-level risk limits (sector caps, gross exposure caps)
- Intraday data support — the engine is daily-only by design, and that's worth changing carefully
