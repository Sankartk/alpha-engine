# alpha-engine

Quantitative strategy backtester with walk-forward validation, transaction cost modeling, and live paper trading via Alpaca.

---

## What it demonstrates

| Signal | Where |
|---|---|
| Vectorised backtesting (no Python loops in hot path) | `alpha_engine/backtest/engine.py` |
| Walk-forward validation (prevents overfitting) | `alpha_engine/backtest/engine.py` |
| Transaction cost modeling (spread + square-root impact) | `alpha_engine/backtest/costs.py` |
| Two strategy implementations with no lookahead bias | `alpha_engine/strategy/` |
| Full performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR) | `alpha_engine/backtest/metrics.py` |
| Live paper trading via Alpaca API | `alpha_engine/live/paper_trader.py` |
| SOLID design: Strategy pattern, Repository pattern, Dependency Injection | throughout |

---

## Strategies

### Momentum
Cross-sectional 12-1 month momentum, inverse-vol scaled, 200-day SMA trend filter.
Long top 20%, short bottom 20%, max 10% per name.

### Mean Reversion
Z-score entry at ±2σ vs 60-day mean, RSI(14) confirmation, exit at z=0.
Max 15% per name.

---

## Quickstart

```bash
pip install -e .

# Run a backtest
python scripts/run_backtest.py --strategy momentum --universe sp500

# Launch the dashboard
streamlit run alpha_engine/dashboard/app.py

# Paper trade (requires ALPACA_API_KEY + ALPACA_SECRET_KEY)
python scripts/run_paper.py --strategy momentum
```

## Architecture

```
DataLoader (cache-aside, yfinance/Alpaca)
     │
     ▼
Strategy.generate_signals()  →  weight matrix (dates × symbols)
     │
     ▼
BacktestEngine
  ├── shift weights by 1 day  (no lookahead)
  ├── compute gross returns
  ├── TransactionCostModel    (spread + square-root impact)
  └── MetricsCalculator       (Sharpe, Sortino, Calmar, VaR, CVaR)
     │
     ▼
BacktestResult
     │
     ├── Streamlit dashboard
     └── PaperTrader (Alpaca) — live execution
```

## Design principles

- **No lookahead bias** — all signals use only past data at time *t*; weights are shifted by 1 day before computing returns.
- **Walk-forward validation** — `engine.walk_forward()` trains on 504 days, tests on 63, steps forward 63. Never tests on training data.
- **Cost realism** — spread cost + Almgren square-root market impact. Ignoring costs inflates Sharpe by 0.3–0.8 on high-turnover strategies.
- **SOLID** — `BaseStrategy` / `BaseDataProvider` are abstract interfaces. `BacktestEngine` receives them via constructor injection. New strategies require zero changes to the engine.
