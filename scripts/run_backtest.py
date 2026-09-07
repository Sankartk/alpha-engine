"""Run a backtest from the command line."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from alpha_engine.backtest import BacktestEngine, TransactionCostModel
from alpha_engine.data import DataLoader, Universe
from alpha_engine.strategy import MeanReversionStrategy, MomentumStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="alpha-engine backtest runner")
    parser.add_argument("--strategy", choices=["momentum", "mean_reversion"], default="momentum")
    parser.add_argument("--universe", choices=["sp500", "etfs"], default="sp500")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    universe = Universe.sp500_sample() if args.universe == "sp500" else Universe.liquid_etfs()
    strategy = MomentumStrategy() if args.strategy == "momentum" else MeanReversionStrategy()

    print(f"Loading {len(universe)} symbols ({args.years}y)…")
    loader = DataLoader()
    start = datetime.now() - timedelta(days=args.years * 365)
    raw = loader.get_many(universe.symbols, start=start)

    import pandas as pd
    prices = pd.DataFrame({s: df["close"] for s, df in raw.items()}).ffill()
    volumes = pd.DataFrame({s: df["volume"] for s, df in raw.items()}).ffill()

    engine = BacktestEngine(
        cost_model=TransactionCostModel(),
        initial_capital=args.capital,
    )

    if args.walk_forward:
        print("Running walk-forward validation…")
        results = engine.walk_forward(strategy, prices, volumes)
        print(f"\n{len(results)} folds completed:")
        for r in results:
            m = r.metrics
            print(
                f"  fold {r.metadata['fold']:02d}  "
                f"{r.start:%Y-%m-%d} → {r.end:%Y-%m-%d}  "
                f"sharpe={m.sharpe:.2f}  ret={m.total_return:+.1%}"
            )
    else:
        print("Running full-period backtest…")
        result = engine.run(strategy, prices, volumes)
        print()
        print(result.summary())


if __name__ == "__main__":
    main()
