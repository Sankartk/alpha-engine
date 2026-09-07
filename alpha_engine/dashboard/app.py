"""
Streamlit dashboard for alpha-engine backtest results.

Usage: streamlit run alpha_engine/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine.backtest import BacktestEngine, MetricsCalculator, TransactionCostModel
from alpha_engine.data import DataLoader, Universe
from alpha_engine.strategy import MeanReversionStrategy, MomentumStrategy

st.set_page_config(page_title="alpha-engine", layout="wide", page_icon="📈")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("alpha-engine")
    st.caption("Quantitative strategy backtester")
    st.divider()

    strategy_name = st.selectbox("Strategy", ["Momentum", "Mean Reversion"])
    universe_name = st.selectbox("Universe", ["S&P 500 Sample", "Liquid ETFs"])
    lookback_years = st.slider("Lookback (years)", 1, 5, 2)
    initial_capital = st.number_input(
        "Initial capital ($)", 100_000, 10_000_000, 1_000_000, 100_000
    )
    spread_bps = st.slider("Spread (bps)", 1, 20, 5)

    st.divider()
    run_btn = st.button("▶ Run backtest", type="primary", use_container_width=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data…")
def load_data(symbols: list[str], years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader()
    start = datetime.now() - timedelta(days=years * 365)
    raw = loader.get_many(symbols, start=start)
    prices = pd.DataFrame({s: df["close"] for s, df in raw.items()}).dropna(how="all").ffill()
    volumes = pd.DataFrame({s: df["volume"] for s, df in raw.items()}).reindex(prices.index).ffill()
    return prices, volumes


def build_strategy(name: str) -> object:
    if name == "Momentum":
        return MomentumStrategy()
    return MeanReversionStrategy()


def nav_chart(nav: pd.Series, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav.values,
        name=name, line=dict(color="#7c3aed", width=2),
    ))
    fig.update_layout(
        title="Net Asset Value", height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_tickformat="$,.0f",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    return fig


def drawdown_chart(nav: pd.Series) -> go.Figure:
    mc = MetricsCalculator()
    dd = mc.drawdown_series(nav)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        fill="tozeroy", name="Drawdown",
        line=dict(color="#f87171", width=1.5),
        fillcolor="rgba(248,113,113,0.15)",
    ))
    fig.update_layout(
        title="Drawdown (%)", height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_ticksuffix="%",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    return fig


def rolling_sharpe_chart(returns: pd.Series) -> go.Figure:
    mc = MetricsCalculator()
    rs = mc.rolling_sharpe(returns, window=63)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs.values,
        name="63d Sharpe", line=dict(color="#22d3ee", width=1.5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
    fig.add_hline(y=1, line_dash="dot", line_color="#4ade80")
    fig.update_layout(
        title="Rolling 63-day Sharpe", height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

st.markdown("## alpha-engine — backtest results")

if not run_btn:
    st.info("Configure the strategy in the sidebar and press **Run backtest**.")
    st.stop()

universe = Universe.sp500_sample() if universe_name == "S&P 500 Sample" else Universe.liquid_etfs()
prices, volumes = load_data(universe.symbols, lookback_years)

strategy = build_strategy(strategy_name)
cost_model = TransactionCostModel(spread_bps=spread_bps)
engine = BacktestEngine(cost_model=cost_model, initial_capital=initial_capital)

with st.spinner("Running backtest…"):
    result = engine.run(strategy, prices, volumes)

m = result.metrics

# KPI row
cols = st.columns(6)
kpis = [
    ("Total Return", f"{m.total_return:+.1%}", m.total_return >= 0),
    ("CAGR", f"{m.cagr:+.1%}", m.cagr >= 0),
    ("Sharpe", f"{m.sharpe:.2f}", m.sharpe >= 1),
    ("Max Drawdown", f"{m.max_drawdown:.1%}", False),
    ("Win Rate", f"{m.win_rate:.1%}", m.win_rate >= 0.5),
    ("Calmar", f"{m.calmar:.2f}", m.calmar >= 1),
]
for col, (label, value, _good) in zip(cols, kpis, strict=True):
    col.metric(label, value)

st.divider()

# Charts
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(nav_chart(result.nav, strategy_name), use_container_width=True)
    st.plotly_chart(drawdown_chart(result.nav), use_container_width=True)
with c2:
    st.plotly_chart(rolling_sharpe_chart(result.net_returns), use_container_width=True)

    st.markdown("#### Full metrics")
    rows = {
        "Total return": f"{m.total_return:+.2%}",
        "CAGR": f"{m.cagr:+.2%}",
        "Ann. volatility": f"{m.ann_vol:.2%}",
        "Sharpe ratio": f"{m.sharpe:.3f}",
        "Sortino ratio": f"{m.sortino:.3f}",
        "Max drawdown": f"{m.max_drawdown:.2%}",
        "Calmar ratio": f"{m.calmar:.3f}",
        "Win rate": f"{m.win_rate:.2%}",
        "Profit factor": f"{m.profit_factor:.2f}",
        "Skew": f"{m.skew:.2f}",
        "VaR 95%": f"{m.var_95:.2%}",
        "CVaR 95%": f"{m.cvar_95:.2%}",
        "Avg daily turnover": f"{result.turnover.mean():.2%}",
        "Total cost drag": f"{result.cost_drag.sum():.2%}",
    }
    st.dataframe(
        pd.DataFrame(rows.items(), columns=["Metric", "Value"]),
        hide_index=True,
        use_container_width=True,
    )

# Weight heatmap (last 60 days)
st.markdown("#### Recent weights (last 60 days)")
recent_w = result.weights.tail(60)
recent_w = recent_w.loc[:, (recent_w != 0).any()]
if not recent_w.empty:
    fig = go.Figure(go.Heatmap(
        z=recent_w.T.values,
        x=recent_w.index,
        y=recent_w.columns,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(tickformat=".1%"),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    st.plotly_chart(fig, use_container_width=True)
