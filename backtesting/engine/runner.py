from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from backtesting.data.base import DataResult
from backtesting.data.loader import load_data
from backtesting.strategies import get_strategy
from backtesting.strategies.buy_and_hold import BuyAndHold

from .params import BacktestParams


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    duration_days: int


@dataclass
class PortfolioStats:
    equity: pd.Series          # daily equity curve
    returns: pd.Series         # daily returns
    trades: list[TradeRecord]
    metrics: dict


@dataclass
class BacktestResult:
    params: BacktestParams
    data: DataResult
    strategy: PortfolioStats
    benchmark: PortfolioStats | None
    run_timestamp: datetime = field(default_factory=datetime.now)


def _simulate(
    prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
    slippage: float,
) -> PortfolioStats:
    """Vectorized single-asset portfolio simulation."""
    cash = init_cash
    shares = 0.0
    equity = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    trades: list[TradeRecord] = []

    for date, price in prices.items():
        if pd.isna(price):
            equity.append(cash + shares * (price if not pd.isna(price) else 0))
            continue

        fill = price * (1 + slippage)

        if not in_position and entries.get(date, False):
            shares = (cash / fill) * (1 - fees)
            cash = 0.0
            in_position = True
            entry_price = fill
            entry_date = date

        elif in_position and exits.get(date, False):
            proceeds = shares * fill * (1 - fees)
            pnl = proceeds - (shares * entry_price)
            pnl_pct = (fill / entry_price - 1) * 100
            duration = (date - entry_date).days if hasattr(date - entry_date, "days") else 0
            trades.append(
                TradeRecord(
                    entry_date=str(entry_date.date()) if hasattr(entry_date, "date") else str(entry_date),
                    exit_date=str(date.date()) if hasattr(date, "date") else str(date),
                    entry_price=round(entry_price, 4),
                    exit_price=round(fill, 4),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    duration_days=duration,
                )
            )
            cash = proceeds
            shares = 0.0
            in_position = False

        equity.append(cash + shares * price)

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    returns = equity_series.pct_change().fillna(0)
    return PortfolioStats(equity=equity_series, returns=returns, trades=trades, metrics={})


class BacktestRunner:
    def run(self, params: BacktestParams) -> BacktestResult:
        data = load_data(params.symbol, params.start, params.end, params.interval)
        prices = data.df["Close"]

        strategy_cls = get_strategy(params.strategy_name)
        strategy = strategy_cls(params.strategy_params)
        entries, exits = strategy.generate_signals(data)

        strat_stats = _simulate(prices, entries, exits, params.init_cash, params.fees, params.slippage)

        benchmark_stats = None
        if params.run_benchmark:
            bh = BuyAndHold()
            b_entries, b_exits = bh.generate_signals(data)
            benchmark_stats = _simulate(prices, b_entries, b_exits, params.init_cash, params.fees, params.slippage)

        return BacktestResult(
            params=params,
            data=data,
            strategy=strat_stats,
            benchmark=benchmark_stats,
        )
