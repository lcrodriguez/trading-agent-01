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
    entry_reason: str = ""
    exit_reason: str = ""


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
    cash_rate: float = 0.0,
    monthly_contribution: float = 0.0,
    auto_invest_contributions: bool = True,
    hard_stop_pct: float = 0.0,
    opens: pd.Series | None = None,
    lows: pd.Series | None = None,
    entry_stops: pd.Series | None = None,
    entry_reasons: pd.Series | None = None,
    exit_reasons: pd.Series | None = None,
) -> PortfolioStats:
    """Vectorized single-asset portfolio simulation.

    auto_invest_contributions=True  (B&H default): contributions immediately buy shares when in position.
    auto_invest_contributions=False (all-in strategies): contributions accumulate in cash and are
        fully deployed on the next entry signal.
    """
    daily_cash_rate = (1 + cash_rate) ** (1 / 252) - 1
    cash = init_cash
    shares = 0.0
    equity = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    active_stop_price = float("nan")
    active_entry_reason = ""
    trades: list[TradeRecord] = []
    last_month: int | None = None

    for date, price in prices.items():
        if monthly_contribution > 0 and hasattr(date, "month"):
            month_key = (date.year, date.month)
            if last_month is None:
                last_month = month_key
            elif month_key != last_month:
                last_month = month_key
                if in_position and auto_invest_contributions and not pd.isna(price):
                    contrib_fill = price * (1 + slippage)
                    new_shares = (monthly_contribution / contrib_fill) * (1 - fees)
                    shares += new_shares
                else:
                    cash += monthly_contribution

        if pd.isna(price):
            equity.append(cash + shares * (price if not pd.isna(price) else 0))
            continue

        if not in_position and daily_cash_rate:
            cash *= (1 + daily_cash_rate)

        fill = price * (1 + slippage)

        if not in_position and entries.get(date, False):
            shares = (cash / fill) * (1 - fees)
            cash = 0.0
            in_position = True
            entry_price = fill
            entry_date = date
            active_entry_reason = entry_reasons.get(date, "") if entry_reasons is not None else ""
            # Per-entry stop: from entry_stops series, else uniform hard_stop_pct
            if entry_stops is not None and not pd.isna(entry_stops.get(date, float("nan"))):
                active_stop_price = entry_stops.get(date)
            elif hard_stop_pct > 0:
                active_stop_price = fill * (1 - hard_stop_pct)
            else:
                active_stop_price = float("nan")

        elif in_position and not pd.isna(active_stop_price) and (
            lows is not None and lows.get(date, price) < active_stop_price
            or lows is None and price < active_stop_price
        ):
            stop_price = active_stop_price
            if opens is not None:
                open_price = opens.get(date, price)
                fill_price = open_price if open_price < stop_price else stop_price
            else:
                fill_price = price
            fill = fill_price * (1 + slippage)
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
                    entry_reason=active_entry_reason,
                    exit_reason="stop_loss",
                )
            )
            cash = proceeds
            shares = 0.0
            in_position = False
            active_stop_price = float("nan")
            active_entry_reason = ""

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
                    entry_reason=active_entry_reason,
                    exit_reason=exit_reasons.get(date, "") if exit_reasons is not None else "",
                )
            )
            cash = proceeds
            shares = 0.0
            in_position = False
            active_stop_price = float("nan")
            active_entry_reason = ""

        equity.append(cash + shares * price)

    if in_position and shares > 0:
        last_date, last_price = list(prices.items())[-1]
        fill = last_price * (1 + slippage)
        proceeds = shares * fill * (1 - fees)
        pnl = proceeds - (shares * entry_price)
        pnl_pct = (fill / entry_price - 1) * 100
        duration = (last_date - entry_date).days if hasattr(last_date - entry_date, "days") else 0
        trades.append(
            TradeRecord(
                entry_date=str(entry_date.date()) if hasattr(entry_date, "date") else str(entry_date),
                exit_date=str(last_date.date()) + " (open)",
                entry_price=round(entry_price, 4),
                exit_price=round(fill, 4),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                duration_days=duration,
                entry_reason=active_entry_reason,
                exit_reason="open",
            )
        )

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    returns = equity_series.pct_change().fillna(0)
    return PortfolioStats(equity=equity_series, returns=returns, trades=trades, metrics={})


def _simulate_dca(
    prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
    slippage: float,
    max_adds: int,
    dca_drop_pct: float,
    cash_rate: float = 0.0,
) -> PortfolioStats:
    """DCA portfolio simulation. Each entry signal adds an equal lot if price is
    >= dca_drop_pct% below the last buy price and lots remain. Full exit on exit signal."""
    daily_cash_rate = (1 + cash_rate) ** (1 / 252) - 1
    lot_size = init_cash / max_adds
    cash = init_cash
    shares = 0.0
    total_cost = 0.0
    lots_used = 0
    in_position = False
    last_buy_price = float("inf")
    entry_date_first = None
    equity = []
    trades: list[TradeRecord] = []

    for date, price in prices.items():
        if pd.isna(price):
            equity.append(cash + shares * 0)
            continue

        if not in_position and daily_cash_rate:
            cash *= (1 + daily_cash_rate)

        fill = price * (1 + slippage)

        if entries.get(date, False):
            if not in_position:
                new_shares = (lot_size / fill) * (1 - fees)
                shares += new_shares
                cash -= lot_size
                total_cost += lot_size
                last_buy_price = fill
                lots_used = 1
                in_position = True
                entry_date_first = date

            elif (
                lots_used < max_adds
                and fill <= last_buy_price * (1 - dca_drop_pct / 100)
                and cash >= lot_size
            ):
                new_shares = (lot_size / fill) * (1 - fees)
                shares += new_shares
                cash -= lot_size
                total_cost += lot_size
                last_buy_price = fill
                lots_used += 1

        if exits.get(date, False) and in_position:
            proceeds = shares * fill * (1 - fees)
            avg_entry = total_cost / shares if shares > 0 else fill
            pnl = proceeds - total_cost
            pnl_pct = (proceeds / total_cost - 1) * 100 if total_cost > 0 else 0.0
            duration = (date - entry_date_first).days if hasattr(date - entry_date_first, "days") else 0
            trades.append(
                TradeRecord(
                    entry_date=str(entry_date_first.date()) if hasattr(entry_date_first, "date") else str(entry_date_first),
                    exit_date=str(date.date()) if hasattr(date, "date") else str(date),
                    entry_price=round(avg_entry, 4),
                    exit_price=round(fill, 4),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    duration_days=duration,
                )
            )
            cash += proceeds
            shares = 0.0
            total_cost = 0.0
            lots_used = 0
            in_position = False
            last_buy_price = float("inf")

        equity.append(cash + shares * price)

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    returns = equity_series.pct_change().fillna(0)
    return PortfolioStats(equity=equity_series, returns=returns, trades=trades, metrics={})


def _simulate_pct_dca(
    prices: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float,
    fees: float,
    slippage: float,
    position_pct: float,
    dca_drop_pct: float,
    annual_contribution: float = 0.0,
    cash_rate: float = 0.0,
) -> PortfolioStats:
    """
    Percentage-based DCA simulation.
    Each entry signal deploys position_pct% of current cash.
    DCA adds require price to be >= dca_drop_pct% below the last buy price.
    annual_contribution/12 is added to cash on the first bar of each new month.
    Idle cash (including partial positions) earns cash_rate at all times.
    """
    daily_cash_rate = (1 + cash_rate) ** (1 / 252) - 1
    monthly_contribution = annual_contribution / 12 if annual_contribution > 0 else 0.0
    cash = init_cash
    shares = 0.0
    total_cost = 0.0
    in_position = False
    last_buy_price = float("inf")
    entry_date_first = None
    equity: list[float] = []
    trades: list[TradeRecord] = []
    last_month: tuple | None = None

    for date, price in prices.items():
        if monthly_contribution > 0 and hasattr(date, "month"):
            month_key = (date.year, date.month)
            if last_month is None:
                last_month = month_key
            elif month_key != last_month:
                last_month = month_key
                cash += monthly_contribution

        if pd.isna(price):
            equity.append(cash + shares * 0)
            continue

        # Cash earns interest at all times (idle cash even while partially in position)
        if daily_cash_rate and cash > 0:
            cash *= (1 + daily_cash_rate)

        fill = price * (1 + slippage)

        if entries.get(date, False):
            if not in_position:
                buy_amount = cash * position_pct
                new_shares = (buy_amount / fill) * (1 - fees)
                shares += new_shares
                cash -= buy_amount
                total_cost += buy_amount
                last_buy_price = fill
                in_position = True
                entry_date_first = date
            elif fill <= last_buy_price * (1 - dca_drop_pct / 100) and cash > 0:
                buy_amount = cash * position_pct
                new_shares = (buy_amount / fill) * (1 - fees)
                shares += new_shares
                cash -= buy_amount
                total_cost += buy_amount
                last_buy_price = fill

        if exits.get(date, False) and in_position:
            proceeds = shares * fill * (1 - fees)
            avg_entry = total_cost / shares if shares > 0 else fill
            pnl = proceeds - total_cost
            pnl_pct = (proceeds / total_cost - 1) * 100 if total_cost > 0 else 0.0
            duration = (date - entry_date_first).days if hasattr(date - entry_date_first, "days") else 0
            trades.append(
                TradeRecord(
                    entry_date=str(entry_date_first.date()) if hasattr(entry_date_first, "date") else str(entry_date_first),
                    exit_date=str(date.date()) if hasattr(date, "date") else str(date),
                    entry_price=round(avg_entry, 4),
                    exit_price=round(fill, 4),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    duration_days=duration,
                )
            )
            cash += proceeds
            shares = 0.0
            total_cost = 0.0
            in_position = False
            last_buy_price = float("inf")

        equity.append(cash + shares * price)

    if in_position and shares > 0:
        last_date, last_price = list(prices.items())[-1]
        fill = last_price * (1 + slippage)
        proceeds = shares * fill * (1 - fees)
        avg_entry = total_cost / shares
        pnl = proceeds - total_cost
        pnl_pct = (proceeds / total_cost - 1) * 100 if total_cost > 0 else 0.0
        duration = (last_date - entry_date_first).days if hasattr(last_date - entry_date_first, "days") else 0
        trades.append(
            TradeRecord(
                entry_date=str(entry_date_first.date()) if hasattr(entry_date_first, "date") else str(entry_date_first),
                exit_date=str(last_date.date()) + " (open)",
                entry_price=round(avg_entry, 4),
                exit_price=round(fill, 4),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                duration_days=duration,
            )
        )

    equity_series = pd.Series(equity, index=prices.index, name="equity")
    returns = equity_series.pct_change().fillna(0)
    return PortfolioStats(equity=equity_series, returns=returns, trades=trades, metrics={})


class BacktestRunner:
    def run(self, params: BacktestParams) -> BacktestResult:
        data = load_data(params.symbol, params.start, params.end, params.interval)
        prices = data.df["Close"]

        strategy_cls = get_strategy(params.strategy_name)
        strategy = strategy_cls(params.strategy_params)
        sig = strategy.generate_signals(data)
        entries, exits = sig[0], sig[1]
        entry_stops = sig[2] if len(sig) > 2 else None
        entry_reasons = sig[3] if len(sig) > 3 else None
        exit_reasons = sig[4] if len(sig) > 4 else None
        opens = data.df["Open"] if "Open" in data.df.columns else None
        lows = data.df["Low"] if "Low" in data.df.columns else None

        if getattr(strategy, "pct_dca_mode", False):
            strat_stats = _simulate_pct_dca(
                prices, entries, exits,
                params.init_cash, params.fees, params.slippage,
                position_pct=float(strategy.get("position_pct")) / 100.0,
                dca_drop_pct=float(strategy.get("dca_drop_pct")),
                annual_contribution=params.annual_contribution,
                cash_rate=params.cash_rate,
            )
        elif getattr(strategy, "dca_mode", False):
            strat_stats = _simulate_dca(
                prices, entries, exits,
                params.init_cash, params.fees, params.slippage,
                max_adds=int(strategy.get("max_adds")),
                dca_drop_pct=float(strategy.get("dca_drop_pct")),
                cash_rate=params.cash_rate,
            )
        else:
            strat_stats = _simulate(
                prices, entries, exits, params.init_cash, params.fees, params.slippage,
                cash_rate=params.cash_rate,
                monthly_contribution=params.annual_contribution / 12,
                auto_invest_contributions=False,
                hard_stop_pct=getattr(strategy, "hard_stop_pct", 0.0),
                opens=opens,
                lows=lows,
                entry_stops=entry_stops,
                entry_reasons=entry_reasons,
                exit_reasons=exit_reasons,
            )

        benchmark_stats = None
        if params.run_benchmark:
            bh = BuyAndHold()
            b_entries, b_exits = bh.generate_signals(data)
            benchmark_stats = _simulate(
                prices, b_entries, b_exits, params.init_cash, params.fees, params.slippage,
                monthly_contribution=params.annual_contribution / 12,
            )

        return BacktestResult(
            params=params,
            data=data,
            strategy=strat_stats,
            benchmark=benchmark_stats,
        )
