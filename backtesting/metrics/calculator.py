import numpy as np
import pandas as pd

from backtesting.engine.runner import PortfolioStats

TRADING_DAYS = 252


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years == 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return float((returns.mean() / downside) * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    # duration in days
    in_dd = drawdown < 0
    if not in_dd.any():
        return float(max_dd * 100), 0
    dd_groups = (~in_dd).cumsum()
    max_dur = in_dd.groupby(dd_groups).sum().max()
    return float(max_dd * 100), int(max_dur)


class MetricsCalculator:
    @staticmethod
    def calculate(stats: PortfolioStats) -> dict:
        equity = stats.equity
        returns = stats.returns
        trades = stats.trades

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        cagr = _cagr(equity) * 100
        sharpe = _sharpe(returns)
        sortino = _sortino(returns)
        max_dd, max_dd_dur = _max_drawdown(equity)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        win_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0.0
        best = max((t.pnl for t in trades), default=0.0)
        worst = min((t.pnl for t in trades), default=0.0)
        avg_duration = (
            sum(t.duration_days for t in trades) / len(trades) if trades else 0
        )

        return {
            "Total Return (%)": round(total_return, 2),
            "CAGR (%)": round(cagr, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Calmar Ratio": round(calmar, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Max Drawdown Duration (days)": max_dd_dur,
            "Win Rate (%)": round(win_rate, 2),
            "Total Trades": len(trades),
            "Best Trade ($)": round(best, 2),
            "Worst Trade ($)": round(worst, 2),
            "Avg Trade Duration (days)": round(avg_duration, 1),
            "Start Value ($)": round(equity.iloc[0], 2),
            "End Value ($)": round(equity.iloc[-1], 2),
        }

    @staticmethod
    def compare(strategy_metrics: dict, benchmark_metrics: dict) -> pd.DataFrame:
        df = pd.DataFrame(
            {"Strategy": strategy_metrics, "Benchmark (Buy & Hold)": benchmark_metrics}
        )
        df.index.name = "Metric"
        return df
