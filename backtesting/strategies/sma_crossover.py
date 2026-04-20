import pandas as pd

from backtesting.data.base import DataResult

from .base import BaseStrategy, StrategyParam
from . import register_strategy


@register_strategy
class SMACrossover(BaseStrategy):
    """
    Golden Cross / Death Cross strategy.
    Entry: fast SMA crosses above slow SMA (bullish crossover).
    Exit:  fast SMA crosses below slow SMA (bearish crossunder).
    Pine Script equivalent: ta.crossover(ta.sma(close, fast), ta.sma(close, slow))
    """

    name = "sma_crossover"
    description = "SMA crossover (Golden Cross). Entry on fast>slow cross, exit on fast<slow cross."

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("fast_period", "int", 50, "Fast SMA period", min_val=5, max_val=200),
            StrategyParam("slow_period", "int", 200, "Slow SMA period", min_val=20, max_val=500),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        close = data.df["Close"]
        fast = int(self.get("fast_period"))
        slow = int(self.get("slow_period"))

        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()

        # crossover: fast crosses above slow
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        # crossunder: fast crosses below slow
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return entries.fillna(False), exits.fillna(False)
