import pandas as pd

from backtesting.data.base import DataResult

from .base import BaseStrategy, StrategyParam
from . import register_strategy


@register_strategy
class BuyAndHold(BaseStrategy):
    name = "buy_and_hold"
    description = "Buy on first bar, hold forever. Benchmark baseline."

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return []

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        idx = data.df.index
        entries = pd.Series(False, index=idx)
        exits = pd.Series(False, index=idx)
        entries.iloc[0] = True
        return entries, exits
