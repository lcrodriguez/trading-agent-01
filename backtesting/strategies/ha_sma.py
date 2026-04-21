import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .ha_sma_dca import _compute_ha, _ha_sma


@register_strategy
class HASMA(BaseStrategy):
    """
    Heikin-Ashi all-in strategy.
    Entry: HA bar flips red→green while HA close is below SMA21.
    Exit:  1% hard stop on actual close price below entry fill (handled by simulator).
    """

    name = "ha_sma"
    hard_stop_pct = 0.01
    description = (
        "Heikin-Ashi reversal. Buys on any red→green HA flip below SMA21. "
        "Exits only on 1% hard stop from entry price."
    )

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "Slow SMA period (entry filter)", min_val=5, max_val=500),
            StrategyParam("exit_sma_period", "int", 100, "SMA period for exit crossunder", min_val=2, max_val=500),
            StrategyParam("consec_stop_limit", "int", 3, "Consecutive stops before pausing entries", min_val=1, max_val=10),
            StrategyParam("pause_bars", "int", 50, "Bars to sit out after hitting consecutive stop limit", min_val=1, max_val=200),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        exit_sma_period = int(self.get("exit_sma_period"))
        ha = _compute_ha(df)
        sma21 = _ha_sma(ha["close"], sma_period)
        exit_sma = _ha_sma(ha["close"], exit_sma_period)

        color = ha["color"]

        # Entry: HA flips red→green while below SMA21
        turns_green = (color == "green") & (color.shift(1) == "red")
        entries = turns_green & (ha["close"] < sma21)

        # Exit: SMA21 crosses below SMA100 (death cross)
        exits = (sma21 < exit_sma) & (sma21.shift(1) >= exit_sma.shift(1))

        return entries.fillna(False), exits.fillna(False)
