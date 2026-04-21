import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .ha_sma_dca import _compute_ha, _ha_sma


@register_strategy
class HASMA(BaseStrategy):
    """
    Heikin-Ashi reversal strategy.
    Entry: HA bar flips red→green while HA close < SMA21 AND actual price < SMA50*(1-filter_pct).
    Exit:  SMA21 crosses below SMA100 (death cross). No hard stop.
    """

    name = "ha_sma"
    hard_stop_pct = 0.0
    description = (
        "Heikin-Ashi reversal below SMA21 (3%+ below SMA50) OR SMA9 golden cross above SMA50. "
        "Exits only on SMA21/SMA100 death cross."
    )

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "SMA period for entry filter and exit", min_val=5, max_val=500),
            StrategyParam("exit_sma_period", "int", 100, "SMA period for death cross exit", min_val=2, max_val=500),
            StrategyParam("filter_sma_period", "int", 50, "SMA period for depth filter and golden cross", min_val=5, max_val=500),
            StrategyParam("filter_pct", "float", 3.0, "Min % price must be below filter SMA to enter (reversal)", min_val=0.0, max_val=30.0),
            StrategyParam("fast_sma_period", "int", 9, "Fast SMA period for golden cross re-entry", min_val=2, max_val=100),
            StrategyParam("confirm_bars", "int", 1, "Green HA bars required before entry (1=first flip, 2=second confirmed)", min_val=1, max_val=5),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        exit_sma_period = int(self.get("exit_sma_period"))
        filter_sma_period = int(self.get("filter_sma_period"))
        filter_pct = float(self.get("filter_pct")) / 100.0
        fast_sma_period = int(self.get("fast_sma_period"))
        confirm_bars = int(self.get("confirm_bars"))

        ha = _compute_ha(df)
        sma21 = _ha_sma(ha["close"], sma_period)
        exit_sma = _ha_sma(ha["close"], exit_sma_period)
        sma50 = df["Close"].rolling(filter_sma_period, min_periods=1).mean()
        sma9 = df["Close"].rolling(fast_sma_period, min_periods=1).mean()

        color = ha["color"]

        # Nth consecutive green bar after a red run (confirm_bars=1 → first flip, =2 → second green)
        green_streak = pd.concat([color.shift(i) for i in range(confirm_bars)], axis=1)
        reversal = (green_streak == "green").all(axis=1) & (color.shift(confirm_bars) == "red")

        # Entry A: HA reversal — Nth green bar below SMA21 AND price at least filter_pct% below SMA50
        ha_entry = reversal & (ha["close"] < sma21) & (df["Close"] < sma50 * (1 - filter_pct))

        # Entry B: SMA9 crosses above SMA50 (fast golden cross — momentum recovery re-entry)
        golden_cross = (sma9 > sma50) & (sma9.shift(1) <= sma50.shift(1))

        entries = ha_entry | golden_cross

        # Exit: SMA21 crosses below SMA100 (death cross)
        exits = (sma21 < exit_sma) & (sma21.shift(1) >= exit_sma.shift(1))

        return entries.fillna(False), exits.fillna(False)
