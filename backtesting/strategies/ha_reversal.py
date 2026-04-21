import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .ha_sma_dca import _compute_ha


@register_strategy
class HAReversal(BaseStrategy):
    """
    Heikin-Ashi reversal strategy (simple).
    Entry A: 2 consecutive green HA bars after a red run, price < SMA200 → 2% hard stop.
    Entry B: price crosses up through SMA130 → 2% hard stop.
    Exit:    cross down SMA130 confirmed by 2 consecutive closes below it.
    """

    name = "ha_reversal"
    hard_stop_pct = 0.0
    description = (
        "HA reversal below SMA200 (2 green bars) or SMA130 cross-up re-entry. "
        "Exits on SMA130 cross-down confirmed by 2 consecutive closes. 2% hard stop."
    )

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("exit_sma_period", "int", 130, "SMA period for price-cross exit and re-entry", min_val=5, max_val=500),
            StrategyParam("entry_sma_period", "int", 200, "SMA period for entry filter (price must be below)", min_val=10, max_val=500),
            StrategyParam("confirm_bars", "int", 2, "Consecutive green HA bars required before entry", min_val=1, max_val=5),
            StrategyParam("stop_pct", "float", 2.0, "Hard stop % below entry price", min_val=0.1, max_val=20.0),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series, pd.Series]:
        df = data.df
        exit_sma_period = int(self.get("exit_sma_period"))
        entry_sma_period = int(self.get("entry_sma_period"))
        confirm_bars = int(self.get("confirm_bars"))
        stop_pct = float(self.get("stop_pct")) / 100.0

        ha = _compute_ha(df)
        color = ha["color"]

        sma130 = df["Close"].rolling(exit_sma_period, min_periods=1).mean()
        sma200 = df["Close"].rolling(entry_sma_period, min_periods=1).mean()

        # Entry A: confirm_bars consecutive green HA bars after a red, price < SMA200
        green_streak = pd.concat([color.shift(i) for i in range(confirm_bars)], axis=1)
        reversal = (green_streak == "green").all(axis=1) & (color.shift(confirm_bars) == "red")
        ha_entry = reversal & (df["Close"] < sma200)

        # Entry B: price crosses up through SMA130 (momentum recovery)
        cross_above_sma130 = (df["Close"] > sma130) & (df["Close"].shift(1) <= sma130.shift(1))

        entries = ha_entry | cross_above_sma130

        # Exit: 2 consecutive closes below SMA130, where the first was the crossing bar
        exits = (
            (df["Close"] < sma130) &
            (df["Close"].shift(1) < sma130.shift(1)) &
            (df["Close"].shift(2) >= sma130.shift(2))
        )

        # Stop: 2% below close at entry
        entry_stop_prices = pd.Series(float("nan"), index=df.index)
        entry_stop_prices[entries] = df["Close"][entries] * (1 - stop_pct)

        entry_reasons = pd.Series("", index=df.index)
        entry_reasons[ha_entry] = "ha_reversal"
        entry_reasons[cross_above_sma130] = "sma130_cross_up"

        exit_reasons = pd.Series("", index=df.index)
        exit_reasons[exits] = "2close<sma130"

        return entries.fillna(False), exits.fillna(False), entry_stop_prices, entry_reasons, exit_reasons
