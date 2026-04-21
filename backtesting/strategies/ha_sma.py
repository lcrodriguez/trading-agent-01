import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .ha_sma_dca import _compute_ha, _ha_sma


@register_strategy
class HASMA(BaseStrategy):
    """
    Heikin-Ashi reversal strategy.
    Entry A: HA red→green flip, HA close < SMA21, price >= filter_pct% below SMA50 → no stop.
    Entry B: SMA9 golden cross above SMA50 → 1% hard stop (above_sma_stop_pct).
    Exit:    SMA21 crosses below SMA100 (death cross).
    """

    name = "ha_sma"
    hard_stop_pct = 0.0
    description = (
        "HA reversal below SMA21 (no stop) OR SMA9 golden cross above SMA50 (1% stop). "
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
            StrategyParam("above_sma_stop_pct", "float", 1.0, "Hard stop % for golden cross entries (price above SMA21)", min_val=0.0, max_val=10.0),
            StrategyParam("ha_trend_exit", "bool", False, "Exit when price crosses below SMA130 by ≥4% (significant breakdown protection)"),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        exit_sma_period = int(self.get("exit_sma_period"))
        filter_sma_period = int(self.get("filter_sma_period"))
        filter_pct = float(self.get("filter_pct")) / 100.0
        fast_sma_period = int(self.get("fast_sma_period"))
        confirm_bars = int(self.get("confirm_bars"))
        above_sma_stop_pct = float(self.get("above_sma_stop_pct")) / 100.0
        ha_trend_exit = bool(self.get("ha_trend_exit"))

        ha = _compute_ha(df)
        sma21 = _ha_sma(ha["close"], sma_period)
        exit_sma = _ha_sma(ha["close"], exit_sma_period)
        sma50 = df["Close"].rolling(filter_sma_period, min_periods=1).mean()
        sma9 = df["Close"].rolling(fast_sma_period, min_periods=1).mean()
        actual_sma21 = df["Close"].rolling(sma_period, min_periods=1).mean()

        color = ha["color"]

        # Entry A: HA reversal — Nth green bar below SMA21 AND price filter_pct% below SMA50
        green_streak = pd.concat([color.shift(i) for i in range(confirm_bars)], axis=1)
        reversal = (green_streak == "green").all(axis=1) & (color.shift(confirm_bars) == "red")
        ha_entry = reversal & (ha["close"] < sma21) & (df["Close"] < sma50 * (1 - filter_pct))

        # Entry B: SMA9 crosses above SMA50 (momentum recovery re-entry)
        golden_cross = (sma9 > sma50) & (sma9.shift(1) <= sma50.shift(1))

        entries = ha_entry | golden_cross

        # Per-entry stop prices
        entry_stop_prices = pd.Series(float("nan"), index=df.index)

        # HA reversal entries (below SMA21): stop = 1% below the lowest actual Low of the preceding red run
        red_run_low = pd.Series(float("nan"), index=df.index)
        current_low = float("inf")
        for i in range(len(color)):
            c = color.iloc[i]
            if c == "red":
                current_low = min(current_low, df["Low"].iloc[i])
            elif c == "green":
                if i > 0 and color.iloc[i - 1] == "red" and current_low < float("inf"):
                    red_run_low.iloc[i] = current_low
                    current_low = float("inf")

        if above_sma_stop_pct > 0:
            ha_stop = red_run_low * (1 - above_sma_stop_pct)
            entry_stop_prices[ha_entry] = ha_stop[ha_entry]

            # Golden cross entries where price > SMA21: stop = 1% below close at entry
            needs_stop = golden_cross & (df["Close"] > actual_sma21)
            entry_stop_prices[needs_stop] = df["Close"][needs_stop] * (1 - above_sma_stop_pct)

        # Exit: SMA21 crosses below SMA100 (death cross)
        exits = (sma21 < exit_sma) & (sma21.shift(1) >= exit_sma.shift(1))

        # Optional: price crosses below SMA130 by ≥4% in a single bar (significant breakdown)
        if ha_trend_exit:
            sma130 = df["Close"].rolling(130, min_periods=1).mean()
            cross_below = (df["Close"] < sma130 * (1 - 0.04)) & (df["Close"].shift(1) >= sma130.shift(1))
            exits = exits | cross_below

        return entries.fillna(False), exits.fillna(False), entry_stop_prices
