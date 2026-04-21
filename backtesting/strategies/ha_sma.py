import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .ha_sma_dca import _compute_ha, _ha_sma


@register_strategy
class HASMA(BaseStrategy):
    """
    Heikin-Ashi reversal strategy.
    Entry A: HA red→green flip (confirm_bars), HA close < SMA21, price >= filter_pct% below SMA50, SMA50 > SMA200 → no stop.
    Entry B: price > SMA21*1.02 AND price > SMA50 (first bar), SMA50 > SMA200 → 1% hard stop.
    Entry C: SMA21 daily enters within sma200w_proximity_pct% above SMA200 weekly → no stop (conviction).
    Exit:    SMA21 crosses below SMA100 (death cross) OR SMA9 crosses below SMA21
             when SMA21 <= SMA200 * (1 + exit_guard_pct).
    """

    name = "ha_sma"
    hard_stop_pct = 0.0
    description = (
        "HA reversal below SMA21 (no stop) OR SMA9 golden cross above SMA50 (1% stop). "
        "Exits on SMA21/SMA100 death cross, or SMA9/SMA21 cross-down guarded by SMA200."
    )

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "SMA period for entry filter and exit", min_val=5, max_val=500),
            StrategyParam("exit_sma_period", "int", 100, "SMA period for death cross exit", min_val=2, max_val=500),
            StrategyParam("filter_sma_period", "int", 50, "SMA period for depth filter and golden cross", min_val=5, max_val=500),
            StrategyParam("filter_pct", "float", 3.0, "Min % price must be below filter SMA to enter (reversal)", min_val=0.0, max_val=30.0),
            StrategyParam("fast_sma_period", "int", 9, "Fast SMA period for golden cross re-entry", min_val=2, max_val=100),
            StrategyParam("confirm_bars", "int", 2, "Green HA bars required before entry (1=first flip, 2=second confirmed)", min_val=1, max_val=5),
            StrategyParam("above_sma_stop_pct", "float", 1.0, "Hard stop % for golden cross entries (price above SMA21)", min_val=0.0, max_val=10.0),
            StrategyParam("sma200w_proximity_pct", "float", 5.0, "Buy when SMA21 daily enters within this % above SMA200 weekly (bear recovery entry)", min_val=1.0, max_val=15.0),
            StrategyParam("sma9_cross_exit", "bool", True, "Exit when SMA9 crosses below SMA21, only if SMA21 <= SMA200 * (1 + exit_guard_pct)"),
            StrategyParam("sma200_period", "int", 200, "SMA period for SMA9-cross exit guard", min_val=50, max_val=500),
            StrategyParam("exit_guard_pct", "float", 2.0, "Max % SMA21 can be above SMA200 to allow SMA9-cross exit", min_val=0.0, max_val=20.0),
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
        sma200w_proximity_pct = float(self.get("sma200w_proximity_pct")) / 100.0
        sma9_cross_exit = bool(self.get("sma9_cross_exit"))
        sma200_period = int(self.get("sma200_period"))
        exit_guard_pct = float(self.get("exit_guard_pct")) / 100.0

        ha = _compute_ha(df)
        sma21 = _ha_sma(ha["close"], sma_period)
        exit_sma = _ha_sma(ha["close"], exit_sma_period)
        sma50 = df["Close"].rolling(filter_sma_period, min_periods=1).mean()
        sma9 = df["Close"].rolling(fast_sma_period, min_periods=1).mean()
        sma200 = df["Close"].rolling(sma200_period, min_periods=1).mean()
        actual_sma21 = df["Close"].rolling(sma_period, min_periods=1).mean()
        weekly_close = df["Close"].resample("W").last()
        sma200w = weekly_close.rolling(200, min_periods=50).mean().reindex(df.index, method="ffill")

        color = ha["color"]

        # Entry A: HA reversal — Nth green bar below SMA21 AND price filter_pct% below SMA50
        green_streak = pd.concat([color.shift(i) for i in range(confirm_bars)], axis=1)
        reversal = (green_streak == "green").all(axis=1) & (color.shift(confirm_bars) == "red")
        ha_entry = reversal & (ha["close"] < sma21) & (df["Close"] < sma50 * (1 - filter_pct)) & (sma50 > sma200)

        # Entry B: price recovers above SMA21 by 2%+ AND above SMA50, only in bull market (SMA50 > SMA200)
        price_recovery = (df["Close"] > actual_sma21 * 1.02) & (df["Close"] > sma50) & (sma50 > sma200) & \
                         ~((df["Close"].shift(1) > actual_sma21.shift(1) * 1.02) & (df["Close"].shift(1) > sma50.shift(1)) & (sma50.shift(1) > sma200.shift(1)))

        # Entry C: SMA21 daily crosses into within sma200w_proximity_pct% above SMA200 weekly (bear recovery)
        sma200w_zone = sma200w * (1 + sma200w_proximity_pct)
        sma21_proximity = (actual_sma21 < sma200w_zone) & (actual_sma21.shift(1) >= sma200w_zone.shift(1))

        entries = ha_entry | price_recovery | sma21_proximity

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

            # Entry B (price recovery): stop = 1% below close at entry
            entry_stop_prices[price_recovery] = df["Close"][price_recovery] * (1 - above_sma_stop_pct)
            # Entry C (SMA200w proximity): no stop — conviction bear-recovery entry

        # Exit: SMA21 crosses below SMA100 (death cross)
        exits = (sma21 < exit_sma) & (sma21.shift(1) >= exit_sma.shift(1))

        # Optional: SMA9 crosses below SMA21 or SMA200, guarded — only fires when SMA21 <= SMA200 * (1 + exit_guard_pct)
        if sma9_cross_exit:
            guard = actual_sma21 <= sma200 * (1 + exit_guard_pct)
            sma9_cross_down_sma21 = (sma9 < actual_sma21) & (sma9.shift(1) >= actual_sma21.shift(1))
            sma9_cross_down_sma200 = (sma9 < sma200) & (sma9.shift(1) >= sma200.shift(1))
            exits = exits | ((sma9_cross_down_sma21 | sma9_cross_down_sma200) & guard)

        entry_reasons = pd.Series("", index=df.index)
        entry_reasons[ha_entry] = "ha_reversal"
        entry_reasons[price_recovery] = "price_recovery"
        entry_reasons[sma21_proximity] = "sma200w_proximity"

        exit_reasons = pd.Series("", index=df.index)
        exit_reasons[(sma21 < exit_sma) & (sma21.shift(1) >= exit_sma.shift(1))] = "death_cross"
        if sma9_cross_exit:
            exit_reasons[sma9_cross_down_sma21 & guard] = "sma9/sma21"
            exit_reasons[sma9_cross_down_sma200 & guard] = "sma9/sma200"

        return entries.fillna(False), exits.fillna(False), entry_stop_prices, entry_reasons, exit_reasons
