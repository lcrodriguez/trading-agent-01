import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam


def _compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi OHLC + color from raw OHLCV."""
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = ha_close.copy().astype(float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    color = pd.Series(
        ["green" if c > o else "red" for c, o in zip(ha_close, ha_open)],
        index=df.index,
    )
    return pd.DataFrame(
        {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close, "color": color},
        index=df.index,
    )


def _ha_sma(ha_close: pd.Series, period: int) -> pd.Series:
    return ha_close.rolling(period, min_periods=1).mean()


def _was_below_during_red_run_ha(
    ha_close: pd.Series,
    sma: pd.Series,
    buffer: float,
    color: pd.Series,
) -> pd.Series:
    """True on the red→green flip bar if ha_close dipped below sma*(1-buffer) during that red run."""
    result = [False] * len(ha_close)
    red_run_hit = False

    for i in range(len(ha_close)):
        c = color.iloc[i]
        prev = color.iloc[i - 1] if i > 0 else None

        if c == "red":
            thresh = sma.iloc[i]
            if not pd.isna(thresh) and ha_close.iloc[i] < thresh * (1 - buffer):
                red_run_hit = True
        elif c == "green":
            if prev == "red":
                result[i] = red_run_hit
            red_run_hit = False

    return pd.Series(result, index=ha_close.index)


def _compute_entry_stops(ha_low: pd.Series, color: pd.Series, stop_pct: float = 0.02) -> pd.Series:
    """For each green bar, stop price = min HA low of the preceding red run * (1 - stop_pct)."""
    n = len(ha_low)
    stop_prices = [float("nan")] * n
    red_run_low = float("inf")
    last_red_run_low = float("inf")

    for i in range(n):
        c = color.iloc[i]
        if c == "red":
            red_run_low = min(red_run_low, ha_low.iloc[i])
        else:
            if i > 0 and color.iloc[i - 1] == "red":
                last_red_run_low = red_run_low
                red_run_low = float("inf")
            if last_red_run_low < float("inf"):
                stop_prices[i] = last_red_run_low * (1 - stop_pct)

    return pd.Series(stop_prices, index=ha_low.index)


def _compute_three_green_entry(
    ha_close: pd.Series,
    sma: pd.Series,
    color: pd.Series,
    max_below: float = 0.0,
) -> pd.Series:
    """True on the 3rd consecutive green HA bar below SMA.
    max_below > 0: bar must be within max_below% of SMA (filters deep-crash bars)."""
    n = len(ha_close)
    result = [False] * n
    count = 0

    for i in range(n):
        hc = ha_close.iloc[i]
        sma_val = sma.iloc[i]
        if pd.isna(sma_val):
            count = 0
            continue
        below = hc < sma_val
        in_range = (max_below == 0) or (hc >= sma_val * (1 - max_below))
        if color.iloc[i] == "green" and below and in_range:
            count += 1
            if count == 3:
                result[i] = True
        else:
            count = 0

    return pd.Series(result, index=ha_close.index)


def _compute_exits_ha(
    ha_close: pd.Series,
    sma: pd.Series,
    color: pd.Series,
    entries: pd.Series,
    exit_sma: pd.Series | None = None,
    red_bars_threshold: int = 3,
    stop_below_entry: bool = False,
    exit_buffer: float = 0.0,
    stop_prices: pd.Series | None = None,
    max_stop_from_entry: float = 0.0,
    sma_zone: float = 0.0,
) -> pd.Series:
    """
    Exit fires on the first of:
      A) Normal exit: >= 1 green bar above SMA after entry, then red_bars_threshold
         consecutive red bars >= exit_buffer% below SMA.
      B) Below-SMA entry stop: ha_close < stop_price (from preceding red run low),
         capped at max_stop_from_entry% below entry if max_stop_from_entry > 0.
      C) Above-SMA entry stop (Entry C crossover): ha_close crosses back below SMA21.

    State resets only on a fresh entry.
    """
    exits = [False] * len(ha_close)
    in_trade = False
    recovered = False
    red_bars_below = 0
    green_after_entry = 0
    entry_ha_close = float("nan")
    stop_price = float("nan")
    entry_above_sma = False
    consec_lower_reds = 0
    prev_red_close = float("inf")

    def _reset():
        nonlocal in_trade, recovered, red_bars_below, green_after_entry, entry_ha_close, stop_price, entry_above_sma, consec_lower_reds, prev_red_close
        in_trade = False
        recovered = False
        red_bars_below = 0
        green_after_entry = 0
        entry_ha_close = float("nan")
        stop_price = float("nan")
        entry_above_sma = False
        consec_lower_reds = 0
        prev_red_close = float("inf")

    for i in range(len(ha_close)):
        if entries.iloc[i] and not in_trade:
            in_trade = True
            recovered = False
            red_bars_below = 0
            green_after_entry = 0
            entry_ha_close = ha_close.iloc[i]
            sma_at_entry = sma.iloc[i]
            entry_above_sma = (not pd.isna(sma_at_entry)) and (entry_ha_close >= sma_at_entry)
            raw_stop = stop_prices.iloc[i] if stop_prices is not None else float("nan")
            if max_stop_from_entry > 0 and not pd.isna(raw_stop) and not pd.isna(entry_ha_close):
                stop_price = max(raw_stop, entry_ha_close * (1 - max_stop_from_entry))
            else:
                stop_price = raw_stop
            continue

        if in_trade:
            hc = ha_close.iloc[i]
            sma_val = sma.iloc[i]
            c = color.iloc[i]

            # Below-SMA entry: 3 consecutive red bars below SMA21, each lower than the previous
            if not entry_above_sma and not pd.isna(sma_val) and not pd.isna(hc):
                if c == "red" and hc < sma_val:
                    if hc < prev_red_close:
                        consec_lower_reds += 1
                    else:
                        consec_lower_reds = 1
                    prev_red_close = hc
                    if consec_lower_reds >= 3:
                        exits[i] = True
                        _reset()
                        continue
                else:
                    consec_lower_reds = 0
                    prev_red_close = float("inf")

            # Entry C stop: SMA crossunder (below dead zone) OR price drops sma_zone% below entry
            sma_exit_threshold = sma_val * (1 - sma_zone) if sma_zone > 0 and not pd.isna(sma_val) else sma_val
            above_entry_stop = entry_ha_close * (1 - sma_zone) if sma_zone > 0 and not pd.isna(entry_ha_close) else float("nan")
            hit_sma_cross = not pd.isna(sma_val) and not pd.isna(hc) and hc < sma_exit_threshold
            hit_price_stop = not pd.isna(above_entry_stop) and not pd.isna(hc) and hc < above_entry_stop
            if entry_above_sma and (hit_sma_cross or hit_price_stop):
                exits[i] = True
                _reset()
                continue

            # Entry A/B stop: HA below precomputed stop (capped at max_stop_from_entry%)
            if (
                not entry_above_sma
                and stop_prices is not None
                and not pd.isna(stop_price)
                and not pd.isna(hc)
                and hc < stop_price
            ):
                exits[i] = True
                _reset()
                continue

            # Legacy stop (stop_below_entry=True, no stop_prices)
            if (
                stop_below_entry
                and stop_prices is None
                and not pd.isna(hc)
                and not pd.isna(entry_ha_close)
                and not pd.isna(sma_val)
                and hc < entry_ha_close
                and hc < sma_val
            ):
                exits[i] = True
                _reset()
                continue

            if exit_sma is not None:
                fast = exit_sma.iloc[i]
                threshold = fast if (not pd.isna(fast) and not pd.isna(sma_val) and fast > sma_val) else sma_val
            else:
                threshold = sma_val

            exit_threshold = threshold * (1 - exit_buffer) if exit_buffer > 0 else threshold
            above_threshold = not pd.isna(threshold) and not pd.isna(hc) and hc >= threshold

            if c == "green":
                green_after_entry += 1

            if above_threshold and green_after_entry >= 1:
                recovered = True
                red_bars_below = 0

            if c == "red" and not pd.isna(exit_threshold) and hc < exit_threshold:
                red_bars_below += 1
            elif c == "green":
                red_bars_below = 0

            if red_bars_below >= red_bars_threshold and recovered:
                exits[i] = True
                in_trade = False
                recovered = False
                red_bars_below = 0
                green_after_entry = 0
                entry_ha_close = float("nan")
                stop_price = float("nan")

    return pd.Series(exits, index=ha_close.index)


@register_strategy
class HASMADca(BaseStrategy):
    """
    Heikin-Ashi DCA strategy.
    Entry: HA bar flips red→green AND ha_close was below SMA(sma_period) during the red run.
    DCA add: same entry signal fires while in position AND price dropped >= dca_drop_pct%.
    Exit: red_bars_exit consecutive red HA bars below fast SMA (SMA exit_sma_period),
          only when fast SMA > slow SMA and after at least 1 green bar above SMA.
    Position sizing: position_pct % of available cash per buy.
    Annual cash injection handled by the engine (annual_contribution param).
    """

    name = "ha_sma_dca"
    description = (
        "Heikin-Ashi red→green DCA. Buys position_pct% of cash on each signal. "
        "Exits on consecutive red HA bars below fast SMA."
    )
    pct_dca_mode = True

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "SMA period (entry filter and exit threshold)", min_val=5, max_val=500),
            StrategyParam("sma_buffer_pct", "float", 3.0, "Min % ha_close must be below SMA to enter", min_val=0.0, max_val=20.0),
            StrategyParam("position_pct", "float", 25.0, "% of available cash to deploy per buy", min_val=5.0, max_val=100.0),
            StrategyParam("dca_drop_pct", "float", 5.0, "Min % drop below last buy price to add", min_val=0.5, max_val=50.0),
            StrategyParam("red_bars_exit", "int", 3, "Consecutive red HA bars below SMA to exit", min_val=1, max_val=20),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        buffer = float(self.get("sma_buffer_pct")) / 100.0
        red_bars_exit = int(self.get("red_bars_exit"))

        ha = _compute_ha(df)
        sma = _ha_sma(ha["close"], sma_period)

        color = ha["color"]
        turns_green = (color == "green") & (color.shift(1) == "red")
        was_below = _was_below_during_red_run_ha(ha["close"], sma, buffer, color)
        entries = turns_green & was_below

        exits = _compute_exits_ha(ha["close"], sma, color, entries, exit_sma=None, red_bars_threshold=red_bars_exit)

        return entries.fillna(False), exits.fillna(False)
