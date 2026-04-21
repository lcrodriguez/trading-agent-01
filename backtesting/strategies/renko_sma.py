import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam


def _compute_renko(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
) -> pd.DataFrame:
    """
    Project OHLCV bars onto Renko bricks.

    Returns a DataFrame with columns:
      - 'color':       'green' / 'red' / None  (None during ATR warm-up)
      - 'brick_close': brick_top for green bricks, brick_bottom for red bricks
                       (matches the SMA source on TradingView Renko charts)
    """
    idx = close.index
    n = len(close)
    colors = [None] * n
    brick_closes = [float("nan")] * n

    first_valid = atr.first_valid_index()
    if first_valid is None:
        return pd.DataFrame({"color": colors, "brick_close": brick_closes}, index=idx)

    k = idx.get_loc(first_valid)
    box = atr.iloc[k]
    brick_bottom = close.iloc[k] - box / 2
    brick_top = close.iloc[k] + box / 2
    current_color = "green"
    colors[k] = current_color
    brick_closes[k] = brick_top  # green → brick_top

    for i in range(k + 1, n):
        h = high.iloc[i]
        l = low.iloc[i]
        new_atr = atr.iloc[i]
        if pd.isna(new_atr) or new_atr <= 0:
            colors[i] = current_color
            brick_closes[i] = brick_top if current_color == "green" else brick_bottom
            continue

        if current_color == "green" and h >= brick_top + box:
            brick_bottom = brick_top
            brick_top = brick_top + new_atr
            box = new_atr

        elif current_color == "red" and l <= brick_bottom - box:
            brick_top = brick_bottom
            brick_bottom = brick_bottom - new_atr
            box = new_atr

        elif current_color == "green" and l <= brick_bottom - box:
            brick_top = brick_bottom
            brick_bottom = brick_bottom - new_atr
            box = new_atr
            current_color = "red"

        elif current_color == "red" and h >= brick_top + box:
            brick_bottom = brick_top
            brick_top = brick_top + new_atr
            box = new_atr
            current_color = "green"

        colors[i] = current_color
        brick_closes[i] = brick_top if current_color == "green" else brick_bottom

    return pd.DataFrame({"color": colors, "brick_close": brick_closes}, index=idx)


def _compute_renko_colors(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
) -> pd.Series:
    """Thin wrapper — returns only the color Series (used by tests)."""
    return _compute_renko(high, low, close, atr)["color"]


def _was_below_during_red_run(
    close: pd.Series,
    sma: pd.Series,
    buffer: float,
    renko: pd.Series,
) -> pd.Series:
    """
    Returns True on bars where Renko flips red→green AND price dipped below
    sma*(1-buffer) at any point during that preceding red brick run.
    Fixes the lag problem: Renko confirmation often arrives after price recovers.
    """
    result = [False] * len(close)
    red_run_hit = False

    for i in range(len(close)):
        color = renko.iloc[i]
        prev_color = renko.iloc[i - 1] if i > 0 else None

        if color == "red":
            thresh = sma.iloc[i]
            if not pd.isna(thresh) and close.iloc[i] < thresh * (1 - buffer):
                red_run_hit = True
        elif color == "green":
            if prev_color == "red":
                result[i] = red_run_hit
            red_run_hit = False

    return pd.Series(result, index=close.index)


def _renko_sma(brick_close: pd.Series, period: int) -> pd.Series:
    """
    SMA of the last `period` Renko bricks, forward-filled to daily frequency.
    Matches TradingView: each brick counts once regardless of how many days it spans.
    """
    is_new_brick = brick_close.notna() & (
        brick_close.isna().shift(1).fillna(True) | (brick_close != brick_close.shift(1))
    )
    brick_events = brick_close[is_new_brick]
    sma_bricks = brick_events.rolling(period, min_periods=1).mean()
    return sma_bricks.reindex(brick_close.index).ffill()


def _compute_exits(
    brick_close: pd.Series,
    sma: pd.Series,
    renko: pd.Series,
    entries: pd.Series,
    exit_sma: pd.Series | None = None,
) -> pd.Series:
    """
    Exit fires when ALL are true:
      1. Active trade where at least 1 new green brick formed AFTER entry
         (brick_close >= exit threshold — the entry bar itself does not count)
      2. At least 2 NEW red bricks have formed while brick_close < exit threshold

    exit_sma: fast SMA used for exits (default: same as sma / slow SMA).
    When exit_sma > sma (fast above slow), exit_sma is used as the exit threshold.
    Otherwise falls back to sma.

    State resets only on a FRESH entry (not while already in trade) to match
    _simulate's behaviour of ignoring repeated entry signals.
    """
    exits = [False] * len(brick_close)
    in_trade = False
    recovered = False
    red_bricks_below = 0
    new_green_after_entry = 0

    for i in range(len(brick_close)):
        # Fresh entry only — don't reset if already in trade (mirrors _simulate)
        if entries.iloc[i] and not in_trade:
            in_trade = True
            recovered = False
            red_bricks_below = 0
            new_green_after_entry = 0
            continue  # skip tracking for the entry bar itself

        if in_trade:
            bc = brick_close.iloc[i]
            sma_val = sma.iloc[i]
            prev_bc = brick_close.iloc[i - 1] if i > 0 else bc
            is_new_brick = bc != prev_bc

            # Pick exit threshold: fast SMA when above slow SMA, else slow SMA
            if exit_sma is not None:
                fast = exit_sma.iloc[i]
                threshold = fast if (not pd.isna(fast) and not pd.isna(sma_val) and fast > sma_val) else sma_val
            else:
                threshold = sma_val

            above_threshold = not pd.isna(threshold) and not pd.isna(bc) and bc >= threshold

            if renko.iloc[i] == "green" and is_new_brick:
                new_green_after_entry += 1

            # recovered = saw at least 1 new green brick above threshold after entry
            if above_threshold and new_green_after_entry >= 1:
                recovered = True
                red_bricks_below = 0

            if renko.iloc[i] == "red" and is_new_brick and not above_threshold:
                red_bricks_below += 1
            elif renko.iloc[i] == "green":
                red_bricks_below = 0

            if red_bricks_below >= 2 and recovered:
                exits[i] = True
                in_trade = False
                recovered = False
                red_bricks_below = 0
                new_green_after_entry = 0

    return pd.Series(exits, index=brick_close.index)


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's RMA-based ATR — matches TradingView ATR exactly."""
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.copy().astype(float)
    atr.iloc[:period] = float("nan")
    atr.iloc[period - 1] = tr.iloc[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr.iloc[i] = alpha * tr.iloc[i] + (1 - alpha) * atr.iloc[i - 1]
    return atr


def _compute_atr_weekly(df: pd.DataFrame, period: int) -> pd.Series:
    """
    ATR computed on weekly-resampled OHLCV, forward-filled back to daily index.
    Matches TradingView Renko [ATR(period), 10] where the second param is weekly resolution.
    Weekly ATR produces box sizes ~2-3× daily ATR, matching TradingView's larger brick sizes.
    """
    weekly = (
        df.resample("W")
        .agg({"High": "max", "Low": "min", "Close": "last", "Open": "first"})
        .dropna()
    )
    atr_weekly = _compute_atr(weekly, period)
    return atr_weekly.reindex(df.index, method="ffill")


@register_strategy
class RenkoSMA(BaseStrategy):
    """
    Entry: price ≥ sma_buffer_pct% below SMA(sma_period) AND Renko ATR brick flips red→green.
    Exit:  bearish SMA crossunder (price falls back below SMA after crossing above it).
    Mirrors Pine Script: ta.crossover logic + Renko reversal confirmation.
    """

    name = "renko_sma"
    description = (
        "Renko ATR red→green flip below SMA. "
        "Entry on reversal confirmation; exit on bearish SMA crossunder."
    )

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "SMA period (entry + slow exit baseline)", min_val=5, max_val=500),
            StrategyParam("atr_period", "int", 10, "ATR period for Renko box size (Wilder's RMA)", min_val=5, max_val=100),
            StrategyParam("sma_buffer_pct", "float", 2.0, "Min % price must be below SMA to enter", min_val=0.0, max_val=20.0),
            StrategyParam("exit_sma_period", "int", 9, "Fast SMA period for exit (0 = use sma_period)", min_val=0, max_val=500),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        atr_period = int(self.get("atr_period"))
        buffer = float(self.get("sma_buffer_pct")) / 100.0
        exit_sma_period = int(self.get("exit_sma_period"))

        atr = _compute_atr_weekly(df, atr_period)
        renko_df = _compute_renko(df["High"], df["Low"], df["Close"], atr)
        renko = renko_df["color"]
        brick_close = renko_df["brick_close"]
        sma = _renko_sma(brick_close, sma_period)

        exit_sma = _renko_sma(brick_close, exit_sma_period) if exit_sma_period and exit_sma_period != sma_period else None

        renko_turns_green = (renko == "green") & (renko.shift(1) == "red")
        was_below = _was_below_during_red_run(brick_close, sma, buffer, renko)
        entries = renko_turns_green & was_below

        exits = _compute_exits(brick_close, sma, renko, entries, exit_sma=exit_sma)

        return entries.fillna(False), exits.fillna(False)
