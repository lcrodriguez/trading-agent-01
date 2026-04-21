import numpy as np
import pandas as pd
import pytest

from backtesting.data.loader import load_data
from backtesting.strategies import STRATEGY_REGISTRY
from backtesting.strategies.buy_and_hold import BuyAndHold
from backtesting.strategies.renko_sma import RenkoSMA, _compute_renko, _compute_renko_colors, _renko_sma, _was_below_during_red_run
from backtesting.strategies.renko_sma_dca import RenkoSmaDCA
from backtesting.strategies.sma_crossover import SMACrossover


@pytest.fixture(scope="module")
def spy_data():
    return load_data("SPY", "2015-01-01", "2024-01-01", "1d")


def test_buy_and_hold_signals(spy_data):
    entries, exits = BuyAndHold().generate_signals(spy_data)
    assert entries.iloc[0] is True or entries.iloc[0] == True
    assert entries.sum() == 1
    assert exits.sum() == 0


def test_sma_crossover_signal_shapes(spy_data):
    entries, exits = SMACrossover({"fast_period": 50, "slow_period": 200}).generate_signals(spy_data)
    assert entries.shape == exits.shape == (len(spy_data.df),)
    assert entries.dtype == bool or entries.dtype == object


def test_sma_crossover_no_same_bar_entry_exit(spy_data):
    entries, exits = SMACrossover({"fast_period": 50, "slow_period": 200}).generate_signals(spy_data)
    assert not (entries & exits).any(), "Entry and exit should not both be True on same bar"


# ── RenkoSMA ──────────────────────────────────────────────────────────────────

def test_renko_sma_registered():
    assert "renko_sma" in STRATEGY_REGISTRY


def test_renko_sma_default_params():
    s = RenkoSMA()
    assert s.get("sma_period") == 21
    assert s.get("atr_period") == 10
    assert s.get("sma_buffer_pct") == 2.0


def test_renko_sma_param_specs():
    specs = RenkoSMA.param_specs()
    names = [p.name for p in specs]
    assert "sma_period" in names
    assert "atr_period" in names
    assert "sma_buffer_pct" in names
    assert "exit_sma_period" in names
    int_params = [p for p in specs if p.type == "int"]
    assert len(int_params) == 3


def test_renko_sma_signal_shapes(spy_data):
    entries, exits = RenkoSMA().generate_signals(spy_data)
    assert len(entries) == len(spy_data.df)
    assert len(exits) == len(spy_data.df)


def test_renko_sma_no_same_bar_entry_exit(spy_data):
    entries, exits = RenkoSMA().generate_signals(spy_data)
    assert not (entries & exits).any()


def test_renko_sma_entry_requires_price_below_sma(spy_data):
    """Entry fires when price dipped below SMA during the red run, even if flip bar is above SMA."""
    from backtesting.strategies.renko_sma import _compute_atr_weekly

    s = RenkoSMA({"sma_period": 21, "sma_buffer_pct": 2.0})
    entries, _ = s.generate_signals(spy_data)
    df = spy_data.df
    atr = _compute_atr_weekly(df, 10)
    renko_df = _compute_renko(df["High"], df["Low"], df["Close"], atr)
    renko = renko_df["color"]
    sma = _renko_sma(renko_df["brick_close"], 21)

    entry_dates = entries[entries].index
    assert len(entry_dates) > 0, "Expected at least one entry"

    for flip_date in entry_dates:
        # Find start of the preceding red run
        loc = df.index.get_loc(flip_date)
        red_run_start = loc - 1
        while red_run_start > 0 and renko.iloc[red_run_start] == "red":
            red_run_start -= 1
        red_run_start += 1  # first red bar

        red_bricks = renko_df["brick_close"].iloc[red_run_start:loc]
        red_smas = sma.iloc[red_run_start:loc]
        threshold = red_smas * (1 - 0.02)
        assert (red_bricks < threshold).any(), (
            f"Entry at {flip_date}: brick_close never dipped below SMA-2% during red run"
        )


def test_renko_color_transition():
    """Synthetic 40-bar series: verify warm-up Nones, then color changes on forced moves."""
    atr_period = 5
    n = 40
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 100.0
    close = pd.Series([base] * n, index=idx, dtype=float)
    high = close.copy()
    low = close.copy()

    # Manual ATR for a flat series → tr=0, atr→NaN until period filled
    tr = pd.Series([0.0] * n, index=idx)
    atr = tr.rolling(atr_period).mean()
    # Inject a non-zero ATR after warmup to allow brick formation
    atr.iloc[atr_period:] = 2.0

    # Force a large upward move at bar 20
    high.iloc[20] = base + 5.0

    colors = _compute_renko_colors(high, low, close, atr)

    # Warm-up bars (0 to k-1) should be None/NaN — k = first valid ATR bar = atr_period-1
    for i in range(atr_period - 1):
        assert pd.isna(colors.iloc[i]) or colors.iloc[i] is None

    # Initialization bar (k = atr_period - 1) gets 'green'
    assert colors.iloc[atr_period - 1] == "green"

    # After upward move bar, brick should still be green (continuation or unchanged)
    assert colors.iloc[20] == "green"

    # Force a downward reversal at bar 25 (low drops by 2× box below brick_bottom)
    low.iloc[25] = base - 5.0
    colors2 = _compute_renko_colors(high, low, close, atr)
    assert colors2.iloc[25] == "red"


# ── RenkoSmaDCA ───────────────────────────────────────────────────────────────

def test_renko_sma_dca_registered():
    assert "renko_sma_dca" in STRATEGY_REGISTRY


def test_renko_sma_dca_default_params():
    s = RenkoSmaDCA()
    assert s.get("max_adds") == 3
    assert s.get("dca_drop_pct") == 5.0
    assert s.dca_mode is True


def test_renko_sma_dca_signal_shapes(spy_data):
    entries, exits = RenkoSmaDCA().generate_signals(spy_data)
    assert len(entries) == len(spy_data.df)
    assert len(exits) == len(spy_data.df)


def test_renko_sma_dca_no_same_bar_entry_exit(spy_data):
    entries, exits = RenkoSmaDCA().generate_signals(spy_data)
    assert not (entries & exits).any()
