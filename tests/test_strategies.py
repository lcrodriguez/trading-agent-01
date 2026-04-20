import pandas as pd
import pytest

from backtesting.data.loader import load_data
from backtesting.strategies.buy_and_hold import BuyAndHold
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
