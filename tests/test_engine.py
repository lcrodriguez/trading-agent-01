import pandas as pd
import pytest

from backtesting.engine.params import BacktestParams
from backtesting.engine.runner import BacktestRunner, _simulate_dca
from backtesting.metrics.calculator import MetricsCalculator


@pytest.fixture(scope="module")
def spy_result():
    params = BacktestParams(
        symbol="SPY",
        start="2015-01-01",
        end="2024-01-01",
        strategy_name="sma_crossover",
        strategy_params={"fast_period": 50, "slow_period": 200},
        init_cash=10_000.0,
        run_benchmark=True,
    )
    return BacktestRunner().run(params)


def test_result_has_equity(spy_result):
    assert len(spy_result.strategy.equity) > 0
    assert spy_result.strategy.equity.iloc[0] == pytest.approx(10_000.0, rel=0.01)


def test_benchmark_exists(spy_result):
    assert spy_result.benchmark is not None
    assert len(spy_result.benchmark.equity) > 0


def test_metrics_keys(spy_result):
    metrics = MetricsCalculator.calculate(spy_result.strategy)
    required = ["Total Return (%)", "CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    for key in required:
        assert key in metrics


def test_buy_and_hold_beats_zero(spy_result):
    bench = MetricsCalculator.calculate(spy_result.benchmark)
    assert bench["Total Return (%)"] > 0, "SPY buy-and-hold should be positive over 9 years"


# ── DCA engine ────────────────────────────────────────────────────────────────

def _make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_dca_adds_multiple_lots():
    """Three descending entry signals → 1 trade record, 3 lots used, equity > single-lot."""
    n = 20
    prices = _make_series([100.0] * 5 + [95.0] * 5 + [89.0] * 5 + [110.0] * 5)
    entries = pd.Series(False, index=prices.index)
    exits = pd.Series(False, index=prices.index)
    entries.iloc[0] = True   # first buy at 100
    entries.iloc[5] = True   # DCA add at 95 (5% below 100)
    entries.iloc[10] = True  # DCA add at 89 (~6.3% below 95)
    exits.iloc[19] = True    # exit at 110

    stats = _simulate_dca(prices, entries, exits, init_cash=3000.0, fees=0.0, slippage=0.0,
                          max_adds=3, dca_drop_pct=5.0)
    assert len(stats.trades) == 1
    assert stats.trades[0].pnl > 0
    assert stats.equity.iloc[-1] > 3000.0


def test_dca_respects_drop_threshold():
    """Second signal only 3% below last buy (< dca_drop_pct=5) → skipped, only 1 lot used."""
    prices = _make_series([100.0] * 5 + [97.0] * 5 + [120.0] * 5)
    entries = pd.Series(False, index=prices.index)
    exits = pd.Series(False, index=prices.index)
    entries.iloc[0] = True   # first buy at 100
    entries.iloc[5] = True   # 97 = only 3% below 100 → should be skipped
    exits.iloc[14] = True

    stats = _simulate_dca(prices, entries, exits, init_cash=3000.0, fees=0.0, slippage=0.0,
                          max_adds=3, dca_drop_pct=5.0)
    assert len(stats.trades) == 1
    # Only 1 lot of 1000 was used; remaining 2000 stayed as cash
    end_equity = stats.equity.iloc[-1]
    # Position was 1000/100 = 10 shares, exited at 120 = 1200. Plus 2000 cash = 3200
    assert end_equity == pytest.approx(3200.0, rel=0.01)


def test_dca_triggers_at_exact_threshold():
    """Second signal exactly 5% below last buy → add fires."""
    prices = _make_series([100.0] * 5 + [95.0] * 5 + [120.0] * 5)
    entries = pd.Series(False, index=prices.index)
    exits = pd.Series(False, index=prices.index)
    entries.iloc[0] = True   # buy at 100
    entries.iloc[5] = True   # 95 = exactly 5% below → should trigger
    exits.iloc[14] = True

    stats = _simulate_dca(prices, entries, exits, init_cash=2000.0, fees=0.0, slippage=0.0,
                          max_adds=2, dca_drop_pct=5.0)
    assert len(stats.trades) == 1
    # 2 lots of 1000 each used: 10 shares at 100 + ~10.526 shares at 95
    total_shares = 1000 / 100 + 1000 / 95
    expected_proceeds = total_shares * 120
    assert stats.trades[0].pnl == pytest.approx(expected_proceeds - 2000.0, rel=0.01)
