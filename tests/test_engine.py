import pytest

from backtesting.engine.params import BacktestParams
from backtesting.engine.runner import BacktestRunner
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
