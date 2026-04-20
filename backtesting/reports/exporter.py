import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtesting.engine.runner import BacktestResult
from backtesting.metrics.calculator import MetricsCalculator


class ResultExporter:
    @staticmethod
    def to_json(result: BacktestResult, path: str) -> None:
        strat_metrics = MetricsCalculator.calculate(result.strategy)
        bench_metrics = MetricsCalculator.calculate(result.benchmark) if result.benchmark else {}

        payload = {
            "params": result.params.model_dump(),
            "run_timestamp": result.run_timestamp.isoformat(),
            "strategy_metrics": strat_metrics,
            "benchmark_metrics": bench_metrics,
            "trades": [asdict(t) for t in result.strategy.trades],
            "equity": result.strategy.equity.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def to_csv(result: BacktestResult, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        parent = Path(path).parent

        trades_df = pd.DataFrame([t.__dict__ for t in result.strategy.trades])
        trades_df.to_csv(parent / f"{stem}_trades.csv", index=False)

        strat_metrics = MetricsCalculator.calculate(result.strategy)
        bench_metrics = MetricsCalculator.calculate(result.benchmark) if result.benchmark else {}
        metrics_df = MetricsCalculator.compare(strat_metrics, bench_metrics)
        metrics_df.to_csv(parent / f"{stem}_metrics.csv")

    @staticmethod
    def load_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
