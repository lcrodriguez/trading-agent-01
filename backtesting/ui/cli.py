import click
from rich.console import Console
from rich.table import Table

from backtesting.engine.params import BacktestParams
from backtesting.engine.runner import BacktestRunner
from backtesting.metrics.calculator import MetricsCalculator
from backtesting.reports.exporter import ResultExporter
from backtesting.strategies import list_strategies

console = Console()


@click.group()
def main():
    """Backtesting CLI — run and analyze trading strategies."""


@main.command()
@click.option("--symbol", default="SPY", show_default=True, help="Ticker (e.g. SPY, AAPL, BTC/USDT)")
@click.option("--strategy", "strategy_name", default="sma_crossover", show_default=True)
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default="2024-01-01", show_default=True)
@click.option("--interval", default="1d", show_default=True, type=click.Choice(["1d", "1wk", "1mo"]))
@click.option("--cash", "init_cash", default=10_000.0, show_default=True, type=float)
@click.option("--fees", default=0.001, show_default=True, type=float, help="Fee fraction per trade")
@click.option("--fast-period", default=50, show_default=True, type=int, help="SMA fast period")
@click.option("--slow-period", default=200, show_default=True, type=int, help="SMA slow period")
@click.option("--output", default=None, help="Save result as JSON (e.g. outputs/result.json)")
@click.option("--no-benchmark", is_flag=True, default=False, help="Skip buy-and-hold benchmark")
def run(symbol, strategy_name, start, end, interval, init_cash, fees, fast_period, slow_period, output, no_benchmark):
    """Run a backtest and print metrics."""
    params = BacktestParams(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        init_cash=init_cash,
        fees=fees,
        strategy_name=strategy_name,
        strategy_params={"fast_period": fast_period, "slow_period": slow_period},
        run_benchmark=not no_benchmark,
    )

    console.print(f"[bold cyan]Running {strategy_name} on {symbol} ({start} → {end})[/bold cyan]")

    with console.status("Fetching data and running backtest…"):
        runner = BacktestRunner()
        result = runner.run(params)

    strat_metrics = MetricsCalculator.calculate(result.strategy)
    bench_metrics = MetricsCalculator.calculate(result.benchmark) if result.benchmark else {}

    table = Table(title=f"{strategy_name} vs Buy & Hold — {symbol}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Strategy", justify="right")
    if bench_metrics:
        table.add_column("Buy & Hold", justify="right")

    for key, val in strat_metrics.items():
        row = [key, str(val)]
        if bench_metrics:
            bval = bench_metrics.get(key, "—")
            row.append(str(bval))
        table.add_row(*row)

    console.print(table)

    if output:
        ResultExporter.to_json(result, output)
        console.print(f"[green]Result saved to {output}[/green]")


@main.command("list-strategies")
def list_strats():
    """List all available strategies."""
    table = Table(title="Available Strategies", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Parameters")

    for s in list_strategies():
        params_str = ", ".join(
            f"{p.name}={p.default}" for p in s["params"]
        ) or "none"
        table.add_row(s["name"], s["description"], params_str)

    console.print(table)


@main.command()
@click.option("--input", "path", required=True, help="Path to saved JSON result")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "csv", "json"]))
def show(path, fmt):
    """Display a saved backtest result."""
    data = ResultExporter.load_json(path)

    if fmt == "json":
        import json
        console.print_json(json.dumps(data["strategy_metrics"], indent=2))
        return

    if fmt == "csv":
        for k, v in data["strategy_metrics"].items():
            click.echo(f"{k},{v}")
        return

    table = Table(title=f"Result: {path}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Strategy", justify="right")
    if data.get("benchmark_metrics"):
        table.add_column("Buy & Hold", justify="right")

    for key, val in data["strategy_metrics"].items():
        row = [key, str(val)]
        if data.get("benchmark_metrics"):
            row.append(str(data["benchmark_metrics"].get(key, "—")))
        table.add_row(*row)

    console.print(table)
