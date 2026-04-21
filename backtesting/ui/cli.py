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


@main.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--symbol", default="SPY", show_default=True, help="Ticker (e.g. SPY, AAPL, BTC/USDT)")
@click.option("--strategy", "strategy_name", default="sma_crossover", show_default=True)
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default="2024-01-01", show_default=True)
@click.option("--interval", default="1d", show_default=True, type=click.Choice(["1d", "1wk", "1mo"]))
@click.option("--cash", "init_cash", default=10_000.0, show_default=True, type=float)
@click.option("--fees", default=0.001, show_default=True, type=float, help="Fee fraction per trade")
@click.option("--output", default=None, help="Save result as JSON (e.g. outputs/result.json)")
@click.option("--no-benchmark", is_flag=True, default=False, help="Skip buy-and-hold benchmark")
@click.option("--trades", is_flag=True, default=False, help="Print individual trade details")
@click.option("--cash-rate", default=0.04, show_default=True, type=float, help="Annual return on idle cash (e.g. 0.04 = 4%)")
@click.argument("extra_params", nargs=-1, type=click.UNPROCESSED)
def run(symbol, strategy_name, start, end, interval, init_cash, fees, output, no_benchmark, trades, cash_rate, extra_params):
    """Run a backtest and print metrics.

    Pass strategy-specific params as --key value pairs, e.g.:
      bt run --strategy renko_sma --sma-period 9 --atr-period 14
      bt run --strategy sma_crossover --fast-period 50 --slow-period 200
    """
    # Parse extra --key value pairs into strategy_params dict
    strategy_params: dict = {}
    it = iter(extra_params)
    for token in it:
        if token.startswith("--"):
            key = token.lstrip("-").replace("-", "_")
            try:
                raw = next(it)
                # Auto-cast: int → float → str
                try:
                    strategy_params[key] = int(raw)
                except ValueError:
                    try:
                        strategy_params[key] = float(raw)
                    except ValueError:
                        strategy_params[key] = raw
            except StopIteration:
                pass

    params = BacktestParams(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        init_cash=init_cash,
        fees=fees,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        run_benchmark=not no_benchmark,
        cash_rate=cash_rate,
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

    if trades:
        trade_table = Table(title="Trades", show_lines=True)
        trade_table.add_column("#", justify="right", style="dim")
        trade_table.add_column("Entry Date", min_width=12)
        trade_table.add_column("Exit Date", min_width=14)
        trade_table.add_column("Entry $", justify="right")
        trade_table.add_column("Exit $", justify="right")
        trade_table.add_column("P&L $", justify="right")
        trade_table.add_column("P&L %", justify="right")
        trade_table.add_column("Days", justify="right")
        trade_table.add_column("Entry Reason", min_width=16)
        trade_table.add_column("Exit Reason", min_width=14)
        from datetime import date as date_type
        trades_list = result.strategy.trades
        for i, t in enumerate(trades_list, 1):
            pnl_color = "green" if t.pnl >= 0 else "red"
            trade_table.add_row(
                str(i),
                t.entry_date,
                t.exit_date,
                f"{t.entry_price:,.4f}",
                f"{t.exit_price:,.4f}",
                f"[{pnl_color}]{t.pnl:+,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{t.pnl_pct:+.2f}%[/{pnl_color}]",
                str(t.duration_days),
                f"[dim]{t.entry_reason}[/dim]" if t.entry_reason else "",
                f"[dim]{t.exit_reason}[/dim]" if t.exit_reason else "",
            )
            # Show days out of market between this exit and next entry
            if i < len(trades_list):
                next_t = trades_list[i]
                try:
                    exit_str = t.exit_date.replace(" (open)", "")
                    d1 = date_type.fromisoformat(exit_str)
                    d2 = date_type.fromisoformat(next_t.entry_date)
                    gap = (d2 - d1).days
                    cash_earned_pct = ((1 + cash_rate) ** (gap / 365) - 1) * 100
                    trade_table.add_row(
                        "", f"[dim]↕ {gap}d out[/dim]", "",
                        "", "", f"[dim]+{cash_earned_pct:.2f}%[/dim]", "[dim]cash[/dim]", "", "", "",
                        end_section=True,
                    )
                except Exception:
                    pass
        console.print(trade_table)

    if output:
        ResultExporter.to_json(result, output)
        console.print(f"[green]Result saved to {output}[/green]")


@main.command()
@click.option("--input", "paths", multiple=True, required=True, help="Saved JSON results to compare (repeat for each)")
def compare(paths):
    """Compare multiple saved backtest results side by side.

    Example:
      bt run --strategy sma_crossover --output outputs/sma.json
      bt run --strategy renko_sma --output outputs/renko.json
      bt compare --input outputs/sma.json --input outputs/renko.json
    """
    results = [ResultExporter.load_json(p) for p in paths]

    all_keys = list(results[0]["strategy_metrics"].keys())

    title = "Strategy Comparison"
    table = Table(title=title, show_lines=True)
    table.add_column("Metric", style="bold")
    for r in results:
        label = f"{r['params']['strategy_name']}\n{r['params']['symbol']} {r['params']['start'][:4]}–{r['params']['end'][:4]}"
        table.add_column(label, justify="right")

    for key in all_keys:
        row = [key]
        vals = [r["strategy_metrics"].get(key) for r in results]

        for i, v in enumerate(vals):
            if v is None:
                row.append("—")
                continue
            cell = str(v)
            # Highlight best value (green) for numeric metrics
            if isinstance(v, (int, float)) and len(vals) > 1:
                numeric = [x for x in vals if isinstance(x, (int, float))]
                if len(numeric) == len(vals):
                    # Higher is better for most metrics except drawdown/duration
                    lower_is_better = "Drawdown" in key or "Duration" in key
                    best = min(numeric) if lower_is_better else max(numeric)
                    if v == best:
                        cell = f"[bold green]{v}[/bold green]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


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
