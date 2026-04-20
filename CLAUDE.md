# CLAUDE.md — backtesting-trading

## What this project is

Python backtesting system for a long-term equity/crypto investor. Runs strategy simulations on historical OHLCV data. Two interfaces: CLI (`bt`) and Streamlit dashboard.

## Architecture

```
Data layer → Strategy layer → Engine (simulation) → Metrics → CLI / Dashboard
```

### Data layer (`backtesting/data/`)

- `base.py`: `DataResult` dataclass (symbol, asset_type, df, interval, start, end), `BaseDataLoader` ABC, `DataLoadError`
- `equity.py`: `EquityLoader` — wraps `yfinance.download()`. Auto-flattens MultiIndex columns.
- `crypto.py`: `CryptoLoader` — wraps ccxt, paginates `fetch_ohlcv()` to get full history.
- `loader.py`: `get_loader(symbol)` dispatches by `/` in symbol → crypto, else equity. `load_data()` is the public entry point.

Symbol format: `SPY` → equity, `BTC/USDT` → crypto (Binance by default).

### Strategy layer (`backtesting/strategies/`)

- `base.py`: `BaseStrategy` ABC with `generate_signals(data) → (entries, exits)` both `pd.Series[bool]`. `StrategyParam` dataclass drives CLI option injection and Streamlit form auto-generation.
- `__init__.py`: `STRATEGY_REGISTRY` dict + `@register_strategy` decorator. New strategies must be imported here.
- `buy_and_hold.py`: `BuyAndHold` — always used as benchmark. Entry on bar 0, never exits.
- `sma_crossover.py`: `SMACrossover` — Golden Cross. `fast_ma > slow_ma` crossover = entry, crossunder = exit. Maps to Pine Script `ta.crossover(ta.sma(close, fast), ta.sma(close, slow))`.

**To add a strategy**: create file, decorate class with `@register_strategy`, add import in `__init__.py`. No other registration needed.

### Engine (`backtesting/engine/`)

- `params.py`: `BacktestParams` Pydantic model. Validates start < end. Fields: symbol, start, end, interval, init_cash, fees, slippage, strategy_name, strategy_params, run_benchmark.
- `runner.py`: `BacktestRunner.run(params) → BacktestResult`. Orchestrates: load data → get strategy → generate signals → `_simulate()` → optionally run BuyAndHold as benchmark.
- `_simulate()`: event-driven loop over prices. Applies fees and slippage on entry/exit. Returns `PortfolioStats` (equity series, returns series, trade records).
- `BacktestResult`: params + data + strategy PortfolioStats + benchmark PortfolioStats + timestamp.

The engine is a simple loop (not vectorbt). Deliberate choice for clarity and pandas 2.x compatibility.

### Metrics (`backtesting/metrics/calculator.py`)

`MetricsCalculator.calculate(stats: PortfolioStats) → dict` — computes CAGR, Sharpe (252-day factor), Sortino, Calmar, Max Drawdown, Win Rate, Best/Worst trade, etc. from the equity series and trade list.

`MetricsCalculator.compare(strategy, benchmark) → pd.DataFrame` — side-by-side table for display.

### Reports (`backtesting/reports/exporter.py`)

`ResultExporter`: `to_json`, `to_csv`, `load_json`. Used by CLI `--output` flag and dashboard export tab.

### UI

- `ui/cli.py`: Click group `main` registered as `bt` entry point. Commands: `run`, `list-strategies`, `show`. `run` takes `--fast-period`/`--slow-period` as strategy params (hardcoded for now — to generalize, use `param_specs()` to inject options dynamically).
- `ui/dashboard.py`: Streamlit app. `st.session_state["result"]` caches the last `BacktestResult` across tab switches. Strategy params rendered dynamically from `param_specs()`.

## Key conventions

- All strategy signal generation uses pure pandas — no vectorbt dependency.
- `entries` and `exits` must never both be `True` on the same bar (tested).
- `DataResult.df` always has columns: Open, High, Low, Close, Volume (dropna applied).
- Fees and slippage applied multiplicatively: `fill = price * (1 + slippage)`, then `shares * (1 - fees)` on entry and `proceeds * (1 - fees)` on exit.
- CAGR uses 252 trading days per year. Crypto uses same factor for consistency.
- Metrics dict keys are human-readable strings used directly as table row labels in CLI and dashboard.

## Tests

```bash
pytest tests/ -v
```

- `test_data.py`: SPY download, column check, invalid symbol raises `DataLoadError`.
- `test_strategies.py`: signal shapes, no same-bar entry+exit, buy-and-hold has exactly 1 entry.
- `test_engine.py`: end-to-end SPY backtest, equity starts at init_cash, benchmark exists, required metric keys present, B&H positive over 9 years.

Run all tests after any change to data, strategy, or engine layer.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Entry point: `bt` CLI registered via `pyproject.toml` `[project.scripts]`.

## What NOT to do

- Do not use vectorbt — removed for pandas 2.x compatibility. Keep the custom engine.
- Do not add error handling for impossible cases inside strategies — validate at the `BacktestParams` level.
- Do not commit API keys — `.env` is gitignored.
- Do not auto-push or auto-commit.
