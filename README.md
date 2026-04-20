# backtesting-trading

Python backtesting system for long-term investors. Run strategy simulations on historical stock and crypto data from the CLI or a Streamlit dashboard.

## Stack

| Layer | Library |
|---|---|
| Data (equities) | yfinance |
| Data (crypto) | ccxt / Binance |
| Indicators | pandas-ta |
| Engine | Custom vectorized (pandas) |
| Metrics | Custom + quantstats |
| CLI | Click + Rich |
| Dashboard | Streamlit + Plotly |

---

## Setup

```bash
git clone git@github.com:lcrodriguez/trading-agent-01.git
cd trading-agent-01

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## CLI

### Run a backtest

```bash
bt run --symbol SPY --strategy sma_crossover --start 2010-01-01 --end 2024-01-01
```

```bash
# Crypto
bt run --symbol BTC/USDT --strategy sma_crossover --start 2018-01-01 --end 2024-01-01

# Custom SMA periods
bt run --symbol AAPL --strategy sma_crossover --fast-period 20 --slow-period 50

# Save result
bt run --symbol SPY --strategy sma_crossover --output outputs/spy_sma.json
```

### List strategies

```bash
bt list-strategies
```

### Show a saved result

```bash
bt show --input outputs/spy_sma.json
bt show --input outputs/spy_sma.json --format csv
bt show --input outputs/spy_sma.json --format json
```

---

## Streamlit Dashboard

```bash
streamlit run backtesting/ui/dashboard.py
```

Opens at `http://localhost:8501`. Configure symbol, strategy, date range, and capital in the sidebar, then click **Run Backtest**.

**Tabs:**
- **Overview** — equity curve vs buy-and-hold, 4 key metric cards
- **Trades** — trade log table, PnL histogram
- **Metrics** — full side-by-side comparison + drawdown chart + monthly returns heatmap
- **Export** — download trades CSV, metrics CSV, or full JSON

---

## Strategies

| Name | Description | Default Params |
|---|---|---|
| `sma_crossover` | Golden Cross (fast SMA crosses above slow SMA) | fast=50, slow=200 |
| `buy_and_hold` | Buy first bar, hold forever. Used as benchmark | — |

### Add a strategy

1. Create `backtesting/strategies/my_strategy.py`:

```python
import pandas as pd
from backtesting.data.base import DataResult
from .base import BaseStrategy, StrategyParam
from . import register_strategy

@register_strategy
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "One-line description shown in CLI and UI."

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("period", "int", 14, "RSI period", min_val=2, max_val=100),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        close = data.df["Close"]
        entries = pd.Series(False, index=close.index)
        exits = pd.Series(False, index=close.index)
        return entries, exits
```

2. Import it in `backtesting/strategies/__init__.py`:

```python
from . import buy_and_hold, sma_crossover, my_strategy  # noqa: F401
```

The strategy auto-registers and appears in `bt list-strategies` and the dashboard dropdown.

---

## Metrics

| Metric | Description |
|---|---|
| Total Return (%) | (end / start - 1) × 100 |
| CAGR (%) | Compound annual growth rate |
| Sharpe Ratio | Annualized risk-adjusted return (252 trading days) |
| Sortino Ratio | Sharpe using only downside deviation |
| Calmar Ratio | CAGR / Max Drawdown |
| Max Drawdown (%) | Largest peak-to-trough decline |
| Max Drawdown Duration (days) | Longest time underwater |
| Win Rate (%) | % of trades with positive PnL |
| Total Trades | Number of round-trip trades |
| Best / Worst Trade ($) | PnL of best and worst trade |
| Avg Trade Duration (days) | Average holding period |

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
backtesting/
├── data/          # Data loaders (yfinance, ccxt) + DataResult
├── strategies/    # BaseStrategy + registered strategies
├── engine/        # BacktestRunner, BacktestParams, simulation
├── metrics/       # MetricsCalculator
├── reports/       # ResultExporter (JSON, CSV)
└── ui/
    ├── cli.py        # Click CLI (bt command)
    └── dashboard.py  # Streamlit app
tests/
outputs/           # Saved results (gitignored)
```

---

## Symbol Format

| Asset | Format | Example |
|---|---|---|
| Equity / ETF | Ticker | `SPY`, `AAPL`, `QQQ` |
| Crypto | `BASE/QUOTE` | `BTC/USDT`, `ETH/USDT` |

The loader auto-detects asset type from the symbol format.
