# backtesting-trading

Python backtesting system for long-term investors. Run strategy simulations on historical stock and crypto data from the CLI or a Streamlit dashboard.

## Stack

| Layer | Library |
|---|---|
| Data (equities) | yfinance |
| Data (crypto) | ccxt / Binance |
| Engine | Custom event-driven (pandas) |
| Metrics | Custom |
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
bt run --symbol SPY --strategy ha_sma --start 2010-01-01 --end 2025-01-01
```

```bash
# Show individual trades
bt run --symbol SPY --strategy ha_sma --trades

# Crypto
bt run --symbol BTC/USDT --strategy sma_crossover --start 2018-01-01 --end 2024-01-01

# Custom strategy params
bt run --symbol SPY --strategy ha_sma --sma-period 21 --exit-sma-period 100

# Save result
bt run --symbol SPY --strategy ha_sma --output outputs/spy_ha_sma.json
```

### Compare saved results

```bash
bt run --symbol SPY --strategy ha_sma --output outputs/ha.json
bt run --symbol SPY --strategy sma_crossover --output outputs/sma.json
bt compare --input outputs/ha.json --input outputs/sma.json
```

### List strategies

```bash
bt list-strategies
```

### Show a saved result

```bash
bt show --input outputs/spy_ha_sma.json
bt show --input outputs/spy_ha_sma.json --format csv
bt show --input outputs/spy_ha_sma.json --format json
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

| Name | Type | Description |
|---|---|---|
| `ha_sma` | All-in | **Preferred.** Heikin-Ashi reversal + SMA9 golden cross with adaptive stops |
| `ha_sma_dca` | % DCA | Heikin-Ashi DCA — adds on each red→green flip below SMA |
| `renko_sma` | All-in | Renko ATR reversal with SMA crossunder exit |
| `renko_sma_dca` | DCA | Renko reversal with equal-lot DCA adds |
| `sma_crossover` | All-in | Golden Cross (fast SMA crosses above slow SMA) |
| `buy_and_hold` | — | Buy first bar, hold forever. Used as benchmark |

---

## Preferred Strategy: `ha_sma`

### Concept

Two complementary entry signals with adaptive risk management:

1. **HA Reversal** — buys when Heikin-Ashi bars flip red→green after a meaningful pullback (≥3% below SMA50). Stop is placed 1% below the lowest low of the preceding red run — the natural support level from the correction. Exit only on the SMA21/SMA100 death cross.

2. **Golden Cross re-entry** — when SMA9 crosses above SMA50 (momentum recovery confirmed), re-enters even if the HA reversal conditions aren't met. Stop is 1% below the entry close price. This catches recovery legs where price never pulled back far enough for the reversal signal but the trend has clearly turned.

The idea: if everything goes right, the HA entry rides up to the golden cross and beyond. If not, the stop at the previous red-run low limits the damage.

### Entry Rules

**Entry A — HA Reversal (below SMA21):**
- Heikin-Ashi bars flip red → green (or `confirm_bars` consecutive greens)
- HA close < SMA21
- Actual price ≥ `filter_pct`% below SMA50 (default 3%)
- **Stop**: 1% below the lowest actual Low of the preceding red run

**Entry B — Golden Cross (above SMA21):**
- SMA9 crosses above SMA50
- **Stop**: 1% below the entry close price

**Exit (both entries):**
- SMA21 crosses below SMA100 (death cross — macro bear signal)
- Or stop is hit (described above per entry type)

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `sma_period` | 21 | SMA period for entry filter and death cross exit |
| `exit_sma_period` | 100 | Slow SMA for death cross exit |
| `filter_sma_period` | 50 | SMA period for depth filter and golden cross |
| `filter_pct` | 3.0 | Min % price must be below SMA50 to enter (reversal) |
| `fast_sma_period` | 9 | Fast SMA for golden cross re-entry |
| `confirm_bars` | 1 | Green HA bars required before entry (1=first flip, 2=second) |
| `above_sma_stop_pct` | 1.0 | Hard stop % for both entry types |

### Backtest Results (SPY, 2020–2025)

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total Return | ~88% | ~71% |
| CAGR | ~12.7% | ~10.7% |
| Sharpe | 0.77 | 0.58 |
| Max Drawdown | -25.8% | -33.7% |
| Trades | 13 | 1 |
| Avg Hold | 139 days | — |

Key trades:
- **2020-03-24 → 2022-02-03**: +88.49% (629 days) — COVID crash bottom, rode full recovery
- **2022-10-14 → 2023-10-05**: +20.57% — 2022 bear market bottom
- **2023-10-31 → 2025-03-12**: +35.90% (498 days) — entire 2024 bull run

### Usage

```bash
# Default run
bt run --symbol SPY --strategy ha_sma --start 2020-01-01 --end 2025-01-01 --trades

# Wait for second confirmed green bar before entry
bt run --symbol SPY --strategy ha_sma --trades --confirm-bars 2

# Custom filter depth
bt run --symbol SPY --strategy ha_sma --trades --filter-pct 5

# Idle cash earns 4% annualized (default) — change with:
bt run --symbol SPY --strategy ha_sma --cash-rate 0.05
```

---

## Strategy Design Findings

### What works

**Heikin-Ashi for reversals** — HA bars smooth noise, making red→green flips at pullback lows more reliable than raw candles. Smoothing means entries fire slightly after the exact bottom, but with higher signal quality.

**SMA50 depth filter (3%)** — requiring price to be 3%+ below SMA50 filters shallow dead-cat bounces (2022 churn) while allowing meaningful crash reversals (2020 COVID). Without this, the strategy takes 13+ losses in 2022 on 1-3 bar fake bounces.

**SMA9 golden cross re-entry** — catches recovery legs where price never pulls back far enough for the HA reversal signal. Enables re-entries after death-cross exits without waiting for a deep pullback.

**Red-run low as stop level** — using the lowest Low of the preceding red run as the stop anchor is the natural support from the correction. If price revisits those lows, the thesis is invalidated.

**Death cross as macro exit** — SMA21 crossing below SMA100 is slow and lagging — which is exactly right for exits. Avoids whipsawing out of multi-month winners during normal pullbacks while catching genuine bear markets.

**4% cash rate on idle periods** — idle cash earns interest (HYSA/T-bill proxy), adding return during out-of-market periods. Shown explicitly in the `--trades` table.

### What doesn't work

**Fixed 1% stop on close price** — too tight for daily bar resolution. A -5% tariff gap-down fires the stop at the open, not at 1%. Using Low/Open for stop detection significantly improves fill accuracy.

**Consecutive stop pause mechanism** — counting stops to trigger a sit-out period is too deterministic and blocks genuine recovery entries (e.g., April 9/22, 2025 after the tariff crash). Removed in favor of the SMA50 depth filter.

**SMA slope filter** — requiring SMA50 to be rising blocks crash recovery entries (SMA50 still falling when the bottom forms). Rejected.

### Gap risk (inherent limitation)

On daily bars, overnight gaps cause fills at the open rather than the stop price. A gap-down through the stop fills at open — still better than close (which is further away), but unavoidable on daily resolution. The stop simulation uses Low to detect trigger and Open to determine fill price.

---

## Add a Strategy

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
    hard_stop_pct: float = 0.0  # set > 0 to enable hard stop in simulator

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
│   ├── ha_sma.py         # Preferred: Heikin-Ashi reversal all-in
│   ├── ha_sma_dca.py     # HA DCA variant + shared HA helpers
│   ├── renko_sma.py      # Renko ATR reversal
│   ├── renko_sma_dca.py  # Renko DCA variant
│   └── sma_crossover.py  # Golden Cross baseline
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
