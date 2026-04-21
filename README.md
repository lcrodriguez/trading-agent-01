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
| `ha_sma` | All-in | **Preferred.** Heikin-Ashi reversal with hard stop and downtrend pause |
| `ha_sma_dca` | % DCA | Heikin-Ashi DCA — adds on each red→green flip below SMA |
| `renko_sma` | All-in | Renko ATR reversal with SMA crossunder exit |
| `renko_sma_dca` | DCA | Renko reversal with equal-lot DCA adds |
| `sma_crossover` | All-in | Golden Cross (fast SMA crosses above slow SMA) |
| `buy_and_hold` | — | Buy first bar, hold forever. Used as benchmark |

---

## Preferred Strategy: `ha_sma`

### Concept

Heikin-Ashi bars smooth price action by averaging OHLC values, making trend reversals easier to spot. This strategy buys when price is in a corrective pullback below the slow SMA and HA bars flip from red to green — a momentum reversal signal. It exits via a macro trend filter (SMA death cross) or a hard stop to cap losses.

### Rules

**Entry** — all must be true on the same bar:
- Heikin-Ashi bar color flips red → green
- HA close is below SMA(21)

**Exit** — first of:
1. **Hard stop**: actual close price drops ≥ 1% below entry fill price
2. **Death cross**: SMA(21) crosses below SMA(100) (macro bear signal)

**Downtrend protection** — consecutive stop pause:
- After 3 hard stops ≤ 3% loss, sit out 50 bars before taking new entries
- Extraordinary gap-downs > 3% are excluded from the consecutive counter (they are market-wide events, not trend signals)

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `sma_period` | 21 | Entry filter SMA period |
| `exit_sma_period` | 100 | Slow SMA for death cross exit |
| `consec_stop_limit` | 3 | Consecutive stops before pausing |
| `pause_bars` | 50 | Bars to sit out after hitting limit |

### Backtest Results (SPY)

| Period | Strategy Return | B&H Return | Notes |
|---|---|---|---|
| 2020–2023 | ~115% | ~56% | COVID crash recovery, 2022 bear, 2023 rebound |
| 2024–2025 | Open position +31%+ | — | Entered April 22, 2025 after tariff crash bottom |

Key characteristics (2020–2023):
- 12 total trades, avg holding period ~103 days
- Long winners, short losers (1% hard stop)
- Sat out most of 2022 bear market via consecutive stop pause

### Usage

```bash
bt run --symbol SPY --strategy ha_sma --start 2020-01-01 --end 2024-01-01 --trades
```

---

## Strategy Design Findings

### What works

**Heikin-Ashi for reversals** — HA bars filter noise better than raw candles for identifying red→green flips at pullback lows. The smoothing effect means entries happen slightly after the actual bottom, but with higher confidence.

**Below-SMA-only entries** — filtering to only take entries when price is below SMA(21) avoids chasing breakouts and keeps the strategy focused on pullback reversals.

**Hard stop over trailing stop** — a fixed 1% hard stop on the actual close price (not HA price) cleanly caps loss per trade. HA smoothing would cause late stops; actual price is used instead.

**Death cross as macro exit** — SMA(21) crossing below SMA(100) is a slow, lagging signal — which is exactly what's wanted for exits. It avoids whipsawing out of multi-month winners during normal pullbacks while catching genuine bear markets.

**Consecutive stop pause** — taking 11 consecutive 1% losses in 2022 was the biggest drawdown risk. A 50-bar sit-out after 3 consecutive stops lets the trend fully resolve before re-engaging. This alone separated the 2022 experience from being crippling.

### What doesn't work

**Dead zone filter on entry** — applying a ±2% SMA buffer to entries blocks too many valid pullback reversals that are inherently near the SMA line. Removed.

**SMA slope filter** — requiring SMA(50) to be rising to allow entries blocked the March 2020 COVID crash recovery (slope was still falling when the bottom formed). Rejected.

**HA-based stop prices** — computing stop prices from HA low values introduces lag. The strategy uses actual close vs entry fill for stop checks instead.

**Tight consecutive stop threshold** — a 2% gap exclusion cut into 2022 stops that should count, reducing 2020–2023 return. 3% threshold correctly excludes only extraordinary single-session gap events (e.g., tariff announcements).

### Gap risk (inherent limitation)

On daily bars, overnight gaps can cause fills significantly below the 1% stop level. A -4.93% tariff-related gap-down in April 2025 filled at -4.93% despite a 1% stop. This is accepted as a property of daily-resolution backtesting — the stop triggers at the open if price gaps below it, not at the stop price.

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
