import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtesting.engine.params import BacktestParams
from backtesting.engine.runner import BacktestResult, BacktestRunner
from backtesting.metrics.calculator import MetricsCalculator
from backtesting.reports.exporter import ResultExporter
from backtesting.strategies import list_strategies

st.set_page_config(page_title="Backtesting Dashboard", page_icon="📈", layout="wide")
st.title("📈 Backtesting Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    symbol = st.text_input("Symbol", value="SPY", help="Equity: SPY, AAPL  |  Crypto: BTC/USDT")
    strategies = list_strategies()
    strategy_names = [s["name"] for s in strategies]
    strategy_descs = {s["name"]: s["description"] for s in strategies}
    strategy_params_map = {s["name"]: s["params"] for s in strategies}

    chosen = st.selectbox("Strategy", strategy_names, format_func=lambda n: f"{n} — {strategy_descs[n]}")

    st.caption(strategy_descs[chosen])

    # Dynamic strategy params
    strategy_params: dict = {}
    param_specs = strategy_params_map[chosen]
    if param_specs:
        st.subheader("Strategy Parameters")
        for p in param_specs:
            if p.type == "int":
                strategy_params[p.name] = st.number_input(
                    p.name, value=int(p.default), step=1,
                    min_value=int(p.min_val) if p.min_val else 1,
                    max_value=int(p.max_val) if p.max_val else 9999,
                    help=p.description,
                )
            elif p.type == "float":
                strategy_params[p.name] = st.number_input(
                    p.name, value=float(p.default), step=0.01, help=p.description
                )
            elif p.type == "bool":
                strategy_params[p.name] = st.checkbox(p.name, value=bool(p.default), help=p.description)
            else:
                strategy_params[p.name] = st.text_input(p.name, value=str(p.default), help=p.description)

    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    start = col1.text_input("Start", value="2010-01-01")
    end = col2.text_input("End", value="2024-01-01")

    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Portfolio")
    init_cash = st.number_input("Initial Capital ($)", value=10_000.0, step=1000.0, min_value=100.0)
    fees = st.number_input("Fees (fraction)", value=0.001, step=0.0001, format="%.4f")
    slippage = st.number_input("Slippage (fraction)", value=0.0005, step=0.0001, format="%.4f")

    run_clicked = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ── Run backtest ──────────────────────────────────────────────────────────────
if run_clicked:
    try:
        params = BacktestParams(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            strategy_name=chosen,
            strategy_params=strategy_params,
            run_benchmark=True,
        )
        with st.spinner("Fetching data and running backtest…"):
            result = BacktestRunner().run(params)
        st.session_state["result"] = result
    except Exception as e:
        st.error(f"Backtest failed: {e}")

result: BacktestResult | None = st.session_state.get("result")

if result is None:
    st.info("Configure your strategy in the sidebar and click **▶ Run Backtest**.")
    st.stop()

# ── Compute metrics once ──────────────────────────────────────────────────────
strat_metrics = MetricsCalculator.calculate(result.strategy)
bench_metrics = MetricsCalculator.calculate(result.benchmark) if result.benchmark else {}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_trades, tab_metrics, tab_export = st.tabs(
    ["Overview", "Trades", "Metrics", "Export"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    p = result.params
    st.caption(f"{p.strategy_name} · {p.symbol} · {p.start} → {p.end} · ${p.init_cash:,.0f}")

    m = strat_metrics
    b = bench_metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Return",
        f"{m['Total Return (%)']:.1f}%",
        delta=f"{m['Total Return (%)'] - b.get('Total Return (%)', 0):.1f}% vs B&H" if b else None,
    )
    c2.metric(
        "CAGR",
        f"{m['CAGR (%)']:.1f}%",
        delta=f"{m['CAGR (%)'] - b.get('CAGR (%)', 0):.1f}% vs B&H" if b else None,
    )
    c3.metric(
        "Sharpe",
        f"{m['Sharpe Ratio']:.2f}",
        delta=f"{m['Sharpe Ratio'] - b.get('Sharpe Ratio', 0):.2f} vs B&H" if b else None,
    )
    c4.metric(
        "Max Drawdown",
        f"{m['Max Drawdown (%)']:.1f}%",
        delta=f"{m['Max Drawdown (%)'] - b.get('Max Drawdown (%)', 0):.1f}% vs B&H" if b else None,
        delta_color="inverse",
    )

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.strategy.equity.index,
        y=result.strategy.equity,
        name=p.strategy_name,
        line=dict(color="#00b4d8", width=2),
    ))
    if result.benchmark:
        fig.add_trace(go.Scatter(
            x=result.benchmark.equity.index,
            y=result.benchmark.equity,
            name="Buy & Hold",
            line=dict(color="#f77f00", width=2, dash="dot"),
        ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Trades ─────────────────────────────────────────────────────────────
with tab_trades:
    trades = result.strategy.trades
    if not trades:
        st.info("No trades executed.")
    else:
        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        st.dataframe(
            df_trades.style.applymap(
                lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else
                          "color: red" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["pnl", "pnl_pct"],
            ),
            use_container_width=True,
        )

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=[t.pnl for t in trades],
            nbinsx=20,
            name="PnL",
            marker_color=["#2dc653" if t.pnl > 0 else "#e63946" for t in trades],
        ))
        fig_hist.update_layout(title="Trade PnL Distribution ($)", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 3: Metrics ────────────────────────────────────────────────────────────
with tab_metrics:
    comparison = MetricsCalculator.compare(strat_metrics, bench_metrics)
    st.dataframe(comparison, use_container_width=True)

    # Drawdown chart
    equity = result.strategy.equity
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy",
        line=dict(color="#e63946", width=1),
        name="Drawdown",
    ))
    fig_dd.update_layout(title="Drawdown (%)", yaxis_title="%", height=280)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Monthly returns heatmap
    monthly = result.strategy.returns.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
    monthly_df = monthly.to_frame("return")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if v == v else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(title="Monthly Returns (%)", height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 4: Export ─────────────────────────────────────────────────────────────
with tab_export:
    import json
    import io

    strat_metrics_calc = MetricsCalculator.calculate(result.strategy)
    bench_metrics_calc = MetricsCalculator.calculate(result.benchmark) if result.benchmark else {}

    trades_df = pd.DataFrame([t.__dict__ for t in result.strategy.trades])
    metrics_df = MetricsCalculator.compare(strat_metrics_calc, bench_metrics_calc)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_buf = io.StringIO()
        trades_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Trades CSV",
            data=csv_buf.getvalue(),
            file_name="trades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        metrics_buf = io.StringIO()
        metrics_df.to_csv(metrics_buf)
        st.download_button(
            "⬇ Metrics CSV",
            data=metrics_buf.getvalue(),
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        json_payload = {
            "params": result.params.model_dump(),
            "strategy_metrics": strat_metrics_calc,
            "benchmark_metrics": bench_metrics_calc,
            "trades": [t.__dict__ for t in result.strategy.trades],
        }
        st.download_button(
            "⬇ Full JSON",
            data=json.dumps(json_payload, indent=2, default=str),
            file_name="result.json",
            mime="application/json",
            use_container_width=True,
        )
