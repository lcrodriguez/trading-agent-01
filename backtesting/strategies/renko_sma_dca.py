import pandas as pd

from backtesting.data.base import DataResult

from . import register_strategy
from .base import BaseStrategy, StrategyParam
from .renko_sma import _compute_atr_weekly, _compute_exits, _compute_renko, _renko_sma, _was_below_during_red_run


@register_strategy
class RenkoSmaDCA(BaseStrategy):
    """
    DCA variant of RenkoSMA. Same entry signal (Renko red→green below SMA buffer),
    but the engine adds equal lots on each subsequent signal as long as the new
    price is ≥ dca_drop_pct% below the last buy price and lots remain.
    Full position exits on bearish SMA crossunder.
    """

    name = "renko_sma_dca"
    description = (
        "Renko ATR red→green DCA strategy. Adds equal lots on each signal "
        "≥ dca_drop_pct% below last buy. Full exit on bearish SMA crossunder."
    )
    dca_mode = True

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return [
            StrategyParam("sma_period", "int", 21, "SMA period", min_val=5, max_val=500),
            StrategyParam("atr_period", "int", 10, "ATR period for Renko box size (Wilder's RMA)", min_val=5, max_val=100),
            StrategyParam("sma_buffer_pct", "float", 2.0, "Min % price must be below SMA to enter", min_val=0.0, max_val=20.0),
            StrategyParam("max_adds", "int", 3, "Max total entries incl. first buy", min_val=2, max_val=10),
            StrategyParam("dca_drop_pct", "float", 5.0, "Min % drop below last buy to add", min_val=0.5, max_val=50.0),
        ]

    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        df = data.df
        sma_period = int(self.get("sma_period"))
        atr_period = int(self.get("atr_period"))
        buffer = float(self.get("sma_buffer_pct")) / 100.0

        atr = _compute_atr_weekly(df, atr_period)
        renko_df = _compute_renko(df["High"], df["Low"], df["Close"], atr)
        renko = renko_df["color"]
        sma = _renko_sma(renko_df["brick_close"], sma_period)

        brick_close = renko_df["brick_close"]
        renko_turns_green = (renko == "green") & (renko.shift(1) == "red")
        was_below = _was_below_during_red_run(brick_close, sma, buffer, renko)
        entries = renko_turns_green & was_below

        exits = _compute_exits(brick_close, sma, renko, entries)

        return entries.fillna(False), exits.fillna(False)
