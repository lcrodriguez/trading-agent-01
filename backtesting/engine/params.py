from pydantic import BaseModel, model_validator


class BacktestParams(BaseModel):
    symbol: str
    start: str
    end: str
    interval: str = "1d"
    init_cash: float = 10_000.0
    fees: float = 0.001
    slippage: float = 0.0005
    strategy_name: str = "sma_crossover"
    strategy_params: dict = {}
    run_benchmark: bool = True
    cash_rate: float = 0.0  # annual return on idle cash (e.g. 0.025 for 2.5% bonds)
    annual_contribution: float = 0.0  # cash injected at the start of each calendar year

    @model_validator(mode="after")
    def check_dates(self) -> "BacktestParams":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self
