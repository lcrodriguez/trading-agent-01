from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Optional

import pandas as pd

from backtesting.data.base import DataResult


@dataclass
class StrategyParam:
    name: str
    type: Literal["int", "float", "bool", "str"]
    default: Any
    description: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class BaseStrategy(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    dca_mode: ClassVar[bool] = False
    hard_stop_pct: ClassVar[float] = 0.0  # hard stop on actual price vs entry fill

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    def get(self, key: str) -> Any:
        spec = {p.name: p for p in self.param_specs()}
        return self.params.get(key, spec[key].default)

    @classmethod
    def param_specs(cls) -> list[StrategyParam]:
        return []

    @abstractmethod
    def generate_signals(self, data: DataResult) -> tuple[pd.Series, pd.Series]:
        """Return (entries, exits) boolean Series aligned to data.df.index."""
        ...
