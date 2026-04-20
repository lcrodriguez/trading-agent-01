from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd


class AssetType(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"


@dataclass
class DataResult:
    symbol: str
    asset_type: AssetType
    df: pd.DataFrame  # columns: Open, High, Low, Close, Volume
    interval: str
    start: datetime
    end: datetime


class DataLoadError(Exception):
    pass


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, symbol: str, start: str, end: str, interval: str = "1d") -> DataResult:
        ...
