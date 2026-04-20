from .base import BaseDataLoader, DataResult
from .crypto import CryptoLoader
from .equity import EquityLoader


def get_loader(symbol: str) -> BaseDataLoader:
    if "/" in symbol:
        return CryptoLoader()
    return EquityLoader()


def load_data(symbol: str, start: str, end: str, interval: str = "1d") -> DataResult:
    return get_loader(symbol).load(symbol, start, end, interval)
