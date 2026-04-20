from datetime import datetime

import ccxt
import pandas as pd

from .base import AssetType, BaseDataLoader, DataLoadError, DataResult

_TIMEFRAME_MAP = {"1d": "1d", "1wk": "1w", "1mo": "1M"}


class CryptoLoader(BaseDataLoader):
    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id

    def load(self, symbol: str, start: str, end: str, interval: str = "1d") -> DataResult:
        try:
            exchange: ccxt.Exchange = getattr(ccxt, self.exchange_id)()
            timeframe = _TIMEFRAME_MAP.get(interval, "1d")
            since = exchange.parse8601(f"{start}T00:00:00Z")

            all_ohlcv = []
            while True:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break

            if not all_ohlcv:
                raise DataLoadError(f"No data returned for {symbol} on {self.exchange_id}")

            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            df.index.name = "Date"
            df = df[start:end]

            return DataResult(
                symbol=symbol,
                asset_type=AssetType.CRYPTO,
                df=df,
                interval=interval,
                start=datetime.fromisoformat(start),
                end=datetime.fromisoformat(end),
            )
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to download {symbol} from {self.exchange_id}: {e}") from e
