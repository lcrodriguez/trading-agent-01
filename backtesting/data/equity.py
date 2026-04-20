from datetime import datetime

import pandas as pd
import yfinance as yf

from .base import AssetType, BaseDataLoader, DataLoadError, DataResult


class EquityLoader(BaseDataLoader):
    def load(self, symbol: str, start: str, end: str, interval: str = "1d") -> DataResult:
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
        except Exception as e:
            raise DataLoadError(f"Failed to download {symbol}: {e}") from e

        if df.empty:
            raise DataLoadError(f"No data returned for symbol '{symbol}' — check the ticker.")

        # Flatten MultiIndex columns (older yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        return DataResult(
            symbol=symbol,
            asset_type=AssetType.EQUITY,
            df=df,
            interval=interval,
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
        )
