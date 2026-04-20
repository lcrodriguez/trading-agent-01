import pytest

from backtesting.data.base import AssetType, DataLoadError
from backtesting.data.loader import load_data


def test_spy_daily_download():
    result = load_data("SPY", "2020-01-01", "2020-12-31", "1d")
    assert result.symbol == "SPY"
    assert result.asset_type == AssetType.EQUITY
    assert not result.df.empty
    assert len(result.df) > 200


def test_data_columns():
    result = load_data("SPY", "2023-01-01", "2023-06-30", "1d")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in result.df.columns


def test_invalid_symbol_raises():
    with pytest.raises(DataLoadError):
        load_data("ZZZZINVALID999XYZ", "2023-01-01", "2023-06-30", "1d")
