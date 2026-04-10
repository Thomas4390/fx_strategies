"""Shared test fixtures -- 1-week data slice for fast tests."""

import pytest
import vectorbtpro as vbt


@pytest.fixture(scope="session")
def raw_and_data():
    """Load 1 week of EUR-USD data for all tests."""
    from utils import load_fx_data

    raw, data = load_fx_data("data/EUR-USD_minute.parquet")
    # Last ~5 trading days
    raw_mini = raw.iloc[-6300:].copy()
    return raw_mini, data


@pytest.fixture(scope="session")
def raw(raw_and_data):
    return raw_and_data[0]


@pytest.fixture(scope="session")
def data(raw_and_data):
    return raw_and_data[1]


@pytest.fixture(scope="session")
def index_ns(raw):
    return vbt.dt.to_ns(raw.index)
