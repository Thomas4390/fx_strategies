"""Contract of the broker (MT5) daily loader.

The loader exists so a sleeve written against the QuantConnect gold export runs
unchanged on a broker export. What is pinned here is exactly what makes that
substitution safe:

1. The index convention — tz-naive New York, strictly increasing, no duplicates.
   A duplicated stamp silently double-counts a session; a tz-aware one shifts
   every 17:00 boundary by the UTC offset.
2. The **refusal to guess a timezone**. A tz-naive export is rejected rather
   than assumed to be UTC, because a wrong guess moves session boundaries and
   nothing downstream says so.
3. The pairs actually available, read from the data directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_EXPECTED_PAIRS: tuple[str, ...] = (
    "AUD-USD",
    "EUR-GBP",
    "EUR-JPY",
    "EUR-USD",
    "GBP-JPY",
    "GBP-USD",
    "NZD-USD",
    "USD-CAD",
    "USD-CHF",
    "USD-JPY",
)


def test_eur_usd_index_and_columns():
    """Naive New York index, strictly increasing, with the broker columns kept."""
    from utils import load_mt5_daily

    raw, data = load_mt5_daily("EUR-USD")

    assert raw.index.tz is None, f"index must be tz-naive New York, got tz={raw.index.tz}"
    assert raw.index.is_monotonic_increasing
    assert not raw.index.has_duplicates, "a duplicated session is counted twice by every lookback"
    assert (np.diff(raw.index.to_numpy()) > np.timedelta64(0, "ns")).all(), (
        "index must be *strictly* increasing"
    )

    for col in ("open", "high", "low", "close", "spread"):
        assert col in raw.columns, f"{col!r} missing from the raw frame: {raw.columns.tolist()}"

    # The vbt.Data mirror carries the same sessions, capitalized.
    assert data.close.shape[0] == len(raw)


def test_eur_jpy_starts_in_november_2022():
    """The EUR-JPY export is shorter than the majors — it starts in 2022-11."""
    from utils import load_mt5_daily

    raw, _ = load_mt5_daily("EUR-JPY")
    first = raw.index[0]

    assert (first.year, first.month) == (2022, 11), (
        f"EUR-JPY starts at {first}, expected 2022-11. A longer history means the "
        "export changed and every cross-pair window has to be rechecked."
    )


def test_tz_naive_export_is_rejected(tmp_path, monkeypatch):
    """A tz-naive frame raises rather than having its timezone guessed.

    VBT localizes a naive parquet index to UTC on its own (``data.tz_localize``
    defaults to ``"utc"``), so the loader's guard is only reachable once that
    auto-localization is off — which is exactly the situation it protects
    against: an index whose timezone nobody stated. Both settings are patched
    because ``tz_convert`` would otherwise raise a TypeError first.
    """
    import vectorbtpro as vbt

    import utils
    from utils import load_mt5_daily

    monkeypatch.setitem(vbt.settings["data"], "tz_localize", None)
    monkeypatch.setitem(vbt.settings["data"], "tz_convert", None)

    idx = pd.date_range("2024-01-01", periods=5, freq="1D")
    idx.name = "time"
    frame = pd.DataFrame(
        {
            "Open": [1.10, 1.11, 1.12, 1.13, 1.14],
            "High": [1.15, 1.16, 1.17, 1.18, 1.19],
            "Low": [1.05, 1.06, 1.07, 1.08, 1.09],
            "Close": [1.12, 1.13, 1.14, 1.15, 1.16],
        },
        index=idx,
    )
    frame.to_parquet(tmp_path / "FAKE_synthetic_mt5.parquet")

    # An absolute template short-circuits the project-root join in the loader,
    # so the fixture never touches the repository's data directory.
    monkeypatch.setattr(
        utils, "MT5_DATA_TEMPLATE", str(tmp_path / "{pair}_{period}_mt5.parquet")
    )

    with pytest.raises(ValueError, match="tz-naive"):
        load_mt5_daily("FAKE", period="synthetic")


def test_list_mt5_pairs_covers_the_ten_pairs():
    """All ten exported pairs are visible to the callers."""
    from utils import list_mt5_pairs

    pairs = list_mt5_pairs()

    missing = [p for p in _EXPECTED_PAIRS if p not in pairs]
    assert not missing, f"missing broker export(s): {missing}"
