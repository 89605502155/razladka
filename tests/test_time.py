"""The time axis: bounds of any kind, unit choice, exact even spacing."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import razladka
from razladka._time import choose_unit, fill_times, to_ns


def gen(n: int, start: object, end: object) -> razladka.GeneratedSeries:
    return razladka.generate(n, 1, start=start, end=end, lower=0, upper=1, seed=0)


def test_year_five() -> None:
    s = gen(1000, "0005-01-01", "2026-10-09")
    assert s.unit == "us"
    assert str(s.times[0]) == "0005-01-01T00:00:00.000000"
    assert str(s.times[-1]) == "2026-10-09T00:00:00.000000"
    assert np.all(np.diff(s.times.astype(np.int64)) > 0)


def test_today_back_to_2008() -> None:
    s = gen(500, "2026-10-09", "2008-01-01")
    assert s.unit == "ns"
    assert str(s.times[0]) == "2008-01-01T00:00:00.000000000"
    assert str(s.times[-1]) == "2026-10-09T00:00:00.000000000"


def test_reversed_bounds_give_the_same_axis() -> None:
    a = gen(100, "2026-01-01", "2026-02-01")
    b = gen(100, "2026-02-01", "2026-01-01")
    assert np.array_equal(a.times, b.times)


def test_bound_types_agree() -> None:
    expected = gen(10, "2026-01-01T00:00:00Z", "2026-01-01T12:00:00Z").times
    variants = [
        (dt.datetime(2026, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 1, 1, 12, tzinfo=dt.UTC)),
        (dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 1, 12)),
        (dt.date(2026, 1, 1), "2026-01-01T12:00"),
        (np.datetime64("2026-01-01"), np.datetime64("2026-01-01T12", "h")),
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01T12:00:00+00:00")),
        ("2026-01-01T03:00:00+03:00", "2026-01-01T15:00:00+03:00"),
    ]
    for start, end in variants:
        assert np.array_equal(gen(10, start, end).times, expected), (start, end)


def test_nanosecond_strings_and_timestamps() -> None:
    assert to_ns("1970-01-01T00:00:00.000000007Z", "start") == 7
    assert to_ns(pd.Timestamp("1970-01-01T00:00:00.000000123"), "start") == 123
    assert to_ns(np.datetime64(5, "s"), "start") == 5 * 10**9
    assert to_ns(np.datetime64("1970-02", "M"), "start") == 31 * 86_400 * 10**9


def test_unit_choice() -> None:
    assert choose_unit(0, 10**18, 10)[0] == "ns"
    assert choose_unit(-(10**19), 0, 10)[0] == "us"
    assert choose_unit(-(10**22), 0, 10)[0] == "ms"
    assert choose_unit(-(10**25), 0, 10)[0] == "s"
    with pytest.raises(ValueError, match="too long"):
        choose_unit(-(10**29), 0, 10)


def test_hundred_million_points_per_day_fit_in_ns() -> None:
    unit, start, end = choose_unit(0, 86_400 * 10**9, 10**8)
    assert unit == "ns"
    out = np.empty(10**6, dtype=np.int64)
    # the first million stamps of the 10**8-point axis, computed the same way
    fill_times(out, start, start + (end - start) * (10**6 - 1) // (10**8 - 1))
    assert np.all(np.diff(out) > 0)


@given(
    start=st.integers(-(2**62), 2**62),
    span=st.integers(1, 2**62),
    n=st.integers(2, 3000),
)
@settings(max_examples=300, deadline=None)
def test_fill_times_is_exact(start: int, span: int, n: int) -> None:
    span = max(span, n - 1)
    end = start + span
    out = np.empty(n, dtype=np.int64)
    fill_times(out, start, end)
    expected = [start + (2 * i * span + (n - 1)) // (2 * (n - 1)) for i in (0, 1, n // 2, n - 1)]
    assert out[[0, 1, n // 2, n - 1]].tolist() == expected
    assert out[0] == start
    assert out[-1] == end
    assert np.all(np.diff(out) > 0)


def test_full_int64_range() -> None:
    out = np.empty(1000, dtype=np.int64)
    fill_times(out, -(2**63) + 1, 2**63 - 1)
    assert out[0] == -(2**63) + 1
    assert out[-1] == 2**63 - 1
    assert np.all(np.diff(out) > 0)


@pytest.mark.slow
def test_hundred_million_points() -> None:
    """Needs about 2 GB of RAM: run with ``pytest -m slow``."""
    s = gen(10**8, "2026-01-01", "2026-01-02")
    assert s.unit == "ns"
    assert len(s) == 10**8
    assert np.all(np.diff(s.times.astype(np.int64)) > 0)
