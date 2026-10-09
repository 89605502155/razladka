"""Time axis: parsing of the bounds, choice of the unit, evenly spaced stamps.

All arithmetic on the bounds is done on exact Python integers (nanoseconds
since the Unix epoch), so dates far outside the ``datetime64[ns]`` range
(e.g. the year 5) are handled without rounding surprises.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from typing import Any

import numpy as np

UNITS: tuple[str, ...] = ("ns", "us", "ms", "s")
"""Supported time units, from the finest to the coarsest."""

NS_PER_UNIT: dict[str, int] = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}

_INT64_MIN = -(2**63) + 1  # -2**63 is NaT in NumPy
_INT64_MAX = 2**63 - 1
_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
_CHUNK = 1 << 20
_FRACTION = re.compile(r"([.,])(\d{7,})")


def to_ns(value: Any, name: str) -> int:
    """Converts a time bound to integer nanoseconds since 1970-01-01 UTC.

    Accepts :class:`datetime.datetime`, :class:`datetime.date`, ISO 8601
    strings, :class:`numpy.datetime64` and :class:`pandas.Timestamp`. Naive
    times are taken as UTC.

    >>> to_ns("1970-01-01T00:00:01Z", "start")
    1000000000
    >>> to_ns("0005-01-01", "start")
    -62009366400000000000
    """
    if isinstance(value, np.datetime64):
        return _datetime64_to_ns(value, name)
    if isinstance(value, dt.datetime):
        ns = _datetime_to_ns(value)
        # pandas.Timestamp keeps nanoseconds that datetime does not
        return ns + int(getattr(value, "nanosecond", 0))
    if isinstance(value, dt.date):
        return _datetime_to_ns(dt.datetime(value.year, value.month, value.day))
    if isinstance(value, str):
        return _string_to_ns(value, name)
    msg = (
        f"{name} must be a datetime, date, ISO 8601 string, numpy.datetime64 "
        f"or pandas.Timestamp, not {type(value).__name__}"
    )
    raise TypeError(msg)


def _datetime_to_ns(value: dt.datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=dt.UTC)
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _datetime64_to_ns(value: np.datetime64, name: str) -> int:
    if np.isnat(value):
        msg = f"{name} must not be NaT"
        raise ValueError(msg)
    unit, count = np.datetime_data(value.dtype)
    if unit not in NS_PER_UNIT:
        # calendar units (Y, M, W, D, h, m) are converted exactly to seconds;
        # finer ones (ps, fs, as) are truncated to nanoseconds
        target = "s" if unit in {"Y", "M", "W", "D", "h", "m"} else "ns"
        value = value.astype(f"datetime64[{target}]")
        unit, count = target, 1
    return int(value.astype(np.int64)) * NS_PER_UNIT[unit] * count


def _string_to_ns(value: str, name: str) -> int:
    text = value.strip()
    # datetime holds microseconds: digits 7-9 of the fraction are added here
    extra_ns = 0
    match = _FRACTION.search(text)
    if match:
        digits = match.group(2)
        extra_ns = int(digits[6:9].ljust(3, "0"))
        text = text[: match.start(2)] + digits[:6] + text[match.end(2) :]
    try:
        return _datetime_to_ns(dt.datetime.fromisoformat(text)) + extra_ns
    except ValueError:
        pass
    # expanded years (ISO 8601 "+YYYYYY" / "-YYYYYY"), which datetime cannot hold
    naive = text[:-1] if text.endswith(("Z", "z")) else text
    if naive[:1] in {"+", "-"}:
        try:
            return _datetime64_to_ns(np.datetime64(naive.lstrip("+")), name) + extra_ns
        except ValueError:
            pass
    msg = f"{name}={value!r} is not an ISO 8601 date or date-time"
    raise ValueError(msg)


def now_ns() -> int:
    """Current time, nanoseconds since 1970-01-01 UTC."""
    return time.time_ns()


def choose_unit(start_ns: int, end_ns: int, n_points: int) -> tuple[str, int, int]:
    """Picks the finest unit in which both bounds fit into int64 and the step
    between ``n_points`` stamps is at least one tick.

    Returns ``(unit, start, end)`` with the bounds rounded to that unit,
    ``start < end``.

    >>> choose_unit(0, 10**9, 11)
    ('ns', 0, 1000000000)
    """
    if start_ns > end_ns:
        start_ns, end_ns = end_ns, start_ns
    for unit in UNITS:
        factor = NS_PER_UNIT[unit]
        start = _round_div(start_ns, factor)
        end = _round_div(end_ns, factor)
        if start < _INT64_MIN or end > _INT64_MAX:
            continue
        if end - start >= n_points - 1:
            return unit, start, end
        # a coarser unit only shortens the span
        break
    msg = (
        f"cannot place {n_points} strictly increasing time stamps between the bounds: "
        "the interval is too short (need at least one nanosecond per step) "
        "or too long for 64-bit seconds"
    )
    raise ValueError(msg)


def _round_div(a: int, b: int) -> int:
    """``a / b`` rounded to the nearest integer, halves away from minus infinity."""
    return (2 * a + b) // (2 * b)


def fill_times(out: np.ndarray, start: int, end: int) -> None:
    """Fills ``out`` (int64) with ``start + round(i * (end - start) / (n - 1))``.

    Exact integer arithmetic in chunks: the span is split as ``q * (n - 1) + r``,
    so no product exceeds 64 bits, and offsets that do not fit into int64 wrap
    around modulo 2**64 and come back into range when added to ``start``.
    """
    n = out.size
    if n == 1:
        out[0] = start
        return
    span = end - start
    q, r = divmod(span, n - 1)
    den = 2 * (n - 1)
    base = np.uint64(start % 2**64)
    q64 = np.uint64(q)
    view = out.view(np.uint64)
    for a in range(0, n, _CHUNK):
        i = np.arange(a, min(n, a + _CHUNK), dtype=np.int64)
        frac = (2 * i * r + (n - 1)) // den  # round(i * r / (n - 1)), < n
        offset = i.astype(np.uint64) * q64 + frac.astype(np.uint64)
        view[a : a + i.size] = base + offset


def format_ns(value: int, unit: str) -> str:
    """An ISO 8601 UTC string for a stamp in ``unit``.

    >>> format_ns(1500, "ms")
    '1970-01-01T00:00:01.500Z'
    """
    tick: Any = unit
    return str(np.datetime_as_string(np.datetime64(value, tick), unit=tick)) + "Z"
