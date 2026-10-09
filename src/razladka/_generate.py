"""The public function :func:`generate`."""

from __future__ import annotations

import math
import numbers
import os
from typing import Any

import numpy as np

from ._enums import FileFormat, OutputType, Trend
from ._markov import MIN_SEGMENT, draw_change_points
from ._result import GeneratedSeries
from ._roughness import fill_weierstrass, n_terms
from ._time import choose_unit, fill_times, now_ns, to_ns

DRIFT_LOW = 0.5
DRIFT_HIGH = 1.5
"""Absolute drift of a regime is uniform on ``[DRIFT_LOW, DRIFT_HIGH)``."""

_CHUNK = 1 << 20


def generate(
    n_points: int,
    n_change_points: int,
    *,
    start: Any,
    end: Any = None,
    lower: float,
    upper: float,
    roughness: float = 0.0,
    roughness_scale: float = 0.1,
    last_trend: Trend | None = None,
    output: OutputType = OutputType.RESULT,
    file: str | os.PathLike[str] | None = None,
    file_format: FileFormat | None = None,
    seed: int | np.random.Generator | None = None,
) -> Any:
    """Generates a time series with exactly ``n_change_points`` change points.

    The moments of the changes are drawn by a Markov regime-switching chain
    with a left-to-right transition matrix; the series is a continuous
    piecewise-linear skeleton whose direction alternates between segments,
    plus an optional Weierstrass-Mandelbrot component of fractal dimension
    ``1 + roughness``. All values lie within ``[lower, upper]``.

    Args:
        n_points: number of points, at least ``2 * (n_change_points + 1)``.
        n_change_points: number of change points ``K >= 0``.
        start: first bound of the time axis (datetime, date, ISO 8601
            string, ``numpy.datetime64`` or ``pandas.Timestamp``; naive
            times are UTC).
        end: second bound, by default the current time. The bounds may come
            in any order: the series always runs forward in time.
        lower: lower bound of the values.
        upper: upper bound of the values, greater than ``lower``.
        roughness: roughness coefficient ``s = D - 1`` in ``[0, 1]``: 0 is a
            series that is smooth inside segments, 1 a nowhere smooth one.
        roughness_scale: amplitude of the rough component at ``s = 1`` as a
            fraction of ``upper - lower``; ``roughness * roughness_scale``
            must be below 0.5.
        last_trend: direction of the most recent segment; random if ``None``.
        output: what to return (see :class:`OutputType`).
        file: if given, the series is also written to this file.
        file_format: format of ``file``; by default taken from its extension.
        seed: seed or generator; the same seed and arguments give the same
            series bit for bit.

    Returns:
        The series in the form chosen by ``output``.

    >>> import razladka
    >>> s = razladka.generate(
    ...     1000, 3, start="2026-01-01", end="2026-02-01", lower=0, upper=10, seed=42
    ... )
    >>> s.n_change_points, len(s), bool(s.values.min() >= 0 and s.values.max() <= 10)
    (3, 1000, True)
    """
    n = _check_int(n_points, "n_points")
    k = _check_int(n_change_points, "n_change_points")
    if k < 0:
        msg = f"n_change_points must be >= 0, got {k}"
        raise ValueError(msg)
    if n < MIN_SEGMENT * (k + 1):
        msg = (
            f"n_points must be at least {MIN_SEGMENT} * (n_change_points + 1) = "
            f"{MIN_SEGMENT * (k + 1)} (every segment needs {MIN_SEGMENT} points), got {n}"
        )
        raise ValueError(msg)
    lo = _check_float(lower, "lower")
    hi = _check_float(upper, "upper")
    if not lo < hi:
        msg = f"upper must be greater than lower, got lower={lo}, upper={hi}"
        raise ValueError(msg)
    s = _check_float(roughness, "roughness")
    if not 0.0 <= s <= 1.0:
        msg = f"roughness must be within [0, 1], got {s}"
        raise ValueError(msg)
    scale = _check_float(roughness_scale, "roughness_scale")
    if scale < 0:
        msg = f"roughness_scale must be >= 0, got {scale}"
        raise ValueError(msg)
    if s * scale >= 0.5:
        msg = (
            "roughness * roughness_scale must be below 0.5 so that the rough component "
            f"fits into [lower, upper], got {s} * {scale} = {s * scale}"
        )
        raise ValueError(msg)
    if last_trend is not None and not isinstance(last_trend, Trend):
        msg = f"last_trend must be a Trend or None, not {type(last_trend).__name__}"
        raise TypeError(msg)
    if not isinstance(output, OutputType):
        msg = f"output must be an OutputType, not {type(output).__name__}"
        raise TypeError(msg)
    if file_format is not None and not isinstance(file_format, FileFormat):
        msg = f"file_format must be a FileFormat or None, not {type(file_format).__name__}"
        raise TypeError(msg)
    if file is None and file_format is not None:
        msg = "file_format is given but file is not; pass file=... to save the series"
        raise ValueError(msg)
    if file is not None and file_format is None:
        file_format = FileFormat.from_path(file)  # fail early on a bad extension

    start_ns = to_ns(start, "start")
    end_ns = now_ns() if end is None else to_ns(end, "end")
    unit, t0, t1 = choose_unit(start_ns, end_ns, n)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    change_points = draw_change_points(n, k, rng)
    if last_trend is None:
        last_trend = Trend.UP if rng.integers(2) == 1 else Trend.DOWN
    trends = _alternate(last_trend, k + 1)
    drifts = rng.uniform(DRIFT_LOW, DRIFT_HIGH, size=k + 1) * np.array(
        [t.sign for t in trends], dtype=np.float64
    )
    amplitude = s * scale * (hi - lo)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_terms(n)) if amplitude > 0 else None

    times = np.empty(n, dtype=np.int64)
    fill_times(times, t0, t1)
    times = times.view(f"datetime64[{unit}]")

    values = np.zeros(n, dtype=np.float64)
    peak = 0.0
    if phases is not None:
        peak = fill_weierstrass(values, s, phases)
    rough = amplitude / peak if peak > 0 else 0.0
    _add_skeleton(
        values,
        change_points=change_points,
        drifts=drifts,
        low=lo + amplitude,
        high=hi - amplitude,
        rough=rough,
    )
    # guard against rounding in the last bit; mathematically a no-op
    np.clip(values, lo, hi, out=values)

    series = GeneratedSeries(
        times=times,
        values=values,
        change_points=change_points,
        change_times=times[change_points],
        trends=trends,
        roughness=s,
        unit=unit,
    )
    if file is not None:
        series.save(file, file_format)
    return _convert(series, output)


def _check_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        msg = f"{name} must be an integer, not {type(value).__name__}"
        raise TypeError(msg)
    return int(value)


def _check_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        msg = f"{name} must be a real number, not {type(value).__name__}"
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result):
        msg = f"{name} must be finite, got {result}"
        raise ValueError(msg)
    return result


def _alternate(last: Trend, count: int) -> tuple[Trend, ...]:
    """Alternating directions ending with ``last``.

    >>> [t.value for t in _alternate(Trend.UP, 3)]
    ['up', 'down', 'up']
    """
    return tuple(last if (count - 1 - j) % 2 == 0 else last.opposite for j in range(count))


def _add_skeleton(
    values: np.ndarray,
    *,
    change_points: np.ndarray,
    drifts: np.ndarray,
    low: float,
    high: float,
    rough: float,
) -> None:
    """``values = skeleton scaled onto [low, high] + rough * values`` in place.

    The skeleton is the cumulative sum of the regime drifts: ``x[0] = 0``,
    ``x[i] = x[i - 1] + drift[segment(i)]``. It is linear inside a segment,
    so its extremes are at the first and last points of segments.
    """
    n = values.size
    firsts = np.concatenate([[0], change_points]).astype(np.int64)
    lasts = np.concatenate([change_points - 1, [n - 1]]).astype(np.int64)
    bases = np.empty(drifts.size, dtype=np.float64)  # x at the first point of a segment
    level = 0.0
    for j, first in enumerate(firsts):
        if j > 0:
            level += drifts[j - 1] * float(first - 1 - firsts[j - 1]) + drifts[j]
        bases[j] = level
    ends = bases + drifts * (lasts - firsts).astype(np.float64)
    extremes = np.concatenate([bases, ends])
    x_min = float(np.min(extremes))
    x_max = float(np.max(extremes))
    factor = (high - low) / (x_max - x_min)
    for j, (first, last) in enumerate(zip(firsts, lasts, strict=True)):
        for a in range(int(first), int(last) + 1, _CHUNK):
            b = min(int(last) + 1, a + _CHUNK)
            offset = np.arange(a - first, b - first, dtype=np.float64)
            skeleton = low + ((bases[j] - x_min) + drifts[j] * offset) * factor
            if rough:
                values[a:b] *= rough
                values[a:b] += skeleton
            else:
                values[a:b] = skeleton


def _convert(series: GeneratedSeries, output: OutputType) -> Any:
    if output is OutputType.RESULT:
        return series
    if output is OutputType.NUMPY:
        return series.to_numpy()
    if output is OutputType.PANDAS:
        return series.to_pandas()
    if output is OutputType.POLARS:
        return series.to_polars()
    return series.to_arrow()
