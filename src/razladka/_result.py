"""The generated series and its conversions to tables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._deps import require
from ._enums import FileFormat, Trend
from ._time import format_ns

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa

METADATA_KEY = "razladka"
"""Key of the ground-truth labels in Arrow schema metadata."""


@dataclass(frozen=True, eq=False)
class GeneratedSeries:
    """A generated time series with its ground-truth change points.

    Segment ``k`` (``0 … K``) runs from ``change_points[k - 1]`` (or 0) up to,
    not including, ``change_points[k]`` (or ``len(series)``).
    """

    times: np.ndarray
    """Time stamps, ``datetime64[unit]``, strictly increasing, UTC."""
    values: np.ndarray
    """Values, ``float64``, all within ``[lower, upper]``."""
    change_points: np.ndarray
    """Indices of the first points of segments ``1 … K`` (``int64``)."""
    change_times: np.ndarray
    """Time stamps of the change points, ``datetime64[unit]``."""
    trends: tuple[Trend, ...]
    """Direction of each of the ``K + 1`` segments."""
    roughness: float
    """Roughness coefficient ``s = D - 1`` used for the series."""
    unit: str
    """Time unit: ``"ns"``, ``"us"``, ``"ms"`` or ``"s"``."""

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        n = len(self)
        span = f"{self.times[0]} … {self.times[-1]}" if n else "empty"
        return (
            f"GeneratedSeries(n_points={n}, n_change_points={self.n_change_points}, "
            f"unit={self.unit!r}, {span}, trends={[t.value for t in self.trends]})"
        )

    @property
    def n_change_points(self) -> int:
        """Number of change points ``K``."""
        return int(self.change_points.size)

    @property
    def segments(self) -> np.ndarray:
        """Segment number of every point (``int64``, ``0 … K``).

        Computed on access from :attr:`change_points` so that a series of
        10**8 points does not carry an extra 800 MB array.
        """
        return segment_numbers(self.change_points, 0, len(self))

    def labels(self) -> dict[str, Any]:
        """Ground truth as plain JSON-compatible values.

        >>> import razladka
        >>> s = razladka.generate(
        ...     10, 1, start="2026-01-01", end="2026-01-10", lower=0, upper=1, seed=1
        ... )
        >>> sorted(s.labels())
        ['change_points', 'change_times', 'n_points', 'roughness', 'trends', 'unit']
        """
        return {
            "n_points": len(self),
            "change_points": [int(i) for i in self.change_points],
            "change_times": [
                format_ns(int(t), self.unit) for t in self.change_times.astype(np.int64)
            ],
            "trends": [t.value for t in self.trends],
            "roughness": float(self.roughness),
            "unit": self.unit,
        }

    # ------------------------------------------------------------ tables

    def to_pandas(self) -> pd.DataFrame:
        """A :class:`pandas.DataFrame` with the columns ``time`` (UTC), ``value``
        and ``segment``; the ground truth is in ``df.attrs``."""
        pd = require("pandas", "pandas")
        frame = pd.DataFrame(
            {
                "time": pd.Series(self.times, copy=False).dt.tz_localize("UTC"),
                "value": self.values,
                "segment": self.segments,
            }
        )
        frame.attrs.update(self.labels())
        return frame

    def to_polars(self) -> pl.DataFrame:
        """A :class:`polars.DataFrame` with the columns ``time`` (UTC), ``value``
        and ``segment``. Polars has no table metadata: the ``segment`` column
        is the ground truth."""
        pl = require("polars", "polars")
        if self.unit == "s":
            # seconds are chosen only when the axis does not fit into int64
            # milliseconds, the coarsest resolution of Polars
            msg = (
                "the time axis spans more than about ±292 million years and does not fit "
                "into Polars' millisecond resolution; use OutputType.RESULT, NUMPY or ARROW"
            )
            raise ValueError(msg)
        times = self.times
        return pl.DataFrame(
            {
                "time": pl.Series("time", times).dt.replace_time_zone("UTC"),
                "value": self.values,
                "segment": self.segments,
            }
        )

    def to_arrow(self) -> pa.Table:
        """A :class:`pyarrow.Table` with the columns ``time`` (UTC), ``value`` and
        ``segment``; the ground truth is JSON in the schema metadata under the
        key ``razladka``."""
        pa = require("pyarrow", "arrow")
        schema = pa.schema(
            [
                pa.field("time", pa.timestamp(self.unit, tz="UTC"), nullable=False),
                pa.field("value", pa.float64(), nullable=False),
                pa.field("segment", pa.int64(), nullable=False),
            ],
            metadata={METADATA_KEY: json.dumps(self.labels())},
        )
        return pa.Table.from_arrays(
            [
                pa.array(self.times.astype(np.int64), pa.int64()).cast(schema.field(0).type),
                pa.array(self.values),
                pa.array(self.segments),
            ],
            schema=schema,
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """``(times, values)``."""
        return self.times, self.values

    # ------------------------------------------------------------ files

    def save(self, path: str | os.PathLike[str], file_format: FileFormat | None = None) -> None:
        """Writes the series and its ground truth to a file (overwriting it).

        The format is taken from the extension unless ``file_format`` is given.
        """
        from ._io import save

        save(self, path, file_format)


def segment_numbers(change_points: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Segment numbers of the points ``start … stop - 1``.

    >>> segment_numbers(np.array([2, 5]), 0, 7)
    array([0, 0, 1, 1, 1, 2, 2])
    """
    index = np.arange(start, stop, dtype=np.int64)
    return np.searchsorted(change_points, index, side="right").astype(np.int64)
