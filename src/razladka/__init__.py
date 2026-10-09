"""razladka — synthetic time series with a given number of change points.

You choose **how many** change points the series has; a Markov
regime-switching chain decides **where** they are, and the true positions
come back with the series so that a change-point detector or a clustering
method can be scored against them::

    import razladka

    s = razladka.generate(10_000, 3, start="2026-01-01", lower=0, upper=100, roughness=0.4)
    s.change_points  # indices of the first points of segments 1..3
    s.save("run.parquet")

The series is a continuous piecewise-linear skeleton whose direction
alternates between segments, plus an optional Weierstrass-Mandelbrot
component whose fractal dimension is ``1 + roughness``.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from ._enums import FileFormat, OutputType, Trend
from ._generate import generate
from ._result import GeneratedSeries

__all__ = ["FileFormat", "GeneratedSeries", "OutputType", "Trend", "__version__", "generate"]

try:
    __version__ = _metadata.version("razladka")
except _metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"
