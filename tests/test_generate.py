"""Properties of the generated series and validation of the arguments."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import razladka
from razladka import FileFormat, OutputType, Trend
from razladka._markov import MIN_SEGMENT, draw_change_points, leave_probability

BASE = {"start": "2026-01-01", "end": "2026-06-01", "lower": 0.0, "upper": 1.0}


def make(n: int = 300, k: int = 3, **kwargs: object) -> razladka.GeneratedSeries:
    return razladka.generate(n, k, **{**BASE, "seed": 1, **kwargs})


@given(
    n_extra=st.integers(0, 400),
    k=st.integers(0, 12),
    roughness=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
    last=st.sampled_from([None, Trend.UP, Trend.DOWN]),
)
@settings(max_examples=200, deadline=None)
def test_structure(n_extra: int, k: int, roughness: float, seed: int, last: Trend | None) -> None:
    n = MIN_SEGMENT * (k + 1) + n_extra
    s = razladka.generate(
        n, k, **{**BASE, "lower": -3.0, "upper": 7.0}, roughness=roughness,
        roughness_scale=0.3, last_trend=last, seed=seed,
    )  # fmt: skip
    assert len(s) == n
    assert s.n_change_points == k
    assert s.change_points.dtype == np.int64
    # every segment has at least MIN_SEGMENT points
    bounds = np.concatenate([[0], s.change_points, [n]])
    assert np.all(np.diff(bounds) >= MIN_SEGMENT)
    segments = s.segments
    assert segments[0] == 0
    assert segments[-1] == k
    assert np.count_nonzero(np.diff(segments)) == k
    assert np.array_equal(np.flatnonzero(np.diff(segments)) + 1, s.change_points)
    assert np.array_equal(s.change_times, s.times[s.change_points])
    # bounds of the values
    assert s.values.dtype == np.float64
    assert s.values.min() >= -3.0
    assert s.values.max() <= 7.0
    # alternating trends, the last one as requested
    assert len(s.trends) == k + 1
    for a, b in zip(s.trends, s.trends[1:], strict=False):
        assert a is not b
    if last is not None:
        assert s.trends[-1] is last
    # strictly increasing time
    assert np.all(np.diff(s.times.astype(np.int64)) > 0)


@pytest.mark.parametrize("k", [0, 1, 4, 20])
def test_smooth_series_is_strictly_monotone_inside_segments(k: int) -> None:
    s = make(n=2000, k=k, roughness=0.0, seed=k)
    bounds = np.concatenate([[0], s.change_points, [len(s)]])
    for j, trend in enumerate(s.trends):
        piece = s.values[bounds[j] : bounds[j + 1]]
        steps = np.diff(piece)
        if trend is Trend.UP:
            assert np.all(steps > 0)
        else:
            assert np.all(steps < 0)


def test_smooth_skeleton_fills_the_band() -> None:
    s = make(n=1000, k=4, roughness=0.0)
    assert s.values.min() == pytest.approx(0.0, abs=1e-12)
    assert s.values.max() == pytest.approx(1.0, abs=1e-12)


def test_rough_series_keeps_a_margin() -> None:
    s = make(n=4000, k=2, lower=10.0, upper=20.0, roughness=0.8, roughness_scale=0.5)
    assert s.values.min() >= 10.0
    assert s.values.max() <= 20.0
    # the rough component is not negligible
    assert np.std(np.diff(s.values)) > 0.01


def test_skeleton_is_continuous_at_change_points() -> None:
    s = make(n=1000, k=5, roughness=0.0)
    steps = np.abs(np.diff(s.values))
    # no level jumps: every step is at most the largest regular step
    assert steps.max() <= 1.6 / 0.5 * np.median(steps)


def test_reproducible() -> None:
    a = make(n=3000, k=4, roughness=0.6, seed=123)
    b = make(n=3000, k=4, roughness=0.6, seed=123)
    assert a.values.tobytes() == b.values.tobytes()
    assert a.times.tobytes() == b.times.tobytes()
    assert np.array_equal(a.change_points, b.change_points)
    assert a.trends == b.trends
    c = make(n=3000, k=4, roughness=0.6, seed=124)
    assert not np.array_equal(a.values, c.values)


def test_generator_seed() -> None:
    a = make(seed=np.random.default_rng(5))
    b = make(seed=np.random.default_rng(5))
    assert np.array_equal(a.values, b.values)


def test_zero_change_points() -> None:
    s = make(n=2, k=0, last_trend=Trend.DOWN)
    assert s.n_change_points == 0
    assert s.trends == (Trend.DOWN,)
    assert s.values[0] > s.values[1]


def test_minimum_length() -> None:
    s = make(n=8, k=3)
    assert np.array_equal(s.change_points, [2, 4, 6])


def test_leave_probability_gives_the_mean_duration() -> None:
    for n, k in [(1000, 4), (10**6, 1), (50, 7)]:
        q = leave_probability(n, k)
        assert MIN_SEGMENT + (1 - q) / q == pytest.approx(n / (k + 1))
    assert leave_probability(10, 4) == 1.0  # every regime lasts the minimum


def test_change_points_spread() -> None:
    """Accepted realisations put change points all over the series.

    Rejecting chains that do not reach the last regime shortens the accepted
    durations a little (mean ~160 instead of 200 for n = 1000, K = 4).
    """
    rng = np.random.default_rng(0)
    lengths = []
    lasts = []
    for _ in range(300):
        cps = draw_change_points(1000, 4, rng)
        lengths.extend(np.diff(np.concatenate([[0], cps])).tolist())
        lasts.append(cps[-1])
    assert 120 < np.mean(lengths) < 200
    assert np.min(lengths) >= MIN_SEGMENT
    assert max(lasts) <= 1000 - MIN_SEGMENT
    assert np.percentile(lasts, 10) < 500 < np.percentile(lasts, 90)


def test_output_types(series: razladka.GeneratedSeries) -> None:
    kwargs = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z", "lower": -5.0,
              "upper": 5.0, "roughness": 0.5, "seed": 7}  # fmt: skip
    times, values = razladka.generate(500, 3, output=OutputType.NUMPY, **kwargs)
    assert np.array_equal(times, series.times)
    assert np.array_equal(values, series.values)
    df = razladka.generate(500, 3, output=OutputType.PANDAS, **kwargs)
    assert list(df.columns) == ["time", "value", "segment"]
    assert df.attrs["change_points"] == series.change_points.tolist()
    assert np.array_equal(df["value"].to_numpy(), series.values)
    pl_df = razladka.generate(500, 3, output=OutputType.POLARS, **kwargs)
    assert pl_df.columns == ["time", "value", "segment"]
    assert np.array_equal(pl_df["segment"].to_numpy(), series.segments)
    table = razladka.generate(500, 3, output=OutputType.ARROW, **kwargs)
    assert table.column_names == ["time", "value", "segment"]
    result = razladka.generate(500, 3, output=OutputType.RESULT, **kwargs)
    assert np.array_equal(result.values, series.values)


def test_file_is_written_and_result_returned(tmp_path: object, series: object) -> None:
    import pathlib

    path = pathlib.Path(str(tmp_path)) / "x.data"
    values = razladka.generate(
        50, 1, **BASE, seed=3, file=path, file_format=FileFormat.CSV, output=OutputType.NUMPY
    )[1]
    assert path.read_text().startswith("time,value,segment\n")
    assert len(values) == 50


def test_repr(series: razladka.GeneratedSeries) -> None:
    text = repr(series)
    assert "n_points=500" in text
    assert "n_change_points=3" in text


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_points": 5, "n_change_points": 3}, ValueError, "at least 2"),
        ({"n_change_points": -1}, ValueError, ">= 0"),
        ({"n_points": 10.0}, TypeError, "integer"),
        ({"n_points": True}, TypeError, "integer"),
        ({"lower": 1.0, "upper": 1.0}, ValueError, "greater than lower"),
        ({"lower": 2.0, "upper": 1.0}, ValueError, "greater than lower"),
        ({"lower": float("nan")}, ValueError, "finite"),
        ({"upper": "1"}, TypeError, "real number"),
        ({"roughness": -0.1}, ValueError, r"\[0, 1\]"),
        ({"roughness": 1.1}, ValueError, r"\[0, 1\]"),
        ({"roughness": 1.0, "roughness_scale": 0.5}, ValueError, "below 0.5"),
        ({"roughness_scale": -1.0}, ValueError, ">= 0"),
        ({"last_trend": "up"}, TypeError, "Trend"),
        ({"output": "numpy"}, TypeError, "OutputType"),
        ({"file_format": FileFormat.CSV}, ValueError, "file is not"),
        ({"file_format": ".csv", "file": "x.csv"}, TypeError, "FileFormat"),
        ({"file": "x.unknown"}, ValueError, "extension"),
        ({"start": 12345}, TypeError, "start must be"),
        ({"start": "yesterday"}, ValueError, "ISO 8601"),
        ({"start": np.datetime64("NaT", "s")}, ValueError, "NaT"),
        ({"start": "2026-01-01", "end": "2026-01-01"}, ValueError, "too short"),
        ({"n_points": 10**6, "start": "2026-01-01T00:00:00", "end": "2026-01-01T00:00:00.0001"},
         ValueError, "too short"),
    ],
)  # fmt: skip
def test_errors(kwargs: dict[str, object], error: type[Exception], match: str) -> None:
    args = {"n_points": 100, "n_change_points": 2, **BASE, **kwargs}
    with pytest.raises(error, match=match):
        razladka.generate(**args)  # type: ignore[arg-type]


def test_end_defaults_to_now() -> None:
    before = np.datetime64(dt.datetime.now(dt.UTC).replace(tzinfo=None), "us")
    s = razladka.generate(10, 1, start="2020-01-01", lower=0, upper=1, seed=0)
    after = np.datetime64(dt.datetime.now(dt.UTC).replace(tzinfo=None), "us")
    assert before <= s.times[-1].astype("datetime64[us]") <= after + np.timedelta64(1, "us")


def test_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "polars":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(__import__("sys").modules, "polars", raising=False)
    with pytest.raises(ImportError, match=r'pip install "razladka\[polars\]"'):
        make(output=OutputType.POLARS)
