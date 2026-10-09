# razladka

[![PyPI](https://img.shields.io/pypi/v/razladka.svg)](https://pypi.org/project/razladka/)
[![Python](https://img.shields.io/pypi/pyversions/razladka.svg)](https://pypi.org/project/razladka/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/89605502155/razladka/blob/main/LICENSE)

**Synthetic time series with a given number of change points, with ground truth.**

*Razladka* (Russian «разладка») means a *change point*: the moment a process
switches to a different regime. You tell razladka **how many** change points
the series must have; a Markov regime-switching chain decides **where** they
are. The true positions come back with the series, so a change-point detector
or a clustering method can be scored against them.

[Русская версия](https://github.com/89605502155/razladka/blob/main/README.ru.md) ·
[Guide](https://github.com/89605502155/razladka/blob/main/docs/guide.md) ·
[API reference](https://github.com/89605502155/razladka/blob/main/docs/api.md) ·
[Examples](https://github.com/89605502155/razladka/tree/main/examples) ·
[Changelog](https://github.com/89605502155/razladka/blob/main/CHANGELOG.md)

## Install

```bash
pip install razladka            # NumPy only
pip install "razladka[all]"     # + pandas, Polars, PyArrow, h5py, h5netcdf, openpyxl, pytrosna, tsfile
```

Extras one by one: `pandas`, `polars`, `arrow` (Parquet, Feather, Arrow IPC),
`hdf5`, `netcdf`, `excel`, `trosna`, `tsfile`. If a feature needs a missing
package, the error message tells you which extra to install.

## Quick start

```python
import razladka
from razladka import Trend

s = razladka.generate(
    1_000,  # points
    3,  # change points
    start="2026-01-01",
    end="2026-03-01",
    lower=0.0,
    upper=100.0,
    roughness=0.4,  # 0 = smooth inside segments, 1 = nowhere smooth
    last_trend=Trend.UP,
    seed=42,
)

print(s.change_points)  # indices of the first points of segments 1..3
print(s.change_times)  # their time stamps
print(s.trends)  # direction of each of the 4 segments, the last is UP
assert s.values.min() >= 0.0 and s.values.max() <= 100.0
```

Get a table instead, or write a file at the same time:

```python
import razladka
from razladka import FileFormat, OutputType

df = razladka.generate(
    500, 2, start="2026-01-01", lower=-1, upper=1, output=OutputType.PANDAS, seed=1
)
print(df.head())  # columns: time (UTC), value, segment
print(df.attrs["change_points"])  # the ground truth

razladka.generate(
    500,
    2,
    start="2026-01-01",
    lower=-1,
    upper=1,
    seed=1,
    file="series.parquet",  # format from the extension…
)
razladka.generate(
    500,
    2,
    start="2026-01-01",
    lower=-1,
    upper=1,
    seed=1,
    file="series.data",
    file_format=FileFormat.CSV,  # …or explicit
)
```

## How the series is built

1. **Time axis.** `n_points` evenly spaced stamps from `start` to `end`
   (now by default). The unit is the finest of ns, µs, ms, s that holds both
   bounds: 10⁸ points in one day use nanoseconds, a series from the year 5
   to today uses microseconds.
2. **Change points.** A chain with `K + 1` regimes and a left-to-right
   transition matrix: from regime *j* it stays or moves to *j + 1*, the last
   regime is absorbing, so there are exactly `K` changes. Each regime lasts at
   least 2 points, on average `n / (K + 1)`.
3. **Smooth skeleton.** Each regime has a drift of random size; directions
   alternate and the most recent one is `last_trend`. The cumulative sum is a
   continuous piecewise-linear curve: a change point is a kink, not a jump.
   It is scaled to fill `[lower + A, upper − A]`.
4. **Roughness.** A Weierstrass–Mandelbrot sum of fractal dimension
   `D = 1 + roughness` with amplitude `A = roughness · roughness_scale ·
   (upper − lower)` is added. Every value stays within `[lower, upper]`.

Details and formulas: [Guide](https://github.com/89605502155/razladka/blob/main/docs/guide.md).

## Output and files

| `output=` | You get |
|---|---|
| `OutputType.RESULT` (default) | `GeneratedSeries`: `times`, `values`, `segments`, `change_points`, `change_times`, `trends`, `.to_pandas()`, `.to_polars()`, `.to_arrow()`, `.save()` |
| `OutputType.NUMPY` | `(times, values)` |
| `OutputType.PANDAS` | `DataFrame` (`time`, `value`, `segment`), ground truth in `df.attrs` |
| `OutputType.POLARS` | `polars.DataFrame` (`time`, `value`, `segment`) |
| `OutputType.ARROW` | `pyarrow.Table`, ground truth in the schema metadata (`razladka`) |

| `FileFormat` | Extension | Ground truth |
|---|---|---|
| `CSV` | `.csv` | `segment` column |
| `JSON` | `.json` | `change_points`, `change_times`, `trends` fields |
| `PARQUET`, `FEATHER`, `ARROW` | `.parquet`, `.feather`, `.arrow` | `segment` + schema metadata `razladka` |
| `HDF5` | `.h5` | datasets `time` (with `unit`), `value`, `segment`; file attributes |
| `NETCDF` | `.nc` | CF `time` variable, global attributes |
| `NPZ` | `.npz` | arrays `change_points`, `trends`, `unit` |
| `EXCEL` | `.xlsx` | sheets `series`, `change_points`, `info` (≤ 1 048 575 points) |
| `TROSNA` | `.trosna` | one annotation per segment (`up` / `down`) + `razladka.*` metadata |
| `TSFILE` | `.tsfile` | `segment` as a tag column (one device per segment) |

[Trosna](https://github.com/89605502155/trosna-file) files are written with
[pytrosna](https://github.com/89605502155/pytrosna) and are verified in the
test suite against the reference Rust implementation.

## Why

Ferubko & Kazakov (2026), *A review of methods for generating synthetic time
series with given change points*, found that Markov regime switching places
change points naturally, yet on 9 October 2026 no maintained Python package
generated such series (statsmodels only fits them). razladka fills that gap.

## License

[MIT](https://github.com/89605502155/razladka/blob/main/LICENSE) © Andrey Ferubko
