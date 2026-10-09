# razladka API reference

[Русская версия](https://github.com/89605502155/razladka/blob/main/docs/api.ru.md) ·
[Guide](https://github.com/89605502155/razladka/blob/main/docs/guide.md)

```python
from razladka import generate, GeneratedSeries, FileFormat, OutputType, Trend
```

## `generate`

```python
generate(
    n_points: int,
    n_change_points: int,
    *,
    start,
    end=None,
    lower: float,
    upper: float,
    roughness: float = 0.0,
    roughness_scale: float = 0.1,
    last_trend: Trend | None = None,
    output: OutputType = OutputType.RESULT,
    file: str | os.PathLike | None = None,
    file_format: FileFormat | None = None,
    seed: int | numpy.random.Generator | None = None,
)
```

| Parameter | Meaning |
|---|---|
| `n_points` | Number of points, `≥ 2 · (n_change_points + 1)`. |
| `n_change_points` | Number of change points `K ≥ 0`; their positions are random. |
| `start` | First bound of the time axis: `datetime`, `date`, ISO 8601 string, `numpy.datetime64` or `pandas.Timestamp`. Naive = UTC. |
| `end` | Second bound; `None` = now (UTC). May be earlier than `start`. |
| `lower`, `upper` | All values lie in `[lower, upper]`; `upper > lower`. |
| `roughness` | Roughness coefficient `s = D − 1 ∈ [0, 1]` (fractal dimension `D` of the rough part). |
| `roughness_scale` | Amplitude of the rough part at `s = 1`, as a fraction of `upper − lower`; `s · roughness_scale < 0.5`. |
| `last_trend` | Direction of the latest segment; `None` = random. Directions alternate. |
| `output` | Return type, see `OutputType`. |
| `file` | Also write the series to this file (overwrites). |
| `file_format` | Format of `file`; `None` = by extension. Error without `file`. |
| `seed` | Seed or generator; same seed → same series. |

Errors: `TypeError` for arguments of a wrong type, `ValueError` for values
out of range, an unknown extension, an interval too short for strictly
increasing stamps, or `file_format` without `file`; `ImportError` (naming the
extra to install) if an optional dependency is missing.

## `GeneratedSeries`

Frozen dataclass returned for `OutputType.RESULT`.

| Attribute | Type | Meaning |
|---|---|---|
| `times` | `ndarray[datetime64[unit]]` | Strictly increasing UTC stamps. |
| `values` | `ndarray[float64]` | The series. |
| `segments` | `ndarray[int64]` | Segment number `0 … K` of every point (computed on access). |
| `change_points` | `ndarray[int64]` | Index of the first point of segments `1 … K`. |
| `change_times` | `ndarray[datetime64[unit]]` | `times[change_points]`. |
| `trends` | `tuple[Trend, ...]` | Direction of each of the `K + 1` segments. |
| `roughness` | `float` | The `roughness` used. |
| `unit` | `str` | `"ns"`, `"us"`, `"ms"` or `"s"`. |
| `n_change_points` | `int` | `K`. |

Methods:

* `len(s)` — number of points.
* `labels() -> dict` — the ground truth as JSON-compatible values
  (`n_points`, `change_points`, `change_times` as ISO strings, `trends`,
  `roughness`, `unit`).
* `to_numpy() -> (times, values)`.
* `to_pandas() -> pandas.DataFrame` — columns `time` (tz-aware UTC),
  `value`, `segment`; `df.attrs` holds `labels()`.
* `to_polars() -> polars.DataFrame` — the same columns; Polars has no table
  metadata. `ValueError` for unit `"s"` (beyond Polars' millisecond range).
* `to_arrow() -> pyarrow.Table` — the same columns, `time` is
  `timestamp[unit, tz=UTC]`; `labels()` as JSON in the schema metadata key
  `razladka`.
* `save(path, file_format=None)` — write a file, format by extension unless
  given.

## `OutputType`

`RESULT`, `NUMPY`, `PANDAS`, `POLARS`, `ARROW`.

## `FileFormat`

| Member | Extension(s) | Needs |
|---|---|---|
| `CSV` | `.csv` | — |
| `JSON` | `.json` | — |
| `PARQUET` | `.parquet` | `razladka[arrow]` |
| `FEATHER` | `.feather` | `razladka[arrow]` |
| `ARROW` | `.arrow`, `.ipc` | `razladka[arrow]` |
| `HDF5` | `.h5`, `.hdf5`, `.hdf` | `razladka[hdf5]` |
| `NETCDF` | `.nc`, `.nc4` | `razladka[netcdf]` |
| `NPZ` | `.npz` | — |
| `EXCEL` | `.xlsx` | `razladka[excel]` |
| `TROSNA` | `.trosna` | `razladka[trosna]` |
| `TSFILE` | `.tsfile`, `.tsf` | `razladka[tsfile]` |

`FileFormat.from_path(path)` picks the format by extension
(case-insensitive); `fmt.extension` gives the usual one.

## `Trend`

`UP`, `DOWN`; `trend.sign` is `+1` / `-1`, `trend.opposite` the other one.
