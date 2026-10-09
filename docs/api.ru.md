# Справочник API razladka

[English](https://github.com/89605502155/razladka/blob/main/docs/api.md) ·
[Руководство](https://github.com/89605502155/razladka/blob/main/docs/guide.ru.md)

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

| Параметр | Смысл |
|---|---|
| `n_points` | Число точек, `≥ 2 · (n_change_points + 1)`. |
| `n_change_points` | Число разладок `K ≥ 0`; где они встанут, решает случай. |
| `start` | Первая граница шкалы времени: `datetime`, `date`, строка ISO 8601, `numpy.datetime64` или `pandas.Timestamp`. Наивное время — UTC. |
| `end` | Вторая граница; `None` — текущий момент (UTC). Может быть раньше `start`. |
| `lower`, `upper` | Все значения лежат в `[lower, upper]`; `upper > lower`. |
| `roughness` | Коэффициент негладкости `s = D − 1 ∈ [0, 1]` (фрактальная размерность `D` негладкой части). |
| `roughness_scale` | Амплитуда негладкой части при `s = 1` в долях `upper − lower`; `s · roughness_scale < 0.5`. |
| `last_trend` | Направление самого позднего участка; `None` — случайно. Направления чередуются. |
| `output` | Что вернуть, см. `OutputType`. |
| `file` | Заодно записать ряд в этот файл (с перезаписью). |
| `file_format` | Формат `file`; `None` — по расширению. Без `file` — ошибка. |
| `seed` | Зерно или генератор; одинаковое зерно — одинаковый ряд. |

Ошибки: `TypeError` — аргумент не того типа; `ValueError` — значение вне
допустимого, неизвестное расширение, слишком короткий интервал для строго
возрастающих меток, `file_format` без `file`; `ImportError` с названием
нужного дополнения — если нет необязательной зависимости.

## `GeneratedSeries`

Неизменяемый dataclass, возвращается при `OutputType.RESULT`.

| Атрибут | Тип | Смысл |
|---|---|---|
| `times` | `ndarray[datetime64[unit]]` | Строго возрастающие метки UTC. |
| `values` | `ndarray[float64]` | Ряд. |
| `segments` | `ndarray[int64]` | Номер участка `0 … K` каждой точки (вычисляется при обращении). |
| `change_points` | `ndarray[int64]` | Индекс первой точки участков `1 … K`. |
| `change_times` | `ndarray[datetime64[unit]]` | `times[change_points]`. |
| `trends` | `tuple[Trend, ...]` | Направление каждого из `K + 1` участков. |
| `roughness` | `float` | Использованный `roughness`. |
| `unit` | `str` | `"ns"`, `"us"`, `"ms"` или `"s"`. |
| `n_change_points` | `int` | `K`. |

Методы:

* `len(s)` — число точек.
* `labels() -> dict` — истинная разметка в виде значений JSON
  (`n_points`, `change_points`, `change_times` строками ISO, `trends`,
  `roughness`, `unit`).
* `to_numpy() -> (times, values)`.
* `to_pandas() -> pandas.DataFrame` — столбцы `time` (UTC с часовым поясом),
  `value`, `segment`; в `df.attrs` — `labels()`.
* `to_polars() -> polars.DataFrame` — те же столбцы; у Polars нет метаданных
  таблицы. Для единицы `"s"` — `ValueError` (за пределами миллисекунд Polars).
* `to_arrow() -> pyarrow.Table` — те же столбцы, `time` типа
  `timestamp[unit, tz=UTC]`; `labels()` в JSON в метаданных схемы под ключом
  `razladka`.
* `save(path, file_format=None)` — записать файл; формат по расширению, если
  не задан.

## `OutputType`

`RESULT`, `NUMPY`, `PANDAS`, `POLARS`, `ARROW`.

## `FileFormat`

| Значение | Расширения | Нужно |
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

`FileFormat.from_path(path)` выбирает формат по расширению (без учёта
регистра); `fmt.extension` — обычное расширение.

## `Trend`

`UP`, `DOWN`; `trend.sign` — `+1` / `-1`, `trend.opposite` — противоположное.
