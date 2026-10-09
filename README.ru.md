# razladka

[![PyPI](https://img.shields.io/pypi/v/razladka.svg)](https://pypi.org/project/razladka/)
[![Python](https://img.shields.io/pypi/pyversions/razladka.svg)](https://pypi.org/project/razladka/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/89605502155/razladka/blob/main/LICENSE)

**Синтетические временные ряды с заданным числом точек разладки и истинной разметкой.**

Вы говорите, **сколько** разладок должно быть в ряде. **Где** они встанут,
решает цепь Маркова с переключением режимов. Истинные моменты возвращаются
вместе с рядом, так что по ним можно оценить детектор разладки или
кластеризатор.

[English](https://github.com/89605502155/razladka/blob/main/README.md) ·
[Руководство](https://github.com/89605502155/razladka/blob/main/docs/guide.ru.md) ·
[Справочник API](https://github.com/89605502155/razladka/blob/main/docs/api.ru.md) ·
[Примеры](https://github.com/89605502155/razladka/tree/main/examples) ·
[Изменения](https://github.com/89605502155/razladka/blob/main/CHANGELOG.md)

## Установка

```bash
pip install razladka            # только NumPy
pip install "razladka[all]"     # + pandas, Polars, PyArrow, h5py, h5netcdf, openpyxl, pytrosna, tsfile
```

Дополнения по отдельности: `pandas`, `polars`, `arrow` (Parquet, Feather,
Arrow IPC), `hdf5`, `netcdf`, `excel`, `trosna`, `tsfile`. Если для функции
не хватает пакета, в сообщении об ошибке сказано, какое дополнение поставить.

## Быстрый старт

```python
import razladka
from razladka import Trend

s = razladka.generate(
    1_000,  # точек
    3,  # разладок
    start="2026-01-01",
    end="2026-03-01",
    lower=0.0,
    upper=100.0,
    roughness=0.4,  # 0 — гладкий внутри участков, 1 — нигде не гладкий
    last_trend=Trend.UP,
    seed=42,
)

print(s.change_points)  # индексы первых точек участков 1..3
print(s.change_times)  # их моменты времени
print(s.trends)  # направление каждого из 4 участков, последний — UP
assert s.values.min() >= 0.0 and s.values.max() <= 100.0
```

Можно сразу получить таблицу или заодно записать файл:

```python
import razladka
from razladka import FileFormat, OutputType

df = razladka.generate(
    500, 2, start="2026-01-01", lower=-1, upper=1, output=OutputType.PANDAS, seed=1
)
print(df.head())  # столбцы: time (UTC), value, segment
print(df.attrs["change_points"])  # истинная разметка

razladka.generate(
    500,
    2,
    start="2026-01-01",
    lower=-1,
    upper=1,
    seed=1,
    file="series.parquet",  # формат по расширению…
)
razladka.generate(
    500,
    2,
    start="2026-01-01",
    lower=-1,
    upper=1,
    seed=1,
    file="series.data",
    file_format=FileFormat.CSV,  # …или явно
)
```

## Как строится ряд

1. **Шкала времени.** `n_points` равноотстоящих меток от `start` до `end`
   (по умолчанию «сейчас»). Единица — самая мелкая из нс, мкс, мс, с, в
   которую помещаются обе границы: 10⁸ точек за сутки — наносекунды, ряд от
   5 года н. э. до сегодня — микросекунды.
2. **Разладки.** Цепь с `K + 1` режимами и левосторонней матрицей
   переходов: из режима *j* — только в *j* или *j + 1*, последний
   поглощающий, поэтому разладок ровно `K`. Каждый режим длится не меньше
   2 точек, в среднем `n / (K + 1)`.
3. **Гладкий каркас.** У каждого режима дрейф случайной величины,
   направления чередуются, самое свежее равно `last_trend`. Накопленная сумма —
   непрерывная кусочно-линейная кривая: разладка — излом, а не скачок. Каркас
   растягивается на `[lower + A, upper − A]`.
4. **Негладкость.** Добавляется сумма Вейерштрасса — Мандельброта с
   фрактальной размерностью `D = 1 + roughness` и амплитудой
   `A = roughness · roughness_scale · (upper − lower)`. Все значения остаются
   в `[lower, upper]`.

Подробности и формулы — в [руководстве](https://github.com/89605502155/razladka/blob/main/docs/guide.ru.md).

## Результат и файлы

| `output=` | Что возвращается |
|---|---|
| `OutputType.RESULT` (по умолчанию) | `GeneratedSeries`: `times`, `values`, `segments`, `change_points`, `change_times`, `trends`, `.to_pandas()`, `.to_polars()`, `.to_arrow()`, `.save()` |
| `OutputType.NUMPY` | `(times, values)` |
| `OutputType.PANDAS` | `DataFrame` (`time`, `value`, `segment`), разметка в `df.attrs` |
| `OutputType.POLARS` | `polars.DataFrame` (`time`, `value`, `segment`) |
| `OutputType.ARROW` | `pyarrow.Table`, разметка в метаданных схемы (`razladka`) |

| `FileFormat` | Расширение | Разметка |
|---|---|---|
| `CSV` | `.csv` | столбец `segment` |
| `JSON` | `.json` | поля `change_points`, `change_times`, `trends` |
| `PARQUET`, `FEATHER`, `ARROW` | `.parquet`, `.feather`, `.arrow` | `segment` + метаданные схемы `razladka` |
| `HDF5` | `.h5` | датасеты `time` (атрибут `unit`), `value`, `segment`; атрибуты файла |
| `NETCDF` | `.nc` | переменная `time` по CF, глобальные атрибуты |
| `NPZ` | `.npz` | массивы `change_points`, `trends`, `unit` |
| `EXCEL` | `.xlsx` | листы `series`, `change_points`, `info` (≤ 1 048 575 точек) |
| `TROSNA` | `.trosna` | аннотация на каждый участок (`up` / `down`) + метаданные `razladka.*` |
| `TSFILE` | `.tsfile` | `segment` — тег (по устройству на участок) |

Файлы [Trosna](https://github.com/89605502155/trosna-file) пишутся через
[pytrosna](https://github.com/89605502155/pytrosna) и в тестах сверяются с
эталонной реализацией на Rust.

## Зачем

В обзоре Ферубко и Казакова (2026) «Обзор методов генерации синтетических
временных рядов с заданными точками разладки» показано, что переключение
режимов по цепи Маркова естественно расставляет разладки. Но на 9 октября
2026 года поддерживаемого пакета на Python, который генерирует такие ряды, не
было: statsmodels их только оценивает. razladka закрывает этот пробел.

## Лицензия

[MIT](https://github.com/89605502155/razladka/blob/main/LICENSE) © Andrey Ferubko
