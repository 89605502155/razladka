"""Writers of the supported file formats.

Every file holds the time, the value and the segment number of each point;
the ground truth (change points, their times, the trends) is stored next to
the data in the way native to the format. Writers that can stream do so in
chunks to keep memory flat for long series.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from ._deps import require
from ._enums import FileFormat
from ._result import METADATA_KEY, segment_numbers
from ._time import format_ns

if TYPE_CHECKING:
    from ._result import GeneratedSeries

EXCEL_MAX_POINTS = 1_048_575
"""Rows of an Excel sheet minus the header row."""

TROSNA_DEVICE = "series"
TSFILE_TABLE_PREFIX = "series_"

_CHUNK = 1 << 16
_CF_UNITS = {"ns": "nanoseconds", "us": "microseconds", "ms": "milliseconds", "s": "seconds"}


def save(
    series: GeneratedSeries, path: str | os.PathLike[str], file_format: FileFormat | None
) -> None:
    """Writes ``series`` to ``path`` in ``file_format`` (by extension if ``None``)."""
    fmt = FileFormat.from_path(path) if file_format is None else file_format
    if not isinstance(fmt, FileFormat):
        msg = f"file_format must be a FileFormat or None, not {type(fmt).__name__}"
        raise TypeError(msg)
    _WRITERS[fmt](series, os.fspath(path))


def _chunks(n: int, size: int = _CHUNK) -> Iterator[tuple[int, int]]:
    for a in range(0, n, size):
        yield a, min(n, a + size)


def _iso_times(series: GeneratedSeries, a: int, b: int) -> np.ndarray:
    unit: Any = series.unit
    return np.char.add(np.datetime_as_string(series.times[a:b], unit=unit), "Z")


def _write_csv(series: GeneratedSeries, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("time,value,segment\n")
        for a, b in _chunks(len(series)):
            times = _iso_times(series, a, b)
            segments = segment_numbers(series.change_points, a, b)
            f.writelines(
                f"{t},{v!r},{s}\n"
                for t, v, s in zip(
                    times.tolist(), series.values[a:b].tolist(), segments.tolist(), strict=True
                )
            )


def _write_json(series: GeneratedSeries, path: str) -> None:
    labels = series.labels()
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key, value in labels.items():
            f.write(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n")

        def stream(name: str, piece: Callable[[int, int], list[object]], last: bool) -> None:
            f.write(f'  "{name}": [')
            sep = ""
            for a, b in _chunks(len(series)):
                f.write(sep + ", ".join(json.dumps(x) for x in piece(a, b)))
                sep = ", "
            f.write("]\n" if last else "],\n")

        stream("time", lambda a, b: _iso_times(series, a, b).tolist(), last=False)
        stream("value", lambda a, b: series.values[a:b].tolist(), last=False)
        stream(
            "segment",
            lambda a, b: segment_numbers(series.change_points, a, b).tolist(),
            last=True,
        )
        f.write("}\n")


def _write_parquet(series: GeneratedSeries, path: str) -> None:
    pq = require("pyarrow.parquet", "arrow")
    pq.write_table(series.to_arrow(), path)


def _write_feather(series: GeneratedSeries, path: str) -> None:
    feather = require("pyarrow.feather", "arrow")
    feather.write_feather(series.to_arrow(), path)


def _write_arrow_ipc(series: GeneratedSeries, path: str) -> None:
    pa = require("pyarrow", "arrow")
    table = series.to_arrow()
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


def _write_hdf5(series: GeneratedSeries, path: str) -> None:
    h5py = require("h5py", "hdf5")
    labels = series.labels()
    n = len(series)
    with h5py.File(path, "w") as f:
        time = f.create_dataset("time", shape=(n,), dtype="i8")
        value = f.create_dataset("value", shape=(n,), dtype="f8")
        segment = f.create_dataset("segment", shape=(n,), dtype="i8")
        for a, b in _chunks(n, 1 << 20):
            time[a:b] = series.times[a:b].astype(np.int64)
            value[a:b] = series.values[a:b]
            segment[a:b] = segment_numbers(series.change_points, a, b)
        time.attrs["unit"] = series.unit
        time.attrs["epoch"] = "1970-01-01T00:00:00Z"
        f.attrs["change_points"] = np.asarray(series.change_points, dtype=np.int64)
        f.attrs["change_times"] = json.dumps(labels["change_times"])
        f.attrs["trends"] = json.dumps(labels["trends"])
        f.attrs["roughness"] = float(series.roughness)
        f.attrs["unit"] = series.unit
        f.attrs[METADATA_KEY] = json.dumps(labels)


def _write_netcdf(series: GeneratedSeries, path: str) -> None:
    h5netcdf = require("h5netcdf", "netcdf")
    labels = series.labels()
    n = len(series)
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": n}
        time = f.create_variable("time", ("time",), "i8")
        value = f.create_variable("value", ("time",), "f8")
        segment = f.create_variable("segment", ("time",), "i8")
        for a, b in _chunks(n, 1 << 20):
            time[a:b] = series.times[a:b].astype(np.int64)
            value[a:b] = series.values[a:b]
            segment[a:b] = segment_numbers(series.change_points, a, b)
        time.attrs["units"] = f"{_CF_UNITS[series.unit]} since 1970-01-01T00:00:00Z"
        time.attrs["calendar"] = "proleptic_gregorian"
        time.attrs["standard_name"] = "time"
        segment.attrs["long_name"] = "segment number (ground truth)"
        f.attrs["change_points"] = np.asarray(series.change_points, dtype=np.int64)
        f.attrs["change_times"] = json.dumps(labels["change_times"])
        f.attrs["trends"] = json.dumps(labels["trends"])
        f.attrs["roughness"] = float(series.roughness)
        f.attrs["unit"] = series.unit
        f.attrs[METADATA_KEY] = json.dumps(labels)


def _write_npz(series: GeneratedSeries, path: str) -> None:
    with open(path, "wb") as f:
        np.savez(
            f,
            time=series.times.astype(np.int64),
            value=series.values,
            segment=series.segments,
            change_points=series.change_points,
            trends=np.array([t.value for t in series.trends]),
            roughness=np.float64(series.roughness),
            unit=np.array(series.unit),
        )


def _write_excel(series: GeneratedSeries, path: str) -> None:
    n = len(series)
    if n > EXCEL_MAX_POINTS:
        msg = (
            f"an Excel sheet holds at most {EXCEL_MAX_POINTS} points, the series has {n}; "
            "use CSV, Parquet or another format"
        )
        raise ValueError(msg)
    openpyxl = require("openpyxl", "excel")
    labels = series.labels()
    book = openpyxl.Workbook(write_only=True)
    sheet = book.create_sheet("series")
    sheet.append(["time", "value", "segment"])
    for a, b in _chunks(n):
        times = _iso_times(series, a, b).tolist()
        segments = segment_numbers(series.change_points, a, b).tolist()
        for row in zip(times, series.values[a:b].tolist(), segments, strict=True):
            sheet.append(list(row))
    truth = book.create_sheet("change_points")
    truth.append(["segment", "first_index", "first_time", "trend"])
    firsts = [0, *labels["change_points"]]
    first_times = [format_ns(int(series.times[0].astype(np.int64)), series.unit)]
    first_times += labels["change_times"]
    for k, (i, t, trend) in enumerate(zip(firsts, first_times, labels["trends"], strict=True)):
        truth.append([k, i, t, trend])
    info = book.create_sheet("info")
    info.append(["key", "value"])
    info.append(["unit", series.unit])
    info.append(["roughness", float(series.roughness)])
    info.append([METADATA_KEY, json.dumps(labels)])
    book.save(path)


def _write_trosna(series: GeneratedSeries, path: str) -> None:
    pytrosna = require("pytrosna", "trosna")
    labels = series.labels()
    n = len(series)
    stamps = series.times.astype(np.int64)
    bounds = [0, *labels["change_points"], n]
    with pytrosna.Writer.create(path, overwrite=True) as w:
        w.create_device(
            pytrosna.DeviceSchema.build(
                TROSNA_DEVICE, series.unit, {"value": "float64", "segment": "int64"}
            )
        )
        for a, b in _chunks(n, 1 << 20):
            w.write(
                TROSNA_DEVICE,
                {
                    "time": stamps[a:b],
                    "value": series.values[a:b],
                    "segment": segment_numbers(series.change_points, a, b),
                },
            )
        for k, trend in enumerate(labels["trends"]):
            w.annotate(
                TROSNA_DEVICE,
                int(stamps[bounds[k]]),
                int(stamps[bounds[k + 1] - 1]),
                trend,
                f"segment {k}",
            )
        w.update_metadata(
            {
                f"{METADATA_KEY}.change_points": json.dumps(labels["change_points"]),
                f"{METADATA_KEY}.change_times": json.dumps(labels["change_times"]),
                f"{METADATA_KEY}.trends": json.dumps(labels["trends"]),
                f"{METADATA_KEY}.roughness": repr(float(series.roughness)),
                f"{METADATA_KEY}.unit": series.unit,
            }
        )
        w.commit(message="razladka.generate")


def _write_tsfile(series: GeneratedSeries, path: str) -> None:
    tsfile = require("tsfile", "tsfile")
    pd = require("pandas", "tsfile")
    if os.path.exists(path):
        os.remove(path)  # the TsFile writer refuses to overwrite
    schema = tsfile.TableSchema(
        TSFILE_TABLE_PREFIX + series.unit,
        [
            tsfile.ColumnSchema("segment", tsfile.TSDataType.STRING, tsfile.ColumnCategory.TAG),
            tsfile.ColumnSchema("value", tsfile.TSDataType.DOUBLE, tsfile.ColumnCategory.FIELD),
        ],
    )
    writer = tsfile.TsFileTableWriter(path, schema)
    try:
        for a, b in _chunks(len(series)):
            writer.write_dataframe(
                pd.DataFrame(
                    {
                        "time": series.times[a:b].astype(np.int64),
                        "segment": segment_numbers(series.change_points, a, b).astype(str),
                        "value": series.values[a:b],
                    }
                )
            )
    finally:
        writer.close()


_WRITERS: dict[FileFormat, Callable[[GeneratedSeries, str], None]] = {
    FileFormat.CSV: _write_csv,
    FileFormat.JSON: _write_json,
    FileFormat.PARQUET: _write_parquet,
    FileFormat.FEATHER: _write_feather,
    FileFormat.ARROW: _write_arrow_ipc,
    FileFormat.HDF5: _write_hdf5,
    FileFormat.NETCDF: _write_netcdf,
    FileFormat.NPZ: _write_npz,
    FileFormat.EXCEL: _write_excel,
    FileFormat.TROSNA: _write_trosna,
    FileFormat.TSFILE: _write_tsfile,
}
