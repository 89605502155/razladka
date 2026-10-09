"""Enumerations of the public API."""

from __future__ import annotations

import enum
import os


class Trend(enum.Enum):
    """Direction of the smooth skeleton on a segment."""

    UP = "up"
    DOWN = "down"

    @property
    def sign(self) -> int:
        """``+1`` for :attr:`UP`, ``-1`` for :attr:`DOWN`."""
        return 1 if self is Trend.UP else -1

    @property
    def opposite(self) -> Trend:
        """The other direction."""
        return Trend.DOWN if self is Trend.UP else Trend.UP


class OutputType(enum.Enum):
    """What :func:`razladka.generate` returns."""

    RESULT = "result"
    """A :class:`razladka.GeneratedSeries` with the series and its ground truth."""
    NUMPY = "numpy"
    """A tuple ``(times, values)`` of NumPy arrays."""
    PANDAS = "pandas"
    """A :class:`pandas.DataFrame` with the columns ``time``, ``value``, ``segment``."""
    POLARS = "polars"
    """A :class:`polars.DataFrame` with the columns ``time``, ``value``, ``segment``."""
    ARROW = "arrow"
    """A :class:`pyarrow.Table` with the columns ``time``, ``value``, ``segment``."""


class FileFormat(enum.Enum):
    """File format; the value is the usual file extension."""

    CSV = ".csv"
    JSON = ".json"
    PARQUET = ".parquet"
    FEATHER = ".feather"
    ARROW = ".arrow"
    HDF5 = ".h5"
    NETCDF = ".nc"
    NPZ = ".npz"
    EXCEL = ".xlsx"
    TROSNA = ".trosna"
    TSFILE = ".tsfile"

    @property
    def extension(self) -> str:
        """The usual file extension, with the dot."""
        return self.value

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> FileFormat:
        """Picks the format by the file extension (case-insensitive).

        >>> FileFormat.from_path("data/run1.parquet")
        <FileFormat.PARQUET: '.parquet'>
        """
        ext = os.path.splitext(os.fspath(path))[1].lower()
        fmt = _EXTENSIONS.get(ext)
        if fmt is None:
            known = ", ".join(sorted(_EXTENSIONS))
            msg = (
                f"cannot tell the file format from the extension {ext!r} of {os.fspath(path)!r}; "
                f"pass file_format=FileFormat.<...> or use one of: {known}"
            )
            raise ValueError(msg)
        return fmt


_EXTENSIONS: dict[str, FileFormat] = {f.value: f for f in FileFormat} | {
    ".hdf5": FileFormat.HDF5,
    ".hdf": FileFormat.HDF5,
    ".nc4": FileFormat.NETCDF,
    ".ipc": FileFormat.ARROW,
    ".tsf": FileFormat.TSFILE,
}
